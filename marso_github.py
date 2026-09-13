"""GitHub-backed orchestration for the existing T4 next-pick model."""
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

from marso_next_pick import NextPickExperiment
from marso_experiment import save_json
from github_store import GitHubStore
from github_data import download_archive


def atomic_checkpoint_patch(source):
    anchor = '''    torch.save({
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')'''
    if source.count(anchor) != 1:
        raise ValueError('Official checkpoint writer changed; refusing an unverified patch.')
    replacement = '''    checkpoint_path = f'runs/{run_name}/checkpoints/{tag}.pt'
    torch.save({
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, checkpoint_path+'.tmp')
    os.replace(checkpoint_path+'.tmp', checkpoint_path)
    from github_store import sync_from_env
    sync_from_env()'''
    patched = source.replace(anchor, replacement)
    compile(patched, 'train_next_pick.py', 'exec')
    return patched


class GitHubExperiment(NextPickExperiment):
    def __init__(self, config, sources):
        super().__init__(config, sources)
        self.project = Path(config['project_dir'])
        self.store = None
        self.remote_enabled = config['profile'] != 'smoke'

    def connect(self):
        if self.remote_enabled:
            tag = 'run-'+self.run_dir.name
            self.store = GitHubStore(self.cfg['github_repository'], tag)
            # Verify access before a long GPU run. A new release is safe to create.
            self.store.load()
            if not any(self.run_dir.glob('*')):
                count = self.store.restore(self.run_dir)
                print(f'GitHub에서 복원한 결과: {count} files')
            else:
                print('현재 런타임의 로컬 결과를 유지합니다. 원격 복원은 빈 출력 폴더에서 수행합니다.')
        super().connect()
        if self.remote_enabled:
            self.sync_common()
            print('GitHub 결과:', f'https://github.com/{self.store.repository}/releases/tag/{self.store.tag}')
        else:
            print('100 iteration 동작 확인은 로컬 checks에만 저장합니다. 전체 학습 결과는 GitHub에 저장됩니다.')

    def notebook_snapshot(self):
        from build_github_notebook import make_notebook
        config = dict(self.cfg)
        config['project_ref'] = config.get('project_commit', config['project_ref'])
        return make_notebook(config)

    def prepare_data(self):
        self._ready()
        source = download_archive(self.project/'data/manifest.json', 'state', self.cfg['download_cache'])
        self.cfg['data_source'] = str(source)
        super().prepare_data()
        self.sync_common()

    def _stage_helpers(self):
        super()._stage_helpers()
        (self.repo/'github_store.py').write_text(self.sources['github_store.py'], encoding='utf-8')

    def training_script(self, level, flags):
        script = super().training_script(level, flags)
        source = atomic_checkpoint_patch(script.read_text(encoding='utf-8'))
        script.write_text(source, encoding='utf-8')
        (self.run_dir/level/'train_next_pick.py').write_text(source, encoding='utf-8')
        (self.base/'github_store.py').write_text(self.sources['github_store.py'], encoding='utf-8')
        save_json(self.run_dir/level/'github_training_provenance.json', dict(
            project_commit=self.cfg.get('project_commit'),
            patched_trainer_sha256=hashlib.sha256(source.encode()).hexdigest(),
            storage_sha256=hashlib.sha256(self.sources['github_store.py'].encode()).hexdigest(),
            resume_training='Weights and EMA only; optimizer/scheduler state is not saved.'))
        return script

    @contextlib.contextmanager
    def sync_environment(self, scope):
        names = ('MOVEBOXES_SYNC_ROOT', 'MOVEBOXES_SYNC_SCOPE', 'MOVEBOXES_GITHUB_REPO', 'MOVEBOXES_RELEASE_TAG')
        previous = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ.pop(name, None)
        if self.remote_enabled and scope in ('easy', 'medium', 'hard'):
            os.environ.update(dict(zip(names, (str(self.run_dir), scope, self.store.repository, self.store.tag))))
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def run(self, command, cwd=None, log=None):
        scope = None
        if log is not None:
            try:
                scope = Path(log).relative_to(self.run_dir).parts[0]
            except ValueError:
                pass
        with self.sync_environment(scope):
            return super().run(command, cwd=cwd, log=log)

    def sync_level(self, level):
        if self.remote_enabled and self.store:
            self.store.sync(self.run_dir, level)

    def sync_common(self):
        if self.remote_enabled and self.store and self.connected:
            files = [p for p in self.run_dir.iterdir() if p.is_file() and not p.name.startswith('.')]
            files += list(self.session.glob('*'))
            self.store.sync(self.run_dir, 'common', files)

    @contextlib.contextmanager
    def persist_operation(self, level):
        try:
            yield
        except BaseException:
            # Preserve the original failure; a second upload error should not hide it.
            try:
                self.sync_level(level)
                self.sync_common()
            except Exception as error:
                print('추가 GitHub 백업 실패; 로컬 파일은 남아 있습니다:', error)
            raise
        else:
            self.sync_level(level)
            self.sync_common()

    def train(self, level):
        with self.persist_operation(level):
            return super().train(level)

    def test(self, level):
        with self.persist_operation(level):
            return super().test(level)

    def evaluate(self, level):
        with self.persist_operation(level):
            return super().evaluate(level)

    def package(self):
        result = super().package()
        self.sync_common()
        return result


def source_bundle():
    from build_next_pick_notebook import sources
    bundle = sources()
    root = Path(__file__).parent
    for name in ('github_store.py', 'github_data.py', 'marso_github.py', 'build_github_notebook.py'):
        bundle[name] = (root/name).read_text(encoding='utf-8')
    evaluator = bundle['colab_eval_modular.py']
    anchor = "                save_json(job['output'],result_for(rows,job,elapsed+time.perf_counter()-tick))"
    if evaluator.count(anchor) != 1:
        raise ValueError('Evaluator checkpoint boundary changed.')
    bundle['colab_eval_modular.py'] = evaluator.replace(anchor, anchor+
        '\n                from github_store import sync_from_env\n                sync_from_env()')
    return bundle
