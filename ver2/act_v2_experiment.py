"""Separate ver2 workflow; legacy DP notebooks and results remain available."""
import hashlib
import json
import sys
from pathlib import Path
from marso_github import GitHubExperiment, github_token, source_bundle as legacy_sources
from marso_train_test import TrainTestExperiment
from marso_experiment import LEVELS, read_json, save_json
from github_store import GitHubStore
from build_github_notebook import CONFIG as LEGACY_CONFIG


class ActV2Experiment(GitHubExperiment):
    def __init__(self, cfg, sources):
        super().__init__(dict(LEGACY_CONFIG, **cfg), sources)

    def connect(self):
        if self.connected:
            return
        c = self.cfg
        if c['width'] % c['heads'] or not 1 <= c['ensemble_window'] <= c['chunk_size']:
            raise ValueError('ACT width/heads or ensemble_window/chunk_size mismatch')
        if min(c['batch_size'],c['history'],c['chunk_size'],c['layers'],c['latent_dim'],
               c['save_freq'],c['validation_batches'],c['warmup_steps'],*c['total_iters'].values()) < 1:
            raise ValueError('Training sizes and intervals must be positive')
        if not c['ensemble_candidates'] or any(not 1 <= n <= c['chunk_size'] for n in c['ensemble_candidates']):
            raise ValueError('Invalid temporal ensemble candidates')
        if c['position_noise'] < 0 or c['temporal_decay'] < 0 or c['kl_weight'] < 0:
            raise ValueError('Noise and regularization must be nonnegative')
        if self.remote_enabled:
            github_token()
            self.store = GitHubStore(c['github_repository'], 'run-'+self.run_dir.name)
            self.store.load()
            if not any(self.run_dir.glob('*')):
                print('GitHub 복원:', self.store.restore(self.run_dir), 'files')
        try:
            TrainTestExperiment.connect(self)
            self.sync_common()
        except BaseException:
            self.connected = False
            raise
        print('ver2: State ACT / 매 스텝 재계획 / XYZ temporal ensemble / 별도 집게 분류')

    def notebook_snapshot(self):
        from build_act_v2_notebook import make_notebook
        cfg = dict(self.cfg, project_ref=self.cfg.get('project_commit', self.cfg['project_ref']))
        return make_notebook(cfg)

    def _stage_helpers(self):
        super()._stage_helpers()
        for name, code in self.sources.items():
            if name.startswith('act_v2_'):
                (self.repo/name).write_text(code, encoding='utf-8')

    def model_config(self, level):
        return dict(state_dim={'easy':54,'medium':72,'hard':90}[level],
                    **{k:self.cfg[k] for k in ('history','chunk_size','width','heads','layers','latent_dim')})

    def policy_config(self, level):
        return dict(model_config=self.model_config(level), temporal_decay=self.cfg['temporal_decay'],
                    ensemble_window=self.cfg['ensemble_window'], act_horizon=1, num_inference_steps=1)

    def train(self, level):
        self._ready()
        assert level in LEVELS
        folder = self.run_dir/level
        data = self.data/level/'trajectory.state.pd_ee_delta_pos.physx_cuda.h5'
        if not data.exists():
            raise FileNotFoundError('05 셀에서 데이터 다운로드를 먼저 실행하세요.')
        script = self.repo/'act_v2_train.py'
        if not script.exists():
            raise FileNotFoundError('04 셀에서 ver2 실행 코드를 설치하세요.')
        cfg = {k:self.cfg[k] for k in ('seed','batch_size','lr','save_freq','warmup_steps','kl_weight',
            'position_noise','validation_batches','amp','console_interval_seconds')}
        smoke = self.cfg['profile']=='smoke'
        cfg['total_iters'] = 100 if smoke else self.cfg['total_iters'][level]
        if smoke:
            cfg.update(save_freq=100, validation_batches=2)
        job = dict(folder=str(folder), data=str(data), model_config=self.model_config(level), train_config=cfg,
            policy_config=self.policy_config(level), num_demos=4 if smoke else self.cfg['num_demos'],
            source_sha256=hashlib.sha256(''.join(self.sources[n] for n in
                ('act_v2_model.py','act_v2_data.py','act_v2_train.py')).encode()).hexdigest())
        folder.mkdir(parents=True, exist_ok=True)
        previous = read_json(folder/'act_train_job.json')
        if (folder/'checkpoints/latest.pt').exists():
            keys = ('model_config','train_config','num_demos','source_sha256')
            if not previous or any(previous.get(k)!=job.get(k) for k in keys):
                raise ValueError('이 run_name의 저장된 학습 설정/코드와 다릅니다. 새 실험은 새 run_name을 사용하세요.')
            if (folder/'training_complete.json').exists():
                print(f'[{level}] 완료된 ver2 모델 재사용. 다음 테스트/평가 셀을 실행하세요.')
                return
        with self.persist_operation(level):
            save_json(folder/'act_train_job.json', job)
            try:
                self.run([sys.executable, str(script), str(folder/'act_train_job.json')], cwd=self.repo,
                         log=folder/'train.log')
            except BaseException as error:
                save_json(folder/'train_status.json', dict(status='interrupted',error=str(error)))
                raise
        print(f'[{level}] ver2 학습 완료. 다음 테스트 셀을 실행하세요.')

    def _test_candidate(self, level):
        folder = self.run_dir/level
        selected = read_json(folder/'selection.json', {}).get('selected')
        if selected:
            checkpoint = folder/'checkpoints'/Path(selected['checkpoint']).name
            if checkpoint.exists() and self.checkpoint_hash(checkpoint)==selected.get('checkpoint_sha256'):
                return checkpoint, selected['policy_config']
        for name in ('best_val.pt','latest.pt'):
            checkpoint = folder/'checkpoints'/name
            if checkpoint.exists():
                job = read_json(folder/'act_train_job.json')
                if not job:
                    raise ValueError('Saved ACT training architecture is missing')
                return checkpoint, dict(job['policy_config'])
        return None

    def _job(self, level, checkpoint, policy, seeds, label):
        job = super()._job(level, checkpoint, policy, seeds, label)
        job['model_sources_sha256'] = hashlib.sha256(self.sources['act_v2_model.py'].encode()).hexdigest()
        identity = {k:v for k,v in job.items() if k not in ('fingerprint','output','video')}
        job['fingerprint'] = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        return job

    def evaluate(self, level):
        self._ready()
        assert level in LEVELS
        with self.persist_operation(level):
            restored = self._saved_final(level)
            if restored:
                print(f'[{level}] 저장된 최종 평가 재사용: {restored[0]["sort_accuracy"]:.1%}')
                return restored[0]
            candidate = self._test_candidate(level)
            if candidate is None:
                print(f'[{level}] 먼저 이 난이도의 학습 셀을 실행하세요.')
                return None
            folder = self.run_dir/level
            trials, hashes = [], set()
            for name in ('best_val.pt','latest.pt'):
                ck = folder/'checkpoints'/name
                if not ck.exists():
                    continue
                digest = self.checkpoint_hash(ck)
                if digest in hashes:
                    continue
                hashes.add(digest)
                for window in self.cfg['ensemble_candidates']:
                    policy = dict(candidate[1], ensemble_window=window)
                    trials.append(self._trial(level, ck, policy, self.seeds(True), f'tune_{ck.stem}_w{window}'))
            winner = max(trials,key=lambda r:r['score'])
            save_json(folder/'selection.json', dict(selected=winner,trials=trials,tuning_seeds=self.seeds(True)))
            save_json(folder/'checkpoints/policy_config.json', winner['policy_config'])
            self._trial(level, Path(winner['checkpoint']), winner['policy_config'], self.seeds(False), 'metrics')
            self._show_progress(level, 'metrics')
            if self.cfg['record_eval_video']:
                self.record_video(level)
            return read_json(folder/'metrics.json')


def source_bundle():
    root = Path(__file__).parent
    bundle = legacy_sources()
    for path in root.glob('act_v2_*.py'):
        bundle[path.name] = path.read_text(encoding='utf-8')
    bundle['build_act_v2_notebook.py'] = (root.parent/'build_act_v2_notebook.py').read_text(encoding='utf-8')
    bundle['colab_policy.py'] = bundle['act_v2_policy.py']
    bundle['colab_eval_modular.py'] = bundle['act_v2_eval.py']
    bundle['colab_trace.py'] = """import json, sys
from pathlib import Path
from act_v2_eval import main
job = json.loads(Path(sys.argv[1]).read_text())
job['trace_only'] = True
main(job)
"""
    bundle['test_policy_runtime.py'] = bundle['act_v2_runtime_check.py']
    return bundle
