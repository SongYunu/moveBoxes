"""Medium v2.2: reuse verified demonstrations and correct pickup height separately."""
import hashlib
import contextlib
import os
from pathlib import Path
from medium_v21 import MediumV21, source_bundle as v21_sources
from github_store import GitHubStore, safe_target
from marso_experiment import read_json, save_json, digest


class MediumV22(MediumV21):
    def __init__(self, cfg, sources):
        super().__init__(cfg, sources)
        if not self.cfg['warm_start']:
            raise ValueError('v2.2 requires a compatible source model')
        for key in ('pick_z_weight','contact_z_weight','contact_gripper_weight'):
            if not 1 <= self.cfg[key] <= 10:
                raise ValueError(f'{key} must be in [1,10]')

    def notebook_snapshot(self):
        from build_medium_v22_notebook import make_notebook
        return make_notebook(dict(self.cfg, project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))

    def _stage_helpers(self):
        super()._stage_helpers()
        for name, source in self.sources.items():
            if name.startswith('zfocus_'):
                (self.repo/name).write_text(source, encoding='utf-8')

    @contextlib.contextmanager
    def sync_environment(self, scope):
        name='MOVEBOXES_COMPACT_MEDIUM_SYNC';previous=os.environ.get(name)
        with super().sync_environment(scope):
            if scope=='medium':os.environ[name]='1'
            try:
                yield
            finally:
                if previous is None:os.environ.pop(name,None)
                else:os.environ[name]=previous

    def sync_level(self, level):
        if level!='medium':
            return super().sync_level(level)
        if self.remote_enabled and self.store:
            from github_store import compact_medium_files
            files=compact_medium_files(self.run_dir)
            if files:self.store.sync(self.run_dir,level,files)

    def run(self, command, cwd=None, log=None):
        # Development work is committed once at its block boundary by run_blocks.
        # Avoid one Release snapshot per internal evaluation episode.
        local_only = log is not None and Path(log).stem.startswith(('train_block_','dev_b','baseline'))
        if local_only:
            from marso_experiment import Experiment
            with self.sync_environment(None):
                return Experiment.run(self,command,cwd=cwd,log=log)
        return super().run(command,cwd=cwd,log=log)

    def prepare(self):
        super().prepare()
        folder = self.run_dir/'medium'
        snapshot = read_json(folder/'v2_source_snapshot.json')
        entries = {e['path']:e for e in snapshot['entries']}
        store = GitHubStore(self.cfg['github_repository'],'run-'+self.cfg['source_run_name']+'_benchmark')
        if not store.load(create=False):
            raise FileNotFoundError('Source Release is unavailable')
        with self.persist_operation('medium'):
            def fetch(relative):
                entry = entries.get(relative)
                if not entry:
                    raise FileNotFoundError(f'Source snapshot missing {relative}')
                destination = safe_target(self.run_dir,relative)
                destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists():
                    if digest(destination) != entry['sha256']:
                        raise ValueError('Imported collection changed; use a new run_name')
                    return destination
                asset = store.assets.get(entry['asset'])
                if not asset or asset['size'] != entry['bytes']:
                    raise ValueError('Incomplete source collection')
                temporary = Path(str(destination)+'.part')
                store.download(asset,temporary)
                if digest(temporary) != entry['sha256']:
                    raise ValueError('Collection checksum mismatch')
                os.replace(temporary,destination)
                return destination
            manifest_path = fetch('medium/collection/manifest.json')
            manifest = read_json(manifest_path)
            if manifest.get('version') != 'curriculum-v21' or not manifest.get('complete'):
                raise ValueError('A completed v2.1 curriculum collection is required')
            for entry in manifest['episodes']:
                # Validate paths before using a remote manifest.
                safe_target(folder/'collection',entry['file'])
                path = fetch('medium/collection/'+entry['file'])
                if digest(path) != entry['sha256']:
                    raise ValueError('Manifest/collection checksum mismatch')
            if manifest['minimum_correct'] != self.cfg['minimum_correct']:
                raise ValueError('Imported success threshold differs from CONFIG')
            save_json(folder/'inputs_ready.json',dict(source_run_name=self.cfg['source_run_name'],
                episodes=len(manifest['episodes']),minimum_correct=manifest['minimum_correct'],evaluation_actions=199))
        print(f"v2.1 시연 {len(manifest['episodes'])}개 가져오기 완료 · 새 시연 수집 없이 Z축 보정")

    def training_job(self):
        job = super().training_job()
        for key in ('pick_z_weight','contact_z_weight','contact_gripper_weight'):
            job['train_config'][key] = self.cfg[key]
        job['source_sha256'] = hashlib.sha256(''.join(self.sources[k] for k in sorted(self.sources)).encode()).hexdigest()
        return job

    def _trial(self, level, checkpoint, policy, seeds, label):
        if label == 'baseline':
            # Compare against the original deployed settings, including its ensemble.
            source = read_json(self.run_dir/'medium/source_best.json', {})
            policy = source.get('policy_config', policy)
        return super()._trial(level, checkpoint, policy, seeds, label)

    def audit_height(self):
        self._ready();self._stage_helpers()
        code = """from curriculum_data import load_data
import torch
data=load_data(__import__('sys').argv[1]);gaps=[]
for t in data:
    active=(t['stage']==0)&(t['actions'][:,3]<0)&(t['obs'][:-1,25]<.5)
    ids=active.nonzero().flatten()
    for i in ids:
        j=int(t['target'][i]);gaps.append(float(t['obs'][i,20]-t['obs'][i,28+7*j]))
if not gaps:raise ValueError('No pre-grasp closing examples')
g=torch.tensor(gaps)*1000
print(f'시연 집기 전 TCP-상자 중심 높이 차: 중앙값 {g.median():.1f}mm / 10~90% {g.quantile(.1):.1f}~{g.quantile(.9):.1f}mm / {len(g)}프레임')
"""
        import sys
        self.run([sys.executable,'-c',code,str(self.run_dir/'medium/collection/manifest.json')],cwd=self.repo,
            log=self.session/'height_audit.log')
        print('Z축 가중치:',self.cfg['pick_z_weight'],'/ 닫힘 접촉 Z:',self.cfg['contact_z_weight'],
            '/ 하강 중 열림·접촉 닫힘:',self.cfg['contact_gripper_weight'])
        print('관측 인코더와 단계 판정은 고정 · 행동 디코더만 보정 · 성공한 4/4 시연끼리만 시간 보너스')


def source_bundle():
    bundle = v21_sources();root=Path(__file__).parent
    bundle['zfocus_model.py']=(root/'zfocus/zfocus_model.py').read_text(encoding='utf-8')
    bundle['stage_train.py']=bundle['curriculum_train.py'].replace(
        'from stage_model import StageACT, stage_loss','from zfocus_model import StageACT, stage_loss')
    for name in ('medium_v22.py','build_medium_v22_notebook.py'):
        bundle[name]=(root/name).read_text(encoding='utf-8')
    return bundle
