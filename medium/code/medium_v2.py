"""Medium deadline v2: bounded collection, short fine-tuning, separate GitHub results."""
import hashlib
import os
import sys
import zipfile
from pathlib import Path
from medium_lab import MediumLab, quality, source_bundle as medium_sources
from marso_experiment import read_json, save_json, digest
from github_store import GitHubStore


class MediumV2(MediumLab):
    def __init__(self,cfg,sources):
        super().__init__(cfg,sources)
        c=self.cfg
        if c['run_name']==c.get('source_run_name'):
            raise ValueError('v2 run_name must differ from the source run')
        if c['deadline_episodes']<4 or c['deadline_max_attempts']<c['deadline_episodes']:
            raise ValueError('At least four demonstrations and sufficient attempts required')
        if not c['teacher_gains'] or any(not 1<=g<=2 for g in c['teacher_gains']):
            raise ValueError('Teacher gains must be in [1,2]')
        if c['pilot_episodes']<1 or c['collection_seconds_per_call']<=0:
            raise ValueError('Collection budgets must be positive')
        collection=set(range(c['pilot_seed_start'],c['pilot_seed_start']+c['pilot_episodes']))
        full=set(range(c['deadline_seed_start'],c['deadline_seed_start']+c['deadline_max_attempts']))
        if collection & full or (collection|full)&set(self.test_seeds()+self.seeds(True)+self.seeds(False)):
            raise ValueError('Collection/development/test/final seeds must be separate')

    def notebook_snapshot(self):
        from build_medium_v2_notebook import make_notebook
        return make_notebook(dict(self.cfg,project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))

    def _stage_helpers(self):
        super()._stage_helpers()
        for name,source in self.sources.items():
            if name.startswith('deadline_'):
                (self.repo/name).write_text(source,encoding='utf-8')

    def prepare(self):
        self._ready();self._stage_helpers()
        folder=self.run_dir/'medium';folder.mkdir(parents=True,exist_ok=True)
        with self.persist_operation('medium'):
            origin=read_json(folder/'v2_origin.json')
            if origin:
                if origin['source_run_name']!=self.cfg.get('source_run_name') or origin['warm_start']!=self.cfg['warm_start']:
                    raise ValueError('Saved source differs. Use a new run_name.')
                if origin.get('checkpoint_sha256') and digest(folder/'checkpoints/initial_model.pt')!=origin['checkpoint_sha256']:
                    raise ValueError('Saved initial model changed')
                print('v2 시작 모델 준비 상태를 재사용합니다.');return
            if not self.cfg['warm_start']:
                save_json(folder/'v2_origin.json',dict(source_run_name=self.cfg.get('source_run_name'),warm_start=False))
                print('새 모델로 시작합니다. 다음 시연 시험 셀을 실행하세요.');return
            store=GitHubStore(self.cfg['github_repository'],'run-'+self.cfg['source_run_name']+'_benchmark')
            if not store.load(create=False):raise FileNotFoundError('원본 Medium 실행 Release가 없습니다.')
            names=sorted(n for n in store.assets if n.startswith('snapshot-medium-'))
            if not names:raise FileNotFoundError('원본 Medium snapshot이 없습니다.')
            snapshot_path=folder/'v2_source_snapshot.json'
            if not snapshot_path.exists():store.download(store.assets[names[-1]],snapshot_path)
            snapshot=read_json(snapshot_path)
            if snapshot.get('scope')!='medium' or snapshot.get('version')!=1:
                raise ValueError('Invalid source snapshot')
            entries={e['path']:e for e in snapshot['entries']}
            def fetch(name,destination):
                entry=entries.get(name)
                if not entry:raise FileNotFoundError(f'Source snapshot missing {name}')
                asset=store.assets.get(entry['asset'])
                if not asset or asset['size']!=entry['bytes']:raise ValueError('Incomplete source snapshot')
                destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists():
                    if digest(destination)!=entry['sha256']:raise ValueError('Saved input changed')
                else:
                    temporary=Path(str(destination)+'.part');store.download(asset,temporary)
                    if digest(temporary)!=entry['sha256']:raise ValueError('Source checksum mismatch')
                    os.replace(temporary,destination)
                return entry
            selection=folder/'source_best.json';fetch('medium/lab_best.json',selection)
            best=read_json(selection)
            # Use the model chosen on development seeds, never the latest optimizer state.
            entry=fetch('medium/checkpoints/'+Path(best['checkpoint']).name,folder/'checkpoints/initial_model.pt')
            if entry['sha256']!=best['checkpoint_sha256']:raise ValueError('Source selection/checkpoint mismatch')
            import torch
            weights=torch.load(folder/'checkpoints/initial_model.pt',map_location='cpu',weights_only=True)
            if weights.get('format')!='moveboxes-stage-act-v1' or weights['model_config']!=self.model_config('medium'):
                raise ValueError('Source architecture differs from CONFIG; keep the original model size')
            save_json(folder/'v2_origin.json',dict(source_run_name=self.cfg['source_run_name'],warm_start=True,
                checkpoint_sha256=entry['sha256'],source_iteration=best.get('iteration'),source_score=best['score']))
        print('기존 최고 Medium 모델 준비 완료. 기존 시연은 복사하지 않습니다.')

    def collect_deadline(self,operation='collect'):
        self._ready();self._stage_helpers()
        if operation not in ('pilot','collect'):raise ValueError('Unknown collection operation')
        folder=self.run_dir/'medium';c=self.cfg
        if not (folder/'v2_origin.json').exists():raise RuntimeError('먼저 05 준비 셀을 실행하세요.')
        config=dict(episodes=c['deadline_episodes'],max_attempts=c['deadline_max_attempts'],
            gains=c['teacher_gains'],pilot_episodes=c['pilot_episodes'],pilot_seed_start=c['pilot_seed_start'],
            seed_start=c['deadline_seed_start'],noise_std=c['deadline_noise_std'],
            noise_probability=c['deadline_noise_probability'],seconds_per_call=c['collection_seconds_per_call'])
        job=dict(operation=operation,folder=str(folder/'collection'),collection_config=config,
            repo_commit=c['repo_commit'],source_sha256=hashlib.sha256(''.join(self.sources[k] for k in
                ('deadline_teacher.py','deadline_collect.py','stage_labels.py')).encode()).hexdigest())
        with self.persist_operation('medium'):
            save_json(folder/'deadline_collection_job.json',job)
            self.run([sys.executable,'deadline_collect.py',str(folder/'deadline_collection_job.json')],
                cwd=self.repo,log=folder/('deadline_'+operation+'.log'))
            manifest=read_json(folder/'collection/manifest.json',{})
            if manifest.get('complete'):
                save_json(folder/'inputs_ready.json',dict(episodes=len(manifest['episodes']),evaluation_actions=199))
        print(f"시연 상태: {manifest.get('status','진행 중')} · 성공 {len(manifest.get('episodes',[]))}/{c['deadline_episodes']}")
        return manifest.get('status')

    def training_job(self):
        folder=self.run_dir/'medium'
        manifest=read_json(folder/'collection/manifest.json',{})
        if not manifest.get('complete'):raise RuntimeError('07 셀에서 제한 내 성공 시연 수집을 완료하세요.')
        job=super().training_job()
        job['data']=str(folder/'collection/manifest.json')
        job.pop('recovery_manifest',None)
        job['num_demos']=None
        if self.cfg['warm_start']:job['warm_start']=str(folder/'checkpoints/initial_model.pt')
        job['source_sha256']=hashlib.sha256(''.join(self.sources[k] for k in sorted(self.sources)
            if k.startswith(('stage_','act_v2_','deadline_'))).encode()).hexdigest()
        return job

    def run_blocks(self):
        self._ready();self._stage_helpers()
        # Refuse long work before checking data readiness.
        self.training_job()
        folder=self.run_dir/'medium'
        if self.cfg['warm_start'] and not (folder/'lab_best.json').exists():
            with self.persist_operation('medium'):
                initial=folder/'checkpoints/initial_model.pt'
                trial=self._trial('medium',initial,self.policy_config('medium'),self.seeds(True),'baseline')
                trial.update(quality(read_json(folder/'baseline.json')),block=0,iteration=0)
                save_json(folder/'lab_best.json',trial)
                print(f"보정 전 동일 시드 기준 분류: {trial['score']:.1%}")
        return super().run_blocks()

    def final_evaluation(self):
        self._ready()
        selected=self._test_candidate('medium')
        if not selected:raise RuntimeError('먼저 08 보정 학습 셀을 실행하세요.')
        folder=self.run_dir/'medium';best=read_json(folder/'lab_best.json')
        if best['score']<=0:
            print('개발 평가 분류 0%: 긴 최종 평가는 생략합니다.');return
        checkpoint,policy=selected
        with self.persist_operation('medium'):
            save_json(folder/'selection.json',dict(selected=best,tuning_seeds=self.seeds(True)))
            self._trial('medium',checkpoint,policy,self.seeds(False),'metrics')
        result=read_json(folder/'metrics.json')
        print(f"별도 최종 {result['n_episodes']}회 분류 정확도: {result['sort_accuracy']:.1%}")
        return result

    def package(self):
        path=super().package()
        if path is None:return
        temporary=Path(str(path)+'.tmp')
        with zipfile.ZipFile(path) as src,zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as dest:
            for entry in src.infolist():
                if not Path(entry.filename).name.startswith('deadline_'):
                    dest.writestr(entry,src.read(entry.filename))
        os.replace(temporary,path);self.sync_common()
        return path


def source_bundle():
    bundle=medium_sources();root=Path(__file__).parent
    for path in (root/'deadline').glob('deadline_*.py'):
        bundle[path.name]=path.read_text(encoding='utf-8')
    # Alias is scoped to this experiment; the original trainer file is untouched.
    bundle['stage_train.py']=bundle['deadline_train.py']
    for name in ('medium_v2.py','build_medium_v2_notebook.py'):
        bundle[name]=(root/name).read_text(encoding='utf-8')
    return bundle
