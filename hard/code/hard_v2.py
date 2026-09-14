"""Target-conditioned Hard training with development selection and compact backups."""
import contextlib
import hashlib
import os
import sys
from pathlib import Path
from hard_lab import HardLab,source_bundle as stage_sources
from marso_experiment import read_json,save_json,digest,Experiment
from github_store import GitHubStore,safe_target,compact_hard_files


class HardV2(HardLab):
    def notebook_snapshot(self):
        from build_hard_v2_notebook import make_notebook
        return make_notebook(dict(self.cfg,project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))

    def model_config(self,level):
        return dict(super().model_config(level),target_conditioning=True)

    def _stage_helpers(self):
        super()._stage_helpers()
        for name,source in self.sources.items():
            if name.startswith(('hard_','shared_stage_train')):(self.repo/name).write_text(source,encoding='utf-8')

    @contextlib.contextmanager
    def sync_environment(self,scope):
        name='MOVEBOXES_COMPACT_HARD_SYNC';previous=os.environ.get(name)
        with super().sync_environment(scope):
            if scope=='hard':os.environ[name]='1'
            try:yield
            finally:
                if previous is None:os.environ.pop(name,None)
                else:os.environ[name]=previous

    def sync_level(self,level):
        if self.remote_enabled and self.store:self.store.sync(self.run_dir,level,compact_hard_files(self.run_dir))

    def run(self,command,cwd=None,log=None):
        if log is not None and Path(log).stem.startswith(('train_block_','dev_b')):
            with self.sync_environment(None):return Experiment.run(self,command,cwd=cwd,log=log)
        return super().run(command,cwd=cwd,log=log)

    def prepare(self):
        self._ready();self.prepare_data();self._stage_helpers()
        folder=self.run_dir/'hard';folder.mkdir(parents=True,exist_ok=True)
        store=GitHubStore(self.cfg['github_repository'],'run-'+self.cfg['source_run_name']+'_benchmark')
        if not store.load(create=False):raise FileNotFoundError('Hard v1 Release is unavailable')
        snapshots=sorted(n for n in store.assets if n.startswith('snapshot-hard-'))
        if not snapshots:raise FileNotFoundError('Hard v1 snapshot is missing')
        path=folder/'source_snapshot.json'
        if not path.exists():store.download(store.assets[snapshots[-1]],path)
        snapshot=read_json(path)
        if snapshot.get('version')!=1 or snapshot.get('scope')!='hard':raise ValueError('Invalid source snapshot')
        entries={e['path']:e for e in snapshot['entries']}
        def fetch(relative):
            entry=entries.get(relative)
            if not entry:raise FileNotFoundError(relative)
            destination=safe_target(self.run_dir,relative);destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                if digest(destination)!=entry['sha256']:raise ValueError('Imported data changed')
            else:
                asset=store.assets.get(entry['asset'])
                if not asset or asset['size']!=entry['bytes']:raise ValueError('Incomplete source snapshot')
                temporary=Path(str(destination)+'.part');store.download(asset,temporary)
                if digest(temporary)!=entry['sha256']:raise ValueError('Source checksum mismatch')
                os.replace(temporary,destination)
            return destination
        with self.persist_operation('hard'):
            manifest=read_json(fetch('hard/collection/manifest.json'))
            if not manifest.get('complete') or len(manifest['episodes'])<2:raise ValueError('Complete successful collection required')
            for entry in manifest['episodes']:
                safe_target(folder/'collection',entry['file'])
                if digest(fetch('hard/collection/'+entry['file']))!=entry['sha256']:raise ValueError('Collection checksum mismatch')
            save_json(folder/'inputs_ready.json',dict(source_run_name=self.cfg['source_run_name'],episodes=len(manifest['episodes'])))
        print(f"Hard v1 성공/복구 시연 {len(manifest['episodes'])}개 재사용 · 새 수집 없이 학습")

    def training_job(self):
        folder=self.run_dir/'hard'
        if not read_json(folder/'collection/manifest.json',{}).get('complete'):raise RuntimeError('05 준비 셀을 실행하세요')
        cfg={k:self.cfg[k] for k in ('seed','batch_size','lr','warmup_steps','kl_weight','position_noise',
            'validation_batches','amp','console_interval_seconds','stage_loss_weight','gate_loss_weight','target_loss_weight')}
        cfg.update(total_iters=self.cfg['block_iters']*self.cfg['max_blocks'],save_freq=self.cfg['block_iters'],action_training_mode='prior')
        return dict(folder=str(folder),data=str(self.data/'hard/trajectory.state.pd_ee_delta_pos.physx_cuda.h5'),
            recovery_manifest=str(folder/'collection/manifest.json'),num_demos=self.cfg['num_demos'],
            model_config=self.model_config('hard'),train_config=cfg,policy_config=self.policy_config('hard'),
            source_sha256=hashlib.sha256(''.join(self.sources[k] for k in sorted(self.sources)).encode()).hexdigest())

    def _job(self,*args,**kwargs):
        job=super()._job(*args,**kwargs)
        job['target_model_sha256']=hashlib.sha256(self.sources['hard_model.py'].encode()).hexdigest()
        identity={k:v for k,v in job.items() if k not in ('fingerprint','output','video')}
        import json
        job['fingerprint']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        return job

    def _test_candidate(self,level):
        self._hard(level);best=read_json(self.run_dir/'hard/lab_best.json')
        if not best:return None
        checkpoint=self.run_dir/'hard/checkpoints'/Path(best['checkpoint']).name
        if digest(checkpoint)!=best['checkpoint_sha256']:raise ValueError('Selected model changed')
        return checkpoint,best['policy_config']

    def run_blocks(self):
        self._ready();self._stage_helpers();folder=self.run_dir/'hard';job=self.training_job()
        identity=dict(job=job,seeds=self.seeds(True))
        old=read_json(folder/'lab_protocol.json')
        if old and old!=identity:raise ValueError('Saved protocol differs; use a new run_name')
        save_json(folder/'lab_protocol.json',identity)
        history=read_json(folder/'lab_history.json',dict(blocks=[],status='running'))
        if history['status']!='running':return self.report()
        with self.persist_operation('hard'):
            for block in range(len(history['blocks'])+1,self.cfg['max_blocks']+1):
                stop=block*self.cfg['block_iters'];save_json(folder/'stage_train_job.json',dict(job,stop_at=stop))
                self.run([sys.executable,'stage_train.py',str(folder/'stage_train_job.json')],cwd=self.repo,log=folder/f'train_block_{block:02d}.log')
                import torch
                checkpoint=folder/'checkpoints'/f'block_{block:02d}.pt'
                if not checkpoint.exists():
                    saved=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
                    if saved['step']!=stop:raise ValueError('Checkpoint step mismatch')
                    temporary=Path(str(checkpoint)+'.tmp')
                    torch.save({k:saved[k] for k in ('format','model_config','model','step')},temporary);os.replace(temporary,checkpoint)
                trial=self._trial('hard',checkpoint,self.policy_config('hard'),self.seeds(True),f'dev_b{block:02d}')
                metrics=read_json(folder/f'dev_b{block:02d}.json')
                trial.update(iteration=stop,first_grasp_rate=sum(r.get('next_pick',{}).get('stable_grasp_cycles',0)>0 for r in metrics['episodes'])/len(metrics['episodes']))
                previous=read_json(folder/'lab_best.json');improved=previous is None or (trial['score'],trial['first_grasp_rate'])>(previous['score'],previous['first_grasp_rate'])
                if improved:save_json(folder/'lab_best.json',trial)
                history['blocks'].append(dict(iteration=stop,best=trial,improved=improved))
                recent=history['blocks'][-self.cfg['plateau_blocks']:]
                plateau=len(recent)==self.cfg['plateau_blocks'] and not any(r['improved'] for r in recent)
                history['status']='plateau' if plateau else 'budget_reached' if block==self.cfg['max_blocks'] else 'running'
                save_json(folder/'lab_history.json',history);self.sync_level('hard')
                print(f"{stop}회 · 첫 집기 {trial['first_grasp_rate']:.0%} · 분류 {trial['score']:.1%}",flush=True)
                if history['status']!='running':break
        return self.report()

    def report(self):
        self._ready();history=read_json(self.run_dir/'hard/lab_history.json',dict(blocks=[],status='not_started'))
        for row in history['blocks']:print(f"{row['iteration']}회 · 분류 {row['best']['score']:.1%} · 첫 집기 {row['best']['first_grasp_rate']:.0%}")
        print('상태:',history['status']);return history

    def final_evaluation(self):
        self._ready();candidate=self._test_candidate('hard')
        if not candidate:raise RuntimeError('먼저 07 학습 셀을 실행하세요')
        folder=self.run_dir/'hard';best=read_json(folder/'lab_best.json')
        if best['score']<=0:print('개발 점수 0%라 긴 최종 평가를 생략합니다');return
        with self.persist_operation('hard'):
            save_json(folder/'selection.json',dict(selected=best,tuning_seeds=self.seeds(True)))
            return self._trial('hard',candidate[0],candidate[1],self.seeds(False),'metrics')


def source_bundle():
    bundle=stage_sources();root=Path(__file__).parent
    for name in ('hard_model.py','hard_data.py','hard_train.py','hard_policy.py','hard_v2.py','build_hard_v2_notebook.py'):
        bundle[name]=(root/name).read_text(encoding='utf-8')
    bundle['shared_stage_train.py']=bundle['stage_train.py'].replace('moveboxes-stage-act-v1','moveboxes-hard-target-act-v2')
    bundle['stage_train.py']=bundle['hard_train.py']
    bundle['stage_policy.py']=bundle['hard_policy.py'];bundle['colab_policy.py']=bundle['hard_policy.py']
    return bundle
