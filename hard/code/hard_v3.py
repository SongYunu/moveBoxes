"""Hard v3: select an Easy/Medium transplant, then fine-tune the proven StageACT."""
import sys
from pathlib import Path
from hard_v2 import HardV2
from hard_lab import source_bundle as stage_sources
from marso_experiment import read_json,save_json,digest
from github_store import GitHubStore


class HardV3(HardV2):
    def notebook_snapshot(self):
        from build_hard_v3_notebook import make_notebook
        return make_notebook(dict(self.cfg,project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))

    def model_config(self,level):
        # Return to the architecture that already learned Easy and Medium.
        return super(HardV2,self).model_config(level)

    def _job(self,*args,**kwargs):
        return super(HardV2,self)._job(*args,**kwargs)

    def _download_donor(self,level,run_name):
        folder=self.run_dir/'hard/transfer_sources'/level;folder.mkdir(parents=True,exist_ok=True)
        store=GitHubStore(self.cfg['github_repository'],'run-'+run_name+'_benchmark')
        if not store.load(create=False):raise FileNotFoundError(f'{level} source Release unavailable')
        snapshots=sorted(n for n in store.assets if n.startswith(f'snapshot-{level}-'))
        if not snapshots:raise FileNotFoundError(f'{level} source snapshot unavailable')
        snap_path=folder/'snapshot.json';store.download(store.assets[snapshots[-1]],snap_path)
        snapshot=read_json(snap_path);entries={e['path']:e for e in snapshot['entries']}
        if snapshot.get('scope')!=level:raise ValueError('Donor snapshot scope mismatch')
        def fetch(relative,name):
            entry=entries.get(relative)
            if not entry:raise FileNotFoundError(relative)
            target=folder/name;asset=store.assets.get(entry['asset'])
            if not asset or asset['size']!=entry['bytes']:raise ValueError('Donor snapshot incomplete')
            if not target.exists():store.download(asset,target)
            if digest(target)!=entry['sha256']:raise ValueError('Donor checksum mismatch')
            return target,entry
        metadata=None
        for name in (f'{level}/lab_best.json',f'{level}/selection.json'):
            if name in entries:
                metadata,_=fetch(name,'selection_source.json');break
        preferred=[]
        if metadata:
            value=read_json(metadata);selected=value.get('selected',value)
            if isinstance(selected,dict) and selected.get('checkpoint'):preferred.append(Path(selected['checkpoint']).name)
        preferred += ['best_val.pt','latest.pt','initial_model.pt']
        checkpoint_entry=None
        for basename in preferred:
            match=next((e for p,e in entries.items() if p.startswith(level+'/') and p.endswith('.pt') and Path(p).name==basename),None)
            if match:checkpoint_entry=match;break
        if checkpoint_entry is None:
            choices=[e for p,e in entries.items() if p.startswith(level+'/') and p.endswith('.pt')]
            if not choices:raise FileNotFoundError(f'{level} checkpoint absent from snapshot')
            checkpoint_entry=choices[0]
        relative=checkpoint_entry['path'];checkpoint,_=fetch(relative,'donor.pt')
        return dict(level=level,run_name=run_name,snapshot=snapshots[-1],checkpoint=str(checkpoint),sha256=digest(checkpoint))

    def prepare(self):
        super().prepare();folder=self.run_dir/'hard'
        donors=[]
        with self.persist_operation('hard'):
            for level,key in (('easy','easy_source_run_name'),('medium','medium_source_run_name')):
                donors.append(self._download_donor(level,self.cfg[key]))
            job=dict(data=str(self.data/'hard/trajectory.state.pd_ee_delta_pos.physx_cuda.h5'),
                recovery_manifest=str(folder/'collection/manifest.json'),num_demos=self.cfg['num_demos'],seed=self.cfg['seed'],
                model_config=self.model_config('hard'),report=str(folder/'transfer_report.json'),donors=[])
            for donor in donors:
                output=folder/'checkpoints'/f"from_{donor['level']}.pt"
                job['donors'].append(dict(level=donor['level'],checkpoint=donor['checkpoint'],output=str(output)))
            job_path=folder/'transfer_job.json';save_json(job_path,job)
            self.run([sys.executable,'hard_transfer.py',str(job_path)],cwd=self.repo,log=folder/'transfer.log')
            save_json(folder/'transfer_sources.json',donors)
        print('Easy/Medium 최고 모델을 Hard 구조로 변환했습니다. 07 셀이 둘을 비교하고 재학습합니다.')

    def _select_transfer(self):
        folder=self.run_dir/'hard';selection=read_json(folder/'transfer_selection.json')
        if selection:return selection
        trials=[]
        for level in ('easy','medium'):
            checkpoint=folder/'checkpoints'/f'from_{level}.pt'
            if not checkpoint.exists():raise RuntimeError('먼저 05 준비 셀을 실행하세요')
            trial=self._trial('hard',checkpoint,self.policy_config('hard'),self.seeds(True),f'transfer_{level}')
            metrics=read_json(folder/f'transfer_{level}.json')
            trial.update(source_level=level,iteration=0,
                first_grasp_rate=sum(r.get('next_pick',{}).get('stable_grasp_cycles',0)>0 for r in metrics['episodes'])/len(metrics['episodes']))
            trials.append(trial)
        winner=max(trials,key=lambda x:(x['score'],x['first_grasp_rate']))
        selection=dict(selected=winner,trials=trials,seeds=self.seeds(True));save_json(folder/'transfer_selection.json',selection)
        save_json(folder/'lab_best.json',winner)
        print(f"전이 시작점: {winner['source_level']} · 첫 집기 {winner['first_grasp_rate']:.0%} · 분류 {winner['score']:.1%}")
        return selection

    def training_job(self):
        job=super().training_job();selection=read_json(self.run_dir/'hard/transfer_selection.json')
        if not selection:raise RuntimeError('전이 모델 비교가 먼저 필요합니다')
        job['warm_start']=selection['selected']['checkpoint']
        return job

    def run_blocks(self):
        self._ready();self._stage_helpers()
        with self.persist_operation('hard'):self._select_transfer()
        return super().run_blocks()

def source_bundle():
    bundle=stage_sources();root=Path(__file__).parent
    for name in ('hard_lab.py','hard_v2.py','hard_v3.py','hard_transfer.py','build_hard_v3_notebook.py'):
        bundle[name]=(root/name).read_text(encoding='utf-8')
    return bundle
