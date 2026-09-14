"""One model trained from scratch on all three difficulties from the first update."""
import hashlib,json,shutil,sys,tempfile
from pathlib import Path
from unified_experiment import UnifiedExperiment,source_bundle as original_sources,LEVELS,COUNTS
from marso_experiment import read_json,save_json,digest
from hard_progress import summarize


def joint_quality(metrics):
    if set(metrics)!=set(LEVELS):raise ValueError('All three task evaluations are required')
    seeds=None
    for level,m in metrics.items():
        rows=m['episodes']
        if not m.get('complete') or not rows:raise ValueError('Incomplete joint evaluation')
        current=[row['seed'] for row in rows]
        if seeds is not None and seeds!=current:raise ValueError('Joint evaluations use the same seed list')
        seeds=current
    scores={k:metrics[k]['sort_accuracy'] for k in LEVELS}
    full={k:summarize(metrics[k],COUNTS[k])['all_correct_rate'] for k in LEVELS}
    return dict(scores=scores,macro_score=sum(scores.values())/3,minimum_score=min(scores.values()),full_success=full)


def rank(quality):
    return quality['macro_score'],quality['minimum_score'],sum(quality['full_success'].values())/3


class JointExperiment(UnifiedExperiment):
    def __init__(self,cfg,sources):
        super().__init__(cfg,sources)
        if self.cfg['profile']=='smoke':
            self.cfg.update(joint_total_iters=100,joint_eval_interval=100)
        if self.cfg['batch_size']<3 or self.cfg['joint_total_iters']<1 or self.cfg['joint_eval_interval']<1:
            raise ValueError('Invalid joint batch or training budget')

    def connect(self):
        from marso_github import github_token
        from github_store import GitHubStore
        from marso_train_test import TrainTestExperiment
        if self.connected:return
        c=self.cfg
        if c['width']%c['heads'] or min(c['history'],c['chunk_size'],c['layers'],c['spatial_layers'],c['validation_batches'],c['warmup_steps'])<1:
            raise ValueError('Invalid Transformer or training dimensions')
        if not 1<=c['ensemble_window']<=c['chunk_size'] or c['position_noise']!=0:
            raise ValueError('Invalid ensemble window or unaligned observation noise')
        if self.remote_enabled:
            github_token();self.store=GitHubStore(c['github_repository'],'run-'+self.run_dir.name);self.store.load()
            if not any(self.run_dir.glob('*')):print('GitHub 복원:',self.store.restore(self.run_dir),'files')
        try:
            TrainTestExperiment.connect(self);self.sync_common()
        except BaseException:
            self.connected=False;raise
        print('Object attention · 같은 모델 하나 · 세 난이도 동시 학습 · 매 제어 스텝 재계획')

    def model_config(self,level):
        return dict(state_dim=90,architecture='object-attention-v1',
            **{k:self.cfg[k] for k in ('history','chunk_size','width','heads','layers','spatial_layers')})

    def policy_config(self,level):
        return dict(model_config=self.model_config(level),temporal_decay=self.cfg['temporal_decay'],
            ensemble_window=self.cfg['ensemble_window'],act_horizon=1,num_inference_steps=1)

    def notebook_snapshot(self):
        from build_joint_notebook import make_notebook
        return make_notebook(dict(self.cfg,project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))

    def _stage_helpers(self):
        super()._stage_helpers()
        for name,source in self.sources.items():
            if name.startswith(('joint_','object_')):(self.repo/name).write_text(source,encoding='utf-8')

    def _job(self,*args,**kwargs):
        job=super()._job(*args,**kwargs)
        job['object_sources_sha256']=hashlib.sha256(''.join(self.sources[k] for k in
            ('object_model.py','object_policy.py','object_observer.py')).encode()).hexdigest()
        identity={k:v for k,v in job.items() if k not in ('fingerprint','output','video')}
        job['fingerprint']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        return job

    def prepare(self):
        self._ready();self.prepare_data();self._stage_helpers()
        for level in LEVELS:(self.run_dir/level).mkdir(exist_ok=True)
        state=read_json(self.run_dir/'curriculum.json')
        if state:
            if state.get('version')!='joint-object-v1':raise ValueError('Use the joint object experiment run_name')
            if state.get('checkpoint'):self._current()
            print('공동 학습 복원: 완료',state['completed_updates'],'회');return
        save_json(self.run_dir/'curriculum.json',dict(version='joint-object-v1',checkpoint=None,checkpoint_sha256=None,
            completed_updates=0,best=None,segment=None,history=[],passed=[],rounds={}))
        self.sync_common();print('세 난이도 원본 데이터 준비. 첫 학습은 무작위 초기화합니다. 기존 PT를 불러오지 않습니다.')

    def _current(self):
        state=read_json(self.run_dir/'curriculum.json')
        if not state or not state.get('checkpoint'):raise RuntimeError('선택 모델이 아직 없습니다. 07 공동 학습을 먼저 실행하세요.')
        return super()._current()

    def _dataset(self):
        sources=[]
        for level in LEVELS:
            path=self.data/level/'trajectory.state.pd_ee_delta_pos.physx_cuda.h5'
            sources.append(dict(level=level,data=str(path),num_demos=self.cfg['num_demos'],data_sha256=digest(path),
                metadata_sha256=digest(path.with_suffix('.json'))))
        return dict(mode='joint',sources=sources)

    def _make_job(self,state):
        c=self.cfg;segment=state.get('segment')
        if not segment or not (self.run_dir/segment['folder']/'checkpoints/latest.pt').exists():
            segment=dict(folder=f"joint/segment_{state['completed_updates']:06d}",start=state['completed_updates'],anchor=state.get('checkpoint'))
            state['segment']=segment
        folder=self.run_dir/segment['folder'];folder.mkdir(parents=True,exist_ok=True)
        manifest=folder/'data.json';save_json(manifest,self._dataset())
        config={k:c[k] for k in ('seed','batch_size','lr','warmup_steps','kl_weight','position_noise','validation_batches','amp',
            'console_interval_seconds','stage_loss_weight','gate_loss_weight')}
        config.update(total_iters=c['joint_total_iters']-segment['start'],save_freq=c['joint_eval_interval'],action_training_mode='prior')
        stop=min(state['completed_updates']+c['joint_eval_interval'],c['joint_total_iters'])-segment['start']
        return dict(folder=str(folder),data=str(manifest),model_config=self.model_config('hard'),train_config=config,
            policy_config=self.policy_config('hard'),stop_at=stop,
            warm_start=str(self.run_dir/segment['anchor']) if segment['anchor'] else None,resume_joint=bool(segment['anchor']),
            source_sha256=hashlib.sha256(''.join(self.sources[k] for k in sorted(self.sources)).encode()).hexdigest())

    def sync_common(self):
        if not (self.remote_enabled and self.store and self.connected):return
        # All metadata and the one selected model belong to one restorable snapshot.
        files=list(self.run_dir.glob('*.json'))+list(self.run_dir.glob('*.ipynb'))
        files+=list((self.run_dir/'joint').rglob('*.json'))
        for level in LEVELS:files+=list((self.run_dir/level).rglob('*.json'))
        state=read_json(self.run_dir/'curriculum.json',{})
        if state.get('checkpoint'):files.append(self.run_dir/state['checkpoint'])
        package=read_json(self.run_dir/'package_info.json',{})
        if package.get('checkpoint_sha256')==state.get('checkpoint_sha256') and (self.run_dir/'unified_policy.zip').exists():
            files.append(self.run_dir/'unified_policy.zip')
        files+=list(self.session.glob('pip-freeze.txt'))
        for level in LEVELS:
            videos=sorted((self.run_dir/level/'test_videos').rglob('*.mp4'),key=lambda p:p.stat().st_mtime)
            if videos:files.append(videos[-1])
        self.store.sync(self.run_dir,'common',sorted(set(files)))

    def train_joint(self):
        self._ready();self._stage_helpers();state=read_json(self.run_dir/'curriculum.json')
        if not state or state.get('version')!='joint-object-v1':raise RuntimeError('05 준비 셀을 먼저 실행하세요')
        while state['completed_updates']<self.cfg['joint_total_iters']:
            with self.persist_operation('hard'):
                job=self._make_job(state);save_json(self.run_dir/'curriculum.json',state)
                job_path=Path(job['folder'])/'job.json';save_json(job_path,job)
                self.run([sys.executable,'stage_train.py',str(job_path)],cwd=self.repo,log=Path(job['folder'])/'train.log')
                import torch
                saved=torch.load(Path(job['folder'])/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
                if saved['step']!=job['stop_at']:raise ValueError('Training segment not complete')
                step=state['segment']['start']+saved['step']
                candidate=self.run_dir/f'joint/candidates/step_{step:06d}.pt';candidate.parent.mkdir(exist_ok=True)
                torch.save({k:saved[k] for k in ('format','model_config','model','step')},candidate)
                metrics={level:self._evaluate(level,candidate,f'joint_{step:06d}') for level in LEVELS}
                quality=joint_quality(metrics)
                accepted=state['best'] is None or rank(quality)>rank(state['best'])
                if accepted:state.update(checkpoint=candidate.relative_to(self.run_dir).as_posix(),checkpoint_sha256=digest(candidate),best=quality)
                state['completed_updates']=step;state['history'].append(dict(step=step,accepted=accepted,**quality))
                save_json(self.run_dir/'curriculum.json',state)
                print(f"공동 학습 {step}/{self.cfg['joint_total_iters']} · "+' / '.join(f'{k} {quality["scores"][k]:.1%}' for k in LEVELS)+f' · 평균 {quality["macro_score"]:.1%} · 최고 교체 {accepted}')
        print('공동 학습 예산 완료. 같은 선택 모델로 08~10 테스트를 실행하세요.')

    def report(self):
        state=read_json(self.run_dir/'curriculum.json',{})
        print('공동 학습 완료 update:',state.get('completed_updates',0))
        print('선택 모델:',state.get('checkpoint'));print('선택 모델 개발 결과:',state.get('best'))
        return state

    def show_results(self,videos=False):return self.report()

    def final_evaluation(self):
        self._ready();_,checkpoint=self._current()
        with self.persist_operation('hard'):
            for level in LEVELS:self._evaluate(level,checkpoint,'metrics',self.seeds(False))

    def check_runtime(self):
        self._ready();self._stage_helpers()
        self.run([sys.executable,'test_policy_runtime.py'],cwd=self.repo,log=self.session/'object_runtime.log')

    def test(self,level):
        with self.persist_operation(level):return super().test(level)

    def package(self):
        self._ready();state,checkpoint=self._current()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoints').mkdir();shutil.copy2(checkpoint,root/'checkpoints/model.pt')
            save_json(root/'checkpoints/policy_config.json',self.policy_config('hard'))
            save_json(root/'submission.yaml',dict(team=self.cfg['team'],state=dict(policy='colab_policy:load_policy',
                levels={k:dict(checkpoint='checkpoints/model.pt') for k in LEVELS})))
            for name in ('object_model.py','object_policy.py','act_v2_model.py','colab_policy.py'):
                (root/name).write_text(self.sources[name],encoding='utf-8')
            output=Path(shutil.make_archive(str(self.run_dir/'unified_policy'),'zip',root))
        save_json(self.run_dir/'package_info.json',dict(checkpoint_sha256=state['checkpoint_sha256'],architecture='object-attention-v1'))
        self.sync_common();print('같은 모델 하나 / 세 난이도 패키지:',output);return output


def source_bundle():
    bundle=original_sources()
    bundle['stage_train.py']=bundle['joint_train.py']
    bundle['stage_model.py']='from object_model import ObjectACT as StageACT, stage_loss\n'
    bundle['shared_stage_train.py']=bundle['shared_stage_train.py'].replace('moveboxes-stage-act-v1','moveboxes-object-act-v1').replace('Stage ACT:','Object ACT:')
    bundle['shared_stage_train.py']=bundle['shared_stage_train.py'].replace(
        '25% alternate previous-stage inputs; action/observation targets unchanged','none; phase labels are not model inputs').replace(
        'stop at stage/parcel boundaries and first perturbed executed action','stop at episode boundary; chunks may cross parcel and stage boundaries').replace(
        'training trajectories only; state plus parcel/TCP and target-bin relative positions','training proprioception only; shared fixed geometry scales for every object')
    evaluator=bundle['stage_eval.py']
    start=evaluator.index('class BaseObserver(');end=evaluator.index('def main(job):')
    evaluator=evaluator[:start]+'from object_observer import Observer\n\n\n'+evaluator[end:]
    evaluator=evaluator.replace('from unified_policy import load_stage','from object_policy import load_stage')
    bundle['stage_eval.py']=evaluator;bundle['colab_eval_modular.py']=evaluator
    bundle['colab_policy.py']='from object_policy import load_policy, load_stage\n'
    bundle['test_policy_runtime.py']=bundle['object_runtime_check.py']
    return bundle
