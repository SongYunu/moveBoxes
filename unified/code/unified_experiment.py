"""One policy, bounded difficulty curriculum, honest full-episode gates and rehearsal."""
import hashlib,json,shutil,sys,tempfile
from pathlib import Path
from stage_experiment import StageExperiment,source_bundle as stage_sources
from marso_experiment import read_json,save_json,digest,Experiment
from hard_v3 import HardV3
from hard_progress import summarize,paired_comparison

LEVELS=('easy','medium','hard')
COUNTS=dict(easy=2,medium=4,hard=6)


class UnifiedExperiment(StageExperiment):
    def __init__(self,cfg,sources):
        super().__init__(cfg,sources)
        c=self.cfg
        if not 0<=c['replay_fraction']<1 or not 0<=c['speed_bonus']<=1:
            raise ValueError('replay_fraction must be in [0,1), speed_bonus in [0,1]')
        if min(c['block_iters'],c['plateau_blocks'],*c['max_blocks'].values())<1:
            raise ValueError('Curriculum budgets must be positive')
        if set(c['max_blocks'])!=set(LEVELS) or set(c['pass_rate'])!=set(LEVELS):
            raise ValueError('Configure all three curriculum levels')
        if any(not 0<value<=1 for value in c['pass_rate'].values()):
            raise ValueError('Pass rates must be in (0,1]')
        confirmation=set(range(c['confirmation_seed_start'],c['confirmation_seed_start']+c['tuning_episodes']))
        if confirmation.intersection(self.seeds(True)+self.test_seeds()+self.seeds(False)):
            raise ValueError('Confirmation seeds must be separate from development/test/final seeds')

    def notebook_snapshot(self):
        from build_unified_notebook import make_notebook
        return make_notebook(dict(self.cfg,project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))

    def _stage_helpers(self):
        super()._stage_helpers()
        for name,source in self.sources.items():
            if name.startswith(('unified_','hard_','shared_stage_')):(self.repo/name).write_text(source,encoding='utf-8')

    def model_config(self,level):return super().model_config('hard')

    def policy_config(self,level):
        policy=super().policy_config(level)
        for source in read_json(self.run_dir/'hard/transfer_sources.json',[]):
            for key in ('gate_threshold','stage_threshold','temporal_decay','ensemble_window'):
                policy[key]=source['policy_config'][key]
        return policy

    def sync_level(self,level):pass  # One atomic common snapshot owns policy + all curriculum metadata.

    def sync_common(self):
        if not (self.remote_enabled and self.store and self.connected):return
        files=list(self.run_dir.glob('*.json'))+list(self.run_dir.glob('*.ipynb'))
        for level in LEVELS:
            files+=list((self.run_dir/level).rglob('*.json'))
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

    def run(self,command,cwd=None,log=None):
        with self.sync_environment(None):return Experiment.run(self,command,cwd=cwd,log=log)

    def _job(self,*args,**kwargs):
        job=super()._job(*args,**kwargs)
        job['unified_sources_sha256']=hashlib.sha256(''.join(self.sources[k] for k in ('unified_policy.py','stage_eval.py','hard_observer.py','hard_progress.py')).encode()).hexdigest()
        identity={k:v for k,v in job.items() if k not in ('fingerprint','output','video')}
        job['fingerprint']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        return job

    def _current(self):
        state=read_json(self.run_dir/'curriculum.json')
        if not state:raise RuntimeError('05 준비 셀부터 실행하세요')
        path=self.run_dir/state['checkpoint']
        if not path.exists() or digest(path)!=state['checkpoint_sha256']:raise ValueError('Current policy is missing or changed')
        return state,path

    def _test_candidate(self,level):
        _,path=self._current();return path,self.policy_config(level)

    def prepare(self):
        self._ready();self.prepare_data();self._stage_helpers()
        for level in LEVELS:(self.run_dir/level).mkdir(exist_ok=True)
        state=read_json(self.run_dir/'curriculum.json')
        if state:
            self._current();print('공유 모델 복원:',state['checkpoint']);return
        donor=HardV3._download_donor(self,'medium',self.cfg['medium_source_run_name'])
        save_json(self.run_dir/'hard/transfer_sources.json',[donor])
        initial=self.run_dir/'hard/checkpoints/medium_initial.pt'
        job=dict(data=str(self.data/'hard/trajectory.state.pd_ee_delta_pos.physx_cuda.h5'),num_demos=self.cfg['num_demos'],
            recovery_manifest=None,seed=self.cfg['seed'],model_config=self.model_config('hard'),
            report=str(self.run_dir/'hard/transfer_report.json'),donors=[dict(level='medium',checkpoint=donor['checkpoint'],output=str(initial))])
        path=self.run_dir/'hard/transfer_job.json';save_json(path,job)
        self.run([sys.executable,'hard_transfer.py',str(path)],cwd=self.repo,log=self.run_dir/'hard/transfer.log')
        save_json(self.run_dir/'curriculum.json',dict(checkpoint=initial.relative_to(self.run_dir).as_posix(),checkpoint_sha256=digest(initial),passed=[],rounds={},history=[]))
        self.sync_common();print('Medium 가중치로 공유 모델 준비. Easy부터 전체 에피소드로 검증합니다.')

    def _train_job(self,level,folder,checkpoint,focus,updates):
        # Medium was pretrained already: rehearse it even during Easy repair.
        included=set(LEVELS[:LEVELS.index(level)+1])|{'medium'}
        sources=[dict(level=k,data=str(self.data/k/'trajectory.state.pd_ee_delta_pos.physx_cuda.h5'),num_demos=self.cfg['num_demos']) for k in LEVELS if k in included]
        for source in sources:
            data=Path(source['data']);metadata=data.with_suffix('.json')
            source.update(data_sha256=digest(data),metadata_sha256=digest(metadata) if metadata.exists() else None)
        manifest=folder/'data.json';folder.mkdir(parents=True,exist_ok=True)
        save_json(manifest,dict(active=level,sources=sources,focus=focus))
        config={k:self.cfg[k] for k in ('seed','batch_size','lr','warmup_steps','kl_weight','position_noise','validation_batches','amp','console_interval_seconds','stage_loss_weight','gate_loss_weight','replay_fraction','speed_bonus')}
        config.update(total_iters=updates,save_freq=updates,action_training_mode='prior')
        return dict(folder=str(folder),data=str(manifest),model_config=self.model_config(level),train_config=config,
            policy_config=self.policy_config(level),warm_start=str(checkpoint),num_demos=None,
            source_sha256=hashlib.sha256(''.join(self.sources[n] for n in sorted(self.sources)).encode()).hexdigest())

    def _evaluate(self,level,checkpoint,label,seeds=None):
        seeds=self.seeds(True) if seeds is None else seeds
        self._trial(level,checkpoint,self.policy_config(level),seeds,label)
        return read_json(self.run_dir/level/(label+'.json'))

    def calibrate(self):
        self._ready();self._stage_helpers();_,checkpoint=self._current()
        folder=self.run_dir/'easy/calibration';job=self._train_job('easy',folder,checkpoint,{},100)
        if not (folder/'resource_report.json').exists():
            path=folder/'job.json';save_json(path,job)
            self.run([sys.executable,'stage_train.py',str(path)],cwd=self.repo,log=folder/'train.log')
        report=read_json(folder/'resource_report.json')
        metrics=self._evaluate('easy',checkpoint,'calibration_eval',self.seeds(True)[:2])
        timing=dict(training_seconds_per_update=report['elapsed_seconds']/100,
            evaluation_seconds_per_episode=metrics['elapsed_seconds']/len(metrics['episodes']),
            peak_allocated_gib=report['peak_allocated_gib'],peak_reserved_gib=report['peak_reserved_gib'])
        save_json(self.run_dir/'timing.json',timing);self.sync_common();return self.estimate_time()

    def estimate_time(self):
        timing=read_json(self.run_dir/'timing.json')
        if not timing:print('06 셀의 100회 실측 후 예상 시간을 표시합니다.');return
        train=timing['training_seconds_per_update']*self.cfg['block_iters']
        evaluation=timing['evaluation_seconds_per_episode']*self.cfg['tuning_episodes']
        if timing['peak_reserved_gib'] is not None:
            print(f"학습 VRAM 최대: 사용 {timing['peak_allocated_gib']:.2f} GiB / 예약 {timing['peak_reserved_gib']:.2f} GiB (시뮬레이터 별도)")
        print(f"학습 {self.cfg['block_iters']}회 약 {train/60:.1f}분 / 개발 평가 {self.cfg['tuning_episodes']}회 약 {evaluation/60:.1f}분")
        for i,level in enumerate(LEVELS):
            rounds=self.cfg['max_blocks'][level]
            # Incumbent + candidate + retention + possible confirmation.
            upper=rounds*(train+(4+max(1,i))*evaluation)
            print(f"{level}: {rounds}구간 최대 예산 약 {upper/60:.0f}분 (Easy 실측 기준, 초기화·다운로드·백업 제외)")
        print('통과/조기 중단으로 단축됩니다. Medium/Hard episode 시간은 이후 실제 평가로 확인해야 합니다.')
        return timing

    def _focus(self,metrics,level):
        votes={i:0 for i in range(COUNTS[level])};stages=[]
        for row in metrics['episodes']:
            if row['mean_sorted']>=COUNTS[level]:continue
            progress=row.get('next_pick',{}).get('parcel_progress',{})
            done={int(i) for i in progress.get('sorted_parcel_first_actions',{})}
            for i in votes:
                if i not in done:votes[i]+=1
            stages.append(row.get('next_pick',{}).get('final_stage',0))
        top=max(votes.values(),default=0)
        return dict(parcel_ids=[i for i,v in votes.items() if v==top and v>0],stage=max(set(stages),key=stages.count) if stages else 0)

    def _pass(self,metrics,level):return summarize(metrics,COUNTS[level])['all_correct_rate']>=self.cfg['pass_rate'][level]

    def _confirm(self,level,checkpoint,state):
        seeds=list(range(self.cfg['confirmation_seed_start'],self.cfg['confirmation_seed_start']+self.cfg['tuning_episodes']))
        result=self._evaluate(level,checkpoint,'confirm_'+digest(checkpoint)[:12],seeds)
        if self._pass(result,level):
            if level not in state['passed']:state['passed'].append(level)
            save_json(self.run_dir/'curriculum.json',state);self.sync_common();print(level,'전체 에피소드 통과');return True
        return False

    def train_level(self,level):
        self._ready();self._stage_helpers();state,current=self._current();index=LEVELS.index(level)
        if any(k not in state['passed'] for k in LEVELS[:index]):
            print('이전 난이도 전체 에피소드 통과가 필요합니다:',list(LEVELS[:index]));return
        if level in state['passed']:print(level,'이미 통과. 다음 난이도 셀을 실행하세요.');return
        stagnant=0
        for number in range(state['rounds'].get(level,0)+1,self.cfg['max_blocks'][level]+1):
            with self.persist_operation(level):
                baseline=self._evaluate(level,current,'baseline_'+digest(current)[:12])
                if self._pass(baseline,level) and self._confirm(level,current,state):return
                focus=self._focus(baseline,level);folder=self.run_dir/level/f'round_{number:02d}'
                job=self._train_job(level,folder,current,focus,self.cfg['block_iters']);path=folder/'job.json';save_json(path,job)
                self.run([sys.executable,'stage_train.py',str(path)],cwd=self.repo,log=folder/'train.log')
                candidate=folder/'checkpoints/candidate.pt'
                if not candidate.exists():
                    import torch
                    saved=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
                    torch.save({k:saved[k] for k in ('format','model_config','model','step')},candidate)
                result=self._evaluate(level,candidate,f'candidate_{number:02d}')
                comparison=paired_comparison(baseline,result);accept=comparison['promote'];reason='score'
                # Speed matters only for identical full-success seed sets.
                if not accept and level!='easy' and all(r['mean_sorted']==COUNTS[level] for r in baseline['episodes']+result['episodes']):
                    def duration(m):return sum(max(r['next_pick']['parcel_progress']['sorted_parcel_first_actions'].values()) for r in m['episodes'])
                    accept=duration(result)<duration(baseline);reason='full_success_speed'
                retention={}
                if accept:
                    for previous in sorted(set(state['passed'])|({'medium'} if level=='easy' else set())):
                        old=self._evaluate(previous,current,'retained_'+digest(current)[:12])
                        new=self._evaluate(previous,candidate,f'retention_{level}_{number:02d}')
                        ok=new['sort_accuracy']+1e-9>=old['sort_accuracy'] and (previous not in state['passed'] or self._pass(new,previous))
                        retention[previous]=ok;accept=accept and ok
                if accept:
                    current=candidate;state.update(checkpoint=current.relative_to(self.run_dir).as_posix(),checkpoint_sha256=digest(current))
                stagnant=0 if accept else stagnant+1
                state['rounds'][level]=number
                state['history'].append(dict(level=level,round=number,score=result['sort_accuracy'],accepted=accept,comparison=comparison,retention=retention,reason=reason,focus=focus))
                save_json(self.run_dir/'curriculum.json',state)
                print(f"{level} {number}구간 · 분류 {result['sort_accuracy']:.1%} · 완주 {summarize(result,COUNTS[level])['all_correct_rate']:.1%} · 채택 {accept}",flush=True)
                if accept and self._pass(result,level) and self._confirm(level,current,state):return
                if stagnant>=self.cfg['plateau_blocks']:print('연속 개선 없음: 현재 모델을 보존하고 이 단계에서 멈춥니다.');break
        print('다음 단계로 넘어가지 않았습니다. 진단 기록:',self.run_dir/'curriculum.json')

    def report(self):
        state,_=self._current();print('현재 공유 모델:',state['checkpoint'])
        for level in LEVELS:
            history=[row for row in state['history'] if row['level']==level]
            status='통과' if level in state['passed'] else '진행/대기'
            print(f"{level}: {status} · 실행 {state['rounds'].get(level,0)}구간 · 채택 {sum(row['accepted'] for row in history)}개")
        print('상세 기록:',self.run_dir/'curriculum.json');return state

    def final_evaluation(self):
        self._ready();state,checkpoint=self._current()
        if set(state['passed'])!=set(LEVELS):print('세 난이도 통과 후 최종 평가를 실행합니다.');return
        with self.persist_operation('hard'):
            for level in LEVELS:self._evaluate(level,checkpoint,'metrics',self.seeds(False))

    def package(self):
        self._ready();state,checkpoint=self._current()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoints').mkdir();shutil.copy2(checkpoint,root/'checkpoints/model.pt')
            save_json(root/'checkpoints/policy_config.json',self.policy_config('hard'))
            save_json(root/'submission.yaml',dict(team=self.cfg['team'],state=dict(policy='colab_policy:load_policy',levels={k:dict(checkpoint='checkpoints/model.pt') for k in LEVELS})))
            save_json(root/'curriculum.json',state)
            for name in ('act_v2_model.py','stage_model.py','stage_policy.py','stage_schema.py','unified_policy.py','colab_policy.py'):
                (root/name).write_text(self.sources[name],encoding='utf-8')
            output=Path(shutil.make_archive(str(self.run_dir/'unified_policy'),'zip',root))
        save_json(self.run_dir/'package_info.json',dict(checkpoint_sha256=state['checkpoint_sha256'],passed=state['passed']))
        self.sync_common();print('공유 모델 하나:',output);return output


def source_bundle():
    root=Path(__file__).parent;project=root.parents[1];bundle=stage_sources()
    for path in root.glob('*.py'):bundle[path.name]=path.read_text(encoding='utf-8')
    for name in ('hard_transfer.py','hard_progress.py','hard_observer.py'):
        bundle[name]=(project/'hard/code'/name).read_text(encoding='utf-8')
    bundle['shared_stage_train.py']=bundle['stage_train.py'];bundle['stage_train.py']=bundle['unified_train.py']
    evaluator=bundle['stage_eval.py'].replace('class Observer(NextPickObserver):','class BaseObserver(NextPickObserver):')
    evaluator=evaluator.replace('def main(job):','from hard_observer import instrument_observer\nObserver = instrument_observer(BaseObserver)\n\ndef main(job):')
    evaluator=evaluator.replace('from stage_policy import load_stage','from unified_policy import load_stage')
    evaluator=evaluator.replace("        agent.trace = job.get('trace_only', False)","        agent.env = env\n        agent.trace = job.get('trace_only', False)")
    evaluator=evaluator.replace('                row = dict(seed=seed, **metrics, next_pick=agent.report())','                agent.capture()\n                row = dict(seed=seed, **metrics, next_pick=agent.report())')
    evaluator=evaluator.replace("    if job.get('video_only'):\n        seed", "    if job.get('video_only'):\n        agent.env = None\n        seed")
    bundle['stage_eval.py']=evaluator;bundle['colab_eval_modular.py']=evaluator;bundle['colab_policy.py']=bundle['unified_policy.py']
    return bundle
