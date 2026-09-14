"""Start from verified Easy; keep a continuous learner separate from the selected policy."""
import sys
from pathlib import Path
from unified_experiment import UnifiedExperiment,source_bundle as original_sources,LEVELS,COUNTS
from hard_v3 import HardV3
from hard_progress import paired_comparison,summarize
from marso_experiment import read_json,save_json,digest


class UnifiedV2(UnifiedExperiment):
    def __init__(self,cfg,sources):
        super().__init__(cfg,sources)
        if not 0<=self.cfg['focus_fraction']<1:raise ValueError('focus_fraction must be in [0,1)')

    def notebook_snapshot(self):
        from build_unified_v2_notebook import make_notebook
        return make_notebook(dict(self.cfg,project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))

    def prepare(self):
        self._ready();self.prepare_data();self._stage_helpers()
        for level in LEVELS:(self.run_dir/level).mkdir(exist_ok=True)
        if read_json(self.run_dir/'curriculum.json'):
            state,_=self._current()
            if state.get('version')!=2:raise ValueError('v2는 별도 run_name을 사용하세요.')
            print('선택 모델과 커리큘럼 복원 완료');return
        donor=HardV3._download_donor(self,'easy',self.cfg['easy_source_run_name'])
        save_json(self.run_dir/'hard/transfer_sources.json',[donor])
        initial=self.run_dir/'easy/checkpoints/easy_initial.pt'
        job=dict(data=str(self.data/'hard/trajectory.state.pd_ee_delta_pos.physx_cuda.h5'),num_demos=self.cfg['num_demos'],
            recovery_manifest=None,seed=self.cfg['seed'],model_config=self.model_config('hard'),
            report=str(self.run_dir/'hard/transfer_report.json'),donors=[dict(level='easy',checkpoint=donor['checkpoint'],output=str(initial))])
        path=self.run_dir/'hard/transfer_job.json';save_json(path,job)
        self.run([sys.executable,'hard_transfer.py',str(path)],cwd=self.repo,log=self.run_dir/'hard/transfer.log')
        save_json(self.run_dir/'curriculum.json',dict(version=2,checkpoint=initial.relative_to(self.run_dir).as_posix(),
            checkpoint_sha256=digest(initial),passed=[],rounds={},history=[],learners={}))
        self.sync_common();print('기존 Easy 최고 모델로 초기화했습니다. 새 시드의 전체 평가로 다시 확인합니다.')

    def _train_job(self,level,folder,checkpoint,focus,updates):
        job=super()._train_job(level,folder,checkpoint,focus,updates)
        manifest=read_json(job['data']);included=set(LEVELS[:LEVELS.index(level)+1])
        manifest['sources']=[s for s in manifest['sources'] if s['level'] in included]
        save_json(job['data'],manifest)
        job['train_config']['focus_fraction']=self.cfg['focus_fraction']
        job['train_config']['save_freq']=min(updates,self.cfg['block_iters'])
        return job

    def verify_initial(self):
        """Compare the untouched native Easy model with its canonical form on the same seeds."""
        self._ready();self._stage_helpers();state,current=self._current()
        if read_json(self.run_dir/'initial_verified.json'):return
        if state['history']:raise RuntimeError('초기 변환 검증 전에 학습 기록이 생겼습니다. 새 run_name을 사용하세요.')
        donor=HardV3._download_donor(self,'easy',self.cfg['easy_source_run_name'])
        label='native_easy_reference';job=self._job('easy',Path(donor['checkpoint']),donor['policy_config'],self.seeds(True),label)
        job_path=self.run_dir/'easy/native_job.json';save_json(job_path,job)
        self.run([sys.executable,'unified_native_eval.py',str(job_path)],cwd=self.repo,log=self.run_dir/'easy/native.log')
        native=read_json(job['output']);converted=self._evaluate('easy',current,'initial_converted')
        if not native.get('complete') or not converted.get('complete') or len(native['episodes'])!=len(converted['episodes']):raise ValueError('초기 검증 미완료')
        for left,right in zip(native['episodes'],converted['episodes']):
            if left['seed']!=right['seed'] or abs(left['sort_accuracy']-right['sort_accuracy'])>1e-7:
                raise RuntimeError('원본 Easy와 변환 모델의 성능이 다릅니다. 보정 학습을 시작하지 않았습니다.')
        save_json(self.run_dir/'initial_verified.json',dict(native_sha256=donor['sha256'],converted_sha256=digest(current),
            native_score=native['sort_accuracy'],converted_score=converted['sort_accuracy'],seeds=self.seeds(True)))
        self.sync_common();print(f"초기 Easy 비교: 원본 {native['sort_accuracy']:.1%} / 공유 모델 {converted['sort_accuracy']:.1%}")

    def _learner_job(self,state,level,number,current,baseline):
        entry=state.setdefault('learners',{}).get(level)
        if not entry or not (self.run_dir/entry['folder']/'checkpoints/latest.pt').exists():
            entry=dict(folder=f'{level}/learner_{number:02d}',start_round=number,
                anchor=current.relative_to(self.run_dir).as_posix(),focus=self._focus(baseline,level))
            state['learners'][level]=entry
            print('연속 학습 구간 시작:',entry['folder'])
        folder=self.run_dir/entry['folder']
        total=(self.cfg['max_blocks'][level]-entry['start_round']+1)*self.cfg['block_iters']
        job=self._train_job(level,folder,self.run_dir/entry['anchor'],entry['focus'],total)
        job['stop_at']=(number-entry['start_round']+1)*self.cfg['block_iters']
        return entry,job

    def train_level(self,level):
        self._ready();self._stage_helpers()
        if level=='easy':self.verify_initial()
        state,current=self._current();index=LEVELS.index(level)
        if any(k not in state['passed'] for k in LEVELS[:index]):
            print('이전 난이도 전체 에피소드 통과 후 실행하세요.');return
        if level in state['passed']:print(level,'이미 통과. 재학습 생략.');return
        stagnant=0
        for number in range(state['rounds'].get(level,0)+1,self.cfg['max_blocks'][level]+1):
            with self.persist_operation(level):
                baseline=self._evaluate(level,current,'baseline_'+digest(current)[:12])
                if self._pass(baseline,level) and self._confirm(level,current,state):return
                learner,job=self._learner_job(state,level,number,current,baseline)
                save_json(self.run_dir/'curriculum.json',state)
                path=Path(job['folder'])/'job.json';save_json(path,job)
                self.run([sys.executable,'stage_train.py',str(path)],cwd=self.repo,log=Path(job['folder'])/'train.log')
                import torch
                saved=torch.load(Path(job['folder'])/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
                if saved['step']!=job['stop_at']:raise RuntimeError('학습 구간 완료 step 불일치')
                candidate=self.run_dir/level/f'candidates/round_{number:02d}.pt';candidate.parent.mkdir(exist_ok=True)
                torch.save({k:saved[k] for k in ('format','model_config','model','step')},candidate)
                result=self._evaluate(level,candidate,f'candidate_{number:02d}')
                comparison=paired_comparison(baseline,result);accept=comparison['promote'];reason='score'
                if not accept and level!='easy' and all(r['mean_sorted']==COUNTS[level] for r in baseline['episodes']+result['episodes']):
                    def duration(m):return sum(max(r['next_pick']['parcel_progress']['sorted_parcel_first_actions'].values()) for r in m['episodes'])
                    accept=duration(result)<duration(baseline);reason='full_success_speed'
                retention={}
                if accept:
                    for previous in state['passed']:
                        old=self._evaluate(previous,current,'retained_'+digest(current)[:12])
                        new=self._evaluate(previous,candidate,f'retention_{level}_{number:02d}')
                        retention[previous]=new['sort_accuracy']+1e-9>=old['sort_accuracy'] and self._pass(new,previous)
                    accept=all(retention.values())
                if accept:
                    current=candidate;state.update(checkpoint=current.relative_to(self.run_dir).as_posix(),checkpoint_sha256=digest(current))
                # The learner retains weights, optimizer and RNG even when the candidate loses.
                learner['focus']=self._focus(result,level)
                state['rounds'][level]=number
                state['history'].append(dict(level=level,round=number,learner_step=saved['step'],score=result['sort_accuracy'],
                    accepted=accept,comparison=comparison,retention=retention,reason=reason,focus=learner['focus']))
                save_json(self.run_dir/'curriculum.json',state)
                print(f"{level} {number}구간 / 이번 연속 학습 {saved['step']}회 · 분류 {result['sort_accuracy']:.1%} · 완주 {summarize(result,COUNTS[level])['all_correct_rate']:.1%} · 최고 모델 교체 {accept}")
                if accept and self._pass(result,level) and self._confirm(level,current,state):return
                stagnant=0 if accept else stagnant+1
                if stagnant>=self.cfg['plateau_blocks']:
                    print('개선 없음: 최고 모델 보존. 예산이 남으면 같은 셀로 현재 학습을 이어갈 수 있습니다.');break
        print('현재 단계 종료. 다음 단계 통과 여부:',state['passed'])


def source_bundle():
    bundle=original_sources()
    shared=bundle['shared_stage_train.py']
    marker="    save_json(folder/'data_audit.json', dict("
    assert shared.count(marker)==1 and shared.count("if saved['signature'] != signature:")==1
    shared=shared.replace(marker,"    from unified_resume import dataset_identity, resume_matches\n    signature['curriculum_dataset_sha256'] = dataset_identity(job)\n"+marker)
    shared=shared.replace("if saved['signature'] != signature:","if not resume_matches(saved['signature'], signature):")
    bundle['shared_stage_train.py']=shared
    bundle['unified_native_eval.py']=bundle['stage_eval.py'].replace('from unified_policy import load_stage','from stage_policy import load_stage')
    return bundle
