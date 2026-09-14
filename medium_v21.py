"""Medium v2.1: collect partial successes, expand the budget, and prefer efficient successes."""
import hashlib
import os
import sys
import zipfile
from pathlib import Path
from medium_v2 import MediumV2, source_bundle as v2_sources
from medium_lab import quality
from marso_experiment import read_json, save_json


def trial_quality(metrics):
    return dict(quality(metrics),mean_completion_actions=metrics['mean_completion_actions'],
        mean_successful_actions=metrics['mean_successful_actions'])


def rank(result):
    # Success remains primary; time is a tie-break only, never a substitute for sorting.
    successful_actions=result.get('mean_successful_actions')
    return result['score'],result['all_sorted_rate'],-(successful_actions if successful_actions is not None else 199),result['first_grasp_rate']


class MediumV21(MediumV2):
    def __init__(self,cfg,sources):
        super().__init__(cfg,sources)
        if self.cfg['minimum_correct'] not in (3,4):raise ValueError('minimum_correct must be 3 or 4')
        if not 0<=self.cfg['efficiency_bonus']<=1:raise ValueError('efficiency_bonus must be in [0,1]')

    def notebook_snapshot(self):
        from build_medium_v21_notebook import make_notebook
        return make_notebook(dict(self.cfg,project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))

    def _stage_helpers(self):
        super()._stage_helpers()
        for name,source in self.sources.items():
            if name.startswith('curriculum_'):(self.repo/name).write_text(source,encoding='utf-8')

    def collect_deadline(self,operation='collect'):
        self._ready();self._stage_helpers()
        if operation not in ('pilot','collect'):raise ValueError('Unknown collection operation')
        folder=self.run_dir/'medium';c=self.cfg
        if not (folder/'v2_origin.json').exists():self.prepare()
        config=dict(episodes=c['deadline_episodes'],max_attempts=c['deadline_max_attempts'],
            gains=c['teacher_gains'],pilot_episodes=c['pilot_episodes'],pilot_seed_start=c['pilot_seed_start'],
            seed_start=c['deadline_seed_start'],noise_std=c['deadline_noise_std'],minimum_correct=c['minimum_correct'],
            noise_probability=c['deadline_noise_probability'],seconds_per_call=c['collection_seconds_per_call'])
        job=dict(operation=operation,folder=str(folder/'collection'),collection_config=config,
            repo_commit=c['repo_commit'],source_sha256=hashlib.sha256(''.join(self.sources[k] for k in
                ('deadline_teacher.py','curriculum_collect.py','stage_labels.py')).encode()).hexdigest())
        with self.persist_operation('medium'):
            save_json(folder/'curriculum_collection_job.json',job)
            self.run([sys.executable,'curriculum_collect.py',str(folder/'curriculum_collection_job.json')],
                cwd=self.repo,log=folder/('curriculum_'+operation+'.log'))
            manifest=read_json(folder/'collection/manifest.json',{})
            if manifest.get('complete'):
                save_json(folder/'inputs_ready.json',dict(episodes=len(manifest['episodes']),evaluation_actions=199,
                    minimum_correct=c['minimum_correct']))
        episodes=manifest.get('episodes',[])
        print(f"시연 상태: {manifest.get('status','진행 중')} · 학습 채택 {len(episodes)}/{c['deadline_episodes']} · "
              f"네 상자 완주 {sum(r['fully_successful'] for r in episodes)}")
        return manifest.get('status')

    def training_job(self):
        job=super().training_job()
        job['train_config']['efficiency_bonus']=self.cfg['efficiency_bonus']
        job['source_sha256']=hashlib.sha256(''.join(self.sources[k] for k in sorted(self.sources)
            if k.startswith(('stage_','act_v2_','deadline_','curriculum_'))).encode()).hexdigest()
        return job

    def report(self):
        self._ready();folder=self.run_dir/'medium'
        manifest=read_json(folder/'collection/manifest.json',{})
        episodes=manifest.get('episodes',[])
        full=sum(r['fully_successful'] for r in episodes)
        print(f'학습 채택 시연 {len(episodes)}개: 부분 성공 {len(episodes)-full} / 네 상자 완주 {full}')
        if full<2:print('완주 시연이 아직 부족해 시간 보너스는 적용되지 않습니다.')
        history=read_json(folder/'lab_history.json',dict(blocks=[],status='not_started'))
        for row in history['blocks']:
            result=row['best'];steps=result.get('mean_successful_actions')
            duration=f'{steps:.1f}행동' if steps is not None else '완주 없음'
            print(f"{row['iteration']}회 · 분류 {result['score']:.1%} · 네 상자 완주 {result['all_sorted_rate']:.0%} · 완주 시간 {duration}")
        print('상태:',history['status'])
        return history

    def run_blocks(self):
        self._ready();self._stage_helpers()
        folder=self.run_dir/'medium'
        if not read_json(folder/'collection/manifest.json',{}).get('complete'):
            print('학습 시연이 아직 부족합니다. 한 차례의 수집 시간 예산만큼 이어서 진행합니다.')
            self.collect_deadline('collect')
            if not read_json(folder/'collection/manifest.json',{}).get('complete'):
                print('학습 대기: 수집 진행은 저장됐습니다. 07 또는 08 셀을 다시 실행하세요.')
                return None
        job=self.training_job()
        identity=dict(job=job,development_seeds=self.seeds(True),windows=self.cfg['ensemble_candidates'],
            target=self.cfg['target_accuracy'],success_streak=self.cfg['success_streak'],plateau=self.cfg['plateau_blocks'],
            ranking='sort_accuracy, all_sorted_rate, lower mean_successful_actions (full successes only), first_grasp_rate')
        previous=read_json(folder/'lab_protocol.json')
        if previous and previous!=identity:raise ValueError('Saved training protocol differs. Use a new run_name.')
        save_json(folder/'lab_protocol.json',identity)
        history=read_json(folder/'lab_history.json',dict(blocks=[],status='running'))
        if history['status'] in ('target_reached','plateau','budget_reached'):return self.report()
        with self.persist_operation('medium'):
            if self.cfg['warm_start'] and not (folder/'lab_best.json').exists():
                initial=folder/'checkpoints/initial_model.pt'
                baseline=self._trial('medium',initial,self.policy_config('medium'),self.seeds(True),'baseline')
                baseline.update(trial_quality(read_json(folder/'baseline.json')),block=0,iteration=0)
                save_json(folder/'lab_best.json',baseline)
                print(f"보정 전 같은 시드 기준 분류: {baseline['score']:.1%}")
            for block in range(len(history['blocks'])+1,self.cfg['max_blocks']+1):
                stop=block*self.cfg['block_iters']
                save_json(folder/'stage_train_job.json',dict(job,stop_at=stop))
                self.run([sys.executable,'stage_train.py',str(folder/'stage_train_job.json')],cwd=self.repo,
                    log=folder/f'train_block_{block:02d}.log')
                import torch
                checkpoint=folder/'checkpoints'/f'block_{block:02d}.pt'
                if not checkpoint.exists():
                    saved=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
                    if saved['step']!=stop:raise ValueError('Training checkpoint step mismatch')
                    temporary=Path(str(checkpoint)+'.tmp')
                    torch.save({k:saved[k] for k in ('format','model_config','model','step')},temporary)
                    os.replace(temporary,checkpoint)
                trials=[]
                for window in self.cfg['ensemble_candidates']:
                    label=f'dev_b{block:02d}_w{window}'
                    trial=self._trial('medium',checkpoint,dict(self.policy_config('medium'),ensemble_window=window),self.seeds(True),label)
                    trial.update(trial_quality(read_json(folder/(label+'.json'))),block=block,iteration=stop)
                    trials.append(trial)
                best=max(trials,key=rank);previous=read_json(folder/'lab_best.json')
                improved=previous is None or rank(best)>rank(previous)
                if improved:save_json(folder/'lab_best.json',best)
                history['blocks'].append(dict(block=block,iteration=stop,best=best,improved=improved,trials=trials))
                recent=history['blocks'][-self.cfg['success_streak']:]
                target=len(recent)==self.cfg['success_streak'] and all(r['best']['all_sorted_rate']>=self.cfg['target_accuracy'] for r in recent)
                stagnant=history['blocks'][-self.cfg['plateau_blocks']:]
                plateau=len(stagnant)==self.cfg['plateau_blocks'] and not any(r['improved'] for r in stagnant)
                history['status']='target_reached' if target else 'plateau' if plateau else 'budget_reached' if block==self.cfg['max_blocks'] else 'running'
                save_json(folder/'lab_history.json',history);self.sync_level('medium')
                print(f"{stop}회 · 분류 {best['score']:.1%} · 네 상자 완주 {best['all_sorted_rate']:.0%} · "
                      f"달성 개수까지 평균 {best['mean_completion_actions']:.1f}행동")
                if history['status']!='running':break
        return self.report()

    def package(self):
        path=super().package()
        if path is None:return
        temporary=Path(str(path)+'.tmp')
        with zipfile.ZipFile(path) as src,zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as dst:
            for entry in src.infolist():
                if not Path(entry.filename).name.startswith('curriculum_'):dst.writestr(entry,src.read(entry.filename))
        os.replace(temporary,path);self.sync_common()
        return path


def source_bundle():
    bundle=v2_sources();root=Path(__file__).parent
    for path in (root/'ver2/curriculum').glob('curriculum_*.py'):
        bundle[path.name]=path.read_text(encoding='utf-8')
    bundle['stage_train.py']=bundle['curriculum_train.py']
    bundle['colab_eval_modular.py']=bundle['curriculum_eval.py']
    bundle['colab_trace.py']=bundle['colab_trace.py'].replace('from stage_eval import main','from curriculum_eval import main')
    for name in ('medium_v21.py','build_medium_v21_notebook.py'):
        bundle[name]=(root/name).read_text(encoding='utf-8')
    return bundle
