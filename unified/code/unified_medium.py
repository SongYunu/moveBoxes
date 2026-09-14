"""Medium adaptation from the exact v2 Easy weights; Easy regression is diagnostic."""
import sys,shutil,tempfile
from pathlib import Path
from unified_v2 import UnifiedV2,source_bundle as v2_sources
from unified_import import SnapshotSource
from marso_experiment import read_json,save_json,digest
from hard_progress import paired_comparison,summarize


def medium_selection(baseline,candidate):
    comparison=paired_comparison(baseline,candidate)
    if comparison['promote']:return True,'sort_accuracy',comparison
    left=baseline['episodes'];right=candidate['episodes']
    # Speed only breaks a tie when every seed has the same number of correct boxes.
    if any(a['mean_sorted']!=b['mean_sorted'] for a,b in zip(left,right)):return False,'score',comparison
    pairs=[(a,b) for a,b in zip(left,right) if a['mean_sorted']==4 and b['mean_sorted']==4]
    if not pairs:return False,'no_shared_full_success',comparison
    def steps(row):return max(row['next_pick']['parcel_progress']['sorted_parcel_first_actions'].values())
    return sum(steps(b) for a,b in pairs)<sum(steps(a) for a,b in pairs),'paired_success_speed',comparison


class UnifiedMedium(UnifiedV2):
    def __init__(self,cfg,sources):
        super().__init__(cfg,sources)
        if self.cfg['easy_check_interval']<1:raise ValueError('easy_check_interval must be positive')

    def notebook_snapshot(self):
        from build_unified_medium_notebook import make_notebook
        return make_notebook(dict(self.cfg,project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))

    def prepare(self):
        self._ready();self.prepare_data();self._stage_helpers()
        for level in ('easy','medium','hard'):(self.run_dir/level).mkdir(exist_ok=True)
        if read_json(self.run_dir/'curriculum.json'):
            state,_=self._current()
            if state.get('version')!='medium-3':raise ValueError('Medium 전용 새 run_name을 사용하세요')
            self._prepare_demonstrations()
            print('Medium 선택 모델 복원. Easy 통과 재실행 없이 학습할 수 있습니다.');return
        origin=SnapshotSource(self.cfg['github_repository'],self.cfg['unified_source_run_name'],'common',self.run_dir/'easy/source')
        verified=read_json(origin.fetch('initial_verified.json'))
        reference=self.run_dir/'easy/checkpoints/easy_reference.pt'
        source_path='easy/checkpoints/easy_initial.pt'
        if origin.entries[source_path]['sha256']!=verified['converted_sha256']:raise ValueError('v2 Easy verification/checkpoint mismatch')
        origin.fetch(source_path,reference)
        provenance=read_json(origin.fetch('hard/transfer_sources.json'))
        save_json(self.run_dir/'hard/transfer_sources.json',provenance)
        episodes=self._prepare_demonstrations()
        save_json(self.run_dir/'curriculum.json',dict(version='medium-3',checkpoint=reference.relative_to(self.run_dir).as_posix(),
            checkpoint_sha256=digest(reference),easy_reference=reference.relative_to(self.run_dir).as_posix(),
            easy_reference_sha256=digest(reference),passed=[],rounds={},history=[],learners={},
            source_snapshot=origin.name,source_easy_score=verified['converted_score']))
        self.sync_common();print(f'v2 Easy 파일 그대로 복원 / 기존 Medium 시연 {len(episodes)}개 / Medium부터 학습')

    def _prepare_demonstrations(self):
        # Reuse the already collected deadline-limited Medium demonstrations.
        demos=SnapshotSource(self.cfg['github_repository'],self.cfg['demonstration_source_run_name'],'medium',self.run_dir/'medium/demo_source')
        manifest=read_json(demos.fetch('medium/collection/manifest.json'))
        from github_store import safe_target
        for entry in manifest['episodes']:
            safe_target(self.run_dir/'medium/collection',entry['file'])
            target=demos.fetch('medium/collection/'+entry['file'],self.run_dir/'medium/collection'/entry['file'])
            if digest(target)!=entry['sha256']:raise ValueError('Demonstration hash mismatch')
        save_json(self.run_dir/'medium/collection/manifest.json',manifest)
        # Validate original deadline labels/shapes before GPU training.
        from unified_deadline_data import load_data
        episodes=load_data(self.run_dir/'medium/collection/manifest.json')
        save_json(self.run_dir/'medium/inputs_ready.json',dict(episodes=len(episodes),
            full_successes=sum(t['fully_successful'] for t in episodes),source_snapshot=demos.name))
        return episodes

    def sync_common(self):
        # Main selection remains one checkpoint; immutable Easy reference is archived separately.
        super().sync_common()
        if self.remote_enabled and self.store and self.connected:
            state=read_json(self.run_dir/'curriculum.json',{})
            if state.get('easy_reference'):
                self.store.sync(self.run_dir,'easy',[self.run_dir/state['easy_reference']])

    def _train_job(self,level,folder,checkpoint,focus,updates):
        job=super()._train_job(level,folder,checkpoint,focus,updates)
        manifest=read_json(job['data']);path=self.run_dir/'medium/collection/manifest.json'
        manifest['sources'].append(dict(level='medium',data=str(path),format='curriculum-v21',manifest_sha256=digest(path)))
        save_json(job['data'],manifest)
        job['train_config']['contact_sampling']=True
        return job

    def train_medium(self):
        self._ready();self._stage_helpers();state,current=self._current();level='medium'
        # No prerequisite gate, no Easy veto, no plateau stop. Finite configured budget remains.
        for number in range(state['rounds'].get(level,0)+1,self.cfg['max_blocks'][level]+1):
            with self.persist_operation(level):
                baseline=self._evaluate(level,current,'baseline_'+digest(current)[:12])
                learner,job=self._learner_job(state,level,number,current,baseline)
                save_json(self.run_dir/'curriculum.json',state)
                path=Path(job['folder'])/'job.json';save_json(path,job)
                self.run([sys.executable,'stage_train.py',str(path)],cwd=self.repo,log=Path(job['folder'])/'train.log')
                import torch
                saved=torch.load(Path(job['folder'])/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
                if saved['step']!=job['stop_at']:raise RuntimeError('Incomplete training segment')
                candidate=self.run_dir/level/f'candidates/round_{number:02d}.pt';candidate.parent.mkdir(exist_ok=True)
                torch.save({k:saved[k] for k in ('format','model_config','model','step')},candidate)
                result=self._evaluate(level,candidate,f'candidate_{number:02d}')
                accept,reason,comparison=medium_selection(baseline,result)
                if accept:
                    current=candidate;state.update(checkpoint=current.relative_to(self.run_dir).as_posix(),checkpoint_sha256=digest(current))
                learner['focus']=self._focus(result,level)
                record=dict(level=level,round=number,learner_step=saved['step'],score=result['sort_accuracy'],
                    accepted=accept,comparison=comparison,reason=reason,focus=learner['focus'],easy_score=None)
                state['rounds'][level]=number;state['history'].append(record)
                save_json(self.run_dir/'curriculum.json',state)
                # Record occasional Easy regression; it does not revoke a Medium improvement.
                if accept and number%self.cfg['easy_check_interval']==0:
                    easy=self._evaluate('easy',current,f'medium_diagnostic_{number:02d}')
                    record['easy_score']=easy['sort_accuracy'];save_json(self.run_dir/'curriculum.json',state)
                print(f"Medium {saved['step']}회 · 분류 {result['sort_accuracy']:.1%} · 네 상자 완주 {summarize(result,4)['all_correct_rate']:.1%} · Medium 최고 교체 {accept}")
        print('설정한 Medium 학습 예산 완료. 08 테스트에서 선택 모델을 확인하세요.')

    def final_evaluation(self):
        self._ready();_,current=self._current()
        with self.persist_operation('medium'):self._evaluate('medium',current,'metrics',self.seeds(False))

    def package(self):
        self._ready();state,current=self._current()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoints').mkdir();shutil.copy2(current,root/'checkpoints/model.pt')
            save_json(root/'checkpoints/policy_config.json',self.policy_config('medium'))
            save_json(root/'submission.yaml',dict(team=self.cfg['team'],state=dict(policy='colab_policy:load_policy',levels={'medium':dict(checkpoint='checkpoints/model.pt')})))
            for name in ('act_v2_model.py','stage_model.py','stage_policy.py','stage_schema.py','unified_policy.py','colab_policy.py'):
                (root/name).write_text(self.sources[name],encoding='utf-8')
            output=Path(shutil.make_archive(str(self.run_dir/'unified_policy'),'zip',root))
        save_json(self.run_dir/'package_info.json',dict(checkpoint_sha256=digest(current),levels=['medium']))
        self.sync_common();print('Medium 전용 패키지:',output);return output


def source_bundle():
    bundle=v2_sources();root=Path(__file__).resolve().parents[2]
    bundle['unified_deadline_data.py']=(root/'medium/code/curriculum/curriculum_data.py').read_text(encoding='utf-8')
    return bundle
