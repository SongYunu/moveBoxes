"""Medium-only bounded training blocks, simulator selection and resumable results."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from stage_experiment import StageExperiment, source_bundle as stage_sources
from marso_experiment import read_json, save_json, digest
from github_store import GitHubStore, safe_target


def quality(metrics):
    rows = metrics['episodes']
    first = sum(r.get('next_pick',{}).get('stable_grasp_cycles',0) >= 1 for r in rows)/len(rows)
    all_sorted = sum(r['mean_sorted'] >= 4 for r in rows)/len(rows)
    return dict(score=metrics['sort_accuracy'], first_grasp_rate=first, all_sorted_rate=all_sorted)


def rank(result):
    return result['score'],result['all_sorted_rate'],result['first_grasp_rate']


class MediumLab(StageExperiment):
    def __init__(self, cfg, sources):
        super().__init__(cfg,sources)
        c = self.cfg
        c['action_training_mode'] = 'prior'
        c['total_iters'] = dict(easy=1,medium=c['block_iters']*c['max_blocks'],hard=1)
        c['tuning_episodes'] = c['development_episodes']
        c['benchmark_episodes'] = c['final_episodes']
        if min(c['block_iters'],c['max_blocks'],c['development_episodes'],c['final_episodes'],
               c['success_streak'],c['plateau_blocks']) < 1:
            raise ValueError('Medium experiment budgets must be positive')
        if not 0 < c['target_accuracy'] <= 1 or not 0 <= c['first_pick_fraction'] < 1:
            raise ValueError('Invalid target_accuracy/first_pick_fraction')
        if c['max_episode_steps']['medium'] != 200:
            raise ValueError('Medium lab keeps the 200-step evaluation protocol')

    def notebook_snapshot(self):
        from build_medium_notebook import make_notebook
        return make_notebook(dict(self.cfg,project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))

    def check_runtime(self):
        self._ready()
        code = """import torch
from warehouse_sort.utils import compose_cfg, make_env
assert torch.cuda.is_available(), 'T4 GPU runtime required'
cfg = compose_cfg(['difficulty=medium', 'num_envs=1'])
env, _ = make_env(cfg, 'state', cfg.randomization, num_envs=1, render_mode='rgb_array')
try:
    obs, _ = env.reset(seed=42)
    assert tuple(obs.shape) == (1,72)
    env.step(torch.zeros((1,4),device='cuda'))
    assert env.render() is not None
    print('Medium GPU/state/render OK')
finally:
    env.close()
"""
        self.run([sys.executable,'-c',code],cwd=self.repo,log=self.session/'medium_gpu_check.log')

    def prepare(self):
        self._ready()
        self.prepare_data()
        folder = self.run_dir/'medium'
        with self.persist_operation('medium'):
            ready = read_json(folder/'inputs_ready.json')
            if not ready and self.cfg.get('source_run_name'):
                tag = 'run-'+self.cfg['source_run_name']+'_benchmark'
                store = GitHubStore(self.cfg['github_repository'],tag)
                if not store.load(create=False):
                    raise FileNotFoundError(f'기존 실행 Release가 없습니다: {tag}. source_run_name을 확인하세요.')
                snapshots = sorted(n for n in store.assets if n.startswith('snapshot-medium-'))
                if not snapshots:
                    raise FileNotFoundError('기존 실행에 Medium snapshot이 없습니다.')
                folder.mkdir(parents=True,exist_ok=True)
                snapshot_path = folder/'input_snapshot.json'
                store.download(store.assets[snapshots[-1]],snapshot_path)
                snapshot = read_json(snapshot_path)
                if snapshot.get('scope') != 'medium' or snapshot.get('version') != 1:
                    raise ValueError('Invalid source snapshot')
                for entry in snapshot['entries']:
                    parts = Path(entry['path']).parts
                    wanted = len(parts)==3 and parts[:2]==('medium','collection') and Path(parts[-1]).suffix in ('.json','.npz')
                    model = entry['path']=='medium/checkpoints/latest.pt' and self.cfg['warm_start']
                    if not wanted and not model:
                        continue
                    destination = folder/'initial_model.pt' if model else safe_target(self.run_dir,entry['path'])
                    asset = store.assets.get(entry['asset'])
                    if not asset or asset['size'] != entry['bytes']:
                        raise ValueError('Incomplete source snapshot')
                    if destination.exists():
                        if digest(destination) != entry['sha256']:
                            raise ValueError('Source snapshot changed. Use a new Medium run_name.')
                    else:
                        temporary = destination.with_name(destination.name+'.part')
                        store.download(asset,temporary)
                        if digest(temporary) != entry['sha256']:
                            raise ValueError('Source data checksum mismatch')
                        os.replace(temporary,destination)
            elif not ready:
                self.collect('medium')
            manifest = read_json(folder/'collection/manifest.json',{})
            if not manifest.get('complete'):
                raise RuntimeError('Medium 성공 시연 수집이 미완료입니다.')
            for entry in manifest['episodes']:
                path = safe_target(folder/'collection',entry['file'])
                if digest(path) != entry['sha256']:
                    raise ValueError('Medium collection checksum mismatch')
            if self.cfg['warm_start'] and not (folder/'initial_model.pt').exists():
                raise RuntimeError('기존 Medium 모델이 없습니다. 새로 학습하려면 warm_start=False로 새 실험을 시작하세요.')
            save_json(folder/'inputs_ready.json',dict(source_run_name=self.cfg.get('source_run_name'),
                episodes=len(manifest['episodes']),warm_start=self.cfg['warm_start']))
        print(f'Medium 성공 시연 {len(manifest["episodes"])}개 준비 완료. 반복 학습 셀을 실행하세요.')

    def training_job(self):
        folder = self.run_dir/'medium'
        if not (folder/'inputs_ready.json').exists():
            raise RuntimeError('먼저 Medium 시연/모델 준비 셀을 실행하세요.')
        cfg = {k:self.cfg[k] for k in ('seed','batch_size','lr','warmup_steps','kl_weight','position_noise',
            'validation_batches','amp','console_interval_seconds','stage_loss_weight','gate_loss_weight','first_pick_fraction')}
        cfg.update(total_iters=self.cfg['total_iters']['medium'],save_freq=self.cfg['block_iters'],action_training_mode='prior')
        names = ('act_v2_model.py','act_v2_data.py','stage_schema.py','stage_labels.py','stage_model.py','stage_data.py','stage_train.py')
        job = dict(folder=str(folder),data=str(self.data/'medium/trajectory.state.pd_ee_delta_pos.physx_cuda.h5'),
            model_config=self.model_config('medium'),train_config=cfg,policy_config=self.policy_config('medium'),
            recovery_manifest=str(folder/'collection/manifest.json'),num_demos=self.cfg['num_demos'],
            source_sha256=hashlib.sha256(''.join(self.sources[n] for n in names).encode()).hexdigest())
        if self.cfg['warm_start']:
            job['warm_start'] = str(folder/'initial_model.pt')
        return job

    def run_blocks(self):
        self._ready()
        self._stage_helpers()
        folder = self.run_dir/'medium'
        job = self.training_job()
        identity = dict(job=job,development_seeds=self.seeds(True),windows=self.cfg['ensemble_candidates'],
                        target=self.cfg['target_accuracy'],success_streak=self.cfg['success_streak'],plateau=self.cfg['plateau_blocks'])
        old = read_json(folder/'lab_protocol.json')
        if old and old != identity:
            raise ValueError('Saved Medium protocol differs. Use a new run_name.')
        save_json(folder/'lab_protocol.json',identity)
        history = read_json(folder/'lab_history.json',dict(blocks=[],status='running'))
        if history['status'] in ('target_reached','plateau','budget_reached'):
            print('저장된 Medium 반복 종료 상태:',history['status'])
            return self.report()
        with self.persist_operation('medium'):
            for block in range(len(history['blocks'])+1,self.cfg['max_blocks']+1):
                stop = block*self.cfg['block_iters']
                save_json(folder/'stage_train_job.json',dict(job,stop_at=stop))
                self.run([sys.executable,'stage_train.py',str(folder/'stage_train_job.json')],cwd=self.repo,
                         log=folder/f'train_block_{block:02d}.log')
                # Freeze this candidate before evaluation; future training cannot change its identity.
                import torch
                checkpoint = folder/'checkpoints'/f'block_{block:02d}.pt'
                if not checkpoint.exists():
                    state = torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
                    if state['step'] != stop:
                        raise ValueError('Checkpoint step differs from the requested block boundary')
                    temporary = Path(str(checkpoint)+'.tmp')
                    torch.save({k:state[k] for k in ('format','model_config','model','step')},temporary)
                    os.replace(temporary,checkpoint)
                trials = []
                for window in self.cfg['ensemble_candidates']:
                    label = f'dev_b{block:02d}_w{window}'
                    policy = dict(self.policy_config('medium'),ensemble_window=window)
                    trial = self._trial('medium',checkpoint,policy,self.seeds(True),label)
                    trial.update(quality(read_json(folder/(label+'.json'))),block=block,iteration=stop)
                    trials.append(trial)
                best = max(trials,key=rank)
                previous = read_json(folder/'lab_best.json')
                improved = previous is None or rank(best)>rank(previous)
                if improved:
                    save_json(folder/'lab_best.json',best)
                history['blocks'].append(dict(block=block,iteration=stop,best=best,improved=improved,trials=trials))
                recent = history['blocks'][-self.cfg['success_streak']:]
                target = len(recent)==self.cfg['success_streak'] and all(r['best']['all_sorted_rate']>=self.cfg['target_accuracy'] for r in recent)
                stagnant = history['blocks'][-self.cfg['plateau_blocks']:]
                plateau = len(stagnant)==self.cfg['plateau_blocks'] and not any(r['improved'] for r in stagnant)
                history['status'] = 'target_reached' if target else 'plateau' if plateau else 'budget_reached' if block==self.cfg['max_blocks'] else 'running'
                save_json(folder/'lab_history.json',history)
                self.sync_level('medium')
                print(f"Medium {stop}회: 첫 집기 {best['first_grasp_rate']:.0%} / 네 상자 정답 {best['all_sorted_rate']:.0%} / 분류 {best['score']:.1%}",flush=True)
                if history['status'] != 'running':
                    print('반복 종료:',history['status'],'/ 최고 모델은 보존됩니다.')
                    break
        return self.report()

    def _test_candidate(self,level):
        if level != 'medium':
            raise ValueError('This notebook runs Medium only')
        best = read_json(self.run_dir/'medium/lab_best.json')
        if not best:
            return None
        checkpoint = self.run_dir/'medium/checkpoints'/Path(best['checkpoint']).name
        if self.checkpoint_hash(checkpoint) != best['checkpoint_sha256']:
            raise ValueError('Best checkpoint changed')
        return checkpoint,dict(best['policy_config'])

    def report(self):
        self._ready()
        history = read_json(self.run_dir/'medium/lab_history.json',dict(blocks=[],status='not_started'))
        for row in history['blocks']:
            result = row['best']
            print(f"{row['iteration']:5d}회 | 집기 {result['first_grasp_rate']:.0%} | 네 상자 정답 {result['all_sorted_rate']:.0%} | 분류 {result['score']:.1%}")
        print('상태:',history['status'])
        return history

    def final_evaluation(self):
        self._ready()
        selected = self._test_candidate('medium')
        if not selected:
            raise RuntimeError('먼저 Medium 반복 학습을 실행하세요.')
        checkpoint,policy = selected
        best = read_json(self.run_dir/'medium/lab_best.json')
        if best['all_sorted_rate'] < self.cfg['target_accuracy']:
            print('네 상자 성공률이 목표에 못 미쳐 최종 평가는 생략합니다. 진단 셀을 확인하세요.')
            return None
        with self.persist_operation('medium'):
            save_json(self.run_dir/'medium/selection.json',dict(selected=best,tuning_seeds=self.seeds(True)))
            self._trial('medium',checkpoint,policy,self.seeds(False),'metrics')
            return read_json(self.run_dir/'medium/metrics.json')


def source_bundle():
    bundle = stage_sources()
    for name in ('medium_lab.py','build_medium_notebook.py'):
        bundle[name] = (Path(__file__).parent/name).read_text(encoding='utf-8')
    return bundle
