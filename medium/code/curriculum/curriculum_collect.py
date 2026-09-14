"""Partial-success curriculum, with unchanged final-horizon scoring and expandable collection budgets."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from deadline_teacher import DeadlineTeacher
from stage_teacher import TeacherFailure
from stage_labels import StageLabels
from stage_schema import DONE, RECOVER
from marso_experiment import read_json, save_json, digest
from github_store import sync_from_env


def collect_episode(env, seed, gain, noise_std=0., noise_probability=0., minimum_correct=3):
    import torch
    if minimum_correct not in (3,4):raise ValueError('minimum_correct must be 3 or 4')
    completion_steps={}
    obs, _ = env.reset(seed=seed)
    base = env.unwrapped
    teacher, labeler = DeadlineTeacher(base.num_parcels,gain), StageLabels(base.num_parcels)
    rng = np.random.default_rng(seed+923)
    states, actions, executed, labels, phases, perturbed = [], [], [], [], [], []
    first_complete, failure = None, None
    for step in range(199):
        raw = obs['state'] if isinstance(obs,dict) else obs
        state = raw[0].detach().cpu().numpy().copy()
        grasped = [bool(base.agent.is_grasping(p)[0].item()) for p in base.parcels]
        label = labeler.observe(state, int(np.argmax(grasped)) if any(grasped) else -1)
        if failure:
            action, recovering = np.array([0,0,0,1],np.float32), False
        else:
            try:
                action, recovering = teacher.action(state,grasped)
            except TeacherFailure as error:
                failure = str(error)
                action, recovering = np.array([0,0,0,1],np.float32), False
        if recovering:
            label['gate'] = RECOVER
        command = action.copy()
        if label['stage'] != DONE and rng.random() < noise_probability:
            command[:3] += rng.normal(0,noise_std,3)
        command = command.clip(-1,1)
        states.append(state); actions.append(action); executed.append(command)
        phases.append(teacher.phase); labels.append(label)
        perturbed.append(bool(np.any(command != action)))
        obs, _, _, truncated, _ = env.step(torch.as_tensor(command[None],device=raw.device))
        correct = int(base.evaluate()['success_count'][0].item())
        for count in range(1,correct+1):completion_steps.setdefault(str(count),step+1)
        if correct == base.num_parcels and first_complete is None:
            first_complete = step+1
        if bool(truncated.any()):
            failure = 'Environment truncated before the required horizon'
            break
    raw = obs['state'] if isinstance(obs,dict) else obs
    states.append(raw[0].detach().cpu().numpy().copy())
    correct = int(base.evaluate()['success_count'][0].item())
    evaluated = len(actions)
    # Observe all 199 actions, but avoid oversampling a long completed idle tail.
    end = evaluated
    done = [i for i,r in enumerate(labels) if r['stage'] == DONE]
    if done:
        end = min(end,done[0]+4)
    arrays = dict(obs=np.asarray(states[:end+1],np.float32),actions=np.asarray(actions[:end],np.float32),
        executed_actions=np.asarray(executed[:end],np.float32),perturbed=np.asarray(perturbed[:end],bool))
    arrays.update({k:np.asarray([r[k] for r in labels[:end]],np.int64) for k in labels[0]})
    report = dict(seed=seed,steps=end,evaluation_actions=evaluated,first_complete_step=first_complete,
        correct=correct,num_parcels=base.num_parcels,accepted=evaluated==199 and correct>=minimum_correct and failure is None,
        fully_successful=evaluated==199 and correct==4 and failure is None,minimum_correct=minimum_correct,
        completion_steps=completion_steps,
        failure=failure,phase_steps={p:phases.count(p) for p in sorted(set(phases))},
        perturbations=sum(perturbed),gain=gain)
    return arrays,report


def collect(job):
    from warehouse_sort.utils import compose_cfg, _gym_make
    cfg, folder = job['collection_config'], Path(job['folder'])
    folder.mkdir(parents=True,exist_ok=True)
    identity = {k:v for k,v in job.items() if k not in ('folder','operation')}
    identity['collection_config']={k:v for k,v in cfg.items() if k not in
        ('episodes','max_attempts','pilot_episodes','seconds_per_call')}
    if cfg['minimum_correct'] not in (3,4) or cfg['episodes']<4 or cfg['max_attempts']<cfg['episodes']:
        raise ValueError('Invalid curriculum threshold or collection budget')
    signature = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path = folder/'manifest.json'
    manifest = read_json(path,dict(version='curriculum-v21',signature=signature,complete=False,pilot=[],attempts=[],episodes=[]))
    if manifest['signature'] != signature:
        raise ValueError('Collection protocol changed. Use a new run_name.')
    for entry in manifest['episodes']:
        if digest(folder/entry['file']) != entry['sha256']:
            raise ValueError('Saved demonstration changed')
    def save():
        save_json(path,manifest); sync_from_env()
    if len(manifest['episodes'])>=cfg['episodes']:
        manifest['complete']=True;manifest['status']='complete';save()
        print('부분/전체 성공을 구분한 기존 시연을 재사용합니다.');return
    if (folder.parent/'checkpoints/latest.pt').exists():
        raise ValueError('Training has started: keep its dataset fixed and use a new run for more data')
    manifest['complete']=False
    manifest['minimum_correct']=cfg['minimum_correct']
    manifest['budgets']={k:cfg[k] for k in ('episodes','max_attempts','pilot_episodes','seconds_per_call')}
    started = time.monotonic()
    env_cfg = compose_cfg(['difficulty=medium','obs_mode=state','max_episode_steps=200'])
    env = _gym_make(env_cfg,'state',env_cfg.randomization,1,None)
    try:
        tasks = [(gain,cfg['pilot_seed_start']+i) for gain in cfg['gains'] for i in range(cfg['pilot_episodes'])]
        completed={(r['gain'],r['seed']) for r in manifest['pilot']}
        if 'selected_gain' not in manifest or any(task not in completed for task in tasks):
            for gain,seed in tasks:
                if (gain,seed) in completed:continue
                _,report = collect_episode(env,seed,gain,minimum_correct=cfg['minimum_correct'])
                manifest['pilot'].append(report); save()
                print(f"시연 시험 gain={gain}: {report['correct']}/4 · 학습 채택={report['accepted']} · 네 상자 완주={report['fully_successful']}",flush=True)
                if time.monotonic()-started >= cfg['seconds_per_call']:
                    manifest['status']='paused';save();print('수집 시간 예산 도달. 같은 셀을 다시 실행하면 이어집니다.'); return
            eligible = [g for g in cfg['gains'] if any(r['accepted'] for r in manifest['pilot'] if r['gain']==g)]
            if not eligible:
                eligible=list(cfg['gains'])
                print('아직 채택 가능한 시연이 없습니다. 가장 진전이 있던 설정으로 다른 시드를 더 시험합니다.',flush=True)
            def ranking(g):
                rows=[r for r in manifest['pilot'] if r['gain']==g]
                return (sum(r['accepted'] for r in rows)/len(rows),sum(r['correct'] for r in rows)/len(rows),
                    -sum(r.get('completion_steps',{}).get(str(cfg['minimum_correct']),200) for r in rows)/len(rows))
            manifest['selected_gain']=max(eligible,key=ranking); manifest['status']='collecting'; save()
        if job['operation']=='pilot':
            print('시연 설정 선택 완료. 다음 수집 셀에서 새 시드로 계속 진행합니다.'); return
        for attempt in range(len(manifest['attempts']),cfg['max_attempts']):
            arrays,report=collect_episode(env,cfg['seed_start']+attempt,manifest['selected_gain'],
                cfg['noise_std'],cfg['noise_probability'],minimum_correct=cfg['minimum_correct'])
            if report['accepted']:
                name=f"episode_{report['seed']}.npz"
                with (folder/(name+'.tmp')).open('wb') as f:
                    np.savez_compressed(f,**arrays)
                os.replace(folder/(name+'.tmp'),folder/name)
                manifest['episodes'].append(dict(file=name,sha256=digest(folder/name),**report))
            manifest['attempts'].append(report)
            manifest['fully_successful_episodes']=sum(r['fully_successful'] for r in manifest['episodes'])
            manifest['complete']=len(manifest['episodes']) >= cfg['episodes']
            manifest['status']='complete' if manifest['complete'] else 'collecting'
            save()
            print(f"학습 시연 {len(manifest['episodes'])}/{cfg['episodes']} (4개 완주 {manifest['fully_successful_episodes']}) · 시도 {attempt+1}/{cfg['max_attempts']}",flush=True)
            if manifest['complete']:return
            if time.monotonic()-started >= cfg['seconds_per_call']:
                manifest['status']='paused';save();print('수집 시간 예산 도달. 같은 셀로 이어서 수집하세요.'); return
        manifest['status']='attempt_budget_reached';save()
        print('수집 시도 한도 도달. 다음 호출에서 max_attempts를 늘려 새 시드로 이어갈 수 있습니다.')
    finally:
        env.close()


if __name__=='__main__':
    collect(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
