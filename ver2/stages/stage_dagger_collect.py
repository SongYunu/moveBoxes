"""Training-only DAgger collection from states visited by a learned policy.

The scripted teacher labels the observation before the environment step.  It is
never copied into a submission package.  Failed rollouts remain useful because
their pre-failure states contain the corrective labels missing from clean demos.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from github_store import sync_from_env
from marso_experiment import digest, read_json, save_json
from stage_labels import StageLabels
from stage_schema import RECOVER
from stage_teacher import CorrectiveTeacher, TeacherFailure


def select_executed(expert, policy, beta, rng):
    if not 0 <= beta <= 1:
        raise ValueError('DAgger beta must be in [0,1]')
    use_teacher = bool(rng.random() < beta)
    executed = expert if use_teacher else policy
    executed = np.asarray(executed, dtype=np.float32).clip(-1, 1)
    if executed.shape != (4,) or not np.isfinite(executed).all():
        raise ValueError('DAgger action must be one finite 4-D action')
    return executed, use_teacher


def collect_episode(env, agent, seed, cfg):
    import torch
    obs, _ = env.reset(seed=seed)
    agent.reset()
    base = env.unwrapped
    n = base.num_parcels
    teacher, labeler = CorrectiveTeacher(n), StageLabels(n)
    rng = np.random.default_rng(seed+1709)
    states, actions, policy_actions, executed_actions = [], [], [], []
    labels, perturbed, teacher_steps = [], [], 0
    failure = None

    def flat(observation):
        return observation['state'] if isinstance(observation, dict) else observation

    for _ in range(cfg['max_steps']):
        raw = flat(obs)
        state = raw[0].detach().cpu().numpy().copy()
        grasped = [bool(base.agent.is_grasping(parcel)[0].item()) for parcel in base.parcels]
        label = labeler.observe(state, int(np.argmax(grasped)) if any(grasped) else -1)
        try:
            expert, recovering = teacher.action(state, grasped)
        except TeacherFailure as error:
            failure = str(error)
            break
        if recovering:
            label['gate'] = RECOVER
        policy = agent.act(obs, deterministic=True)[0].detach().cpu().numpy()
        executed, used_teacher = select_executed(expert, policy, cfg['beta'], rng)
        states.append(state)
        actions.append(expert)
        policy_actions.append(policy)
        executed_actions.append(executed)
        labels.append(label)
        changed = not np.allclose(executed, expert, atol=1e-7, rtol=0)
        perturbed.append(changed)
        teacher_steps += int(used_teacher)
        obs, _, term, trunc, _ = env.step(torch.as_tensor(executed[None], device=raw.device))
        if bool(term.any()) or bool(trunc.any()):
            break

    states.append(flat(obs)[0].detach().cpu().numpy().copy())
    metrics = base.evaluate()
    correct = int(metrics['success_count'][0].item())
    arrays = dict(obs=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        policy_actions=np.asarray(policy_actions, dtype=np.float32),
        executed_actions=np.asarray(executed_actions, dtype=np.float32),
        perturbed=np.asarray(perturbed, dtype=bool))
    if labels:
        arrays.update({key:np.asarray([row[key] for row in labels], dtype=np.int64)
                       for key in labels[0]})
    valid = bool(labels) and len(states) == len(actions)+1
    report = dict(seed=seed, steps=len(actions), correct=correct, num_parcels=n,
        accepted=valid, valid_for_training=valid, success=correct == n,
        failure=failure, teacher_steps=teacher_steps,
        policy_steps=len(actions)-teacher_steps, disagreement_steps=sum(perturbed),
        recovery_labels=sum(row['gate'] == RECOVER for row in labels))
    return arrays, report


def collect(job):
    from stage_policy import load_stage
    from warehouse_sort.utils import compose_cfg, _gym_make

    cfg, folder = job['collection_config'], Path(job['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    identity = {key:value for key,value in job.items() if key != 'folder'}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = folder/'manifest.json'
    manifest = read_json(manifest_path, dict(kind='dagger-v1', signature=signature,
        complete=False, episodes=[], attempts=[]))
    if manifest.get('kind') != 'dagger-v1' or manifest.get('signature') != signature:
        raise ValueError('DAgger collection configuration changed; use a new run suffix')
    for entry in manifest['episodes']:
        if digest(folder/entry['file']) != entry['sha256']:
            raise ValueError('DAgger episode checksum mismatch')
    if manifest.get('complete'):
        print('완료된 DAgger 수집을 재사용합니다.', flush=True)
        return

    env_cfg = compose_cfg(['difficulty='+job['level'], 'obs_mode=state',
                           'max_episode_steps='+str(cfg['max_steps'])])
    env = _gym_make(env_cfg, 'state', env_cfg.randomization, 1, None)
    try:
        sample_obs, _ = env.reset(seed=cfg['seed_start'])
        agent = load_stage(job['checkpoint'], sample_obs, env.single_action_space,
                           cfg['device'], **job['policy_config'])
        for index in range(len(manifest['attempts']), cfg['episodes']):
            seed = cfg['seed_start']+index
            arrays, report = collect_episode(env, agent, seed, cfg)
            if report['valid_for_training']:
                name = f'episode_{seed}.npz'
                with (folder/(name+'.tmp')).open('wb') as handle:
                    np.savez_compressed(handle, **arrays)
                os.replace(folder/(name+'.tmp'), folder/name)
                manifest['episodes'].append(dict(file=name, sha256=digest(folder/name), **report))
            manifest['attempts'].append(report)
            manifest['complete'] = len(manifest['attempts']) >= cfg['episodes']
            save_json(manifest_path, manifest)
            sync_from_env()
            print(f"{job['level']}: DAgger {index+1}/{cfg['episodes']} · "
                  f"정답 {report['correct']}/{report['num_parcels']} · "
                  f"policy {report['policy_steps']}/{report['steps']} · "
                  f"교정 라벨 {report['recovery_labels']}", flush=True)
    finally:
        env.close()
    if not manifest['complete'] or len(manifest['episodes']) < 2:
        raise RuntimeError('DAgger collection did not produce at least two valid labeled episodes')


if __name__ == '__main__':
    collect(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
