"""Collect state-aligned corrective demonstrations, checkpointed per episode.

The expert labels the actual perturbed state. The noisy executed action is saved
separately; it is NEVER the imitation target. Only fully successful episodes enter
the training manifest. This is noisy-expert augmentation, not full DAgger/DART.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from marso_experiment import read_json, save_json, digest
from github_store import sync_from_env
from stage_labels import StageLabels
from stage_schema import CARRY, DONE, RECOVER
from stage_teacher import CorrectiveTeacher, TeacherFailure


def perturb_action(action, stage, rng, cfg, drop_budget):
    executed = action.copy()
    if stage != DONE and rng.random() < cfg['noise_probability']:
        executed[:3] += rng.normal(0, cfg['action_noise_std'], 3)
    dropped = bool(stage == CARRY and drop_budget and rng.random() < cfg['drop_probability'])
    if dropped:
        executed[3] = 1
    executed = executed.clip(-1, 1)
    return executed, bool(np.any(executed != action)), dropped


def collect_episode(env, seed, cfg):
    import torch
    obs, _ = env.reset(seed=seed)
    base = env.unwrapped
    n = base.num_parcels
    teacher, labeler = CorrectiveTeacher(n), StageLabels(n)
    rng = np.random.default_rng(seed+923)
    states, actions, executed_actions, labels, disturbed = [], [], [], [], []
    drops, done_steps, failure = 0, 0, None

    def flat(observation):
        return observation['state'] if isinstance(observation, dict) else observation

    for _ in range(cfg['max_steps']):
        raw = flat(obs)
        state = raw[0].detach().cpu().numpy().copy()
        grasped = [bool(base.agent.is_grasping(p)[0].item()) for p in base.parcels]
        label = labeler.observe(state, int(np.argmax(grasped)) if any(grasped) else -1)
        try:
            expert, recovering = teacher.action(state, grasped)
        except TeacherFailure as error:
            failure = str(error)
            break
        if recovering:
            label['gate'] = RECOVER
        executed, changed, dropped = perturb_action(expert, label['stage'], rng, cfg, drops < 2)
        drops += int(dropped)
        states.append(state)
        actions.append(expert)
        labels.append(label)
        executed_actions.append(executed)
        disturbed.append(changed)
        obs, _, term, trunc, _ = env.step(torch.as_tensor(executed[None], device=raw.device))
        # Save real terminal idle samples to teach the DONE decoder, never fabricated tails.
        done_steps = done_steps+1 if label['stage'] == DONE else 0
        # Base env may report success termination; keep a few real settling frames.
        if done_steps >= 6 or bool(trunc.any()):
            break
    states.append(flat(obs)[0].detach().cpu().numpy().copy())
    metrics = base.evaluate()
    correct = int(metrics['success_count'][0].item())
    arrays = dict(obs=np.asarray(states, dtype=np.float32), actions=np.asarray(actions, dtype=np.float32),
                  executed_actions=np.asarray(executed_actions, dtype=np.float32), perturbed=np.asarray(disturbed, dtype=bool))
    if labels:
        arrays.update({k:np.asarray([r[k] for r in labels], dtype=np.int64) for k in labels[0]})
    report = dict(seed=seed, steps=len(actions), correct=correct, num_parcels=n,
                  accepted=correct == n and bool(labels), failure=failure,
                  perturbations=sum(disturbed), drop_commands=drops,
                  recovery_labels=sum(r['gate'] == RECOVER for r in labels))
    return arrays, report


def collect(job):
    from warehouse_sort.utils import compose_cfg, _gym_make
    cfg, folder = job['collection_config'], Path(job['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps({k:v for k,v in job.items() if k != 'folder'}, sort_keys=True).encode()).hexdigest()
    path = folder/'manifest.json'
    manifest = read_json(path, dict(signature=signature, complete=False, episodes=[], attempts=[]))
    if manifest['signature'] != signature:
        raise ValueError('Collection configuration/code changed. Use a new run_name.')
    for entry in manifest['episodes']:
        if digest(folder/entry['file']) != entry['sha256']:
            raise ValueError('Recovery dataset digest mismatch')
    if manifest.get('complete'):
        print('복구 시연 수집 완료 상태를 재사용합니다.', flush=True)
        return
    env_cfg = compose_cfg(['difficulty='+job['level'], 'obs_mode=state', 'max_episode_steps='+str(cfg['max_steps'])])
    env = _gym_make(env_cfg, 'state', env_cfg.randomization, 1, None)
    try:
        for attempt in range(len(manifest['attempts']), cfg['max_attempts']):
            seed = cfg['seed_start']+attempt
            arrays, report = collect_episode(env, seed, cfg)
            if report['accepted']:
                name = f'episode_{seed}.npz'
                with (folder/(name+'.tmp')).open('wb') as handle:
                    np.savez_compressed(handle, **arrays)
                os.replace(folder/(name+'.tmp'), folder/name)
                manifest['episodes'].append(dict(file=name, sha256=digest(folder/name), **report))
            manifest['attempts'].append(report)
            accepted = len(manifest['episodes'])
            manifest['complete'] = accepted >= cfg['episodes']
            save_json(path, manifest)
            sync_from_env()
            print(f"{job['level']}: 복구 시연 {accepted}/{cfg['episodes']}, "
                  f"시도 {attempt+1}/{cfg['max_attempts']}, 정답 {report['correct']}/{report['num_parcels']}, "
                  f"복구 라벨 {report['recovery_labels']}", flush=True)
            if manifest['complete']:
                break
    finally:
        env.close()
    if not manifest['complete']:
        raise RuntimeError(f"성공 시연 {len(manifest['episodes'])}/{cfg['episodes']}. "
                           "collection/manifest.json의 실패 원인을 확인하세요. 실패 시연은 학습에 섞지 않습니다.")


if __name__ == '__main__':
    collect(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
