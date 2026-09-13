"""Same state DP rollout; persist each completed episode for runtime recovery."""
import json
import random
import sys
import time
from pathlib import Path

from marso_experiment import read_json, save_json

METRICS = ('sort_accuracy','mean_sorted','all_placed_rate','mean_steps','mis_sort_rate')


def restore_rows(saved, job):
    if not saved or saved.get('protocol',{}).get('fingerprint') != job['fingerprint']:
        return []
    rows = saved.get('episodes',[])
    seeds = [row.get('seed') for row in rows]
    if seeds != job['seeds'][:len(rows)] or len(rows) > len(job['seeds']):
        raise ValueError('저장된 평가 에피소드와 요청한 seed가 일치하지 않습니다.')
    if any(any(k not in row for k in METRICS) for row in rows):
        raise ValueError('저장된 평가 에피소드에 지표가 누락되었습니다.')
    return rows


def result_for(rows, job, elapsed):
    values = {key:sum(row[key] for row in rows)/len(rows) for key in METRICS}
    return dict(values,n_episodes=len(rows),episodes=rows,protocol=job,
                elapsed_seconds=elapsed,complete=len(rows)==len(job['seeds']))


def main(job):
    import numpy as np
    import torch
    from colab_policy import ChunkPolicy
    from warehouse_sort.il_policy import load_dp
    from warehouse_sort.utils import compose_cfg, make_env, rollout_metrics, record_eval_video

    cfg = compose_cfg(['difficulty='+job['level'],'obs_mode=state',
                       'max_episode_steps='+str(job['max_steps'])])
    device = torch.device('cuda')
    torch.backends.cudnn.deterministic = True
    previous = read_json(job['output'])
    rows = restore_rows(previous,job)
    elapsed = previous.get('elapsed_seconds',0) if rows else 0
    tick = time.perf_counter()
    env, _ = make_env(cfg,'state',cfg.randomization,num_envs=1)
    try:
        obs, _ = env.reset(seed=0)
        # Exactly the same loader architecture and chunk implementation as t4_policy.py.
        policy_cfg = dict(job['policy_config'])
        chunk = policy_cfg.pop('act_horizon')
        action_space = env.single_action_space
        agent = ChunkPolicy(load_dp(job['checkpoint'],obs,action_space,device,**policy_cfg),chunk)
        if not job.get('video_only'):
            for seed in job['seeds'][len(rows):]:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                agent.reset()
                result = rollout_metrics(env,agent,device,1,[seed],job['max_steps'])
                rows.append(dict(seed=seed,**result))
                save_json(job['output'],result_for(rows,job,elapsed+time.perf_counter()-tick))
                print(f"{job['level']}: {len(rows)}/{len(job['seeds'])}, sort_accuracy={result['sort_accuracy']:.3f}",flush=True)
    finally:
        env.close()
    if job.get('video_only'):
        seed = job['seeds'][0]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        agent.reset()
        record_eval_video(cfg,'state',cfg.randomization,agent,device,
                          job.get('video_dir',str(Path(job['output']).parent/'videos')),n_envs=1,seed=seed)


if __name__=='__main__':
    main(json.loads(Path(sys.argv[1]).read_text()))
