"""Official WarehouseSort metric with the learned stage policy; independent test/final seeds."""
import json
import random
import sys
import time
from pathlib import Path
from marso_experiment import read_json, save_json
from github_store import sync_from_env
from next_pick_diagnostics import NextPickObserver

METRICS = ('sort_accuracy','mean_sorted','all_placed_rate','mean_steps','mis_sort_rate')


class Observer(NextPickObserver):
    def reset(self):
        super().reset()
        self.frames = []
        self.decisions = []

    def act(self, obs, deterministic=True):
        action = super().act(obs, deterministic)
        decision = {k:v[0].detach().cpu().item() for k,v in self.policy.last_decision.items()}
        self.decisions.append(decision)
        if getattr(self, 'trace', False):
            state = obs['state'] if isinstance(obs,dict) else obs
            self.frames.append(dict(step=len(self.frames), state=state[0].detach().cpu().tolist(),
                                    action=action[0].detach().cpu().tolist(), decision=decision))
        return action

    def report(self):
        result = super().report()
        result['stages'] = dict(
            visits={name:sum(d['stage']==i for d in self.decisions) for i,name in enumerate(('pick','carry','place','done'))},
            transitions=sum(d['previous']!=d['stage'] for d in self.decisions),
            recovery_decisions=sum(d['accepted'] and d['gate']==2 for d in self.decisions),
            rejected_proposals=sum(not d['accepted'] and d['proposed_stage']!=d['stage'] for d in self.decisions))
        return result


def main(job):
    import numpy as np
    import torch
    from stage_policy import load_stage
    from warehouse_sort.utils import compose_cfg, make_env, rollout_metrics, record_eval_video
    cfg = compose_cfg(['difficulty='+job['level'], 'obs_mode=state', 'max_episode_steps='+str(job['max_steps'])])
    previous = read_json(job['output'], {}) if not job.get('trace_only') else {}
    rows = previous.get('episodes', []) if previous.get('protocol',{}).get('fingerprint')==job['fingerprint'] else []
    if [r.get('seed') for r in rows] != job['seeds'][:len(rows)] or len(rows)>len(job['seeds']):
        raise ValueError('Stored evaluation seeds do not match the requested prefix')
    elapsed = previous.get('elapsed_seconds',0) if rows else 0
    tick = time.monotonic()
    device = torch.device('cuda')
    torch.backends.cudnn.deterministic = True
    env, _ = make_env(cfg, 'state', cfg.randomization, num_envs=1)
    try:
        obs, _ = env.reset(seed=0)
        agent = Observer(load_stage(job['checkpoint'], obs, env.single_action_space, device, **job['policy_config']))
        agent.trace = job.get('trace_only', False)
        if not job.get('video_only'):
            for seed in job['seeds'][len(rows):]:
                trace_path = Path(job.get('trace_dir','.'))/f'seed_{seed}.json'
                if agent.trace:
                    saved_trace = read_json(trace_path,{})
                    if saved_trace.get('complete') and saved_trace.get('protocol')==job:
                        continue
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                agent.reset()
                metrics = rollout_metrics(env, agent, device, 1, [seed], job['max_steps'])
                row = dict(seed=seed, **metrics, next_pick=agent.report())
                if agent.trace:
                    save_json(trace_path, dict(complete=True, seed=seed, protocol=job,
                        metrics=metrics, next_pick=agent.report(), frames=agent.frames))
                else:
                    rows.append(row)
                    average = {key:sum(r[key] for r in rows)/len(rows) for key in METRICS}
                    save_json(job['output'], dict(average, n_episodes=len(rows), episodes=rows,
                        protocol=job, complete=len(rows)==len(job['seeds']), elapsed_seconds=elapsed+time.monotonic()-tick))
                sync_from_env()
                print(f"{job['level']}: seed={seed}, sorted={metrics['mean_sorted']}, "
                      f"accuracy={metrics['sort_accuracy']:.3f}, grasps={agent.report()['stable_grasp_cycles']}", flush=True)
    finally:
        env.close()
    if job.get('video_only'):
        seed = job['seeds'][0]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        agent.reset()
        record_eval_video(cfg, 'state', cfg.randomization, agent, device,
                          job.get('video_dir',str(Path(job['output']).parent/'videos')), n_envs=1, seed=seed)


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
