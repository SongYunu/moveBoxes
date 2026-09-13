"""Separate test-seed diagnostics; never alters benchmark evaluator or policy."""
import json
import random
import sys
from pathlib import Path

from marso_experiment import read_json, save_json
from next_pick_diagnostics import NextPickObserver


class TraceObserver(NextPickObserver):
    def reset(self):
        super().reset()
        self.frames = []

    def act(self, obs, deterministic=True):
        action = super().act(obs, deterministic=deterministic)
        state = obs['state'] if isinstance(obs, dict) else obs
        self.frames.append(dict(step=len(self.frames),
                                state=state[0].detach().cpu().tolist(),
                                action=action[0].detach().cpu().tolist()))
        return action


def main(job):
    import numpy as np
    import torch
    from colab_policy import ChunkPolicy
    from warehouse_sort.il_policy import load_dp
    from warehouse_sort.utils import compose_cfg, make_env, rollout_metrics
    from github_store import sync_from_env

    cfg = compose_cfg(['difficulty='+job['level'], 'obs_mode=state',
                       'max_episode_steps='+str(job['max_steps'])])
    target = Path(job['trace_dir'])
    pending = []
    for seed in job['seeds']:
        previous = read_json(target/f'seed_{seed}.json', {})
        if (previous.get('complete') is True and previous.get('seed') == seed
                and previous.get('protocol') == job):
            print(f"{job['level']} trace {seed}: 저장 기록 재사용", flush=True)
        else:
            pending.append(seed)
    if not pending:
        return

    device = torch.device('cuda')
    torch.backends.cudnn.deterministic = True
    env, _ = make_env(cfg, 'state', cfg.randomization, num_envs=1)
    try:
        obs, _ = env.reset(seed=0)
        policy_cfg = dict(job['policy_config'])
        chunk = policy_cfg.pop('act_horizon')
        policy = load_dp(job['checkpoint'], obs, env.single_action_space, device, **policy_cfg)
        agent = TraceObserver(ChunkPolicy(policy, chunk))
        for seed in pending:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            agent.reset()
            metrics = rollout_metrics(env, agent, device, 1, [seed], job['max_steps'])
            save_json(target/f'seed_{seed}.json', dict(
                complete=True, seed=seed, protocol=job, metrics=metrics,
                next_pick=agent.report(), frames=agent.frames,
                state_dim=int(obs.shape[-1]),
                description='State before each action; action [delta_x, delta_y, delta_z, gripper]. '
                            'State index 25 is is_grasped. Terminal observation is excluded. '
                            'Uses test seeds only; not included in the final score.'))
            sync_from_env()
            print(f"{job['level']} trace {seed}: {len(agent.frames)}스텝, "
                  f"정답 {metrics['mean_sorted']}, 집기 사이클 {agent.report()['stable_grasp_cycles']}", flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
