"""Read-only rollout trace of the packaged policy. Does not alter its actions."""
import argparse
import json
from pathlib import Path

import torch


def decision_row(step, state, action, decision):
    """Raw observations and decisions, not a success score or a controller."""
    state = state.detach().cpu()[0]
    action = action.detach().cpu()[0]
    n = (state.numel()-36)//9
    if n not in (2, 4, 6):
        raise ValueError('Requires WarehouseSort state dimensions 54/72/90')
    tcp = state[18:21]
    boxes = state[26:26+7*n].reshape(n, 7)[:, :3]
    distance = torch.linalg.vector_norm(boxes[:, :2]-tcp[:2], dim=-1)
    nearest = int(distance.argmin())
    row = dict(step=step, tcp_xyz=tcp.tolist(), parcel_xyz=boxes.tolist(),
               nearest_parcel_xy=nearest, nearest_xy_distance=float(distance[nearest]),
               tcp_minus_nearest_parcel_z=float(tcp[2]-boxes[nearest, 2]),
               finger_qpos=state[7:9].tolist(), is_grasped=bool(state[25] > .5),
               action=action.tolist())
    for key, value in decision.items():
        row[key] = value.detach().cpu()[0].item() if torch.is_tensor(value) else value
    return row


@torch.no_grad()
def diagnose(checkpoint, config_dir, level, output, seed=61000, steps=120):
    from warehouse_sort.utils import compose_cfg, make_env
    from stage_chunk_policy import load_policy
    from stage_schema import STAGES, GATES

    config = compose_cfg(['difficulty='+level, 'obs_mode=state', 'max_episode_steps=200'],
                         config_dir=str(config_dir))
    env, _ = make_env(config, 'state', config.randomization, num_envs=1)
    rows = []
    try:
        obs, _ = env.reset(seed=seed)
        policy = load_policy(checkpoint, obs, env.single_action_space, 'cuda')
        for step in range(steps):
            state = obs['state'] if isinstance(obs, dict) else obs
            # Verify the state layout against the simulator before interpreting Z.
            torch.testing.assert_close(state[:, 18:21], env.unwrapped.agent.tcp_pose.p,
                                       rtol=0, atol=1e-5)
            torch.testing.assert_close(state[:, 26:29], env.unwrapped.parcels[0].pose.p,
                                       rtol=0, atol=1e-5)
            action = policy.act(obs)
            row = decision_row(step, state, action, policy.last_decision)
            row['stage_name'] = STAGES[row['stage']]
            row['proposed_stage_name'] = STAGES[row['proposed_stage']]
            row['gate_name'] = GATES[row['gate']]
            if hasattr(policy, 'action_buffer') and policy.action_buffer is not None:
                row['gripper_logit'] = float(policy.action_buffer[0, policy.cursor[0]-1, 3])
            rows.append(row)
            if step % 10 == 0 or row['accepted'] or step == steps-1:
                print(f"{step:03d} tcp_z={row['tcp_xyz'][2]:.3f} "
                      f"xy_gap={row['nearest_xy_distance']:.3f} "
                      f"z_above_center={row['tcp_minus_nearest_parcel_z']:.3f} "
                      f"action={row['action']} held={row['is_grasped']} "
                      f"stage={row['stage_name']} proposed={row['proposed_stage_name']} "
                      f"gate={row['gate_name']} accepted={row['accepted']}", flush=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(terminated.any() or truncated.any()):
                break
    finally:
        env.close()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(dict(checkpoint=str(checkpoint), difficulty=level,
            seed=seed, rows=rows), indent=2), encoding='utf-8')
        print('Raw pick trace:', output, flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config-dir', required=True, type=Path)
    parser.add_argument('--level', required=True, choices=('easy', 'medium', 'hard'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', default=61000, type=int)
    parser.add_argument('--steps', default=120, type=int, choices=range(1, 200))
    args = parser.parse_args()
    diagnose(args.checkpoint, args.config_dir, args.level, args.output, args.seed, args.steps)
