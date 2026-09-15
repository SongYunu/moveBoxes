"""Group-relative RL for a PICK-only residual on top of an immutable Stage ACT."""
import json
import os
import random
import shutil
import sys
from collections import deque
from pathlib import Path

import torch
from torch.distributions import Normal

from github_store import sync_from_env
from marso_experiment import save_json
from stage_model import PickResidualStageACT, StageACT
from stage_schema import HOLD, PICK


def group_relative_advantages(scores, group_size):
    """Normalize terminal success counts only against identical-reset peers."""
    if scores.ndim != 1 or type(group_size) is not int or group_size < 2:
        raise ValueError('scores must be 1-D and group_size >= 2')
    if len(scores) % group_size:
        raise ValueError('number of scores must be divisible by group_size')
    grouped = scores.view(-1, group_size)
    centered = grouped-grouped.mean(1, keepdim=True)
    scale = grouped.std(1, unbiased=False, keepdim=True)
    return torch.where(scale > 1e-6, centered/scale.clamp_min(1e-6), centered).flatten()


def trailing_pick_mask(stages, window):
    """Keep the last approach actions in each contiguous PICK attempt."""
    if stages.ndim != 2 or type(window) is not int or window < 1:
        raise ValueError('stages must be [time,env] and window positive')
    result = torch.zeros_like(stages, dtype=torch.bool)
    for env_index in range(stages.shape[1]):
        start = None
        for step in range(stages.shape[0]+1):
            is_pick = step < stages.shape[0] and int(stages[step, env_index]) == PICK
            if is_pick and start is None:
                start = step
            if not is_pick and start is not None:
                result[max(start, step-window):step, env_index] = True
                start = None
    return result


def _state(obs, device):
    value = obs['state'] if isinstance(obs, dict) else obs
    return value.float().to(device)


@torch.no_grad()
def collect(env, model, cfg, device):
    groups = cfg['num_envs']//cfg['group_size']
    seeds = []
    for group in range(groups):
        seeds.extend([cfg['seed']+cfg['iteration']*100003+group]*cfg['group_size'])
    obs, _ = env.reset(seed=seeds)
    state = _state(obs, device)
    previous_success = env.unwrapped.evaluate()['success_count'].detach().clone()
    history = deque(maxlen=model.cfg['history'])
    stage = torch.full((cfg['num_envs'],), PICK, dtype=torch.long, device=device)
    rows = {key:[] for key in ('history','previous_stage','stage','raw_xyz','log_prob','reward')}
    for _ in range(cfg['max_steps']-1):
        history.append(state.clone())
        while len(history) < history.maxlen:
            history.appendleft(state.clone())
        stacked = torch.stack(list(history), 1)
        x, memory, stage_logits, gate_logits = model.encode(stacked, stage)
        phase_p, phase = stage_logits.softmax(-1).max(-1)
        gate_p, gate = gate_logits.softmax(-1).max(-1)
        accepted = ((gate != HOLD) & (gate_p >= cfg['gate_threshold']) &
                    (phase_p >= cfg['stage_threshold']))
        previous = stage.clone()
        stage = torch.where(accepted, phase, stage)
        prediction, _ = model.decode(x, memory, stage)
        mean = prediction[:, 0, :3]
        normal = Normal(mean, cfg['xyz_std'])
        sampled = mean+cfg['xyz_std']*torch.randn_like(mean)
        raw_xyz = torch.where((stage == PICK)[:, None], sampled, mean)
        grip = torch.where(prediction[:, 0, 3:4] >= 0, 1., -1.)
        action = torch.cat((raw_xyz.clamp(-1, 1), grip), -1)
        obs, _, _, _, _ = env.step(action)
        current_success = env.unwrapped.evaluate()['success_count'].detach()
        rows['history'].append(stacked.cpu())
        rows['previous_stage'].append(previous.cpu())
        rows['stage'].append(stage.cpu())
        rows['raw_xyz'].append(raw_xyz.cpu())
        rows['log_prob'].append(normal.log_prob(raw_xyz).sum(-1).cpu())
        rows['reward'].append((current_success-previous_success).clamp_min(0).float().cpu())
        previous_success = current_success.clone()
        state = _state(obs, device)
    rollout = {key:torch.stack(value) for key,value in rows.items()}
    rollout['score'] = rollout['reward'].sum(0)
    rollout['credit'] = trailing_pick_mask(rollout['stage'], cfg['pick_credit_steps'])
    return rollout


def group_update(model, anchor, rollout, cfg, optimizer, device):
    env_advantage = group_relative_advantages(rollout['score'].to(device), cfg['group_size'])
    advantage = env_advantage[None].expand(rollout['stage'].shape[0], -1)
    active = rollout['credit'].to(device) & (advantage.abs() > 1e-6)
    if not active.any():
        return dict(sorted_parcels=float(rollout['score'].sum()), policy_loss=0.,
                    anchor_kl=0., active_steps=0, skipped=True)
    indices = active.flatten().nonzero().flatten().cpu()
    flat = {key:value.flatten(0, 1) for key,value in rollout.items()
            if key not in ('reward','score','credit')}
    advantages = advantage.flatten().cpu()
    losses, kls = [], []
    # Keep the frozen anchor path deterministic; eval mode still records residual gradients.
    model.eval()
    for _ in range(cfg['update_epochs']):
        order = indices[torch.randperm(len(indices))]
        for start in range(0, len(order), cfg['minibatch_size']):
            index = order[start:start+cfg['minibatch_size']]
            history = flat['history'][index].to(device)
            previous = flat['previous_stage'][index].to(device)
            stage = flat['stage'][index].to(device)
            raw_xyz = flat['raw_xyz'][index].to(device)
            old_log_prob = flat['log_prob'][index].to(device)
            batch_advantage = advantages[index].to(device)
            x, memory, _, _ = model.encode(history, previous)
            prediction, _ = model.decode(x, memory, stage)
            mean = prediction[:, 0, :3]
            log_prob = Normal(mean, cfg['xyz_std']).log_prob(raw_xyz).sum(-1)
            ratio = (log_prob-old_log_prob).clamp(-10, 10).exp()
            clipped = ratio.clamp(1-cfg['clip_ratio'], 1+cfg['clip_ratio'])
            policy_loss = -torch.minimum(ratio*batch_advantage, clipped*batch_advantage).mean()
            with torch.no_grad():
                ax, amemory, _, _ = anchor.encode(history, previous)
                anchor_prediction, _ = anchor.decode(ax, amemory, stage)
            anchor_kl = ((mean-anchor_prediction[:, 0, :3]).square()/
                         (2*cfg['xyz_std']**2)).sum(-1).mean()
            loss = policy_loss+cfg['anchor_kl_weight']*anchor_kl
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite group RL loss; checkpoint was not written')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.pick_residual.parameters(), .5)
            optimizer.step()
            losses.append(float(policy_loss.detach())); kls.append(float(anchor_kl.detach()))
    return dict(sorted_parcels=float(rollout['score'].sum()),
                policy_loss=sum(losses)/len(losses), anchor_kl=sum(kls)/len(kls),
                active_steps=int(active.sum()), skipped=False)


def atomic_save(path, value):
    path = Path(path)
    torch.save(value, str(path)+'.tmp')
    os.replace(str(path)+'.tmp', path)


def train(job, stop_after=None):
    from warehouse_sort.utils import compose_cfg, make_env
    cfg = dict(job['rl_config'])
    target_iteration = cfg['iterations'] if stop_after is None else int(stop_after)
    if not 1 <= target_iteration <= cfg['iterations']:
        raise ValueError('stop_after must be in 1..configured iterations')
    if cfg['num_envs'] % cfg['group_size']:
        raise ValueError('num_envs must be divisible by group_size')
    device = torch.device(job.get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('T4 GPU 런타임을 선택하세요.')
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
    saved = torch.load(job['anchor'], map_location='cpu', weights_only=True)
    if saved.get('format') != 'moveboxes-stage-act-v1':
        raise ValueError('Immutable Stage ACT anchor required')
    model_cfg = dict(saved['model_config'], residual_hidden=cfg['residual_hidden'],
                     residual_scale=cfg['residual_scale'])
    model = PickResidualStageACT(model_cfg)
    missing, unexpected = model.load_state_dict(saved['model'], strict=False)
    if unexpected or any(not name.startswith('pick_residual.') for name in missing):
        raise ValueError('Anchor/residual architecture mismatch')
    anchor = StageACT(saved['model_config']); anchor.load_state_dict(saved['model'])
    model.requires_grad_(False); model.pick_residual.requires_grad_(True)
    anchor.requires_grad_(False).eval()
    model.to(device); anchor.to(device)
    optimizer = torch.optim.Adam(model.pick_residual.parameters(), lr=cfg['lr'])
    folder = Path(job['folder']); ckdir = folder/'checkpoints'; ckdir.mkdir(parents=True, exist_ok=True)
    latest = ckdir/'latest.pt'; boundary = ckdir/f'iteration_{target_iteration:04d}.pt'
    start = 0; history = []
    if latest.exists():
        resume = torch.load(latest, map_location='cpu', weights_only=True)
        if resume.get('job_signature') != job['job_signature']:
            raise ValueError('Saved group-RL configuration differs; use a new run name')
        model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['torch_rng'])
        if device.type == 'cuda' and resume['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        start = resume.get('rl_iteration', resume.get('iteration'))
        if type(start) is not int or start < 0:
            raise ValueError('Saved group-RL checkpoint has no valid iteration counter')
        history = json.loads((folder/'rl_progress.json').read_text())['history']
    if start > target_iteration:
        if not boundary.is_file():
            raise FileNotFoundError(f'Missing earlier round checkpoint: {boundary}')
        print(f'GROUP RL {target_iteration}/{cfg["iterations"]} round checkpoint reused', flush=True)
        return
    env_cfg = compose_cfg(['difficulty='+job['level'], 'num_envs='+str(cfg['num_envs'])], job['config_dir'])
    env, _ = make_env(env_cfg, 'state', env_cfg.randomization, num_envs=cfg['num_envs'])
    try:
        for iteration in range(start, target_iteration):
            rollout = collect(env, model.eval(), dict(cfg, iteration=iteration), device)
            metrics = group_update(model, anchor, rollout, cfg, optimizer, device)
            metrics['iteration'] = iteration+1; history.append(metrics)
            weights = dict(format='moveboxes-stage-pick-residual-v1', model_config=model_cfg,
                model=model.state_dict(), step=saved.get('step', 0), rl_iteration=iteration+1,
                job_signature=job['job_signature'], optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
            atomic_save(latest, weights)
            save_json(folder/'rl_progress.json', dict(history=history,
                reward='group-relative official terminal success_count',
                total_sorted_parcels=sum(item['sorted_parcels'] for item in history)))
            print(f"GROUP RL {iteration+1}/{cfg['iterations']} sorted={metrics['sorted_parcels']:.0f} "
                  f"active={metrics['active_steps']} policy={metrics['policy_loss']:.4f} "
                  f"anchor_kl={metrics['anchor_kl']:.4f}", flush=True)
            try:
                sync_from_env()
            except Exception as exc:
                print(f'GitHub 백업 지연; 로컬 학습은 계속합니다: {exc}', flush=True)
    finally:
        env.close()
    pending = Path(str(boundary)+'.tmp'); shutil.copy2(latest, pending); os.replace(pending, boundary)
    if target_iteration == cfg['iterations']:
        save_json(folder/'training_complete.json', dict(iterations=cfg['iterations'],
            reward='group-relative terminal success_count'))
        (folder/'training_paused.json').unlink(missing_ok=True)
    else:
        save_json(folder/'training_paused.json', dict(iteration=target_iteration,
            configured_iterations=cfg['iterations']))
    try:
        sync_from_env()
    except Exception as exc:
        print(f'GitHub 최종 백업 지연; 로컬 체크포인트는 유지됩니다: {exc}', flush=True)


if __name__ == '__main__':
    train(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')),
          int(sys.argv[2]) if len(sys.argv) > 2 else None)
