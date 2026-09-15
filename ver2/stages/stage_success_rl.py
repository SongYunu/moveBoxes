"""Conservative on-policy refinement using only the environment's sparse sort reward."""
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

import torch
from torch.distributions import Bernoulli, Normal, kl_divergence

from github_store import sync_from_env
from marso_experiment import save_json
from stage_model import StageACT
from stage_schema import HOLD, PICK, RECOVER


def discounted_returns(rewards, gamma):
    """Reward-to-go from the official sparse reward; no shaped reward is added."""
    if rewards.ndim != 2 or not 0 < gamma <= 1:
        raise ValueError('rewards must be [time,env] and gamma in (0,1]')
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1], device=rewards.device)
    for step in range(len(rewards)-1, -1, -1):
        running = rewards[step]+gamma*running
        returns[step] = running
    return returns


def _distribution(model, history, previous_stage, stage, xyz_std, grip_temperature):
    x, memory, _, _ = model.encode(history, previous_stage)
    prediction, _ = model.decode(x, memory, stage)
    mean = prediction[:, 0, :3]
    grip_logits = prediction[:, 0, 3]/grip_temperature
    return Normal(mean, xyz_std), Bernoulli(logits=grip_logits)


def _state(obs, device):
    state = obs['state'] if isinstance(obs, dict) else obs
    return state.float().to(device)


def sparse_success_delta(current, previous):
    """Positive change in the official primary-metric counter."""
    if current.shape != previous.shape:
        raise ValueError('success counters must have matching shapes')
    return (current-previous).clamp_min(0).float()


@torch.no_grad()
def collect(env, model, cfg, device):
    obs, _ = env.reset(seed=[cfg['seed']+cfg['iteration']*100003+i for i in range(cfg['num_envs'])])
    state = _state(obs, device)
    previous_success = env.unwrapped.evaluate()['success_count'].detach().clone()
    history = deque(maxlen=model.cfg['history'])
    stage = torch.full((cfg['num_envs'],), PICK, dtype=torch.long, device=device)
    rows = {key:[] for key in ('history','previous_stage','stage','raw_xyz','grip','log_prob','reward')}
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
        bernoulli = Bernoulli(logits=prediction[:, 0, 3]/cfg['grip_temperature'])
        # Generator-backed sampling makes checkpoints exactly resumable.
        raw_xyz = mean+cfg['xyz_std']*torch.randn_like(mean)
        grip = torch.bernoulli(bernoulli.probs)
        log_prob = normal.log_prob(raw_xyz).sum(-1)+bernoulli.log_prob(grip)
        action = torch.cat((raw_xyz.clamp(-1, 1), grip[:, None]*2-1), -1)
        obs, _, _, _, _ = env.step(action)
        current_success = env.unwrapped.evaluate()['success_count'].detach()
        reward = sparse_success_delta(current_success, previous_success)
        previous_success = current_success.clone()
        rows['history'].append(stacked.cpu())
        rows['previous_stage'].append(previous.cpu())
        rows['stage'].append(stage.cpu())
        rows['raw_xyz'].append(raw_xyz.cpu())
        rows['grip'].append(grip.cpu())
        rows['log_prob'].append(log_prob.cpu())
        rows['reward'].append(reward.float().cpu())
        state = _state(obs, device)
    return {key:torch.stack(value) for key,value in rows.items()}


def ppo_update(model, reference, rollout, cfg, optimizer, device):
    returns = discounted_returns(rollout['reward'].to(device), cfg['gamma'])
    reward_total = float(rollout['reward'].sum())
    if reward_total <= 0:
        return dict(sorted_parcels=0., policy_loss=0., reference_kl=0., skipped=True)
    advantages = returns-returns.mean(1, keepdim=True)
    advantages = (advantages-advantages.mean())/advantages.std().clamp_min(1e-6)
    count = advantages.numel()
    flat = {key:value.flatten(0,1) for key,value in rollout.items() if key != 'reward'}
    advantages = advantages.flatten()
    losses, kls = [], []
    model.train()
    for _ in range(cfg['update_epochs']):
        order = torch.randperm(count)
        for start in range(0, count, cfg['minibatch_size']):
            index = order[start:start+cfg['minibatch_size']]
            history = flat['history'][index].to(device)
            previous = flat['previous_stage'][index].to(device)
            stage = flat['stage'][index].to(device)
            raw_xyz = flat['raw_xyz'][index].to(device)
            grip = flat['grip'][index].to(device)
            old_log_prob = flat['log_prob'][index].to(device)
            advantage = advantages[index].to(device)
            normal, bernoulli = _distribution(model, history, previous, stage,
                                               cfg['xyz_std'], cfg['grip_temperature'])
            log_prob = normal.log_prob(raw_xyz).sum(-1)+bernoulli.log_prob(grip)
            ratio = (log_prob-old_log_prob).clamp(-10,10).exp()
            clipped = ratio.clamp(1-cfg['clip_ratio'], 1+cfg['clip_ratio'])
            policy_loss = -torch.minimum(ratio*advantage, clipped*advantage).mean()
            with torch.no_grad():
                ref_normal, ref_grip = _distribution(reference, history, previous, stage,
                                                      cfg['xyz_std'], cfg['grip_temperature'])
            reference_kl = (kl_divergence(normal, ref_normal).sum(-1)+
                            kl_divergence(bernoulli, ref_grip)).mean()
            entropy = (normal.entropy().sum(-1)+bernoulli.entropy()).mean()
            loss = policy_loss+cfg['reference_kl_weight']*reference_kl-cfg['entropy_weight']*entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), .5)
            optimizer.step()
            losses.append(float(policy_loss.detach()))
            kls.append(float(reference_kl.detach()))
    return dict(sorted_parcels=reward_total, policy_loss=sum(losses)/len(losses),
                reference_kl=sum(kls)/len(kls), skipped=False)


def atomic_save(path, value):
    path = Path(path)
    torch.save(value, str(path)+'.tmp')
    os.replace(str(path)+'.tmp', path)


def train(job):
    from warehouse_sort.utils import compose_cfg, make_env

    cfg = dict(job['rl_config'])
    device = torch.device(job.get('device','cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('T4 GPU 런타임을 선택하세요.')
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
    saved = torch.load(job['anchor'], map_location='cpu', weights_only=True)
    if saved.get('format') != 'moveboxes-stage-act-v1':
        raise ValueError('Stage ACT anchor required')
    model = StageACT(saved['model_config']); model.load_state_dict(saved['model'])
    reference = StageACT(saved['model_config']); reference.load_state_dict(saved['model'])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == 'queries' or name.startswith(('decoder.','output.')))
    reference.requires_grad_(False).eval()
    model.to(device); reference.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg['lr'])
    folder = Path(job['folder']); ckdir = folder/'checkpoints'; ckdir.mkdir(parents=True, exist_ok=True)
    latest = ckdir/'latest.pt'; start = 0; history = []
    if latest.exists():
        resume = torch.load(latest, map_location='cpu', weights_only=True)
        if resume.get('job_signature') != job['job_signature']:
            raise ValueError('Saved RL configuration differs; use a new run name')
        model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['torch_rng'])
        if device.type == 'cuda' and resume['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        start = resume['iteration']
        history = json.loads((folder/'rl_progress.json').read_text())['history']
    env_cfg = compose_cfg(['difficulty='+job['level'], 'num_envs='+str(cfg['num_envs'])], job['config_dir'])
    env, _ = make_env(env_cfg, 'state', env_cfg.randomization, num_envs=cfg['num_envs'])
    try:
        for iteration in range(start, cfg['iterations']):
            rollout_cfg = dict(cfg, iteration=iteration)
            rollout = collect(env, model.eval(), rollout_cfg, device)
            metrics = ppo_update(model, reference, rollout, cfg, optimizer, device)
            metrics['iteration'] = iteration+1
            history.append(metrics)
            weights = dict(format='moveboxes-stage-act-v1', model_config=saved['model_config'],
                model=model.state_dict(), step=saved.get('step',0), rl_iteration=iteration+1,
                job_signature=job['job_signature'], optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
            atomic_save(latest, weights)
            save_json(folder/'rl_progress.json', dict(history=history, reward='official sparse delta success_count',
                total_sorted_parcels=sum(item['sorted_parcels'] for item in history)))
            print(f"RL {iteration+1}/{cfg['iterations']} sorted={metrics['sorted_parcels']:.0f} "
                  f"policy={metrics['policy_loss']:.4f} anchor_kl={metrics['reference_kl']:.4f}", flush=True)
            sync_from_env()
    finally:
        env.close()
    save_json(folder/'training_complete.json', dict(iterations=cfg['iterations'],
        reward='environment sparse reward only'))
    sync_from_env()


if __name__ == '__main__':
    train(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
