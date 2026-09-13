"""Replan every step. Average recent XYZ proposals; use the latest gripper logit."""
import json
from collections import deque
from pathlib import Path
import torch
from act_v2_model import StateACT


class ACTPolicy:
    def __init__(self, model, temporal_decay=.25, ensemble_window=4):
        if temporal_decay < 0 or not 1 <= ensemble_window <= model.cfg['chunk_size']:
            raise ValueError('Invalid temporal ensemble configuration')
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.decay, self.window = temporal_decay, ensemble_window
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.model.cfg['history'])
        self.predictions = deque(maxlen=self.window)
        self.step = 0
        self.batch = None

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        state = (obs['state'] if isinstance(obs, dict) else obs).float().to(self.device)
        if self.batch != len(state):
            self.reset()
            self.batch = len(state)
        self.history.append(state.clone())
        while len(self.history) < self.history.maxlen:
            self.history.appendleft(state.clone())
        prediction, _ = self.model(torch.stack(list(self.history), 1))
        self.predictions.append((self.step, prediction))
        ages = torch.tensor([self.step-t for t, _ in self.predictions], device=self.device)
        weights = torch.exp(-self.decay*ages.float())  # Larger age = less weight.
        xyz = torch.stack([p[:, self.step-t, :3] for t, p in self.predictions], 1)
        xyz = (xyz*weights[None, :, None]).sum(1)/weights.sum()
        grip = torch.where(prediction[:, 0, 3:4] >= 0, 1., -1.)
        self.step += 1
        return torch.cat((xyz.clamp(-1, 1), grip), -1)


def load_act(checkpoint, sample_obs, action_space, device, **policy_cfg):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('format') != 'moveboxes-act-ver2':
        raise ValueError('This loader requires a ver2 ACT checkpoint, not a DP checkpoint.')
    state = sample_obs['state'] if isinstance(sample_obs, dict) else sample_obs
    if state.shape[-1] != saved['model_config']['state_dim'] or tuple(action_space.shape) != (4,):
        raise ValueError('Checkpoint/environment shape mismatch')
    if 'model_config' in policy_cfg and policy_cfg['model_config'] != saved['model_config']:
        raise ValueError('Requested architecture differs from the trained checkpoint')
    model = StateACT(saved['model_config'])
    model.load_state_dict(saved['model'])
    return ACTPolicy(model.to(device), policy_cfg.get('temporal_decay', .25), policy_cfg.get('ensemble_window', 4))


def load_policy(checkpoint, sample_obs, action_space, device):
    cfg = json.loads((Path(checkpoint).parent/'policy_config.json').read_text())
    return load_act(checkpoint, sample_obs, action_space, device, **cfg)
