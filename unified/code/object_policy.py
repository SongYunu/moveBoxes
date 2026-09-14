"""One observation-driven policy for all levels, replanned every control step."""
import json
from collections import deque
from pathlib import Path
import torch
from object_model import ObjectACT, FORMAT, canonical_state


class ObjectPolicy:
    def __init__(self, model, temporal_decay=.25, ensemble_window=2):
        if temporal_decay < 0 or not 1 <= ensemble_window <= model.cfg['chunk_size']:
            raise ValueError('Invalid action ensemble settings')
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.decay, self.window = temporal_decay, ensemble_window
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.model.cfg['history'])
        self.predictions = deque(maxlen=self.window)
        self.step, self.batch = 0, None

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        state = canonical_state((obs['state'] if isinstance(obs, dict) else obs).float().to(self.device))
        if state.ndim != 2:
            raise ValueError('Policy expects a batched observation')
        if self.batch != len(state):
            self.reset()
            self.batch = len(state)
        self.history.append(state.clone())
        while len(self.history) < self.history.maxlen:
            self.history.appendleft(state.clone())
        prediction = self.model(torch.stack(list(self.history), 1))[0]
        self.predictions.append((self.step, prediction))
        proposals, weights = [], []
        for start, past in self.predictions:
            age = self.step-start
            proposals.append(past[:, age, :3])
            weights.append(state.new_tensor(-self.decay*age).exp())
        weights = torch.stack(weights)
        xyz = (torch.stack(proposals, 1)*weights[None, :, None]).sum(1)/weights.sum()
        grip = torch.where(prediction[:, 0, 3:4] >= 0, 1., -1.)
        self.step += 1
        return torch.cat((xyz.clamp(-1, 1), grip), -1)


def load_stage(checkpoint, sample_obs, action_space, device, **cfg):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('format') != FORMAT:
        raise ValueError('Use the new jointly trained object-attention checkpoint')
    if cfg.get('model_config', saved['model_config']) != saved['model_config']:
        raise ValueError('Requested architecture differs from trained checkpoint')
    canonical_state(sample_obs['state'] if isinstance(sample_obs, dict) else sample_obs)
    if tuple(action_space.shape) != (4,):
        raise ValueError('Expected xyz/gripper control')
    model = ObjectACT(saved['model_config'])
    model.load_state_dict(saved['model'])
    return ObjectPolicy(model.to(device), **{k:cfg[k] for k in ('temporal_decay','ensemble_window') if k in cfg})


def load_policy(checkpoint, sample_obs, action_space, device):
    cfg = json.loads((Path(checkpoint).parent/'policy_config.json').read_text())
    return load_stage(checkpoint, sample_obs, action_space, device, **cfg)
