"""Deployment: observations -> learned gate -> learned actions. No scripted expert."""
import json
from collections import deque
from pathlib import Path
import torch
from stage_model import StageACT
from stage_model import PickResidualStageACT
from stage_schema import PICK, HOLD, RECOVER, STAGES, GATES


class StagePolicy:
    def __init__(self, model, gate_threshold=.65, stage_threshold=.6, temporal_decay=.25, ensemble_window=4,
                 auto_reset_steps=None):
        if not 0 < gate_threshold < 1 or not 0 < stage_threshold < 1:
            raise ValueError('Gate confidence thresholds must be in (0,1)')
        if temporal_decay < 0 or not 1 <= ensemble_window <= model.cfg['chunk_size']:
            raise ValueError('Invalid temporal ensemble')
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.gate_threshold, self.stage_threshold = gate_threshold, stage_threshold
        self.decay, self.window = temporal_decay, ensemble_window
        if auto_reset_steps is not None and (type(auto_reset_steps) is not int or auto_reset_steps < 1):
            raise ValueError('auto_reset_steps must be a positive integer or None')
        self.auto_reset_steps = auto_reset_steps
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.model.cfg['history'])
        self.predictions = deque(maxlen=self.window)
        self.step, self.batch = 0, None
        self.last_decision = None

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        state = (obs['state'] if isinstance(obs, dict) else obs).float().to(self.device)
        if self.auto_reset_steps is not None and self.step >= self.auto_reset_steps:
            self.reset()
        if self.batch != len(state):
            self.reset()
            self.batch = len(state)
            self.stage = torch.full((self.batch,), PICK, dtype=torch.long, device=self.device)
            self.generation = torch.zeros_like(self.stage)
        self.history.append(state.clone())
        while len(self.history) < self.history.maxlen:
            self.history.appendleft(state.clone())
        x, memory, stage_logits, gate_logits = self.model.encode(torch.stack(list(self.history), 1), self.stage)
        phase_p, phase = stage_logits.softmax(-1).max(-1)
        gate_p, gate = gate_logits.softmax(-1).max(-1)
        accepted = (gate != HOLD) & (gate_p >= self.gate_threshold) & (phase_p >= self.stage_threshold)
        old = self.stage.clone()
        self.stage = torch.where(accepted, phase, self.stage)
        reset = accepted & ((self.stage != old) | (gate == RECOVER))
        # Invalidate stale chunks separately for each environment, including same-stage recovery.
        self.generation += reset.long()
        prediction, _ = self.model.decode(x, memory, self.stage)
        self.predictions.append((self.step, prediction, self.generation.clone()))
        proposals, weights = [], []
        for start, past, generation in self.predictions:
            age = self.step-start
            proposals.append(past[:, age, :3])
            weights.append((generation == self.generation).float()*torch.exp(state.new_tensor(-self.decay*age)))
        weights = torch.stack(weights, 1)
        xyz = (torch.stack(proposals, 1)*weights[..., None]).sum(1)/weights.sum(1)[:, None]
        grip = torch.where(prediction[:, 0, 3:4] >= 0, 1., -1.)
        self.last_decision = dict(previous=old, stage=self.stage.clone(), proposed_stage=phase,
                                  gate=gate, accepted=accepted, gate_confidence=gate_p, stage_confidence=phase_p)
        self.step += 1
        return torch.cat((xyz.clamp(-1, 1), grip), -1)


def load_stage(checkpoint, sample_obs, action_space, device, **cfg):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    checkpoint_format = saved.get('format')
    if checkpoint_format not in ('moveboxes-stage-act-v1', 'moveboxes-stage-pick-residual-v1'):
        raise ValueError('Requires a stage ACT or pick-residual checkpoint')
    state = sample_obs['state'] if isinstance(sample_obs, dict) else sample_obs
    if state.shape[-1] != saved['model_config']['state_dim'] or tuple(action_space.shape) != (4,):
        raise ValueError('Checkpoint/environment shape mismatch')
    if cfg.get('model_config', saved['model_config']) != saved['model_config']:
        raise ValueError('Requested architecture differs from trained checkpoint')
    model_type = PickResidualStageACT if checkpoint_format == 'moveboxes-stage-pick-residual-v1' else StageACT
    model = model_type(saved['model_config'])
    model.load_state_dict(saved['model'])
    return StagePolicy(model.to(device), **{k:cfg[k] for k in
        ('gate_threshold','stage_threshold','temporal_decay','ensemble_window','auto_reset_steps') if k in cfg})


def load_policy(checkpoint, sample_obs, action_space, device):
    cfg = json.loads((Path(checkpoint).parent/'policy_config.json').read_text())
    return load_stage(checkpoint, sample_obs, action_space, device, **cfg)
