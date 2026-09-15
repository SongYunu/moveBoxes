"""Optional chunk execution for existing StageACT weights; legacy policy stays intact.

The supervisor still observes every environment step. Only the action decoder is
cached. Gripper filtering follows learned logits, never geometry or a stage/open
lookup table. Call reset() after every environment reset.
"""
import json
from pathlib import Path
import torch
from stage_policy import StagePolicy, load_stage
from stage_schema import PICK, HOLD, RECOVER, STAGES


class ChunkStagePolicy(StagePolicy):
    def __init__(self, model, stage_horizons=None, gripper_fsm=False,
                 gripper_margin=.5, gripper_confirm_steps=2, **kwargs):
        # Conservative starts: pick/place each contain multiple fine motor phases.
        horizons = dict(pick=2, carry=6, place=2, done=1)
        if stage_horizons is not None:
            if set(stage_horizons) != set(STAGES):
                raise ValueError('stage_horizons must specify pick/carry/place/done')
            horizons = dict(stage_horizons)
        if any(type(v) is not int or not 1 <= v <= model.cfg['chunk_size'] for v in horizons.values()):
            raise ValueError('Each execution horizon must be within the trained chunk_size')
        if not 0 <= gripper_margin < float('inf') or type(gripper_confirm_steps) is not int or gripper_confirm_steps < 1:
            raise ValueError('Invalid gripper filter settings')
        self.stage_horizons = horizons
        self.gripper_fsm = bool(gripper_fsm)
        self.gripper_margin = gripper_margin
        self.gripper_confirm_steps = gripper_confirm_steps
        super().__init__(model, **kwargs)

    def reset(self):
        super().reset()
        self.action_buffer = self.cursor = self.remaining = None
        self.grip_state = self.grip_pending = self.grip_count = None
        self.decoder_calls = 0

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        state = (obs['state'] if isinstance(obs, dict) else obs).float().to(self.device)
        if state.ndim != 2 or state.shape[-1] != self.model.cfg['state_dim'] or not torch.isfinite(state).all():
            raise ValueError('Expected finite batched state matching the checkpoint')
        if self.batch != len(state):
            self.reset()
            self.batch = len(state)
            self.stage = torch.full((self.batch,), PICK, dtype=torch.long, device=self.device)
            self.generation = torch.zeros_like(self.stage)
            self.cursor = torch.zeros_like(self.stage)
            self.remaining = torch.zeros_like(self.stage)
            self.action_buffer = state.new_zeros(self.batch, self.model.cfg['chunk_size'], 4)
            self.grip_state = state.new_zeros(self.batch)
            self.grip_pending = state.new_zeros(self.batch)
            self.grip_count = torch.zeros_like(self.stage)
        self.history.append(state.clone())
        while len(self.history) < self.history.maxlen:
            self.history.appendleft(state.clone())
        x, memory, phase_logits, gate_logits = self.model.encode(torch.stack(list(self.history), 1), self.stage)
        phase_p, phase = phase_logits.softmax(-1).max(-1)
        gate_p, gate = gate_logits.softmax(-1).max(-1)
        accepted = (gate != HOLD) & (gate_p >= self.gate_threshold) & (phase_p >= self.stage_threshold)
        old = self.stage.clone()
        self.stage = torch.where(accepted, phase, self.stage)
        invalidate = accepted & ((self.stage != old) | (gate == RECOVER))
        self.generation += invalidate.long()
        self.remaining[invalidate] = 0
        self.action_buffer[invalidate] = 0
        replan = self.remaining == 0
        if replan.any():
            prediction, _ = self.model.decode(x[replan], memory[replan], self.stage[replan])
            if prediction.shape != (int(replan.sum()), self.model.cfg['chunk_size'], 4) or not torch.isfinite(prediction).all():
                raise ValueError('Invalid action chunk from model')
            self.action_buffer[replan] = prediction
            self.cursor[replan] = 0
            lengths = torch.tensor([self.stage_horizons[s] for s in STAGES], device=self.device)
            self.remaining[replan] = lengths[self.stage[replan]]
            self.decoder_calls += 1
        command = self.action_buffer[torch.arange(self.batch, device=self.device), self.cursor].clone()
        logits = command[:, 3]
        grip = torch.where(logits >= 0, 1., -1.)
        if self.gripper_fsm:
            # Reset the filter on transitions/recovery: a new learned command is
            # applied immediately instead of inheriting an old-stage latch.
            fresh = (self.grip_state == 0) | invalidate
            self.grip_state[fresh] = grip[fresh]
            self.grip_count[fresh] = 0
            confident_change = (grip != self.grip_state) & (logits.abs() >= self.gripper_margin) & ~fresh
            consecutive = confident_change & (grip == self.grip_pending)
            self.grip_count = torch.where(confident_change,
                torch.where(consecutive, self.grip_count + 1, 1), 0)
            self.grip_pending = grip.clone()
            switch = self.grip_count >= self.gripper_confirm_steps
            self.grip_state = torch.where(switch, grip, self.grip_state)
            grip = self.grip_state.clone()
        self.cursor += 1
        self.remaining -= 1
        self.last_decision = dict(previous=old, stage=self.stage.clone(), proposed_stage=phase,
            gate=gate, accepted=accepted, gate_confidence=gate_p, stage_confidence=phase_p,
            buffer_reset=invalidate, replanned=replan, buffer_remaining=self.remaining.clone())
        self.step += 1
        return torch.cat((command[:, :3].clamp(-1, 1), grip[:, None]), -1)


def load_chunk_stage(checkpoint, sample_obs, action_space, device, **cfg):
    baseline = load_stage(checkpoint, sample_obs, action_space, device, **cfg)
    if not cfg.get('stage_aware_chunk', False):
        if cfg.get('gripper_fsm', False):
            raise ValueError('gripper_fsm requires stage_aware_chunk')
        return baseline
    keys = ('gate_threshold', 'stage_threshold', 'temporal_decay', 'ensemble_window',
            'stage_horizons', 'gripper_fsm', 'gripper_margin', 'gripper_confirm_steps')
    return ChunkStagePolicy(baseline.model, **{k: cfg[k] for k in keys if k in cfg})


def load_policy(checkpoint, sample_obs, action_space, device):
    cfg = json.loads((Path(checkpoint).parent / 'policy_config.json').read_text())
    return load_chunk_stage(checkpoint, sample_obs, action_space, device, **cfg)
