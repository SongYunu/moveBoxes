"""State diffusion policy with coherent action chunks and explicit reset."""
import json
from collections import deque
from pathlib import Path
import torch


class ChunkPolicy:
    def __init__(self, base, act_horizon):
        self.base = base
        self.obs_horizon = base.obs_horizon
        self.act_horizon = int(act_horizon)
        if not 1 <= self.act_horizon <= base.pred_horizon - self.obs_horizon + 1:
            raise ValueError('Invalid execution horizon')
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.obs_horizon)
        self.actions = deque()
        self.batch_size = None

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        b = self.base
        cur = (obs['state'] if isinstance(obs, dict) else obs).float().to(b.device)
        if self.batch_size != cur.shape[0]:
            self.reset()
            self.batch_size = cur.shape[0]
        self.history.append(cur.clone())
        while len(self.history) < self.obs_horizon:
            self.history.appendleft(cur.clone())
        if not self.actions:
            cond = torch.stack(list(self.history), dim=1).flatten(start_dim=1)
            seq = torch.randn((cur.shape[0], b.pred_horizon, b.act_dim), device=b.device)
            for k in b.scheduler.timesteps:
                eps = b.net(sample=seq, timestep=k, global_cond=cond)
                seq = b.scheduler.step(model_output=eps, timestep=k, sample=seq).prev_sample
            start = self.obs_horizon - 1
            self.actions.extend(seq[:, start:start+self.act_horizon].clamp(-1, 1).unbind(1))
        return self.actions.popleft()


def load_policy(checkpoint, sample_obs, action_space, device):
    from warehouse_sort.il_policy import load_dp
    cfg = json.loads((Path(checkpoint).parent/'policy_config.json').read_text())
    chunk = cfg.pop('act_horizon', 8)
    return ChunkPolicy(load_dp(checkpoint, sample_obs, action_space, device, **cfg), chunk)
