"""Small ACT with a learned completion/recovery gate and a stage-conditioned decoder."""
import torch
from torch import nn
from torch.nn import functional as F
from act_v2_model import StateACT, state_features, action_loss
from stage_schema import STAGES, GATES


class StageACT(StateACT):
    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg['width']
        self.stage_embedding = nn.Embedding(len(STAGES), d)
        self.supervisor = nn.Sequential(nn.LayerNorm(d*2), nn.Linear(d*2, d), nn.GELU())
        self.stage_head = nn.Linear(d, len(STAGES))
        self.gate_head = nn.Linear(d, len(GATES))

    def encode(self, obs, previous_stage):
        x = ((state_features(obs)-self.obs_mean)/self.obs_std).clamp(-10, 10)
        memory = self.encoder(self.obs_proj(x)+self.history_pos)
        context = self.supervisor(torch.cat((memory[:, -1], self.stage_embedding(previous_stage)), -1))
        return x, memory, self.stage_head(context), self.gate_head(context)

    def decode(self, x, memory, stage, actions=None, mask=None):
        kl = x.new_zeros(())
        if actions is None:
            z = x.new_zeros(len(x), self.cfg['latent_dim'])
        else:
            target = torch.cat((actions*mask[..., None], mask[..., None]), -1).flatten(1)
            mu, logvar = self.posterior(torch.cat((x.flatten(1), target), -1)).chunk(2, -1)
            logvar = logvar.clamp(-10, 6)
            z = mu+torch.exp(.5*logvar)*torch.randn_like(mu)
            kl = -.5*(1+logvar-mu.square()-logvar.exp()).mean()
        context = torch.cat((memory, self.latent_proj(z)[:, None]), 1)
        queries = self.queries+self.stage_embedding(stage)[:, None]
        return self.output(self.decoder(queries, context)), kl

    def forward(self, obs, previous_stage, stage, actions=None, mask=None):
        x, memory, phase, gate = self.encode(obs, previous_stage)
        prediction, kl = self.decode(x, memory, stage, actions, mask)
        return prediction, kl, phase, gate


def stage_loss(outputs, actions, mask, stages, gates, cfg):
    prediction, kl, phase, gate = outputs
    loss, parts = action_loss(prediction, actions, mask, kl, cfg['kl_weight'])
    phase_loss = F.cross_entropy(phase, stages)
    gate_loss = F.cross_entropy(gate, gates)
    loss = loss+cfg['stage_loss_weight']*phase_loss+cfg['gate_loss_weight']*gate_loss
    parts.update(stage=phase_loss.detach(), gate=gate_loss.detach())
    return loss, parts
