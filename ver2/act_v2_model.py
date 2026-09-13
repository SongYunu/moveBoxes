"""Small state-only ACT adaptation: CVAE training, deterministic prior at inference."""
import torch
from torch import nn
from torch.nn import functional as F


def state_features(state):
    dim = state.shape[-1]
    if dim not in (54, 72, 90):
        raise ValueError(f'WarehouseSort state shape required, got {dim}')
    n = (dim-36)//9
    xyz = state[..., 26:26+7*n].reshape(*state.shape[:-1], n, 7)[..., :3]
    tags = state[..., 26+7*n:26+9*n].reshape(*state.shape[:-1], n, 2)
    bins = state[..., 26+9*n:32+9*n].reshape(*state.shape[:-1], 2, 3)
    targets = tags @ bins
    relative = xyz-state[..., None, 18:21]
    return torch.cat((state, relative.flatten(-2), (targets-xyz).flatten(-2)), -1)


class StateACT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        d, h, k, z = cfg['width'], cfg['history'], cfg['chunk_size'], cfg['latent_dim']
        features = cfg['state_dim']+6*((cfg['state_dim']-36)//9)
        self.register_buffer('obs_mean', torch.zeros(features))
        self.register_buffer('obs_std', torch.ones(features))
        self.obs_proj = nn.Linear(features, d)
        self.history_pos = nn.Parameter(torch.randn(1, h, d)*.02)
        self.queries = nn.Parameter(torch.randn(1, k, d)*.02)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, cfg['heads'], d*4, dropout=0., activation='gelu', batch_first=True,
            norm_first=True), cfg['layers'], enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d, cfg['heads'], d*4, dropout=0., activation='gelu', batch_first=True,
            norm_first=True), cfg['layers'])
        self.posterior = nn.Sequential(nn.Linear(features*h+k*5, d*2), nn.GELU(), nn.Linear(d*2, z*2))
        self.latent_proj = nn.Linear(z, d)
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4))

    def forward(self, obs, actions=None, mask=None):
        x = ((state_features(obs)-self.obs_mean)/self.obs_std).clamp(-10, 10)
        batch = len(x)
        kl = x.new_zeros(())
        if actions is None:
            latent = x.new_zeros(batch, self.cfg['latent_dim'])
        else:
            target = torch.cat((actions*mask[..., None], mask[..., None]), -1).flatten(1)
            mu, logvar = self.posterior(torch.cat((x.flatten(1), target), -1)).chunk(2, -1)
            logvar = logvar.clamp(-10, 6)
            latent = mu+torch.exp(.5*logvar)*torch.randn_like(mu)
            kl = -.5*(1+logvar-mu.square()-logvar.exp()).mean()
        tokens = self.obs_proj(x)+self.history_pos
        tokens = torch.cat((self.latent_proj(latent)[:, None], tokens), 1)
        memory = self.encoder(tokens)
        return self.output(self.decoder(self.queries.expand(batch, -1, -1), memory)), kl


def action_loss(prediction, target, mask, kl, kl_weight=.001):
    # Mask every loss term: padded tails are not fake "stop here" examples.
    weights = mask*torch.exp(-.03*torch.arange(mask.shape[1], device=mask.device))[None]
    denom = weights.sum().clamp_min(1)
    xyz = F.smooth_l1_loss(prediction[..., :3], target[..., :3], reduction='none', beta=.1).mean(-1)
    grip = F.binary_cross_entropy_with_logits(prediction[..., 3], (target[..., 3]+1)/2, reduction='none')
    xyz, grip = (xyz*weights).sum()/denom, (grip*weights).sum()/denom
    return xyz+.25*grip+kl_weight*kl, dict(xyz=xyz.detach(), gripper=grip.detach(), kl=kl.detach())
