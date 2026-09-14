"""Shared object attention and temporal action chunks, without parcel-slot IDs.

Only proprioception uses fitted statistics. Geometry uses common physical scales,
so reordering parcels/bins cannot select different normalization parameters.
"""
import torch
from torch import nn
from torch.nn import functional as F
from act_v2_model import action_loss

FORMAT = 'moveboxes-object-act-v1'
ARCHITECTURE = 'object-attention-v1'


def canonical_state(state):
    dimension = state.shape[-1]
    if dimension == 90:
        return state
    if dimension not in (54, 72):
        raise ValueError('Expected Easy/Medium/Hard state (54/72/90)')
    n = (dimension-36)//9
    result = state.new_zeros((*state.shape[:-1], 90))
    result[..., :26] = state[..., :26]
    result[..., 26:26+7*n] = state[..., 26:26+7*n]
    result[..., 68:68+2*n] = state[..., 26+7*n:26+9*n]
    result[..., 80:90] = state[..., 26+9*n:]
    return result


def rotation_features(quaternion):
    # First two rotation-matrix columns: q and -q describe the same rotation.
    w, x, y, z = F.normalize(quaternion, dim=-1, eps=1e-8).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
                        2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), -1)


class ObjectACT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.get('architecture') != ARCHITECTURE or cfg['state_dim'] != 90:
            raise ValueError('This model requires object-attention-v1, canonical state_dim=90')
        self.cfg = dict(cfg)
        d, h, k = cfg['width'], cfg['history'], cfg['chunk_size']
        self.register_buffer('obs_mean', torch.zeros(26))
        self.register_buffer('obs_std', torch.ones(26))
        self.robot_proj = nn.Linear(26, d)
        self.parcel_proj = nn.Sequential(nn.Linear(14, d), nn.GELU(), nn.Linear(d, d))
        self.bin_proj = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        # Types identify robot/parcel/bin, never an individual object or task.
        self.type_embedding = nn.Parameter(torch.randn(3, d)*.02)
        self.history_pos = nn.Parameter(torch.randn(1, h, d)*.02)
        self.queries = nn.Parameter(torch.randn(1, k, d)*.02)
        def encoder(layers):
            return nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d, cfg['heads'], 4*d, dropout=0., activation='gelu',
                batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
        self.spatial_encoder = encoder(cfg.get('spatial_layers', 1))
        self.temporal_encoder = encoder(cfg['layers'])
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d, cfg['heads'], 4*d, dropout=0., activation='gelu',
            batch_first=True, norm_first=True), cfg['layers'])
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4))

    def scene_tokens(self, obs):
        shape = obs.shape[:-1]
        poses = obs[..., 26:68].reshape(*shape, 6, 7)
        tags = obs[..., 68:80].reshape(*shape, 6, 2)
        bins = obs[..., 80:86].reshape(*shape, 2, 3)
        colors = obs[..., 86:90].reshape(*shape, 2, 2)
        present = tags.sum(-1) > 0
        bin_present = colors.sum(-1) > 0
        poses = torch.where(present[..., None], poses, 0.)
        bins = torch.where(bin_present[..., None], bins, 0.)
        tcp = obs[..., None, 18:21]
        # Match actual bin colors; bin array order can change without changing actions.
        matching = tags @ colors.transpose(-1, -2)
        target = (matching @ bins)/matching.sum(-1, keepdim=True).clamp_min(1.)
        parcel_features = torch.cat(((poses[..., :3]-tcp)/.3,
            rotation_features(poses[..., 3:]), tags, (target-poses[..., :3])/.3), -1)
        parcel_features = torch.where(present[..., None], parcel_features, 0.)
        bin_features = torch.cat(((bins-tcp)/.3, colors), -1)
        bin_features = torch.where(bin_present[..., None], bin_features, 0.)
        robot = self.robot_proj(((obs[..., :26]-self.obs_mean)/self.obs_std).clamp(-10, 10))
        tokens = torch.cat(((robot+self.type_embedding[0])[..., None, :],
            self.parcel_proj(parcel_features)+self.type_embedding[1],
            self.bin_proj(bin_features)+self.type_embedding[2]), -2)
        padding = torch.cat((torch.zeros_like(present[..., :1]), ~present, ~bin_present), -1)
        return tokens, padding

    def forward(self, obs, previous_stage=None, stage=None, actions=None, mask=None):
        # No expert phase, future action, difficulty ID, or scripted gate enters the policy.
        if obs.shape[1:] != (self.cfg['history'], 90):
            raise ValueError('ObjectACT requires [batch, history, 90] observations')
        b, h = obs.shape[:2]
        tokens, padding = self.scene_tokens(obs)
        spatial = self.spatial_encoder(tokens.flatten(0, 1),
            src_key_padding_mask=padding.flatten(0, 1)).reshape(b, h, 9, -1)
        temporal = self.temporal_encoder(spatial[:, :, 0]+self.history_pos)
        # Keep individual current objects available to the action decoder as well.
        memory = torch.cat((temporal, spatial[:, -1, 1:]), 1)
        memory_padding = torch.cat((padding.new_zeros(b, h), padding[:, -1, 1:]), 1)
        prediction = self.output(self.decoder(self.queries.expand(b, -1, -1), memory,
            memory_key_padding_mask=memory_padding))
        # Adapter for the shared optimizer; these unused diagnostics never gate actions.
        return prediction, prediction.new_zeros(()), prediction.new_zeros(b, 4), prediction.new_zeros(b, 3)


StageACT = ObjectACT  # Shared trainer interface; checkpoint format remains distinct.


def stage_loss(outputs, actions, mask, stages, gates, cfg):
    return action_loss(outputs[0], actions, mask, outputs[1], 0.)
