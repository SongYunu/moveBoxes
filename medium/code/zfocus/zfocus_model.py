"""Decoder-only height/contact correction, compatible with existing Stage ACT weights."""
import torch
from torch.nn import functional as F
from stage_model import StageACT as OriginalACT
from stage_schema import PICK


class StageACT(OriginalACT):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Preserve observation normalization, representation and completion gates.
        for name, parameter in self.named_parameters():
            if not name.startswith(('decoder.', 'output.', 'queries', 'latent_proj.')):
                parameter.requires_grad_(False)


def stage_loss(outputs, actions, mask, stages, gates, cfg):
    prediction, kl, phase, gate = outputs
    weights = mask*torch.exp(-.03*torch.arange(mask.shape[1], device=mask.device))[None]
    denom = weights.sum().clamp_min(1)
    pick = (stages == PICK)[:, None]
    closing = pick & (actions[..., 3] < 0)
    descending = pick & (actions[..., 3] > 0) & (actions[..., 2] < -.05)
    error = F.smooth_l1_loss(prediction[..., :3], actions[..., :3], reduction='none', beta=.1)
    z_multiplier = 1+(cfg['pick_z_weight']-1)*pick+(cfg['contact_z_weight']-cfg['pick_z_weight'])*closing
    xyz = ((error[..., 0]+error[..., 1]+error[..., 2]*z_multiplier)*weights).sum()/(3*denom)
    grip_error = F.binary_cross_entropy_with_logits(prediction[..., 3], (actions[..., 3]+1)/2, reduction='none')
    contact = closing | descending
    grip = (grip_error*(1+(cfg['contact_gripper_weight']-1)*contact)*weights).sum()/denom
    phase_loss, gate_loss = F.cross_entropy(phase, stages), F.cross_entropy(gate, gates)
    loss = xyz+.25*grip+cfg['kl_weight']*kl+cfg['stage_loss_weight']*phase_loss+cfg['gate_loss_weight']*gate_loss
    return loss, dict(xyz=xyz.detach(), z=(error[..., 2]*weights).sum().detach()/denom,
        gripper=grip.detach(), kl=kl.detach(), stage=phase_loss.detach(), gate=gate_loss.detach())
