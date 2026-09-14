"""Learned parcel pointer and parcel-conditioned actions; no geometric action override."""
import torch
from torch import nn
from torch.nn import functional as F
from stage_model import StageACT as OriginalACT, stage_loss as original_loss


def parcel_features(obs):
    n=(obs.shape[-1]-36)//9
    pose=obs[...,26:26+7*n].reshape(*obs.shape[:-1],n,7)
    tags=obs[...,26+7*n:26+9*n].reshape(*obs.shape[:-1],n,2)
    bins=obs[...,26+9*n:32+9*n].reshape(*obs.shape[:-1],2,3)
    return torch.cat((pose,tags,pose[...,:3]-obs[...,None,18:21],tags@bins-pose[...,:3]),-1)


class StageACT(OriginalACT):
    def __init__(self,cfg):
        super().__init__(cfg)
        d=cfg['width']
        self.parcel_proj=nn.Sequential(nn.Linear(15,d),nn.GELU(),nn.Linear(d,d))
        self.pointer=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Linear(d,1))

    def encode(self,obs,previous_stage):
        x,memory,phase,gate=super().encode(obs,previous_stage)
        tokens=self.parcel_proj(parcel_features(obs[:,-1]))
        context=memory[:,-1,None].expand_as(tokens)
        scores=self.pointer(torch.cat((context,tokens),-1)).squeeze(-1)
        return x,memory,phase,gate,scores,tokens

    def decode_target(self,x,memory,stage,target,tokens):
        selected=tokens[torch.arange(len(tokens),device=tokens.device),target]
        context=torch.cat((memory,selected[:,None]),1)
        queries=self.queries+self.stage_embedding(stage)[:,None]+selected[:,None]
        return self.output(self.decoder(queries,context)),x.new_zeros(())

    def forward(self,obs,previous_stage,stage,actions=None,mask=None,target=None,use_target=True):
        x,memory,phase,gate,scores,tokens=self.encode(obs,previous_stage)
        chosen=target if target is not None and use_target else scores.argmax(-1)
        prediction,kl=self.decode_target(x,memory,stage,chosen,tokens)
        return prediction,kl,phase,gate,scores,target


def stage_loss(outputs,actions,mask,stages,gates,cfg):
    loss,parts=original_loss(outputs[:4],actions,mask,stages,gates,cfg)
    scores,target=outputs[4:]
    if target is None:raise ValueError('Parcel labels are required for training/validation')
    pointer_loss=F.cross_entropy(scores,target)
    parts['target']=pointer_loss.detach()
    return loss+cfg['target_loss_weight']*pointer_loss,parts
