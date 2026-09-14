"""Learned target selection invalidates stale chunks when the target changes."""
import json
from collections import deque
from pathlib import Path
import torch
from hard_model import StageACT
from stage_schema import PICK,HOLD,RECOVER


class HardPolicy:
    def __init__(self,model,gate_threshold=.65,stage_threshold=.6,temporal_decay=.25,ensemble_window=4,**unused):
        self.model=model.eval();self.device=next(model.parameters()).device
        self.gate_threshold,self.stage_threshold=gate_threshold,stage_threshold
        self.decay,self.window=temporal_decay,ensemble_window
        if not 1<=self.window<=model.cfg['chunk_size']:raise ValueError('Invalid ensemble window')
        self.reset()

    def reset(self):
        self.history=deque(maxlen=self.model.cfg['history']);self.predictions=deque(maxlen=self.window)
        self.step,self.batch,self.last_decision=0,None,None

    @torch.no_grad()
    def act(self,obs,deterministic=True):
        state=(obs['state'] if isinstance(obs,dict) else obs).float().to(self.device)
        if self.batch!=len(state):
            self.reset();self.batch=len(state)
            self.stage=torch.full((self.batch,),PICK,dtype=torch.long,device=self.device)
            self.generation=torch.zeros_like(self.stage);self.target=torch.zeros_like(self.stage)
        self.history.append(state.clone())
        while len(self.history)<self.history.maxlen:self.history.appendleft(state.clone())
        x,memory,phase_logits,gate_logits,scores,tokens=self.model.encode(torch.stack(list(self.history),1),self.stage)
        phase_p,phase=phase_logits.softmax(-1).max(-1);gate_p,gate=gate_logits.softmax(-1).max(-1)
        accepted=(gate!=HOLD)&(gate_p>=self.gate_threshold)&(phase_p>=self.stage_threshold)
        old=self.stage.clone();old_target=self.target.clone()
        self.stage=torch.where(accepted,phase,self.stage)
        target_p,target=scores.softmax(-1).max(-1)
        self.target=torch.where((target_p>=self.stage_threshold)|(self.step==0),target,self.target)
        reset=(accepted&((self.stage!=old)|(gate==RECOVER)))|(self.target!=old_target)
        self.generation+=reset.long()
        prediction,_=self.model.decode_target(x,memory,self.stage,self.target,tokens)
        self.predictions.append((self.step,prediction,self.generation.clone()))
        proposals=[];weights=[]
        for start,past,generation in self.predictions:
            age=self.step-start;proposals.append(past[:,age,:3])
            weights.append((generation==self.generation).float()*torch.exp(state.new_tensor(-self.decay*age)))
        weights=torch.stack(weights,1)
        xyz=(torch.stack(proposals,1)*weights[...,None]).sum(1)/weights.sum(1)[:,None]
        grip=torch.where(prediction[:,0,3:4]>=0,1.,-1.)
        self.last_decision=dict(previous=old,stage=self.stage.clone(),proposed_stage=phase,gate=gate,
            accepted=accepted,gate_confidence=gate_p,stage_confidence=phase_p,target=self.target.clone(),target_confidence=target_p)
        self.step+=1
        return torch.cat((xyz.clamp(-1,1),grip),-1)


def load_stage(checkpoint,sample_obs,action_space,device,**cfg):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('format')!='moveboxes-hard-target-act-v2':raise ValueError('Requires Hard v2 target ACT weights')
    state=sample_obs['state'] if isinstance(sample_obs,dict) else sample_obs
    if state.shape[-1]!=saved['model_config']['state_dim'] or tuple(action_space.shape)!=(4,):raise ValueError('Shape mismatch')
    model=StageACT(saved['model_config']);model.load_state_dict(saved['model'])
    return HardPolicy(model.to(device),**{k:cfg[k] for k in ('gate_threshold','stage_threshold','temporal_decay','ensemble_window') if k in cfg})


def load_policy(checkpoint,sample_obs,action_space,device):
    cfg=json.loads((Path(checkpoint).parent/'policy_config.json').read_text())
    return load_stage(checkpoint,sample_obs,action_space,device,**cfg)
