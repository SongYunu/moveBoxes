"""One StageACT checkpoint for all three difficulties; pad absent parcel slots."""
from pathlib import Path
import json
import torch
from stage_model import StageACT
from stage_policy import StagePolicy


def canonical_state(state):
    dimension=state.shape[-1]
    if dimension==90:return state
    if dimension not in (54,72):raise ValueError('Expected Easy/Medium/Hard state')
    n=(dimension-36)//9
    result=state.new_zeros((*state.shape[:-1],90))
    result[...,:26]=state[...,:26]
    result[...,26:26+7*n]=state[...,26:26+7*n]
    result[...,68:68+2*n]=state[...,26+7*n:26+9*n]
    result[...,80:90]=state[...,26+9*n:]
    # Real parcel tags are one-hot; zero tags explicitly identify absent slots.
    return result


class UnifiedPolicy(StagePolicy):
    def act(self,obs,deterministic=True):
        state=obs['state'] if isinstance(obs,dict) else obs
        return super().act(canonical_state(state),deterministic)


def load_stage(checkpoint,sample_obs,action_space,device,**cfg):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('format')!='moveboxes-stage-act-v1' or saved['model_config']['state_dim']!=90:
        raise ValueError('Unified checkpoint must have the canonical 90-dimensional state')
    state=sample_obs['state'] if isinstance(sample_obs,dict) else sample_obs
    canonical_state(state)
    if tuple(action_space.shape)!=(4,):raise ValueError('Expected xyz/gripper control')
    model=StageACT(saved['model_config']);model.load_state_dict(saved['model'])
    return UnifiedPolicy(model.to(device),**{k:cfg[k] for k in ('gate_threshold','stage_threshold','temporal_decay','ensemble_window') if k in cfg})


def load_policy(checkpoint,sample_obs,action_space,device):
    cfg=json.loads((Path(checkpoint).parent/'policy_config.json').read_text())
    return load_stage(checkpoint,sample_obs,action_space,device,**cfg)
