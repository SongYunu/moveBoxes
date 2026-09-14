"""Evaluate unchanged native Medium weights on Easy by padding only the observation."""
import json
from pathlib import Path
from stage_policy import load_stage as native_load


def medium_observation(obs):
    state=obs['state'] if isinstance(obs,dict) else obs
    if state.shape[-1]==72:return state
    if state.shape[-1]!=54:raise ValueError('This probe supports Easy/Medium only')
    padded=state.new_zeros((*state.shape[:-1],72))
    padded[...,:40]=state[...,:40]       # Proprioception and two parcel poses.
    padded[...,54:58]=state[...,40:44]  # Two real parcel tags; two absent tags remain zero.
    padded[...,62:72]=state[...,44:54]  # Bin positions and colours.
    return padded


class ProbePolicy:
    def __init__(self,policy):self.policy=policy
    def reset(self):return self.policy.reset()
    @property
    def last_decision(self):return self.policy.last_decision
    def act(self,obs,deterministic=True):return self.policy.act(medium_observation(obs),deterministic)


def load_stage(checkpoint,sample_obs,action_space,device,**cfg):
    return ProbePolicy(native_load(checkpoint,medium_observation(sample_obs),action_space,device,**cfg))


def load_policy(checkpoint,sample_obs,action_space,device):
    cfg=json.loads((Path(checkpoint).parent/'policy_config.json').read_text())
    return load_stage(checkpoint,sample_obs,action_space,device,**cfg)
