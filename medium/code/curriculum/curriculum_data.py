"""Explicit partial-success curriculum; never equate accepted data with four-box completion."""
import json
from pathlib import Path
import numpy as np
import torch
from stage_data import StageWindows as OriginalWindows, split_data
from act_v2_data import normalization
from marso_experiment import digest
from stage_schema import PICK, CARRY, PLACE, DONE, HOLD, COMPLETE, RECOVER


def load_data(path, limit=None, recovery_manifest=None):
    path=Path(path)
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('version')!='curriculum-v21' or not manifest.get('complete') or len(manifest['episodes'])<4:
        raise ValueError('At least four complete deadline-verified demonstrations are required')
    minimum=manifest.get('minimum_correct')
    if minimum not in (3,4):raise ValueError('Invalid curriculum success threshold')
    result=[]
    for entry in manifest['episodes']:
        file=path.parent/entry['file']
        if file.resolve().parent!=path.parent.resolve() or digest(file)!=entry['sha256']:
            raise ValueError('Invalid demonstration path/digest')
        if (not entry['accepted'] or not minimum<=entry['correct']<=4 or entry['evaluation_actions']!=199
                or entry.get('minimum_correct')!=minimum or entry.get('failure') is not None
                or entry.get('fully_successful')!=(entry['correct']==4)):
            raise ValueError('Demonstration does not meet the evaluation deadline')
        with np.load(file,allow_pickle=False) as a:
            item={k:torch.from_numpy(a[k].copy()) for k in ('obs','actions','previous_stage','stage','gate','target','perturbed')}
        n=len(item['actions'])
        if not 1<=n<=199 or item['actions'].shape!=(n,4) or item['obs'].shape!=(n+1,72):
            raise ValueError('Demonstration alignment/shape mismatch')
        if not torch.isfinite(item['obs']).all() or not torch.isfinite(item['actions']).all() or (item['actions'].abs()>1).any():
            raise ValueError('Invalid state/action values')
        for key,bound in (('stage',4),('previous_stage',4),('gate',3),('target',4)):
            if item[key].shape!=(n,) or (item[key]<0).any() or (item[key]>=bound).any():
                raise ValueError('Invalid demonstration labels')
        if item['perturbed'].shape!=(n,):raise ValueError('Invalid disturbance mask')
        duration=entry.get('completion_steps',{}).get(str(entry['correct']))
        if not isinstance(duration,int) or not 1<=duration<=199:
            raise ValueError('Measured completion step required')
        item.update(name='recovery/'+entry['file'],source='recovery',clipped_fraction=0.,soft_gripper_fraction=0.,
            completion_actions=duration,fully_successful=entry['fully_successful'],correct=entry['correct'])
        result.append(item)
    return result


class StageWindows(OriginalWindows):
    def __init__(self, trajectories, ids, history, chunk, training=True, first_pick_fraction=0., efficiency_bonus=.5):
        super().__init__(trajectories,ids,history,chunk,training,first_pick_fraction=0.)
        # The inherited sampler reserves 35% for this pool, 25% for transitions,
        # and 40% for all frames. Include contact on EVERY parcel, not only #1.
        self.contact=[(i,t) for i,t in self.indices if int(trajectories[i]['stage'][t])==PICK
                      and float(trajectories[i]['obs'][t,20])<.13]
        self.recovery=self.contact or self.indices
        if not 0<=efficiency_bonus<=1:raise ValueError('Efficiency bonus must be in [0,1]')
        self.efficiency={i:1. for i in ids}
        full=[i for i in ids if trajectories[i]['fully_successful']]
        if training and len(full)>1:
            times=[trajectories[i]['completion_actions'] for i in full]
            shortest,longest=min(times),max(times)
            if longest>shortest:
                for i in full:
                    self.efficiency[i]+=efficiency_bonus*(longest-trajectories[i]['completion_actions'])/(longest-shortest)


    def batch(self, size, generator, device, noise=0.):
        rows = {k:[] for k in ('obs','actions','mask','previous_stage','stage','gate','efficiency_weight')}
        for _ in range(size):
            p = float(torch.rand((), generator=generator)) if self.training else 1.
            if self.training and self.first_pick and p < self.first_pick_fraction:
                pool = self.first_pick
            else:
                if self.training and self.first_pick:
                    p = (p-self.first_pick_fraction)/(1-self.first_pick_fraction)
                pool = self.recovery if p < .35 and self.recovery else self.transitions if p < .60 and self.transitions else self.indices
            i,t = pool[int(torch.randint(len(pool), (), generator=generator))]
            traj = self.trajectories[i]
            stage, gate = int(traj['stage'][t]), int(traj['gate'][t])
            previous = int(traj['previous_stage'][t])
            # A missed gate must remain recoverable after the one transition frame.
            # This augments supervisor memory only; observations and action labels stay real.
            if self.training and float(torch.rand((), generator=generator)) < .25:
                previous = int(torch.randint(4, (), generator=generator))
                if previous == stage:
                    gate = HOLD
                else:
                    gate = COMPLETE if stage == DONE or (previous,stage) in ((PICK,CARRY),(PICK,PLACE),(CARRY,PLACE)) else RECOVER
            valid = min(self.chunk, len(traj['actions'])-t)
            for dt in range(valid):
                if int(traj['stage'][t+dt]) != stage or int(traj['target'][t+dt]) != int(traj['target'][t]):
                    valid = dt
                    break
                if bool(traj['perturbed'][t+dt]):
                    # The clean label here is valid; later observations follow another action.
                    valid = dt+1
                    break
            a = torch.zeros(self.chunk,4)
            a[:valid] = traj['actions'][t:t+valid]
            rows['obs'].append(traj['obs'][torch.arange(t-self.history+1,t+1).clamp_min(0)])
            rows['actions'].append(a)
            rows['mask'].append((torch.arange(self.chunk)<valid).float())
            rows['previous_stage'].append(torch.tensor(previous))
            rows['stage'].append(torch.tensor(stage))
            rows['gate'].append(torch.tensor(gate))
            rows['efficiency_weight'].append(torch.tensor(self.efficiency[i]))
        rows = {k:torch.stack(v) for k,v in rows.items()}
        if noise:
            n = (rows['obs'].shape[-1]-36)//9
            columns = [18,19,20]+[26+7*i+j for i in range(n) for j in range(3)]
            rows['obs'][...,columns] += torch.randn((size,1,len(columns)), generator=generator)*noise
        return {k:v.to(device) for k,v in rows.items()}
