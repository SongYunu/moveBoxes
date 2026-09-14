"""Parcel labels and balanced first-pick/contact/recovery sampling."""
import torch
from stage_data import load_data,split_data
from act_v2_data import normalization
from stage_schema import PICK,CARRY,DONE,HOLD,COMPLETE,RECOVER


class StageWindows:
    def __init__(self,trajectories,ids,history,chunk,training=True,first_pick_fraction=0.):
        self.trajectories,self.history,self.chunk,self.training=trajectories,history,chunk,training
        self.indices=[(i,t) for i in ids for t in range(len(trajectories[i]['actions']))]
        self.first_pick=[];self.contact=[];self.recovery=[]
        for i in ids:
            traj=trajectories[i];lift=(traj['stage']==CARRY).nonzero().flatten()
            end=int(lift[0])+1 if len(lift) else len(traj['actions'])
            self.first_pick.extend((i,t) for t in range(end))
            self.contact.extend((i,t) for t in range(len(traj['actions']))
                if int(traj['stage'][t])==PICK and float(traj['obs'][t,20])<.13 and float(traj['obs'][t,25])<.5)
            if traj['source']=='recovery':
                starts=(traj['gate']==RECOVER).nonzero().flatten().tolist()
                frames={t for start in starts for t in range(start,min(start+32,len(traj['actions'])))}
                self.recovery.extend((i,t) for t in sorted(frames))

    def batch(self,size,generator,device,noise=0.):
        rows={k:[] for k in ('obs','actions','mask','previous_stage','stage','gate','target')}
        for _ in range(size):
            p=float(torch.rand((),generator=generator)) if self.training else 1.
            pool=self.first_pick if p<.25 else self.contact if p<.50 else self.recovery if p<.70 else self.indices
            pool=pool or self.indices
            i,t=pool[int(torch.randint(len(pool),(),generator=generator))];traj=self.trajectories[i]
            stage,gate,target=(int(traj[k][t]) for k in ('stage','gate','target'))
            previous=int(traj['previous_stage'][t])
            if self.training and float(torch.rand((),generator=generator))<.25:
                previous=int(torch.randint(4,(),generator=generator))
                gate=HOLD if previous==stage else COMPLETE if stage==DONE or (previous,stage) in ((PICK,CARRY),(PICK,2),(CARRY,2)) else RECOVER
            valid=min(self.chunk,len(traj['actions'])-t)
            for dt in range(valid):
                if int(traj['stage'][t+dt])!=stage or int(traj['target'][t+dt])!=target:
                    valid=dt;break
                if bool(traj['perturbed'][t+dt]):valid=dt+1;break
            action=torch.zeros(self.chunk,4);action[:valid]=traj['actions'][t:t+valid]
            rows['obs'].append(traj['obs'][torch.arange(t-self.history+1,t+1).clamp_min(0)])
            rows['actions'].append(action);rows['mask'].append((torch.arange(self.chunk)<valid).float())
            for key,value in (('previous_stage',previous),('stage',stage),('gate',gate),('target',target)):
                rows[key].append(torch.tensor(value))
        rows={k:torch.stack(v) for k,v in rows.items()}
        if noise:
            n=(rows['obs'].shape[-1]-36)//9
            cols=[18,19,20]+[26+7*i+j for i in range(n) for j in range(3)]
            rows['obs'][...,cols]+=torch.randn((size,1,len(cols)),generator=generator)*noise
        return {k:v.to(device) for k,v in rows.items()}
