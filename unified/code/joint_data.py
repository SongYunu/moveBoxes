"""Equal task counts per minibatch; no active difficulty or curriculum gate."""
import torch


class JointWindows:
    def __init__(self,trajectories,ids,history,chunk,training=True):
        self.levels=('easy','medium','hard')
        if set(ids)!=set(self.levels) or any(not ids[k] for k in self.levels):
            raise ValueError('Every task requires independent train and validation episodes')
        self.trajectories,self.history,self.chunk=trajectories,history,chunk
        self.pools={k:[(i,t) for i in ids[k] for t in range(len(trajectories[i]['actions']))] for k in self.levels}
        self.indices=[row for pool in self.pools.values() for row in pool]

    def counts(self,size,generator):
        if size<3:raise ValueError('Joint batch_size must be at least three')
        counts=[size//3]*3
        for i in torch.randperm(3,generator=generator)[:size%3]:counts[int(i)]+=1
        return counts

    def batch(self,size,generator,device,noise=0.):
        if noise:raise ValueError('Joint training uses original aligned observations/actions; position_noise must be zero')
        counts=self.counts(size,generator)
        rows={k:[] for k in ('obs','actions','mask','previous_stage','stage','gate')}
        for level,n in zip(self.levels,counts):
            pool=self.pools[level]
            for index in torch.randint(len(pool),(n,),generator=generator):
                i,t=pool[int(index)];traj=self.trajectories[i]
                valid=min(self.chunk,len(traj['actions'])-t)
                action=torch.zeros(self.chunk,4);action[:valid]=traj['actions'][t:t+valid]
                rows['obs'].append(traj['obs'][torch.arange(t-self.history+1,t+1).clamp_min(0)])
                rows['actions'].append(action);rows['mask'].append((torch.arange(self.chunk)<valid).float())
                # Metadata only, excluded from ObjectACT inputs and loss.
                for key in ('previous_stage','stage','gate'):rows[key].append(traj[key][t])
        order=torch.randperm(size,generator=generator)
        return {k:torch.stack(v)[order].to(device) for k,v in rows.items()}
