"""Task-balanced rehearsal and expert time-window crops selected by development failures."""
import json
from pathlib import Path
import torch
from stage_data import load_data,split_data,StageWindows
from unified_policy import canonical_state
from marso_experiment import digest


def load_curriculum(sources,seed):
    train={};valid={};trajectories=[]
    for source in sources:
        if 'data_sha256' in source:
            path=Path(source['data']);metadata=path.with_suffix('.json')
            if digest(path)!=source['data_sha256'] or (digest(metadata) if metadata.exists() else None)!=source['metadata_sha256']:
                raise ValueError('Dataset or success metadata changed after the training job was created')
        level=source['level']
        if source.get('format')=='curriculum-v21':
            from unified_deadline_data import load_data as load_deadline
            if digest(source['data'])!=source['manifest_sha256']:raise ValueError('Deadline manifest changed')
            items=load_deadline(source['data'])
        else:items=load_data(source['data'],source.get('num_demos'))
        train_ids,val_ids=split_data(items,seed)
        metadata_path=Path(source['data']).with_suffix('.json')
        metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        episodes={int(e['episode_id']):e for e in metadata.get('episodes',[]) if 'episode_id' in e}
        offset=len(trajectories)
        for item in items:
            item['obs']=canonical_state(item['obs']);item['level']=level
            episode=episodes.get(int(item['name'].split('_')[-1]),{}) if source.get('format')!='curriculum-v21' else {}
            # Only explicit dataset success metadata permits speed weighting.
            info=episode.get('info') or {};item['verified_success']=item.get('fully_successful') is True or episode.get('success') is True or info.get('success') is True
            item['name']=level+'/'+item['name']
        trajectories+=items
        train.setdefault(level,[]).extend(offset+i for i in train_ids);valid.setdefault(level,[]).extend(offset+i for i in val_ids)
    return trajectories,train,valid


class CurriculumWindows:
    def __init__(self,trajectories,ids,history,chunk,active,focus=None,training=True,replay_fraction=.3,speed_bonus=.5,focus_fraction=.5,contact_sampling=False):
        self.trajectories=trajectories;self.ids=ids;self.active=active;self.training=training;self.replay=replay_fraction
        self.focus_fraction=focus_fraction
        self.pools={level:StageWindows(trajectories,indices,history,chunk,training=training) for level,indices in ids.items()}
        self.focus=None;self.speed=[];focus=focus or {}
        base=self.pools[active]
        if training and contact_sampling:
            contact=[(i,t) for i,t in base.indices if int(trajectories[i]['stage'][t])==0 and float(trajectories[i]['obs'][t,20])<.13]
            deadline_contact=[(i,t) for i,t in contact if trajectories[i].get('source')=='recovery']
            if contact:base.recovery=deadline_contact or contact
        wanted=set(focus.get('parcel_ids',[]));phase=focus.get('stage',0)
        if training and wanted:
            rows=set()
            for i in ids[active]:
                item=trajectories[i]
                hits=torch.nonzero(torch.isin(item['target'],torch.tensor(sorted(wanted))) & (item['stage']==phase)).flatten().tolist()
                for t in hits:
                    rows.update((i,u) for u in range(max(0,t-history),min(len(item['actions']),t+chunk+1)))
            if rows:
                self.focus=StageWindows(trajectories,ids[active],history,chunk,training=training)
                self.focus.indices=sorted(rows);self.focus.recovery=[];self.focus.transitions=[]
        successful=[i for i in ids[active] if trajectories[i]['verified_success']]
        if training and active!='easy' and len(successful)>1:
            duration=lambda i:trajectories[i].get('completion_actions',len(trajectories[i]['actions']))
            median=sorted(duration(i) for i in successful)[len(successful)//2]
            # Duplicate only verified successes, only in the general active-task pool.
            for i,t in base.indices:
                if trajectories[i]['verified_success'] and duration(i)<median:
                    self.speed.append((i,t))
            if speed_bonus>0 and self.speed:
                base.indices+=self.speed[::max(1,round(1/min(1,speed_bonus)))]
        self.indices=[pair for pool in self.pools.values() for pair in pool.indices]

    def batch(self,size,generator,device,noise=0.):
        parts=[];prior=[k for k in self.pools if k!=self.active]
        replay=min(size-1,round(size*self.replay)) if prior and self.training else 0
        for j in range(replay):
            level=prior[int(torch.randint(len(prior),(1,),generator=generator))]
            parts.append(self.pools[level].batch(1,generator,device,noise))
        remaining=size-replay;crop=min(remaining-1,int(remaining*self.focus_fraction)) if self.focus is not None else 0
        if crop:parts.append(self.focus.batch(crop,generator,device,noise))
        parts.append(self.pools[self.active].batch(remaining-crop,generator,device,noise))
        return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}
