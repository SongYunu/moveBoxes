"""Stage targets, corrective data, disjoint episode splits, boundary-aware chunks."""
import json
from pathlib import Path
import numpy as np
import torch
from act_v2_data import load_trajectories, normalization
from marso_experiment import digest
from stage_labels import annotate
from stage_schema import PICK, CARRY, PLACE, DONE, HOLD, COMPLETE, RECOVER


def load_data(path, limit=None, recovery_manifest=None):
    trajectories = load_trajectories(path, limit)
    for traj in trajectories:
        traj.update({k:torch.from_numpy(v) for k,v in annotate(traj['obs'].numpy()).items()})
        traj.update(source='base', name='base/'+traj['name'], perturbed=torch.zeros(len(traj['actions']), dtype=torch.bool))
    if recovery_manifest:
        path = Path(recovery_manifest)
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if not manifest.get('complete') or len(manifest['episodes']) < 2:
            raise ValueError('Complete collection of at least two successful episodes required')
        for entry in manifest['episodes']:
            file = path.parent/entry['file']
            if file.resolve().parent != path.parent.resolve() or not entry['accepted'] or digest(file) != entry['sha256']:
                raise ValueError('Invalid recovery manifest entry')
            with np.load(file, allow_pickle=False) as arrays:
                item = {k:torch.from_numpy(np.array(arrays[k], copy=True)) for k in
                        ('obs','actions','previous_stage','stage','gate','target','perturbed')}
            length = len(item['actions'])
            if item['obs'].shape != (length+1, trajectories[0]['obs'].shape[1]) or item['actions'].shape != (length,4):
                raise ValueError('Recovery observation/action alignment mismatch')
            for key in ('previous_stage','stage','gate','target','perturbed'):
                if item[key].shape != (length,):
                    raise ValueError(f'Invalid recovery label shape: {key}')
            for key, bound in (('previous_stage',4),('stage',4),('gate',3),('target',(item['obs'].shape[1]-36)//9)):
                if torch.any(item[key] < 0) or torch.any(item[key] >= bound):
                    raise ValueError(f'Out-of-range recovery label: {key}')
            if not torch.isfinite(item['obs']).all() or not torch.isfinite(item['actions']).all():
                raise ValueError('Non-finite recovery data')
            item['actions'] = item['actions'].float().clamp(-1,1)
            item.update(source='recovery', name='recovery/'+entry['file'], clipped_fraction=0.,
                        soft_gripper_fraction=0.)
            trajectories.append(item)
    return trajectories


def split_data(trajectories, seed):
    train, valid = [], []
    generator = torch.Generator().manual_seed(seed)
    # Each source contributes its own held-out episodes; no adjacent-window leakage.
    for source in ('base','recovery'):
        ids = [i for i,t in enumerate(trajectories) if t['source'] == source]
        if not ids:
            continue
        if len(ids) < 2:
            raise ValueError('Each data source needs at least two disjoint episodes')
        order = torch.randperm(len(ids), generator=generator).tolist()
        count = max(1, round(len(ids)*.1))
        valid.extend(ids[i] for i in order[:count])
        train.extend(ids[i] for i in order[count:])
    return train, valid


class StageWindows:
    def __init__(self, trajectories, ids, history, chunk, training=True):
        self.trajectories, self.history, self.chunk, self.training = trajectories, history, chunk, training
        self.indices = [(i,t) for i in ids for t in range(len(trajectories[i]['actions']))]
        self.recovery = [(i,t) for i,t in self.indices if trajectories[i]['source'] == 'recovery']
        self.transitions = [(i,t) for i,t in self.indices if int(trajectories[i]['gate'][t]) != HOLD]

    def batch(self, size, generator, device, noise=0.):
        rows = {k:[] for k in ('obs','actions','mask','previous_stage','stage','gate')}
        for _ in range(size):
            p = float(torch.rand((), generator=generator)) if self.training else 1.
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
        rows = {k:torch.stack(v) for k,v in rows.items()}
        if noise:
            n = (rows['obs'].shape[-1]-36)//9
            columns = [18,19,20]+[26+7*i+j for i in range(n) for j in range(3)]
            rows['obs'][...,columns] += torch.randn((size,1,len(columns)), generator=generator)*noise
        return {k:v.to(device) for k,v in rows.items()}
