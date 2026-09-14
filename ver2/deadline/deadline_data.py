"""Train only on deadline-verified, state-aligned demonstrations; emphasize all picks."""
import json
from pathlib import Path
import numpy as np
import torch
from stage_data import StageWindows as OriginalWindows, split_data
from act_v2_data import normalization
from marso_experiment import digest
from stage_schema import PICK


def load_data(path, limit=None, recovery_manifest=None):
    path=Path(path)
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('version')!='deadline-v2' or not manifest.get('complete') or len(manifest['episodes'])<4:
        raise ValueError('At least four complete deadline-verified demonstrations are required')
    result=[]
    for entry in manifest['episodes']:
        file=path.parent/entry['file']
        if file.resolve().parent!=path.parent.resolve() or digest(file)!=entry['sha256']:
            raise ValueError('Invalid demonstration path/digest')
        if not entry['accepted'] or entry['correct']!=4 or entry['evaluation_actions']!=199:
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
        item.update(name='recovery/'+entry['file'],source='recovery',clipped_fraction=0.,soft_gripper_fraction=0.)
        result.append(item)
    return result


class StageWindows(OriginalWindows):
    def __init__(self, trajectories, ids, history, chunk, training=True, first_pick_fraction=0.):
        super().__init__(trajectories,ids,history,chunk,training,first_pick_fraction=0.)
        # The inherited sampler reserves 35% for this pool, 25% for transitions,
        # and 40% for all frames. Include contact on EVERY parcel, not only #1.
        self.contact=[(i,t) for i,t in self.indices if int(trajectories[i]['stage'][t])==PICK
                      and float(trajectories[i]['obs'][t,20])<.13]
        self.recovery=self.contact or self.indices
