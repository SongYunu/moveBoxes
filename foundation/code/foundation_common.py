"""Framework-independent data, checkpoint metadata and local process protocol."""
import hashlib,json,os
from pathlib import Path
import numpy as np

LEVELS=('easy','medium','hard')
INSTRUCTION='Pick up every parcel and place it in the bin matching its colored label.'
SMOL_REV='c83c3163b8ca9b7e67c509fffd9121e66cb96205'
VLM_REV='7b375e1b73b11138ff12fe22c8f2822d8fe03467'
OCTO_REV='dc9aa3019f764726c770814b27e4ab0fc6e32a58'
OCTO_CODE='241fb3514b7c40957a86d869fecb7c7fc353f540'
T5_REV='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'


def read(path,default=None):
    path=Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def proprio(group):
    return np.concatenate((group['agent/qpos'][:],group['agent/qvel'][:],
        group['extra/tcp_pose'][:],group['extra/is_grasped'][:].reshape(-1,1)),axis=-1).astype(np.float32)


class RGBData:
    """Keep only small state/action arrays in RAM; fetch RGB frames from H5 on demand."""
    def __init__(self,root,seed=42):
        import h5py
        self.handles=[];self.items=[];self.ids={k:dict(train=[],valid=[]) for k in LEVELS};self.hashes={}
        for level in LEVELS:
            path=Path(root)/level/'trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5'
            self.hashes[level]=digest(path)
            handle=h5py.File(path,'r');self.handles.append(handle)
            names=sorted((k for k in handle if k.startswith('traj_')),key=lambda x:int(x.split('_')[-1]))
            if len(names)<2:raise ValueError(f'{level}: at least two episodes required')
            order=np.random.default_rng(seed).permutation(len(names));held=set(order[:max(1,round(.1*len(names)))])
            for j,name in enumerate(names):
                g=handle[name];obs=g['obs'];images=obs['sensor_data/scene_camera/rgb']
                p=proprio(obs);a=np.clip(g['actions'][:].astype(np.float32),-1,1)
                if p.shape!=(len(a)+1,26) or images.shape!=(len(a)+1,128,128,3) or a.shape[1]!=4:
                    raise ValueError(f'{level}/{name}: unexpected camera or state/action alignment')
                if not np.isfinite(p).all() or not np.isfinite(a).all():raise ValueError('Non-finite demonstrations')
                self.ids[level]['valid' if j in held else 'train'].append(len(self.items))
                self.items.append(dict(name=f'{level}/{name}',proprio=p,actions=a,images=images))
        self.pools={split:{k:[(i,t) for i in self.ids[k][split] for t in range(len(self.items[i]['actions']))]
            for k in LEVELS} for split in ('train','valid')}
        all_p=np.concatenate([self.items[i]['proprio'][:-1] for k in LEVELS for i in self.ids[k]['train']])
        all_a=np.concatenate([self.items[i]['actions'] for k in LEVELS for i in self.ids[k]['train']])
        self.stats=dict(proprio_mean=all_p.mean(0).tolist(),proprio_std=np.maximum(all_p.std(0),.01).tolist(),
            action_mean=all_a.mean(0).tolist(),action_std=np.maximum(all_a.std(0),.05).tolist())
        self.stats['action_mean'][3]=0.;self.stats['action_std'][3]=1.

    def close(self):
        for f in self.handles:f.close()

    def audit(self):
        return dict(hashes=self.hashes,stats=self.stats,episodes={k:{s:[self.items[i]['name'] for i in ids] for s,ids in splits.items()}
            for k,splits in self.ids.items()},track='rgb',camera='scene_camera',resolution=[128,128],proprio_dim=26,action_dim=4)

    def batch(self,levels,rng,history,chunk,split='train'):
        rows=[]
        for level in levels:
            pool=self.pools[split][level];i,t=pool[int(rng.integers(len(pool)))];item=self.items[i]
            times=np.arange(t-history+1,t+1);indices=times.clip(0)
            images=np.stack([item['images'][int(u)] for u in indices])
            actions=np.zeros((history,chunk,4),np.float32);mask=np.zeros((history,chunk),bool)
            for h,u in enumerate(indices):
                n=min(chunk,len(item['actions'])-u);actions[h,:n]=item['actions'][u:u+n];mask[h,:n]=True
            rows.append(dict(images=images,proprio=item['proprio'][indices],actions=actions,
                action_mask=mask,time_mask=times>=0))
        return {k:np.stack([r[k] for r in rows]) for k in rows[0]}


def normalize(value,stats,prefix):
    return ((value-np.array(stats[prefix+'_mean'],np.float32))/np.array(stats[prefix+'_std'],np.float32)).astype(np.float32)


def action_to_env(value,stats):
    value=np.asarray(value,np.float32)*np.array(stats['action_std'],np.float32)+np.array(stats['action_mean'],np.float32)
    if not np.isfinite(value).all():raise ValueError('Non-finite policy action')
    value=np.clip(value,-1,1);value[...,3]=np.where(value[...,3]>=0,1.,-1.)
    return value


def checkpoint_meta(folder,job,stats,step,rng):
    write(Path(folder)/'metadata.json',dict(backend=job['cfg']['backend'],step=step,stats=stats,cfg=job['cfg'],
        signature=job['signature'],sample_rng=rng.bit_generator.state))


def wire_array(array):
    array=np.ascontiguousarray(array)
    return dict(dtype=str(array.dtype),shape=array.shape,data=array.tobytes())


def unwire_array(value):
    return np.frombuffer(value['data'],dtype=value['dtype']).reshape(value['shape']).copy()
