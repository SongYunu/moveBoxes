"""Official RGB metric in simulator Python; inference uses an isolated process."""
import json,os,random,sys,time
from pathlib import Path
from multiprocessing.connection import Client
import numpy as np
import torch
from foundation_common import read,write,wire_array


class RemotePolicy:
    def __init__(self,connection):self.connection=connection;self.frames=[]
    def reset(self,seed):
        self.frames=[];self.connection.send(dict(mode='reset',seed=seed))
        if not self.connection.poll(600):raise TimeoutError('Policy reset timed out')
        self.connection.recv()
    def act(self,obs,deterministic=True):
        image=obs['rgb'][0].detach().cpu().numpy()
        state=obs['state'][0].detach().cpu().numpy().astype(np.float32)
        if image.shape!=(128,128,3) or state.shape!=(26,):raise ValueError(f'Unexpected RGB input: {image.shape}/{state.shape}')
        if getattr(self,'video',False):self.frames.append(image.copy())
        self.connection.send(dict(mode='act',image=wire_array(image),state=wire_array(state)))
        if not self.connection.poll(600):raise TimeoutError('Model inference timed out; inspect policy log')
        action=np.array(self.connection.recv()['action'],np.float32)
        if action.shape!=(4,) or not np.isfinite(action).all() or (np.abs(action)>1).any():raise ValueError('Invalid model action')
        return torch.tensor(action[None],device='cuda')


def main(job):
    from warehouse_sort.utils import compose_cfg,make_env,rollout_metrics
    ready=read(job['ready']);auth=bytes.fromhex(os.environ['MOVEBOXES_IPC_KEY'])
    with Client((ready['host'],ready['port']),authkey=auth) as connection:
        policy=RemotePolicy(connection)
        cfg=compose_cfg(['difficulty='+job['level'],'obs_mode=rgb','camera.width=128','camera.height=128',
            'max_episode_steps='+str(job['max_steps'])])
        env,_=make_env(cfg,'rgb',cfg.randomization,num_envs=1)
        try:
            previous=read(job['output'],{})
            rows=previous.get('episodes',[]) if previous.get('fingerprint')==job['fingerprint'] else []
            if [r['seed'] for r in rows]!=job['seeds'][:len(rows)]:raise ValueError('Saved evaluation seed mismatch')
            tick=time.monotonic()
            for seed in job['seeds'][len(rows):]:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);policy.reset(seed)
                policy.video=bool(job.get('video') and seed==job['seeds'][0])
                metrics=rollout_metrics(env,policy,torch.device('cuda'),1,[seed],job['max_steps'])
                rows.append(dict(seed=seed,**metrics))
                write(job['output'],dict(complete=len(rows)==len(job['seeds']),fingerprint=job['fingerprint'],episodes=rows,
                    sort_accuracy=sum(r['sort_accuracy'] for r in rows)/len(rows),checkpoint=job['checkpoint'],
                    track='rgb',elapsed_seconds=time.monotonic()-tick))
                if policy.video:
                    import imageio.v2 as imageio
                    imageio.mimsave(str(Path(job['output']).with_suffix('.mp4')),policy.frames,fps=20,macro_block_size=1)
                print(f'{job["level"]}: {len(rows)}/{len(job["seeds"])} sorted={metrics["mean_sorted"]} score={metrics["sort_accuracy"]:.1%}',flush=True)
        finally:env.close();connection.send(dict(mode='close'))


if __name__=='__main__':main(read(sys.argv[1]))
