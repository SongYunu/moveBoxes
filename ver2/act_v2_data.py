"""CPU-resident trajectories, trajectory-level split and masked future targets."""
import h5py
import numpy as np
import torch
from act_v2_model import state_features


def load_trajectories(path, limit=None):
    trajectories = []
    with h5py.File(path, 'r') as handle:
        keys = sorted((k for k in handle if k.startswith('traj_')), key=lambda k:int(k.split('_')[-1]))
        for name in keys[:limit]:
            obs = np.asarray(handle[name]['obs'], dtype=np.float32)
            actions = np.asarray(handle[name]['actions'], dtype=np.float32)
            if len(obs) != len(actions)+1 or obs.shape[1] not in (54,72,90) or actions.shape[1] != 4:
                raise ValueError(f'Invalid state/action alignment: {name}')
            if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                raise ValueError(f'Non-finite demonstration: {name}')
            trajectories.append(dict(name=name, obs=torch.from_numpy(obs),
                actions=torch.from_numpy(actions.clip(-1,1)),
                soft_gripper_fraction=float(np.mean(np.abs(actions[:, 3])<1)),
                clipped_fraction=float(np.mean(np.abs(actions[:, :3])>1))))
    if len(trajectories) < 2:
        raise ValueError('At least two trajectories required for a disjoint validation split')
    return trajectories


def split_trajectories(trajectories, seed):
    order = torch.randperm(len(trajectories), generator=torch.Generator().manual_seed(seed)).tolist()
    nval = max(1, round(len(order)*.1))
    return order[nval:], order[:nval]


def normalization(trajectories, ids):
    # Statistics are fitted on training trajectories only.
    values = torch.cat([state_features(trajectories[i]['obs'][:-1]) for i in ids])
    return values.mean(0), values.std(0, unbiased=False).clamp_min(.05)


class WindowData:
    def __init__(self, trajectories, ids, history, chunk):
        self.trajectories, self.history, self.chunk = trajectories, history, chunk
        self.indices = [(i,t) for i in ids for t in range(len(trajectories[i]['actions']))]

    def batch(self, batch_size, generator, device, noise=0.):
        chosen = torch.randint(len(self.indices), (batch_size,), generator=generator).tolist()
        obs, actions, masks = [], [], []
        for index in chosen:
            i,t = self.indices[index]
            traj = self.trajectories[i]
            obs.append(traj['obs'][torch.arange(t-self.history+1, t+1).clamp_min(0)])
            valid = min(self.chunk, len(traj['actions'])-t)
            a = torch.zeros(self.chunk,4)
            a[:valid] = traj['actions'][t:t+valid]
            actions.append(a)
            masks.append(torch.arange(self.chunk)<valid)
        obs = torch.stack(obs)
        if noise:
            # Small observation perturbations only; never corrupt tags/grasp bits.
            n = (obs.shape[-1]-36)//9
            columns = [18,19,20]+[26+7*i+j for i in range(n) for j in range(3)]
            jitter = torch.randn((batch_size,1,len(columns)), generator=generator)*noise
            obs[...,columns] += jitter
        return (obs.to(device), torch.stack(actions).to(device), torch.stack(masks).float().to(device))
