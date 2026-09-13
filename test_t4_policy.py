"""Small CPU tensor check, also embedded in Colab before expensive training."""
from types import SimpleNamespace
import torch
from colab_policy import ChunkPolicy

calls = []
class Net:
    def __call__(self, sample, timestep, global_cond):
        calls.append(global_cond.clone())
        return torch.zeros_like(sample)
class Scheduler:
    timesteps = [0]
    def step(self, model_output, timestep, sample):
        seq = torch.arange(6, dtype=torch.float32).view(1,6,1).expand(sample.shape)/10
        return SimpleNamespace(prev_sample=seq)
base = SimpleNamespace(obs_horizon=2, pred_horizon=6, act_dim=4,
                       device='cpu', net=Net(), scheduler=Scheduler())
policy = ChunkPolicy(base, 3)
for t, expected in enumerate([0.1,0.2,0.3,0.1]):
    action = policy.act(torch.full((1,2), float(t)))
    assert torch.allclose(action, torch.full((1,4), expected))
assert len(calls)==2, 'Chunk must be generated only when its buffer is empty'
assert torch.equal(calls[-1], torch.tensor([[2.,2.,3.,3.]])), 'History must update during buffered execution'
policy.reset()
policy.act(torch.full((1,2), 9.))
assert torch.equal(calls[-1], torch.full((1,4), 9.)), 'Reset must clear old history and actions'
policy.act(torch.zeros((2,2)))
assert calls[-1].shape==(2,4), 'Batch size change must clear state'
print('Chunk execution, history, reset and batch-size checks passed')
