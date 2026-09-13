"""All three model shapes, backward, checkpoint loading and reset; no simulator."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from stage_model import StageACT, stage_loss
from stage_policy import load_stage


def run_checks():
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with tempfile.TemporaryDirectory() as directory:
        for dim in (54,72,90):
            cfg = dict(state_dim=dim, history=4, chunk_size=16, width=32, heads=4, layers=1, latent_dim=8)
            model = StageACT(cfg).to(device)
            obs = torch.zeros(2,4,dim,device=device)
            actions = torch.zeros(2,16,4,device=device)
            actions[...,3] = 1
            mask = torch.ones(2,16,device=device)
            previous = torch.zeros(2,dtype=torch.long,device=device)
            stage = previous+1
            loss,_ = stage_loss(model(obs,previous,stage,actions,mask), actions,mask,stage,stage,
                                dict(kl_weight=.001,stage_loss_weight=.3,gate_loss_weight=.3))
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            path = Path(directory)/f'{dim}.pt'
            torch.save(dict(format='moveboxes-stage-act-v1',model_config=cfg,model=model.cpu().state_dict()),path)
            policy = load_stage(path,obs[:1,-1],SimpleNamespace(shape=(4,)),device)
            action = policy.act(obs[:1,-1])
            assert action.shape == (1,4) and action.abs().max() <= 1 and action[0,3].item() in (-1.,1.)
            policy.act(obs[:1,-1]+.01)
            policy.reset()
            torch.testing.assert_close(policy.act(obs[:1,-1]),action)
            policy.reset()
            assert not policy.history and not policy.predictions
            print(f'Stage ACT {dim}: backward/load/action/reset OK ({device})',flush=True)


if __name__ == '__main__':
    run_checks()
