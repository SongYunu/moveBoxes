"""Small tensor and checkpoint contract check, also run by Colab cell 05."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from act_v2_model import StateACT, action_loss
from act_v2_policy import load_act


def run_checks():
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with tempfile.TemporaryDirectory() as directory:
        for dim in (54,72,90):
            cfg = dict(state_dim=dim, history=4, chunk_size=16, width=32, heads=4, layers=1, latent_dim=8)
            model = StateACT(cfg).to(device)
            obs = torch.zeros(2,4,dim,device=device)
            target = torch.zeros(2,16,4,device=device)
            target[...,3] = 1
            mask = torch.ones(2,16,device=device)
            prediction,kl = model(obs,target,mask)
            loss,_ = action_loss(prediction,target,mask,kl)
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            path = Path(directory)/f'{dim}.pt'
            torch.save(dict(format='moveboxes-act-ver2',model_config=cfg,model=model.cpu().state_dict()),path)
            policy = load_act(path,obs[:1,-1],SimpleNamespace(shape=(4,)),device)
            action = policy.act(obs[:1,-1])
            assert action.shape==(1,4) and action.abs().max() <= 1
            assert action[0,3].item() in (-1.,1.)
            policy.act(obs[:1,-1]+.01)
            policy.reset()
            assert torch.allclose(policy.act(obs[:1,-1]),action,atol=1e-6)
            policy.reset()
            assert not policy.history and not policy.predictions
            print(f'ACT ver2 {dim}: loss/backward/load/action/reset OK ({device})',flush=True)


if __name__ == '__main__':
    run_checks()
