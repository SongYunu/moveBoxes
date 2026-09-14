"""Default T4 batch backward, single-checkpoint loading for 2/4/6 objects, reset."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from object_model import ObjectACT, FORMAT, canonical_state, stage_loss
from object_policy import load_stage


def sample_state(dimension, batch=2, device='cpu'):
    n=(dimension-36)//9
    obs=torch.zeros(batch,dimension,device=device)
    obs[:,18:21]=obs.new_tensor([0.,0.,.17])
    for i in range(n):
        obs[:,26+7*i:29+7*i]=obs.new_tensor([-.05,.04*i,.03])
        obs[:,29+7*i]=1
        obs[:,26+7*n+2*i+i%2]=1
    obs[:,26+9*n:32+9*n]=obs.new_tensor([0.,-.36,0.,0.,.36,0.])
    obs[:,-4:]=obs.new_tensor([1.,0.,0.,1.])
    return obs


def run_checks():
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg=dict(architecture='object-attention-v1',state_dim=90,history=8,chunk_size=8,width=128,heads=4,layers=2,spatial_layers=1)
    model=ObjectACT(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    obs=sample_state(90,32,device)[:,None].expand(-1,8,-1)
    actions=torch.zeros(32,8,4,device=device);actions[...,3]=1
    mask=torch.ones(32,8,device=device)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type,enabled=device.type=='cuda'):
            loss,_=stage_loss(model(obs),actions,mask,None,None,{})
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        scaler.step(optimizer);scaler.update()
    params=sum(p.numel() for p in model.parameters())
    print(f'Object attention: {params/1e6:.2f}M parameters / batch 32 / backward OK ({device})')
    if device.type=='cuda':print(f'테스트 GPU 최대 할당: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'model.pt'
        torch.save(dict(format=FORMAT,model_config=cfg,model=model.cpu().state_dict(),step=3),path)
        for dim in (54,72,90):
            raw=sample_state(dim,1,device)
            policy=load_stage(path,raw,SimpleNamespace(shape=(4,)),device)
            action=policy.act(raw)
            assert action.shape==(1,4) and torch.isfinite(action).all() and action.abs().max()<=1
            policy.act(raw+.001);policy.reset()
            torch.testing.assert_close(action,policy.act(raw))
            print(f'{dim}차원 입력: 동일 모델 로딩 / 행동 / reset OK')
    print('연산·로딩 확인 완료. 이 검사는 실제 집기 성공률 평가가 아닙니다.')


if __name__=='__main__':run_checks()
