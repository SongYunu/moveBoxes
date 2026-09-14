"""Pinned SmolVLA checkpoint, frozen VLM, FP32 master weights and FP16 AMP on T4."""
import os,time
from pathlib import Path
import numpy as np
import torch
from foundation_common import *


def load(cfg,checkpoint=None):
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import PolicyFeature,FeatureType
    from safetensors.torch import load_model
    base=snapshot_download('lerobot/smolvla_base',revision=SMOL_REV,allow_patterns=['*.json','*.safetensors'])
    vlm=snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct',revision=VLM_REV,
        allow_patterns=['*.json','*.txt','*.model','tokenizer*'])
    config=SmolVLAConfig.from_pretrained(base)
    # Full SmolVLA weights are restored below. Avoid a second BF16 VLM load on T4.
    config.load_vlm_weights=False;config.vlm_model_name=vlm;config.device='cpu'
    config.freeze_vision_encoder=True;config.train_expert_only=True;config.train_state_proj=True
    config.chunk_size=cfg['chunk'];config.n_action_steps=cfg['execute_steps'];config.num_steps=cfg['inference_steps']
    config.resize_imgs_with_padding=(cfg['image_size'],cfg['image_size'])
    config.input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(26,)),
        'observation.images.camera1':PolicyFeature(type=FeatureType.VISUAL,shape=(3,128,128))}
    config.output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(4,))}
    model=SmolVLAPolicy(config)
    load_model(model,str(Path(base)/'model.safetensors'),strict=True)
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    if checkpoint:
        saved=torch.load(Path(checkpoint)/'adapter.pt',map_location='cpu',weights_only=True)
        if set(saved)!=trainable:raise ValueError('Saved SmolVLA adapter architecture differs')
        missing,unexpected=model.load_state_dict(saved,strict=False)
        if unexpected or trainable.intersection(missing):raise ValueError('Incomplete SmolVLA adapter')
    # Preserve FP32 optimizer/master weights; autocast controls matmuls.
    return model.float().to(cfg.get('device','cuda'))


def tokens(model,size,device):
    tokenizer=model.model.vlm_with_expert.processor.tokenizer
    encoded=tokenizer([INSTRUCTION]*size,padding='max_length',truncation=True,
        max_length=model.config.tokenizer_max_length,return_tensors='pt')
    return encoded['input_ids'].to(device),encoded['attention_mask'].bool().to(device)


def batch_to_torch(batch,stats,device,language):
    ids,attention=language
    actions=normalize(batch['actions'][:,-1],stats,'action')
    actions[~batch['action_mask'][:,-1]]=0.
    return {'observation.images.camera1':torch.from_numpy(batch['images'][:,-1].copy()).to(device).permute(0,3,1,2).float()/255.,
        'observation.state':torch.from_numpy(normalize(batch['proprio'][:,-1],stats,'proprio')).to(device),
        'action':torch.from_numpy(actions).to(device),
        'valid':torch.from_numpy(batch['action_mask'][:,-1]).to(device),
        'observation.language.tokens':ids,'observation.language.attention_mask':attention}


def masked_loss(model,batch):
    images,masks=model.prepare_images(batch)
    # Compute per-action losses directly: this version's policy.forward averages
    # 32 padded action dimensions; the environment has only four real dimensions.
    losses=model.model(images,masks,batch['observation.language.tokens'],batch['observation.language.attention_mask'],
        model.prepare_state(batch),model.prepare_action(batch))[:,:,:4]
    valid=batch['valid'][...,None]
    return (losses.float()*valid).sum()/(valid.sum().clamp_min(1)*4)


def training_mode(model):
    """Keep the frozen VLM deterministic while the action expert is trained."""
    model.train()
    wrapped=model.model.vlm_with_expert
    if hasattr(wrapped,'vlm'):wrapped.vlm.eval()


def train(job):
    cfg=job['cfg'];torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    device=cfg.get('device','cuda')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('SmolVLA CUDA GPU is unavailable')
    data=RGBData(job['data'],cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    checkpoint=job.get('checkpoint');meta=read(Path(checkpoint)/'metadata.json') if checkpoint else None
    if meta and meta['signature']!=job['signature']:raise ValueError('Training config/data/code changed; use a new run_name')
    stats=meta['stats'] if meta else data.stats
    model=load(cfg,checkpoint);params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=1e-4)
    amp=device=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=amp)
    start=meta['step'] if meta else 0
    if meta:rng.bit_generator.state=meta['sample_rng']
    resume=Path(checkpoint)/'resume.pt' if checkpoint else None
    if resume and resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=True)
        optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);torch.set_rng_state(state['torch_rng'])
        if amp and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    language=tokens(model,cfg['micro_batch'],device);tick=last=time.monotonic()
    probe_name,probe_parameter=next((n,p) for n,p in model.named_parameters() if p.requires_grad and n.endswith('action_out_proj.weight'))
    probe_before=probe_parameter.detach().cpu().clone()
    if amp:torch.cuda.reset_peak_memory_stats()
    print(f'SmolVLA pretrained: total={sum(p.numel() for p in model.parameters())/1e6:.1f}M trainable={sum(p.numel() for p in params)/1e6:.1f}M / {start}->{job["stop"]}',flush=True)
    for step in range(start,job['stop']):
        training_mode(model);optimizer.zero_grad(set_to_none=True);loss_value=0.
        for micro in range(cfg['accumulate']):
            levels=[LEVELS[(micro*cfg['micro_batch']+j)%3] for j in range(cfg['micro_batch'])]
            batch=batch_to_torch(data.batch(levels,rng,1,cfg['chunk']),stats,device,language)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):loss=masked_loss(model,batch)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite SmolVLA loss; previous checkpoint retained')
            scaler.scale(loss/cfg['accumulate']).backward();loss_value+=float(loss.detach())/cfg['accumulate']
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update()
        if time.monotonic()-last>=30 or step+1==job['stop']:
            print(f'{step+1}/{cfg["updates"]} loss={loss_value:.4f} {(step+1-start)/(time.monotonic()-tick):.2f} update/s',flush=True);last=time.monotonic()
    model.eval();valid_rng=np.random.default_rng(cfg['seed']+1000)
    parameter_delta=float((probe_parameter.detach().cpu()-probe_before).float().norm())
    if job['stop']>start and not parameter_delta>0:raise RuntimeError(f'{probe_name} did not update; checkpoint was not saved')
    validation=[]
    with torch.no_grad():
        for level in LEVELS:
            b=batch_to_torch(data.batch([level]*cfg['micro_batch'],valid_rng,1,cfg['chunk'],'valid'),stats,device,language)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):validation.append(float(masked_loss(model,b)))
    dest=Path(job['destination']);dest.mkdir(parents=True,exist_ok=True)
    torch.save({n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},dest/'adapter.pt.tmp')
    os.replace(dest/'adapter.pt.tmp',dest/'adapter.pt')
    torch.save(dict(optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if amp else None),dest/'resume.pt.tmp');os.replace(dest/'resume.pt.tmp',dest/'resume.pt')
    checkpoint_meta(dest,job,stats,job['stop'],rng)
    write(dest/'resource.json',dict(seconds=time.monotonic()-tick,updates=job['stop']-start,
        validation_by_level=dict(zip(LEVELS,validation)),parameter_delta_norm=parameter_delta,
        max_gpu_gib=torch.cuda.max_memory_allocated()/2**30 if amp else None))
    data.close()


class Policy:
    def __init__(self,cfg,checkpoint):
        self.cfg=cfg;self.model=load(cfg,checkpoint).eval();self.stats=read(Path(checkpoint)/'metadata.json')['stats']
        self.device=next(self.model.parameters()).device;self.language=tokens(self.model,1,self.device)

    def reset(self,seed):
        self.model.reset();torch.manual_seed(seed)

    def act(self,image,state):
        dummy=dict(images=image[None,None],proprio=state[None,None],
            actions=np.zeros((1,1,self.cfg['chunk'],4),np.float32),action_mask=np.ones((1,1,self.cfg['chunk']),bool))
        batch=batch_to_torch(dummy,self.stats,self.device,self.language)
        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):
            action=self.model.select_action(batch)[0].float().cpu().numpy()
        return action_to_env(action,self.stats)
