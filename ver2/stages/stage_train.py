"""Resumable stage ACT training. No simulator or renderer occupies training VRAM."""
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import torch
from stage_model import StageACT, stage_loss
from stage_data import load_data, split_data, normalization, StageWindows
from marso_experiment import save_json, digest
from github_store import sync_from_env


def atomic_save(path, value):
    path = Path(path)
    torch.save(value, str(path)+'.tmp')
    os.replace(str(path)+'.tmp', path)


def training_forward(model, batch, cfg):
    # The executed policy has no access to future expert actions. Train that same
    # zero-latent path directly; a low posterior loss did not train this path.
    mode = cfg.get('action_training_mode', 'prior')
    if mode == 'prior':
        return model(batch['obs'], batch['previous_stage'], batch['stage'])
    if mode == 'posterior':  # Explicit legacy reproduction only.
        return model(batch['obs'], batch['previous_stage'], batch['stage'], batch['actions'], batch['mask'])
    raise ValueError(f'Unknown action_training_mode: {mode}')


@torch.no_grad()
def validate(model, data, cfg, device):
    model.eval()
    generator = torch.Generator().manual_seed(cfg['seed']+991)
    losses = []
    for _ in range(cfg['validation_batches']):
        batch = data.batch(cfg['batch_size'], generator, device)
        outputs = model(batch['obs'], batch['previous_stage'], batch['stage'])
        loss, _ = stage_loss(outputs, batch['actions'], batch['mask'], batch['stage'], batch['gate'], cfg)
        losses.append(float(loss))
    model.train()
    return sum(losses)/len(losses)


def train(job):
    cfg, arch = job['train_config'], job['model_config']
    device = torch.device(job.get('device','cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('T4 GPU 런타임을 선택하세요.')
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    folder = Path(job['folder'])
    ckdir = folder/'checkpoints'
    ckdir.mkdir(parents=True, exist_ok=True)
    trajectories = load_data(job['data'], job.get('num_demos'), job.get('recovery_manifest'))
    if any(t['obs'].shape[-1] != arch['state_dim'] for t in trajectories):
        raise ValueError('Training difficulty and observation dimensions differ')
    train_ids, val_ids = split_data(trajectories, cfg['seed'])
    train_data = StageWindows(trajectories, train_ids, arch['history'], arch['chunk_size'],
                              first_pick_fraction=cfg.get('first_pick_fraction',0.))
    val_data = StageWindows(trajectories, val_ids, arch['history'], arch['chunk_size'], training=False)
    signature = dict(model_config=arch, train_config=cfg, data_sha256=digest(job['data']),
                     train_ids=train_ids, val_ids=val_ids, source_sha256=job['source_sha256'],
                     recovery_sha256=digest(job['recovery_manifest']) if job.get('recovery_manifest') else None,
                     warm_start_sha256=digest(job['warm_start']) if job.get('warm_start') else None)
    save_json(folder/'data_audit.json', dict(
        train_trajectories=[trajectories[i]['name'] for i in train_ids],
        validation_trajectories=[trajectories[i]['name'] for i in val_ids],
        train_windows=len(train_data.indices), validation_windows=len(val_data.indices),
        stage_counts=torch.bincount(torch.cat([t['stage'] for t in trajectories]), minlength=4).tolist(),
        gate_counts=torch.bincount(torch.cat([t['gate'] for t in trajectories]), minlength=3).tolist(),
        recovery_episodes=sum(t['source']=='recovery' for t in trajectories),
        labeling='base: weak geometric labels; collection: observed state + exact grasp flag',
        supervisor_memory_augmentation='25% alternate previous-stage inputs; action/observation targets unchanged',
        chunk_masking='stop at stage/parcel boundaries and first perturbed executed action',
        action_training_mode=cfg.get('action_training_mode','prior'),
        first_pick_fraction=cfg.get('first_pick_fraction',0.),
        raw_xyz_outside_action_bounds_fraction=sum(t['clipped_fraction'] for t in trajectories)/len(trajectories),
        soft_gripper_fraction=sum(t['soft_gripper_fraction'] for t in trajectories)/len(trajectories),
        gripper_labels='(clip(g,-1,1)+1)/2 soft BCE labels; latest inference logit chooses open/close',
        target_preprocessing='clip raw actions to environment bounds [-1,1]; mask padded future actions',
        normalization='training trajectories only; state plus parcel/TCP and target-bin relative positions'))
    model = StageACT(arch)
    mean, std = normalization(trajectories, train_ids)
    model.obs_mean.copy_(mean)
    model.obs_std.copy_(std)
    if job.get('warm_start'):
        initial = torch.load(job['warm_start'], map_location='cpu', weights_only=True)
        if initial.get('format') != 'moveboxes-stage-act-v1' or initial['model_config'] != arch:
            raise ValueError('Warm-start checkpoint architecture/format mismatch')
        model.load_state_dict(initial['model'])
        # Keep the saved normalization with the saved weights.
        print(f"보정 학습: 기존 {initial.get('step','?')} step 모델에서 시작, optimizer는 새로 초기화", flush=True)
    freeze_modules = tuple(cfg.get('freeze_modules', ()))
    if any(not isinstance(prefix, str) or not prefix for prefix in freeze_modules):
        raise ValueError('freeze_modules must contain non-empty module prefixes')
    if freeze_modules:
        names = [name for name,_ in model.named_parameters()]
        missing = [prefix for prefix in freeze_modules
                   if not any(name == prefix or name.startswith(prefix+'.') for name in names)]
        if missing:
            raise ValueError(f'Unknown freeze_modules prefixes: {missing}')
        for name, parameter in model.named_parameters():
            if any(name == prefix or name.startswith(prefix+'.') for prefix in freeze_modules):
                parameter.requires_grad_(False)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError('freeze_modules left no trainable parameters')
    optimizer = torch.optim.AdamW(trainable, lr=cfg['lr'], weight_decay=1e-4)
    amp = bool(cfg['amp'] and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    generator = torch.Generator().manual_seed(cfg['seed']+1)
    start, best = 0, float('inf')
    latest = ckdir/'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('Saved training configuration/data/code differs. Use a new run_name.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        generator.set_state(saved['sample_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start, best = saved['step'], saved['best_validation']
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    print(f"Stage ACT: {params/1e6:.2f}M parameters ({trainable_params/1e6:.2f}M trainable) / "
          f"{device} / AMP={amp} / resume {start}/{cfg['total_iters']}", flush=True)
    save_json(folder/'model_info.json', dict(parameters=params, trainable_parameters=trainable_params,
        frozen_modules=list(freeze_modules), model_config=arch, train_config=cfg))
    save_json(ckdir/'policy_config.json', job['policy_config'])
    save_json(folder/'train_status.json', dict(status='running', step=start))
    model.train()
    tick = last_print = time.monotonic()
    history = []
    stop = int(job.get('stop_at',cfg['total_iters']))
    if not start <= stop <= cfg['total_iters']:
        raise ValueError('stop_at must be between the saved step and total_iters')
    for step in range(start, stop):
        warmup = min(1., (step+1)/cfg['warmup_steps'])
        progress = max(0., (step-cfg['warmup_steps'])/max(1,cfg['total_iters']-cfg['warmup_steps']))
        lr = cfg['lr']*warmup*(.1+.9*(1+math.cos(math.pi*progress))/2)
        for group in optimizer.param_groups:
            group['lr'] = lr
        batch = train_data.batch(cfg['batch_size'], generator, device, cfg['position_noise'])
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            outputs = training_forward(model, batch, cfg)
            loss, parts = stage_loss(outputs, batch['actions'], batch['mask'], batch['stage'], batch['gate'], cfg)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss; last complete checkpoint is retained.')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not amp)
        scaler.step(optimizer)
        scaler.update()
        done = step+1
        if time.monotonic()-last_print >= cfg['console_interval_seconds'] or done == stop:
            rate = (done-start)/max(.001,time.monotonic()-tick)
            print(f"{done}/{cfg['total_iters']} loss={float(loss.detach()):.4f} "
                  f"{rate:.1f} it/s ETA={(cfg['total_iters']-done)/rate/60:.1f} min", flush=True)
            last_print = time.monotonic()
        if done % cfg['save_freq'] == 0 or done == stop:
            val_loss = validate(model, val_data, cfg, device)
            weights = dict(format='moveboxes-stage-act-v1', model_config=arch, model=model.state_dict(), step=done)
            if val_loss < best:
                best = val_loss
                atomic_save(ckdir/'best_val.pt', dict(weights, validation_loss=val_loss))
            atomic_save(latest, dict(weights, signature=signature, optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(), sample_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None, best_validation=best))
            history.append(dict(step=done, loss=float(loss.detach()), validation_loss=val_loss,
                                **{k:float(v) for k,v in parts.items()}))
            save_json(folder/'training_progress.json', dict(step=done, total=cfg['total_iters'],
                best_validation=best, session_history=history, resumed_from=start))
            save_json(folder/'train_status.json', dict(status='running', step=done))
            print(f'Checkpoint {done}: prior validation loss={val_loss:.4f}', flush=True)
            sync_from_env()
    if stop == cfg['total_iters']:
        save_json(folder/'training_complete.json', dict(total_iters=cfg['total_iters'], model_config=arch))
    save_json(folder/'train_status.json', dict(status='complete' if stop == cfg['total_iters'] else 'paused', step=stop))
    sync_from_env()


if __name__ == '__main__':
    train(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
