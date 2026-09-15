"""Orchestrate conservative DAgger rounds from checksum-verified Stage ACT anchors."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

from marso_experiment import digest, read_json, save_json
from stage_anchor_continue import ANCHORS


FROZEN_MODULES = ('obs_proj', 'history_pos', 'encoder', 'posterior', 'latent_proj')


def _policy_config(saved):
    return dict(model_config=saved['model_config'], temporal_decay=.25, ensemble_window=4,
        gate_threshold=.65, stage_threshold=.6, act_horizon=1,
        num_inference_steps=1, auto_reset_steps=199)


def prepare_dagger(base, level='medium', *, run_suffix='dagger_v1', rounds=4,
                   episodes_per_round=24, train_iters=2000, lr=1e-5,
                   betas=(.5, .3, .15, 0.)):
    if level not in ('medium', 'hard') or not run_suffix:
        raise ValueError('DAgger is configured for Medium/Hard anchors')
    if (type(rounds) is not int or rounds < 1 or type(episodes_per_round) is not int or
            episodes_per_round < 2 or type(train_iters) is not int or train_iters < 1 or
            not 0 < lr <= 5e-5 or len(betas) < rounds or
            any(not 0 <= beta <= 1 for beta in betas[:rounds])):
        raise ValueError('Invalid DAgger configuration')
    root = base.run_dir/level/run_suffix
    root.mkdir(parents=True, exist_ok=True)
    anchor = base.run_dir/level/'anchor.pt'
    if not anchor.is_file() or digest(anchor) != ANCHORS[level]['sha256']:
        raise ValueError('Verified anchor is missing or changed')
    identity = dict(kind='stage-dagger-v1', level=level,
        anchor_sha256=ANCHORS[level]['sha256'], rounds=rounds,
        episodes_per_round=episodes_per_round, train_iters=train_iters,
        lr=lr, betas=list(betas[:rounds]), frozen_modules=list(FROZEN_MODULES),
        collector_sha256=hashlib.sha256(base.sources['stage_dagger_collect.py'].encode()).hexdigest(),
        trainer_sha256=hashlib.sha256(base.sources['stage_train.py'].encode()).hexdigest())
    identity['signature'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = root/'dagger_job.json'
    previous = read_json(path)
    if previous and previous.get('signature') != identity['signature']:
        raise ValueError('Saved DAgger method differs; change run_suffix')
    save_json(path, identity)
    save_json(root/'method.json', dict(
        objective='expert actions on states actually visited by the learned policy',
        reinforcement_learning=False, scripted_controller_in_submission=False,
        failed_rollouts_retained=True, clean_demonstration_rehearsal=True,
        frozen_modules=list(FROZEN_MODULES), anchor_sha256=ANCHORS[level]['sha256']))
    return root


def _copy_entry(source_folder, entry, target_folder, name):
    source = source_folder/entry['file']
    if source.resolve().parent != source_folder.resolve() or digest(source) != entry['sha256']:
        raise ValueError(f'Invalid DAgger source entry: {source}')
    target = target_folder/name
    if target.exists() and digest(target) != entry['sha256']:
        raise ValueError(f'Existing aggregate data differs: {target}')
    if not target.exists():
        shutil.copy2(source, target)
    return dict(file=name, sha256=entry['sha256'], accepted=True,
                original_success=bool(entry.get('success', entry.get('accepted', False))))


def materialize_training_manifest(base, level, dagger_root, through_round):
    target = dagger_root/f'round_{through_round:02d}'/'training_data'
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    original_folder = base.run_dir/level/'collection'
    original = read_json(original_folder/'manifest.json')
    if not original.get('complete'):
        raise ValueError('Historical successful recovery collection is incomplete')
    for index, entry in enumerate(original['episodes']):
        entries.append(_copy_entry(original_folder, entry, target,
            f'clean_{index:03d}_{Path(entry["file"]).name}'))
    for number in range(1, through_round+1):
        folder = dagger_root/f'round_{number:02d}'/'collection'
        manifest = read_json(folder/'manifest.json')
        if manifest.get('kind') != 'dagger-v1' or not manifest.get('complete'):
            raise ValueError(f'DAgger round {number} collection is incomplete')
        for index, entry in enumerate(manifest['episodes']):
            if not entry.get('valid_for_training'):
                raise ValueError('Invalid episode cannot enter DAgger training')
            entries.append(_copy_entry(folder, entry, target,
                f'dagger_r{number:02d}_{index:03d}_{Path(entry["file"]).name}'))
    manifest = dict(kind='dagger-training-v1', complete=True, episodes=entries,
        clean_episodes=len(original['episodes']), dagger_rounds=through_round,
        dagger_episodes=len(entries)-len(original['episodes']))
    save_json(target/'manifest.json', manifest)
    return target/'manifest.json'


def run_dagger_round(base, level, dagger_root, round_number, start_checkpoint):
    dagger_root, start_checkpoint = Path(dagger_root), Path(start_checkpoint)
    config = read_json(dagger_root/'dagger_job.json')
    if not 1 <= round_number <= config['rounds']:
        raise ValueError('round_number outside configured DAgger budget')
    saved = torch.load(start_checkpoint, map_location='cpu', weights_only=True)
    if saved.get('format') != 'moveboxes-stage-act-v1':
        raise ValueError('DAgger requires a Stage ACT checkpoint')
    round_dir = dagger_root/f'round_{round_number:02d}'
    collection = round_dir/'collection'
    collection_job = dict(kind='dagger-v1', level=level,
        checkpoint=str(start_checkpoint), checkpoint_sha256=digest(start_checkpoint),
        policy_config=_policy_config(saved), folder=str(collection),
        collection_config=dict(episodes=config['episodes_per_round'], max_steps=200,
            seed_start=300000+round_number*1000+(0 if level == 'medium' else 100000),
            beta=config['betas'][round_number-1], device='cuda'),
        source_sha256=config['collector_sha256'])
    previous = read_json(round_dir/'collection_job.json')
    if previous and previous != collection_job:
        raise ValueError(f'Round {round_number} collection lineage changed')
    save_json(round_dir/'collection_job.json', collection_job)
    if not read_json(collection/'manifest.json', {}).get('complete'):
        with base.persist_operation(level):
            base.run([sys.executable, str(base.repo/'stage_dagger_collect.py'),
                      str(round_dir/'collection_job.json')], cwd=base.repo,
                     log=round_dir/'collection.log')
    else:
        print(f'[{level}] DAgger round {round_number} collection reused')

    training_manifest = materialize_training_manifest(base, level, dagger_root, round_number)
    train_folder = round_dir/'train'
    train_cfg = dict(seed=42+round_number, batch_size=64, lr=config['lr'],
        total_iters=config['train_iters'], save_freq=500, warmup_steps=100,
        kl_weight=.001, stage_loss_weight=.3, gate_loss_weight=.3,
        position_noise=.001, validation_batches=8, amp=True,
        console_interval_seconds=10, action_training_mode='prior',
        first_pick_fraction=.3, freeze_modules=config['frozen_modules'])
    source_names = ('act_v2_model.py','act_v2_data.py','stage_schema.py','stage_labels.py',
                    'stage_model.py','stage_data.py','stage_train.py')
    train_job = dict(folder=str(train_folder),
        data=str(base.data/level/'trajectory.state.pd_ee_delta_pos.physx_cuda.h5'),
        model_config=saved['model_config'], train_config=train_cfg,
        policy_config=_policy_config(saved), num_demos=None,
        recovery_manifest=str(training_manifest), warm_start=str(start_checkpoint),
        source_sha256=hashlib.sha256(''.join(base.sources[name] for name in source_names).encode()).hexdigest())
    save_json(round_dir/'train_job.json', train_job)
    if not read_json(train_folder/'training_complete.json'):
        with base.persist_operation(level):
            base.run([sys.executable, str(base.repo/'stage_train.py'),
                      str(round_dir/'train_job.json')], cwd=base.repo,
                     log=round_dir/'train.log')
    else:
        print(f'[{level}] DAgger round {round_number} training reused')
    checkpoint = train_folder/'checkpoints/latest.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint
