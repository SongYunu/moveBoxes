"""Prepare a separate short pick-focused run from the packaged submission weights."""
import copy
import hashlib
import shutil
import sys
from pathlib import Path

from marso_experiment import digest, read_json, save_json


def prepare_pick_finetune(experiment, candidate, level='easy', *, iterations=2000,
                          lr_factor=.2, first_pick_fraction=.5, run_suffix='_pick_refine_v1'):
    if level not in ('easy', 'medium', 'hard'):
        raise ValueError('Unknown difficulty')
    if type(iterations) is not int or iterations < 1 or not 0 < lr_factor <= 1:
        raise ValueError('Invalid fine-tune iterations or learning-rate factor')
    if not 0 < first_pick_fraction < 1 or not run_suffix:
        raise ValueError('Use a separate run suffix and a pick fraction in (0,1)')
    candidate = Path(candidate)
    source = experiment.run_dir/level
    checkpoint = candidate/'checkpoints'/level/'model.pt'
    policy = read_json(checkpoint.parent/'policy_config.json')
    provenance = read_json(candidate/'manifest.json')
    previous = read_json(source/'stage_train_job.json')
    recovery = read_json(source/'collection/manifest.json')
    if not checkpoint.is_file() or not policy or not previous or not recovery or not recovery.get('complete'):
        raise RuntimeError('Requires the current packaged checkpoint and completed recovery data')
    if digest(checkpoint) != provenance['levels'][level]['checkpoint_sha256']:
        raise ValueError('Packaged checkpoint differs from its manifest')
    if policy['model_config'] != previous['model_config']:
        raise ValueError('Packaged and trained architectures differ')
    # Preserve the source of the original training job for reproducible resume.
    names = ('act_v2_model.py','act_v2_data.py','stage_schema.py','stage_labels.py',
             'stage_model.py','stage_data.py','stage_train.py')
    source_hash = hashlib.sha256(''.join(experiment.sources[n] for n in names).encode()).hexdigest()
    if source_hash != previous['source_sha256']:
        raise ValueError('Loaded training source differs from the original job')
    for entry in recovery['episodes']:
        path = source/'collection'/entry['file']
        if path.resolve().parent != (source/'collection').resolve() or digest(path) != entry['sha256']:
            raise ValueError('Recovery data path/hash mismatch')

    train_config = copy.deepcopy(previous['train_config'])
    train_config.update(total_iters=iterations, lr=previous['train_config']['lr']*lr_factor,
                        warmup_steps=100, save_freq=500, first_pick_fraction=first_pick_fraction)
    cfg = copy.deepcopy(experiment.cfg)
    cfg.update(run_name=cfg['run_name']+run_suffix, **{k:v for k,v in
               previous['model_config'].items() if k != 'state_dim'})
    cfg.update({k:v for k,v in train_config.items() if k != 'total_iters'})
    cfg['total_iters'] = {name:iterations for name in ('easy','medium','hard')}
    child = type(experiment)(cfg, experiment.sources)
    child.connect()
    child._stage_helpers()
    target = child.run_dir/level
    with child.persist_operation(level):
        pairs = [(checkpoint, target/'initial_model.pt')]
        pairs += [(p,target/'collection'/p.name) for p in (source/'collection').iterdir()
                  if p.suffix in ('.json','.npz')]
        for src, dst in pairs:
            if dst.exists():
                if digest(src) != digest(dst):
                    raise ValueError('Existing fine-tune input differs; use a new run_suffix')
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        job = dict(folder=str(target), data=previous['data'],
                   model_config=previous['model_config'], train_config=train_config,
                   policy_config=policy, num_demos=previous['num_demos'],
                   source_sha256=source_hash, warm_start=str(target/'initial_model.pt'),
                   warm_start_sha256=digest(target/'initial_model.pt'),
                   recovery_manifest=str(target/'collection/manifest.json'),
                   recovery_manifest_sha256=digest(target/'collection/manifest.json'))
        job_path = target/'stage_train_job.json'
        if job_path.exists() and read_json(job_path) != job:
            raise ValueError('Existing fine-tune settings differ; use a new run_suffix')
        save_json(job_path, job)
        save_json(target/'pick_finetune_origin.json', dict(source_run=str(experiment.run_dir),
            candidate=str(candidate), checkpoint_sha256=digest(checkpoint),
            first_pick_fraction=first_pick_fraction,
            optimizer='new AdamW at lower learning rate; subsequent resume restores optimizer/RNG',
            inference='same packaged policy configuration; stage horizons unchanged'))
    print(f'[{level}] prepared {iterations} steps / lr={train_config["lr"]:g} / '
          f'first_pick_fraction={first_pick_fraction} / {child.run_dir}', flush=True)
    return child


def train_pick_finetune(child, level):
    """Run the prepared job; do not regenerate it through StageExperiment.train."""
    folder = child.run_dir/level
    job_path = folder/'stage_train_job.json'
    job = read_json(job_path)
    if not job or 'first_pick_fraction' not in job['train_config']:
        raise RuntimeError('Call prepare_pick_finetune first')
    completed = read_json(folder/'training_complete.json')
    if completed:
        print(level, 'pick fine-tune already complete; run official evaluation', flush=True)
        return
    with child.persist_operation(level):
        child.run([sys.executable, str(child.repo/'stage_train.py'), str(job_path)],
                  cwd=child.repo, log=folder/'train.log')
