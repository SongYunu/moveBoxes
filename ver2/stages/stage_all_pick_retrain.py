"""Fresh StageACT training with all-parcel pick focus; no warm-start weights."""
import copy
import hashlib
import shutil
import sys
from pathlib import Path
from marso_experiment import digest, read_json, save_json


def prepare_retrain(experiment,level,*,iterations=12000,focus_fraction=.5,
                    run_suffix='_all_pick_retrain_v1'):
    if level not in ('easy','medium','hard') or type(iterations) is not int or iterations < 1:
        raise ValueError('Invalid retraining level/budget')
    if not run_suffix or not 0 < focus_fraction < 1:
        raise ValueError('Use a new run suffix and focus fraction in (0,1)')
    source = experiment.run_dir/level
    previous = read_json(source/'stage_train_job.json')
    manifest = read_json(source/'collection/manifest.json')
    if not previous or not manifest or not manifest.get('complete'):
        raise RuntimeError('Restore the parent dataset and completed recovery collection first')
    for entry in manifest['episodes']:
        path = source/'collection'/entry['file']
        if path.resolve().parent != (source/'collection').resolve() or not entry['accepted'] or digest(path) != entry['sha256']:
            raise ValueError('Recovery collection path/hash/acceptance mismatch')
    sources = dict(experiment.sources)
    sources['stage_policy.py'] = Path(__file__).with_name('stage_policy.py').read_text(encoding='utf-8')
    names = ('act_v2_model.py','act_v2_data.py','stage_schema.py','stage_labels.py',
             'stage_model.py','stage_data.py','stage_train.py')
    for name in ('stage_pick_sampling.py','stage_pick_train.py'):
        sources[name] = Path(__file__).with_name(name).read_text(encoding='utf-8')
    source_hash = hashlib.sha256(''.join(sources[n] for n in
        names+('stage_pick_sampling.py','stage_pick_train.py')).encode()).hexdigest()
    cfg = copy.deepcopy(experiment.cfg)
    cfg.update(run_name=cfg['run_name']+run_suffix,action_training_mode='prior',lr=1e-4,
        batch_size=64,warmup_steps=100,save_freq=500,console_interval_seconds=10,
        first_pick_fraction=focus_fraction)
    cfg['total_iters'] = {name:iterations for name in ('easy','medium','hard')}
    child = type(experiment)(cfg,sources)
    child.connect()
    child._stage_helpers()
    target = child.run_dir/level
    if (target/'initial_model.pt').exists():
        raise ValueError('Fresh retraining forbids warm-start input; use a clean new run')
    train_config = copy.deepcopy(previous['train_config'])
    train_config.update(total_iters=iterations,lr=1e-4,batch_size=64,warmup_steps=100,
        save_freq=500,console_interval_seconds=10,first_pick_fraction=focus_fraction,
        pick_sampling='all_picks',action_training_mode='prior')
    policy = dict(model_config=previous['model_config'],ensemble_window=4,temporal_decay=.25,
        gate_threshold=.65,stage_threshold=.6,act_horizon=1,num_inference_steps=1,auto_reset_steps=199)
    with child.persist_operation(level):
        for entry in manifest['episodes']:
            src = source/'collection'/entry['file']
            dst = target/'collection'/entry['file']
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists() and digest(dst) != entry['sha256']:
                raise ValueError('Existing retraining data differs; use a new run suffix')
            if not dst.exists():
                shutil.copy2(src,dst)
        destination = target/'collection/manifest.json'
        if destination.exists() and digest(destination) != digest(source/'collection/manifest.json'):
            raise ValueError('Existing collection manifest differs')
        shutil.copy2(source/'collection/manifest.json',destination)
        job = dict(folder=str(target),data=previous['data'],num_demos=previous['num_demos'],
            model_config=previous['model_config'],train_config=train_config,policy_config=policy,
            source_sha256=source_hash,trainer='stage_pick_train.py',
            recovery_manifest=str(destination),recovery_manifest_sha256=digest(destination))
        old = read_json(target/'stage_train_job.json')
        if old and old != job:
            raise ValueError('Retraining settings differ; use a new run suffix')
        if (target/'checkpoints/latest.pt').exists() and not old:
            raise ValueError('Checkpoint lacks matching retraining job')
        save_json(target/'stage_train_job.json',job)
        save_json(target/'retrain_origin.json',dict(source_data_run=str(experiment.run_dir),
            initialization='random; no current or historical checkpoint imported',
            sampling='50% default target-order-balanced picks; remaining full/recovery/transitions',
            execution='native StagePolicy; every-step XYZ ensemble and latest gripper, horizon 1'))
    print(f'[{level}] fresh retraining / {iterations} steps / lr=1e-4 / batch64 / {child.run_dir}',flush=True)
    return child


def train_retrain(child,level):
    folder = child.run_dir/level
    if read_json(folder/'training_complete.json'):
        print(level,'retraining complete; evaluate the new model',flush=True)
        return
    with child.persist_operation(level):
        child.run([sys.executable,str(child.repo/'stage_pick_train.py'),str(folder/'stage_train_job.json')],
                  cwd=child.repo,log=folder/'train.log')


def package_retrain(child,levels):
    import torch
    if not levels or len(set(levels)) != len(levels) or any(level not in ('easy','medium','hard') for level in levels):
        raise ValueError('Select distinct trained difficulty levels')
    target = child.run_dir/'retrained_candidate'
    target.mkdir(parents=True,exist_ok=True)
    for name in ('stage_policy.py','stage_model.py','stage_schema.py','act_v2_model.py'):
        (target/name).write_text(child.sources[name],encoding='utf-8')
    manifest = dict(policy='native_stage_act',run_name=child.cfg['run_name'],levels={})
    lines = []
    for level in levels:
        source = child.run_dir/level/'checkpoints/latest.pt'
        job = read_json(child.run_dir/level/'stage_train_job.json')
        saved = torch.load(source,map_location='cpu',weights_only=True)
        if saved['format'] != 'moveboxes-stage-act-v1' or saved['model_config'] != job['model_config']:
            raise ValueError('Retrained checkpoint architecture mismatch')
        signature = saved.get('signature',{})
        if any(signature.get(k) != job[k] for k in ('model_config','train_config','source_sha256')) or signature.get('warm_start_sha256'):
            raise ValueError('Checkpoint is not from this fresh retraining job')
        ckdir = target/'checkpoints'/level
        ckdir.mkdir(parents=True,exist_ok=True)
        # Optimizer/RNG remain in the new run; deploy only inference weights.
        torch.save({k:saved[k] for k in ('format','model_config','model','step')},ckdir/'model.pt')
        save_json(ckdir/'policy_config.json',job['policy_config'])
        manifest['levels'][level] = dict(source=str(source),source_sha256=digest(source),
            checkpoint_sha256=digest(ckdir/'model.pt'),step=saved['step'],policy_config=job['policy_config'])
        lines.append(f'    {level}: {{ checkpoint: checkpoints/{level}/model.pt }}')
    (target/'submission.yaml').write_text('team: '+__import__('json').dumps(child.cfg['team'])+
        '\nstate:\n  policy: stage_policy:load_policy\n  levels:\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    save_json(target/'manifest.json',manifest)
    return target
