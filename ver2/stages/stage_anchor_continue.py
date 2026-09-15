"""Continue each difficulty from its checksum-verified historical StageACT anchor."""
import copy
import json
import os
import shutil
from pathlib import Path

from github_store import GitHubStore, safe_target
from marso_experiment import digest, read_json, save_json


ANCHORS = {
    'easy': dict(run='moveboxes_easy_lab_v1_benchmark', checkpoint='block_03.pt',
        sha256='0fbbef92aacc5cbe5642568ec5bd349979934e50f3d03d5120ccd5350720ef19',
        recorded_score=1.0, recorded_episodes=100),
    'medium': dict(run='moveboxes_medium_lab_v1_benchmark', checkpoint='block_02.pt',
        sha256='f7c0d5edf5b2bd6b93918d10af43b9c4ee023befd3162139e391b371e1541730',
        recorded_score=.40625, recorded_episodes=8),
    'hard': dict(run='moveboxes_hard_stage_v1_benchmark', checkpoint='best_val.pt',
        sha256='96d30fe99cdf6c684d8ad5a0e55fbc9b663d76b6ad02f2e6b331a1ea02da30bc',
        recorded_score=1/24, recorded_episodes=8),
}
DIMS = {'easy':54, 'medium':72, 'hard':90}


def _download(store, entry, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != entry['sha256']:
            raise ValueError(f'Existing imported file differs: {target}')
        return target
    asset = store.assets.get(entry['asset'])
    if not asset or asset['size'] != entry['bytes']:
        raise ValueError(f'Missing/incomplete source asset: {entry["path"]}')
    pending = Path(str(target)+'.part')
    store.download(asset, pending)
    if pending.stat().st_size != entry['bytes'] or digest(pending) != entry['sha256']:
        raise ValueError(f'Source checksum mismatch: {entry["path"]}')
    os.replace(pending, target)
    return target


def _snapshot_with_anchor(store, level, known, folder, pin=None):
    names = sorted(n for n in store.assets if n.startswith(f'snapshot-{level}-'))
    if pin:
        names = [pin]
    for name in reversed(names):
        temp = Path(folder)/f'{level}_source_snapshot.json'
        store.download(store.assets[name], temp)
        snapshot = read_json(temp)
        if snapshot.get('version') != 1 or snapshot.get('scope') != level:
            continue
        entries = {entry['path']:entry for entry in snapshot['entries']}
        key = f'{level}/checkpoints/{known["checkpoint"]}'
        if key in entries and entries[key]['sha256'] == known['sha256']:
            return name, entries
    raise FileNotFoundError(f'Exact historical anchor absent: {level} {known["sha256"]}')


def prepare(experiment, *, run_suffix='_anchor_continue_v1', iterations=None, lr=2e-5):
    """Restore immutable anchors plus their accepted recovery collections."""
    budgets = {'medium':8000, 'hard':12000}
    budgets.update(iterations or {})
    if not run_suffix or lr <= 0 or any(type(budgets[k]) is not int or budgets[k] < 1 for k in ('medium','hard')):
        raise ValueError('Invalid anchor continuation settings')
    cfg = copy.deepcopy(experiment.cfg)
    cfg.update(run_name=cfg['run_name']+run_suffix, lr=lr, warmup_steps=100,
        save_freq=500, first_pick_fraction=.3, action_training_mode='prior',
        console_interval_seconds=10)
    cfg['total_iters'] = dict(easy=1, medium=budgets['medium'], hard=budgets['hard'])
    child = type(experiment)(cfg, dict(experiment.sources))
    child.connect()
    child._stage_helpers()
    imported = {}
    for level, known in ANCHORS.items():
        target = child.run_dir/level
        target.mkdir(parents=True, exist_ok=True)
        origin_path = target/'anchor_origin.json'
        origin = read_json(origin_path, {})
        expected = dict(run=known['run'], checkpoint=known['checkpoint'], sha256=known['sha256'])
        if origin and any(origin.get(k) != v for k,v in expected.items()):
            raise ValueError(f'{level}: pinned anchor changed; use a new run suffix')
        store = GitHubStore(cfg['github_repository'], 'run-'+known['run'])
        if not store.load(create=False):
            raise FileNotFoundError(f'Missing anchor Release: {known["run"]}')
        snap_name, entries = _snapshot_with_anchor(store, level, known, target, origin.get('snapshot'))
        anchor_entry = entries[f'{level}/checkpoints/{known["checkpoint"]}']
        anchor = _download(store, anchor_entry, target/'anchor.pt')
        import torch
        saved = torch.load(anchor, map_location='cpu', weights_only=True)
        if saved.get('format') != 'moveboxes-stage-act-v1' or saved['model_config']['state_dim'] != DIMS[level]:
            raise ValueError(f'{level}: anchor architecture mismatch')
        collection_entries = [entry for path,entry in entries.items()
            if path.startswith(f'{level}/collection/') and Path(path).suffix in ('.json','.npz')]
        if f'{level}/collection/manifest.json' not in entries:
            raise FileNotFoundError(f'{level}: accepted recovery manifest absent')
        for entry in collection_entries:
            relative = Path(entry['path']).relative_to(level)
            _download(store, entry, safe_target(target, relative.as_posix()))
        manifest = read_json(target/'collection/manifest.json')
        if not manifest.get('complete'):
            raise ValueError(f'{level}: source recovery collection incomplete')
        for item in manifest['episodes']:
            path = safe_target(target/'collection', item['file'])
            if digest(path) != item['sha256']:
                raise ValueError(f'{level}: recovery episode checksum mismatch')
        if level in ('medium','hard') and not (target/'checkpoints/latest.pt').exists():
            initial = target/'initial_model.pt'
            if not initial.exists():
                shutil.copy2(anchor, initial)
            if digest(initial) != known['sha256']:
                raise ValueError(f'{level}: warm-start copy differs from anchor')
        save_json(origin_path, dict(level=level, snapshot=snap_name, **expected,
            recorded_score=known['recorded_score'], recorded_episodes=known['recorded_episodes'],
            role='immutable baseline; continuation is stored separately'))
        imported[level] = anchor
    child.sync_common()
    print('난이도별 검증 앵커 복원:', {k:str(v) for k,v in imported.items()})
    return child


def train(child, level):
    if level not in ('medium','hard'):
        raise ValueError('Easy anchor is frozen; continue only Medium or Hard')
    child.train(level)


def package(child, *, use_trained=None, folder_name='anchor_candidate'):
    """Package one implementation with independently selected per-level weights."""
    import torch
    selected = {'medium':False, 'hard':False}
    selected.update(use_trained or {})
    target = child.run_dir/folder_name
    target.mkdir(parents=True, exist_ok=True)
    for name in ('stage_policy.py','stage_model.py','stage_schema.py','act_v2_model.py'):
        (target/name).write_text(child.sources[name], encoding='utf-8')
    manifest = dict(policy='native_stage_act_per_difficulty', run_name=child.cfg['run_name'], levels={})
    lines = []
    for level, known in ANCHORS.items():
        anchor = child.run_dir/level/'anchor.pt'
        trained = child.run_dir/level/'checkpoints/latest.pt'
        source = trained if level != 'easy' and selected[level] else anchor
        if not source.is_file():
            raise FileNotFoundError(source)
        saved = torch.load(source, map_location='cpu', weights_only=True)
        config = dict(model_config=saved['model_config'], temporal_decay=.25, ensemble_window=4,
            gate_threshold=.65, stage_threshold=.6, act_horizon=1, num_inference_steps=1,
            auto_reset_steps=199)
        out = target/'checkpoints'/level
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, out/'model.pt')
        save_json(out/'policy_config.json', config)
        manifest['levels'][level] = dict(source=str(source), checkpoint_sha256=digest(source),
            selection='trained' if source == trained else 'anchor', step=saved.get('step'),
            policy_config=config, historical_anchor_sha256=known['sha256'])
        lines.append(f'    {level}: {{ checkpoint: checkpoints/{level}/model.pt }}')
    (target/'submission.yaml').write_text('team: '+json.dumps(child.cfg['team'])+
        '\nstate:\n  policy: stage_policy:load_policy\n  levels:\n'+'\n'.join(lines)+'\n', encoding='utf-8')
    save_json(target/'manifest.json', manifest)
    return target
