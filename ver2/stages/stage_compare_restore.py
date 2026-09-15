"""Restore only known Stage ACT baseline weights from existing Release snapshots.

No release creation, uploads, model selection or fallback to a different hash.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from github_store import GitHubStore, sha256

BASELINES = {
    'easy': dict(run='moveboxes_easy_lab_v1_benchmark', checkpoint='block_03.pt',
        sha256='0fbbef92aacc5cbe5642568ec5bd349979934e50f3d03d5120ccd5350720ef19'),
    'medium': dict(run='moveboxes_medium_lab_v1_benchmark', checkpoint='block_02.pt',
        sha256='f7c0d5edf5b2bd6b93918d10af43b9c4ee023befd3162139e391b371e1541730'),
    'hard': dict(run='moveboxes_hard_stage_v1_benchmark', checkpoint='best_val.pt',
        sha256='96d30fe99cdf6c684d8ad5a0e55fbc9b663d76b6ad02f2e6b331a1ea02da30bc'),
}


def restore_baselines(root, levels=('easy', 'medium', 'hard'), repository='SongYunu/moveBoxes', token=None):
    root = Path(root).resolve()
    result = {}
    for level in levels:
        known = BASELINES[level]
        folder = root/level
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/known['checkpoint']
        if path.exists() and sha256(path) != known['sha256']:
            raise FileExistsError(f'Existing checkpoint differs: {path}; use a fresh restore directory')
        if not path.exists():
            store = GitHubStore(repository, 'run-'+known['run'], token=token)
            if not store.load(create=False):
                raise FileNotFoundError(f'Missing baseline release: {known["run"]}')
            snapshots = sorted((n for n in store.assets if n.startswith('snapshot-'+level+'-')), reverse=True)
            entry = None
            # Search metadata only; a later run may have selected another checkpoint.
            for name in snapshots:
                manifest = folder/'restore_snapshot.json'
                store.download(store.assets[name], manifest)
                snapshot = json.loads(manifest.read_text(encoding='utf-8'))
                if snapshot.get('version') != 1 or snapshot.get('scope') != level:
                    raise ValueError('Invalid baseline snapshot')
                entry = next((e for e in snapshot['entries'] if e['sha256'] == known['sha256']
                    and e['path'] == level+'/checkpoints/'+known['checkpoint']), None)
                if entry:
                    break
            if entry is None:
                raise FileNotFoundError(f'Exact baseline absent from snapshots: {level} {known["sha256"]}')
            asset = store.assets.get(entry['asset'])
            if asset is None or asset['size'] != entry['bytes']:
                raise ValueError('Missing or incomplete checkpoint asset')
            partial = path.with_suffix('.pt.part')
            store.download(asset, partial)
            if sha256(partial) != known['sha256'] or partial.stat().st_size != entry['bytes']:
                raise ValueError('Downloaded baseline hash/size mismatch')
            os.replace(partial, path)
        # Use settings from the recorded evaluation, not a mutable latest sidecar.
        result[level] = dict(checkpoint=str(path), checkpoint_sha256=known['sha256'],
            policy_config=dict(ensemble_window=4, temporal_decay=.25, gate_threshold=.65, stage_threshold=.6))
        print(f'{level}: exact baseline restored ({known["sha256"][:12]})', flush=True)
    return result
