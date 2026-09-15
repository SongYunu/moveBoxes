"""Restore exact State Stage ACT baselines from public GitHub Release assets.

The asset URLs, byte sizes and SHA256 values are pinned. No authentication,
release creation, uploads, model selection or fallback to another weight file.
"""
import hashlib
import os
from pathlib import Path
import urllib.request


BASELINES = {
    'easy': dict(checkpoint='block_03.pt', bytes=5132112,
        sha256='0fbbef92aacc5cbe5642568ec5bd349979934e50f3d03d5120ccd5350720ef19',
        url='https://github.com/SongYunu/moveBoxes/releases/download/run-moveboxes_easy_lab_v1_benchmark/file-e8495370f5a21be0-0fbbef92aacc5cbe5642568ec5bd349979934e50f3d03d5120ccd5350720ef19.pt'),
    'medium': dict(checkpoint='block_02.pt', bytes=5639248,
        sha256='f7c0d5edf5b2bd6b93918d10af43b9c4ee023befd3162139e391b371e1541730',
        url='https://github.com/SongYunu/moveBoxes/releases/download/run-moveboxes_medium_lab_v1_benchmark/file-13ed57ec591572a3-f7c0d5edf5b2bd6b93918d10af43b9c4ee023befd3162139e391b371e1541730.pt'),
    'hard': dict(checkpoint='best_val.pt', bytes=6147472,
        sha256='96d30fe99cdf6c684d8ad5a0e55fbc9b663d76b6ad02f2e6b331a1ea02da30bc',
        url='https://github.com/SongYunu/moveBoxes/releases/download/run-moveboxes_hard_stage_v1_benchmark/file-d1ba41539999b597-96d30fe99cdf6c684d8ad5a0e55fbc9b663d76b6ad02f2e6b331a1ea02da30bc.pt'),
}


def sha256(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def restore_baselines(root, levels=('easy', 'medium', 'hard')):
    root = Path(root).resolve()
    result = {}
    for level in levels:
        known = BASELINES[level]
        folder = root/level
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/known['checkpoint']
        if path.exists() and (path.stat().st_size != known['bytes'] or sha256(path) != known['sha256']):
            raise FileExistsError(f'Existing checkpoint differs: {path}; use a fresh restore directory')
        if not path.exists():
            partial = path.with_suffix(path.suffix+'.part')
            try:
                request = urllib.request.Request(known['url'], headers={'User-Agent':'moveboxes-colab'})
                with urllib.request.urlopen(request, timeout=180) as response, partial.open('wb') as handle:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
                if partial.stat().st_size != known['bytes'] or sha256(partial) != known['sha256']:
                    raise ValueError(f'Downloaded {level} baseline hash/size mismatch')
                os.replace(partial, path)
            finally:
                partial.unlink(missing_ok=True)
        result[level] = dict(checkpoint=str(path), checkpoint_sha256=known['sha256'],
            policy_config=dict(ensemble_window=4, temporal_decay=.25, gate_threshold=.65, stage_threshold=.6))
        print(f'{level}: exact baseline restored ({known["sha256"][:12]})', flush=True)
    return result
