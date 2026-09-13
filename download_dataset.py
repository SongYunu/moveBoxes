"""Download official Kaggle data; no credentials are stored in this project."""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

COMPETITION = 'marso-hack-berlin-2026-robot-parcel-sorting-challenge'
LEVELS = ('easy', 'medium', 'hard')


def stage(source, dest):
    source, dest = Path(source), Path(dest)
    if not source.exists():
        raise FileNotFoundError(f'Data source not found: {source}. Upload the ZIP to Drive or update CONFIG data_source.')
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        roots = [source]
        if source.is_file() and source.suffix == '.zip':
            target = Path(tmp) / 'zip'
            target.mkdir()
            with zipfile.ZipFile(source) as zf:
                for member in zf.infolist():
                    if not (target / member.filename).resolve().is_relative_to(target.resolve()):
                        raise ValueError('Unsafe ZIP member path')
                zf.extractall(target)
            roots = [target]
        for i, archive in enumerate(source.rglob('*.tar.gz')):
            target = Path(tmp) / str(i)
            target.mkdir()
            with tarfile.open(archive) as tf:
                tf.extractall(target, filter='data')
            roots.append(target)
        found = {}
        for root in roots:
            for h5 in root.rglob('trajectory.*.pd_ee_delta_pos.physx_cuda.h5'):
                level = next((p for p in reversed(h5.parts[:-1]) if p in LEVELS), None)
                if level is None:
                    raise ValueError(f'Cannot infer difficulty: {h5}')
                key = (level, h5.name)
                if key in found:
                    raise ValueError(f'Ambiguous duplicate dataset: {key}')
                found[key] = h5
        if not found:
            raise FileNotFoundError('No official trajectory HDF5 files found in downloaded data')
        for (level, name), h5 in found.items():
            metadata = h5.with_suffix('.json')
            if not metadata.exists():
                raise FileNotFoundError(f'Missing trajectory metadata: {metadata}')
            folder = dest / level
            folder.mkdir(exist_ok=True)
            for src in (h5, metadata):
                dst = folder / src.name
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
    return inspect(dest)


def inspect(dest):
    import h5py
    rows = []
    for h5 in sorted(Path(dest).glob('*/trajectory.*.h5')):
        info = json.loads(h5.with_suffix('.json').read_text())
        with h5py.File(h5, 'r') as f:
            keys = sorted(k for k in f if k.startswith('traj_'))
            if not keys:
                raise ValueError(f'No trajectories: {h5}')
            for k in keys:
                actions = f[k]['actions']
                if actions.ndim != 2 or actions.shape[1] != 4:
                    raise ValueError(f'Unexpected actions: {h5}/{k}')
                if '.state.' in h5.name:
                    obs = f[k]['obs']
                    expected = {'easy': 54, 'medium': 72, 'hard': 90}[h5.parent.name]
                    if obs.shape != (actions.shape[0] + 1, expected):
                        raise ValueError(f'Unexpected state shape: {h5}/{k}: {obs.shape}')
            if len(info['episodes']) != len(keys):
                raise ValueError(f'JSON/HDF5 episode count mismatch: {h5}')
        with h5.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        rows.append(dict(level=h5.parent.name, file=str(h5), episodes=len(keys),
                         bytes=h5.stat().st_size, sha256=digest))
    (Path(dest) / 'dataset_manifest.json').write_text(json.dumps(rows, indent=2))
    return rows


def download(dest):
    import kagglehub
    source = kagglehub.competition_download(COMPETITION)
    return stage(source, dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', default='data')
    parser.add_argument('--source', help='Already downloaded directory or ZIP')
    args = parser.parse_args()
    rows = stage(args.source, args.dest) if args.source else download(args.dest)
    print(json.dumps(rows, indent=2))
