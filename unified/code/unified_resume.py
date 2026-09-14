"""Allow deliberate focus changes while retaining optimizer/RNG and fixed data splits."""
import hashlib,json
from pathlib import Path


def dataset_identity(job):
    manifest=json.loads(Path(job['data']).read_text(encoding='utf-8'))
    # Focus is the only changing input allowed within a continuous learning segment.
    manifest.pop('focus',None)
    return hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()


def resume_matches(saved,current):
    if saved==current:return True
    if not saved.get('curriculum_dataset_sha256'):return False
    old=dict(saved);new=dict(current)
    old.pop('data_sha256',None);new.pop('data_sha256',None)
    return old==new
