"""Read-only import of the verified Easy reference, excluded from current submission."""
import shutil
from pathlib import Path
from github_store import GitHubStore, safe_target
from marso_experiment import digest, read_json, save_json

SNAPSHOT = 'snapshot-easy-01789351450673167331-76cc84c95774a970eec14f6f1e44f6a8548715d52086b5151ecdfe736e5bc77f.json'
REFERENCE_SHA256 = '0fbbef92aacc5cbe5642568ec5bd349979934e50f3d03d5120ccd5350720ef19'


def prepare_reference_check(repository,candidate,folder):
    candidate,folder = Path(candidate),Path(folder)
    if folder.resolve() == candidate.resolve() or folder.resolve().is_relative_to(candidate.resolve()):
        raise ValueError('Reference must be outside the current candidate')
    store = GitHubStore(repository,'run-moveboxes_easy_lab_v1_benchmark')
    if not store.load(create=False) or SNAPSHOT not in store.assets:
        raise FileNotFoundError('Pinned successful Easy reference snapshot missing')
    folder.mkdir(parents=True,exist_ok=True)
    store.download(store.assets[SNAPSHOT],folder/'source_snapshot.json')
    snapshot = read_json(folder/'source_snapshot.json')
    if snapshot.get('scope') != 'easy' or snapshot.get('version') != 1:
        raise ValueError('Reference snapshot schema mismatch')
    entries = {e['path']:e for e in snapshot['entries']}
    relative = 'easy/checkpoints/block_03.pt'
    entry = entries[relative]
    if entry['sha256'] != REFERENCE_SHA256:
        raise ValueError('Reference checkpoint differs from verified success record')
    ckdir = folder/'checkpoints/easy'
    ckdir.mkdir(parents=True,exist_ok=True)
    target = ckdir/'model.pt'
    if not target.exists():
        asset = store.assets[entry['asset']]
        if asset['size'] != entry['bytes']:
            raise ValueError('Reference asset size mismatch')
        pending = safe_target(folder,'checkpoints/easy/model.pt.part')
        store.download(asset,pending)
        if digest(pending) != REFERENCE_SHA256:
            raise ValueError('Reference checkpoint checksum mismatch')
        pending.replace(target)
    elif digest(target) != REFERENCE_SHA256:
        raise ValueError('Existing reference checkpoint differs')
    # Identical StageACT architecture/training source; the original execution
    # replans each step and takes its latest gripper prediction without a latch.
    policy = dict(model_config=dict(state_dim=54,history=16,chunk_size=16,width=128,
        heads=4,layers=2,latent_dim=16),temporal_decay=.25,ensemble_window=4,
        gate_threshold=.65,stage_threshold=.6,act_horizon=1,num_inference_steps=1)
    save_json(ckdir/'policy_config.json',policy)
    for name in ('stage_policy.py','stage_model.py','stage_schema.py','act_v2_model.py'):
        shutil.copy2(candidate/name,folder/name)
    (folder/'stage_chunk_policy.py').write_text('''from stage_policy import load_policy as load_native

class EpisodePolicy:
    def __init__(self,policy):
        self.policy = policy
    def act(self,obs,deterministic=True):
        if self.policy.step >= 199:
            self.policy.reset()
        return self.policy.act(obs,deterministic)
    def reset(self):
        self.policy.reset()

def load_policy(checkpoint,sample_obs,action_space,device):
    return EpisodePolicy(load_native(checkpoint,sample_obs,action_space,device))
''',encoding='utf-8')
    save_json(folder/'reference.json',dict(snapshot=SNAPSHOT,checkpoint_sha256=REFERENCE_SHA256,
        purpose='comparison only; never installed into current candidate',
        episode_reset='clear policy memory at the official 199-action boundary'))
    print('Reference only:',folder,flush=True)
    return folder
