"""Fetch only explicitly selected, checksum-verified Release files."""
import os
from pathlib import Path
from github_store import GitHubStore,safe_target
from marso_experiment import read_json,save_json,digest


class SnapshotSource:
    def __init__(self,repository,run_name,scope,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.store=GitHubStore(repository,'run-'+run_name+'_benchmark')
        if not self.store.load(create=False):raise FileNotFoundError(f'Source Release missing: {run_name}')
        pin=read_json(self.folder/'origin.json')
        if pin and (pin['run_name']!=run_name or pin['scope']!=scope):raise ValueError('Source changed; use a new run_name')
        names=sorted(n for n in self.store.assets if n.startswith(f'snapshot-{scope}-'))
        if not names:raise FileNotFoundError(f'No {scope} source snapshot')
        self.name=pin['snapshot'] if pin else names[-1]
        self.store.download(self.store.assets[self.name],self.folder/'snapshot.json')
        snap=read_json(self.folder/'snapshot.json')
        if snap['scope']!=scope:raise ValueError('Snapshot scope mismatch')
        self.entries={e['path']:e for e in snap['entries']}
        save_json(self.folder/'origin.json',dict(run_name=run_name,scope=scope,snapshot=self.name))

    def fetch(self,relative,destination=None):
        entry=self.entries.get(relative)
        if entry is None:raise FileNotFoundError(f'Source snapshot lacks {relative}')
        safe_target(self.folder,relative)
        target=Path(destination) if destination else safe_target(self.folder,relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target)!=entry['sha256']:raise ValueError('Local imported file differs from source')
            return target
        asset=self.store.assets.get(entry['asset'])
        if not asset or asset['size']!=entry['bytes']:raise ValueError('Source asset missing or incomplete')
        pending=Path(str(target)+'.part');self.store.download(asset,pending)
        if digest(pending)!=entry['sha256']:raise ValueError('Source asset checksum mismatch')
        os.replace(pending,target);return target
