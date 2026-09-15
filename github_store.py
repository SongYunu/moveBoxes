"""Immutable Release assets with publish-last snapshots for Colab persistence.

Tokens stay in environment variables and are never serialized. Upload a snapshot
only at a paused checkpoint/episode boundary, after its files are closed.
"""
import hashlib
import http.client
import json
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected


def safe_target(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or '\\' in relative or ':' in relative:
        raise ValueError('Invalid snapshot path')
    path = (root/relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Snapshot path escapes run directory')
    return path


def compact_medium_files(root):
    """Files sufficient to restore a selected Medium model without training debris."""
    root = Path(root).resolve()
    folder = root/'medium'
    names = {
        'v2_origin.json','v2_source_snapshot.json','source_best.json','inputs_ready.json',
        'lab_best.json','lab_history.json','lab_protocol.json','selection.json',
        'stage_train_job.json','data_audit.json','model_info.json','test_metrics.json','metrics.json',
    }
    files = [folder/name for name in names if (folder/name).is_file()]
    best_path = folder/'lab_best.json'
    checkpoints = {folder/'checkpoints/initial_model.pt'}
    if best_path.exists():
        try:
            best = json.loads(best_path.read_text(encoding='utf-8'))
            candidate = folder/'checkpoints'/Path(best['checkpoint']).name
            if candidate.resolve().parent != (folder/'checkpoints').resolve():
                raise ValueError('Invalid selected checkpoint path')
            checkpoints.add(candidate)
        except (KeyError, json.JSONDecodeError):
            pass
    files += [path for path in checkpoints if path.is_file()]
    policy = folder/'checkpoints/policy_config.json'
    if policy.is_file():
        files.append(policy)
    files += [path for path in folder.glob('*.mp4') if path.is_file()]
    return sorted(set(files))


def compact_hard_files(root):
    root=Path(root).resolve();folder=root/'hard'
    names=('source_snapshot.json','inputs_ready.json','lab_best.json','lab_history.json','lab_protocol.json',
        'stage_train_job.json','model_info.json','data_audit.json','test_metrics.json','metrics.json','selection.json')
    names += ('transfer_sources.json','transfer_selection.json','transfer_report.json','transfer_job.json')
    files=[folder/name for name in names if (folder/name).is_file()]
    best=folder/'lab_best.json'
    if best.exists():
        data=json.loads(best.read_text(encoding='utf-8'))
        checkpoint=folder/'checkpoints'/Path(data['checkpoint']).name
        if checkpoint.is_file():files.append(checkpoint)
    policy=folder/'checkpoints/policy_config.json'
    if policy.is_file():files.append(policy)
    files.extend(folder.glob('*.mp4'))
    # Two diagnostic traces are small enough to retain evidence for future fixes.
    files.extend((folder/'traces').rglob('seed*.json'))
    return sorted(set(files))


class AssetUploadError(RuntimeError):
    def __init__(self,status):
        self.status = status
        super().__init__(f'GitHub asset upload failed ({status}); local files are retained.')


class GitHubStore:
    def __init__(self, repository, tag, token=None):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
            raise ValueError('Invalid GitHub repository name')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', tag):
            raise ValueError('Invalid release tag')
        self.repository, self.tag = repository, tag
        self.token = token or os.environ.get('GH_TOKEN')
        if not self.token:
            raise RuntimeError('Colab 보안 비밀에 GH_TOKEN을 추가하고 노트북 액세스를 허용하세요. '
                               'moveBoxes 저장소의 Contents: Read and write 권한이 필요합니다.')
        self.release = None
        self.assets = {}
        self.file_cache = {}
        self.part = 1
        self.write_asset_count = 0

    def headers(self):
        return {'Authorization': 'Bearer '+self.token, 'Accept': 'application/vnd.github+json',
                'User-Agent': 'moveBoxes-colab', 'X-GitHub-Api-Version': '2022-11-28'}

    def request(self, method, path, body=None):
        payload = json.dumps(body).encode() if body is not None else None
        headers = self.headers()
        if payload is not None:
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request('https://api.github.com'+path, data=payload, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404 and method == 'GET':
                return None
            # Do not include headers or token-bearing requests in exception messages.
            raise RuntimeError(f'GitHub {method} failed ({error.code}); check token permissions and repository access.') from None

    def _release_for_part(self, part, create=False):
        base = '/repos/'+self.repository+'/releases'
        tag = self.tag if part == 1 else f'{self.tag}-part-{part:03d}'
        release = self.request('GET', base+'/tags/'+tag)
        if release is None:
            if not create:
                return None
            try:
                release = self.request('POST', base, dict(tag_name=tag, name=tag,
                    body='Colab experiment snapshots for '+self.tag+'. Referenced files may be in earlier parts.',
                    draft=False, prerelease=True, make_latest='false'))
            except RuntimeError:
                # Another notebook may have created the same next part meanwhile.
                release = self.request('GET', base+'/tags/'+tag)
                if release is None:
                    raise
        return release

    def _assets_for_release(self, release):
        items_by_name = {}
        page = 1
        while True:
            items = self.request('GET', f"/repos/{self.repository}/releases/{release['id']}/assets?per_page=100&page={page}")
            if items is None:
                raise RuntimeError('Cannot read release assets')
            items_by_name.update({item['name']: item for item in items})
            if len(items) < 100:
                break
            page += 1
        return items_by_name

    def load(self, create=True):
        release = self._release_for_part(1,create=create)
        if release is None:
            return False
        self.assets = {}
        part = 1
        while release is not None:
            assets = self._assets_for_release(release)
            self.assets.update(assets)
            self.release,self.part,self.write_asset_count = release,part,len(assets)
            part += 1
            release = self._release_for_part(part)
        return True

    def _next_part(self):
        self.part += 1
        self.release = self._release_for_part(self.part,create=True)
        assets = self._assets_for_release(self.release)
        self.assets.update(assets)
        self.write_asset_count = len(assets)
        print(f'GitHub 저장 공간 이어쓰기: {self.release["tag_name"]}',flush=True)

    def upload(self, path, name):
        path = Path(path)
        if self.release is None:
            self.load()
        if name in self.assets:
            if self.assets[name]['size'] != path.stat().st_size:
                raise RuntimeError('Existing immutable asset has an unexpected size')
            return self.assets[name]
        # GitHub permits 1,000 assets per Release. Rotate before exhausting it;
        # old assets remain addressable and every complete snapshot still restores.
        while self.write_asset_count >= 950:
            self._next_part()
        try:
            return self._upload_once(path,name)
        except AssetUploadError as error:
            if error.status != 422:
                raise
            # GitHub can return 422 while a just-finished upload is still absent
            # from the release asset listing. Refresh once after propagation; if
            # it is still absent, continue in a new release part. The immutable
            # content name makes either outcome safe to restore.
            for delay in (0, 2):
                if delay:
                    time.sleep(delay)
                self.load(create=False)
                if name in self.assets:
                    if self.assets[name]['size'] != path.stat().st_size:
                        raise RuntimeError('Existing immutable asset has an unexpected size') from None
                    return self.assets[name]
            self._next_part()
            return self._upload_once(path,name)

    def _upload_once(self, path, name):
        endpoint = f"/repos/{self.repository}/releases/{self.release['id']}/assets?name="+urllib.parse.quote(name)
        headers = self.headers() | {'Content-Type': 'application/octet-stream',
                                   'Content-Length': str(path.stat().st_size)}
        connection = http.client.HTTPSConnection('uploads.github.com', timeout=300)
        try:
            with path.open('rb') as handle:
                connection.request('POST', endpoint, body=handle, headers=headers)
                response = connection.getresponse()
                body = response.read()
            if response.status != 201:
                raise AssetUploadError(response.status)
            asset = json.loads(body)
            self.assets[name] = asset
            self.write_asset_count += 1
            return asset
        finally:
            connection.close()

    def download(self, asset, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(asset['url'], headers=self.headers() | {'Accept': 'application/octet-stream'})
        opener = urllib.request.build_opener(SafeRedirect())
        with opener.open(request, timeout=120) as response, path.open('wb') as handle:
            while chunk := response.read(1024*1024):
                handle.write(chunk)

    def sync(self, root, scope, files=None, *, refresh=True):
        root = Path(root).resolve()
        if scope not in ('common', 'easy', 'medium', 'hard'):
            raise ValueError('Invalid snapshot scope')
        if refresh or self.release is None:
            self.load()
        if files is None:
            files = (root/scope).rglob('*')
        entries = []
        for path in sorted(files):
            if not path.is_file() or path.is_symlink() or path.suffix in ('.tmp', '.part'):
                continue
            path = path.resolve()
            relative = path.relative_to(root).as_posix()
            safe_target(root, relative)
            if '__pycache__' in path.parts or 'events.out.tfevents' in path.name:
                continue
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            cache_key = (str(root), str(path))
            cached = self.file_cache.get(cache_key)
            if cached and cached[0] == signature and cached[1]['asset'] in self.assets:
                entries.append(cached[1])
                continue
            # Logs can still be appended by the notebook while the trainer pauses.
            # Hash and upload the same frozen bytes, never a moving source file.
            with tempfile.TemporaryDirectory() as staging:
                frozen = Path(staging)/path.name
                shutil.copy2(path, frozen)
                digest = sha256(frozen)
                name = 'file-'+hashlib.sha256(relative.encode()).hexdigest()[:16]+'-'+digest+path.suffix
                asset = self.upload(frozen, name)
            entry = dict(path=relative, asset=name, sha256=digest, bytes=asset['size'])
            entries.append(entry)
            after = path.stat()
            if signature == (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
                self.file_cache[cache_key] = (signature, entry)
        if not entries:
            return None
        snapshot = dict(version=1, scope=scope, entries=entries)
        # Content hash prevents duplicate manifests when nothing changed.
        content_hash = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
        previous = sorted(name for name in self.assets if name.startswith('snapshot-'+scope+'-'))
        if previous and previous[-1].endswith('-'+content_hash+'.json'):
            return None
        name = f'snapshot-{scope}-{time.time_ns():020d}-{content_hash}.json'
        temporary = root/('.snapshot-'+uuid.uuid4().hex+'.tmp')
        try:
            temporary.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
            self.upload(temporary, name)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'GitHub 백업 완료: {scope} ({len(entries)} files)', flush=True)
        return name

    def restore(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        if not self.load(create=False):
            return 0
        restored = 0
        for scope in ('common', 'easy', 'medium', 'hard'):
            names = sorted(name for name in self.assets if name.startswith('snapshot-'+scope+'-'))
            if not names:
                continue
            temporary = root/('.restore-'+uuid.uuid4().hex+'.tmp')
            try:
                self.download(self.assets[names[-1]], temporary)
                snapshot = json.loads(temporary.read_text(encoding='utf-8'))
            finally:
                temporary.unlink(missing_ok=True)
            if snapshot.get('version') != 1 or snapshot.get('scope') != scope:
                raise ValueError('Invalid snapshot manifest')
            for entry in snapshot['entries']:
                path = safe_target(root, entry['path'])
                if scope != 'common' and Path(entry['path']).parts[0] != scope:
                    raise ValueError('Cross-scope snapshot path')
                asset = self.assets.get(entry['asset'])
                if asset is None or asset['size'] != entry['bytes']:
                    raise ValueError('Incomplete snapshot: missing asset')
                if path.exists():
                    if sha256(path) == entry['sha256']:
                        continue
                    raise FileExistsError(f'Local/remote results differ: {path}. Use a fresh local output_root or a new run_name.')
                partial = path.with_name(path.name+'.part')
                self.download(asset, partial)
                if partial.stat().st_size != entry['bytes'] or sha256(partial) != entry['sha256']:
                    partial.unlink(missing_ok=True)
                    raise ValueError('Downloaded checkpoint hash mismatch')
                os.replace(partial, path)
                restored += 1
        return restored


_SYNC_STORES = {}


def sync_from_env():
    """Only called by subprocesses at completed checkpoint/episode boundaries."""
    if not os.environ.get('MOVEBOXES_SYNC_ROOT'):
        return
    key = (os.environ['MOVEBOXES_GITHUB_REPO'], os.environ['MOVEBOXES_RELEASE_TAG'],
           os.environ['MOVEBOXES_SYNC_ROOT'], os.environ['MOVEBOXES_SYNC_SCOPE'], os.environ.get('GH_TOKEN'))
    if key not in _SYNC_STORES:
        _SYNC_STORES[key] = GitHubStore(key[0], key[1])
    # One GPU subprocess is the writer during an operation. Keep the uploaded
    # asset index and unchanged-file cache between episodes; the parent refreshes
    # the index after the subprocess exits. Every episode still publishes a snapshot.
    root = Path(key[2])
    files = compact_medium_files(root) if os.environ.get('MOVEBOXES_COMPACT_MEDIUM_SYNC')=='1' and key[3]=='medium' else None
    if os.environ.get('MOVEBOXES_COMPACT_HARD_SYNC')=='1' and key[3]=='hard':files=compact_hard_files(root)
    try:
        _SYNC_STORES[key].sync(root, key[3], files, refresh=False)
    except AssetUploadError as error:
        # A completed local checkpoint is more valuable than aborting GPU work
        # because GitHub temporarily rejects a Release asset. The next boundary
        # retries the immutable upload, while latest.pt remains resumable locally.
        print('GitHub 체크포인트 백업 지연; 로컬 학습은 계속합니다:', error, flush=True)
