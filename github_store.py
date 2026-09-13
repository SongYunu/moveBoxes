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

    def load(self, create=True):
        base = '/repos/'+self.repository+'/releases'
        self.release = self.request('GET', base+'/tags/'+self.tag)
        if self.release is None:
            if not create:
                return False
            self.release = self.request('POST', base, dict(tag_name=self.tag, name=self.tag,
                body='Colab experiment snapshots. Each snapshot is published after all referenced files are uploaded.',
                draft=False, prerelease=True, make_latest='false'))
        self.assets = {}
        page = 1
        while True:
            items = self.request('GET', base+f"/{self.release['id']}/assets?per_page=100&page={page}")
            if items is None:
                raise RuntimeError('Cannot read release assets')
            self.assets.update({item['name']: item for item in items})
            if len(items) < 100:
                break
            page += 1
        return True

    def upload(self, path, name):
        path = Path(path)
        if name in self.assets:
            if self.assets[name]['size'] != path.stat().st_size:
                raise RuntimeError('Existing immutable asset has an unexpected size')
            return self.assets[name]
        if self.release is None:
            self.load()
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
                raise RuntimeError(f'GitHub asset upload failed ({response.status}); local files are retained.')
            asset = json.loads(body)
            self.assets[name] = asset
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
    _SYNC_STORES[key].sync(Path(key[2]), key[3], refresh=False)
