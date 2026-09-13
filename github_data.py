"""Download versioned GitHub Release datasets and verify their SHA-256."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def download_archive(manifest_path, track='state', cache='/content/moveboxes_data_cache'):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    item = manifest['archives'][track]
    destination = Path(cache)/item['filename']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and file_hash(destination) == item['sha256']:
        print('검증된 데이터 캐시 재사용:', destination)
        return destination
    partial = destination.with_suffix(destination.suffix+'.part')
    request = urllib.request.Request(item['url'], headers={'User-Agent': 'moveBoxes-colab'})
    print(f"GitHub 데이터 다운로드: {track} ({item['bytes']/1024**2:.1f} MiB)", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as handle:
        while chunk := response.read(1024*1024):
            handle.write(chunk)
    if partial.stat().st_size != item['bytes'] or file_hash(partial) != item['sha256']:
        partial.unlink(missing_ok=True)
        raise ValueError('데이터 크기/해시 불일치. 다운로드를 다시 실행하세요.')
    os.replace(partial, destination)
    return destination
