import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
from github_store import GitHubStore,AssetUploadError


class PartsStore(GitHubStore):
    def __init__(self,db):
        super().__init__('owner/repo','run-test','local-test-token')
        self.db=db
        self.fail_snapshot=False
    def request(self,method,path,body=None):
        if method=='POST':
            release=dict(id=len(self.db)+1,tag_name=body['tag_name'],files={})
            self.db[body['tag_name']]=release
            return release
        if '/tags/' in path:
            return self.db.get(path.rsplit('/',1)[-1])
        release_id=int(path.split('/releases/')[1].split('/')[0])
        release=next(r for r in self.db.values() if r['id']==release_id)
        page=int(parse_qs(urlsplit(path).query)['page'][0])
        return [dict(name=k,size=len(v),url=f'{release_id}/{k}') for k,v in release['files'].items()][(page-1)*100:page*100]
    def _upload_once(self,path,name):
        if self.fail_snapshot and name.startswith('snapshot-'):
            raise RuntimeError('interrupted')
        release=self.db[self.release['tag_name']]
        if name in release['files']:
            raise AssetUploadError(422)
        release['files'][name]=Path(path).read_bytes()
        asset=dict(name=name,size=Path(path).stat().st_size,url=f'{release["id"]}/{name}')
        self.assets[name]=asset
        self.write_asset_count+=1
        return asset
    def download(self,asset,path):
        ident,name=asset['url'].split('/',1)
        release=next(r for r in self.db.values() if r['id']==int(ident))
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_bytes(release['files'][name])


class ReleasePartTests(unittest.TestCase):
    def fill(self,db):
        files=db['run-test']['files']
        for i in range(1000-len(files)):
            files[f'filler-{i}']=b'x'

    def test_full_release_rotates_and_new_snapshot_restores_old_and_new_blobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';(source/'easy').mkdir(parents=True)
            (source/'easy/model.pt').write_bytes(b'original-model')
            metric=source/'easy/metrics.json';metric.write_bytes(b'old-metric')
            db={};store=PartsStore(db);store.sync(source,'easy')
            self.fill(db)
            metric.write_bytes(b'new-metric')
            store.sync(source,'easy')
            self.assertEqual(len(db['run-test']['files']),1000)
            self.assertIn('run-test-part-002',db)
            restored=root/'restored'
            PartsStore(db).restore(restored)
            self.assertEqual((restored/'easy/model.pt').read_bytes(),b'original-model')
            self.assertEqual((restored/'easy/metrics.json').read_bytes(),b'new-metric')

    def test_interruption_in_new_part_preserves_last_complete_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'source/easy').mkdir(parents=True)
            model=root/'source/easy/model.pt';model.write_bytes(b'old')
            db={};store=PartsStore(db);store.sync(root/'source','easy');self.fill(db)
            model.write_bytes(b'new');store.fail_snapshot=True
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                store.sync(root/'source','easy')
            PartsStore(db).restore(root/'restore')
            self.assertEqual((root/'restore/easy/model.pt').read_bytes(),b'old')

    def test_stale_asset_list_reuses_concurrent_same_name_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x';path.write_bytes(b'weights')
            db={};one=PartsStore(db);two=PartsStore(db)
            one.load();two.load()
            first=one.upload(path,'same-name')
            second=two.upload(path,'same-name')
            self.assertEqual(first,second)
