import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'hard/code'),str(ROOT/'ver2/stages'),str(ROOT/'ver2'),str(ROOT)]
from hard_lab import HardLab,source_bundle
from build_hard_notebook import CONFIG,make_notebook


class HardLayoutTests(unittest.TestCase):
    def test_notebook_contains_hard_only_and_bootstraps_after_move(self):
        nb=make_notebook();self.assertEqual(len(nb['cells']),12)
        for i,cell in enumerate(nb['cells']):
            code=''.join(cell['source']);compile(code,str(i),'exec')
            if i>=5:
                self.assertNotIn('"easy"',code);self.assertNotIn('"medium"',code)
        self.assertIn("PROJECT/'hard'/'code'",''.join(nb['cells'][1]['source']))
        self.assertIn('stage_train.py',source_bundle())

    def test_bootstrap_creates_hard_facade(self):
        nb=make_notebook();scope={};exec(''.join(nb['cells'][0]['source']),scope)
        scope['CFG'].update(project_dir=str(ROOT),profile='smoke')
        def git(command,**kwargs):return 'https://github.com/SongYunu/moveBoxes.git\n' if 'remote' in command else 'a'*40+'\n'
        previous=list(sys.path)
        try:
            with patch('subprocess.run'),patch('subprocess.check_output',side_effect=git),\
                    patch('importlib.reload',side_effect=lambda m:m),contextlib.redirect_stdout(io.StringIO()):
                exec(''.join(nb['cells'][1]['source']),scope)
            self.assertIsInstance(scope['experiment'],HardLab)
            with self.assertRaises(ValueError):scope['experiment']._hard('medium')
        finally:sys.path[:]=previous


if __name__=='__main__':unittest.main()
