"""Build the standalone evaluation-only Colab notebook; keep training notebooks intact."""
import json
from pathlib import Path


def make_notebook():
    cells = []
    def cell(kind, source):
        item = dict(cell_type=kind, id=f'compare-{len(cells)}', metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell('markdown', '''# State Stage ACT: 학습 없는 A/B/C 비교

A: 기존 best + temporal ensemble. B: stage별 chunk. C: B + 학습 gripper logit 필터.
기존 4개 stage와 가중치를 유지합니다. 고정 OPEN/CLOSE stage 제어기는 사용하지 않습니다.
T4 런타임에서 순서대로 실행하세요. 결과는 별도 폴더에 저장하며 자동 업로드/재학습하지 않습니다.
실제 성능 개선은 아직 측정하지 않았습니다. Easy 100%, Medium 40.625%, Hard 4.167%는 과거 서로 다른 평가 기록이며 이번 비교 결과가 아닙니다.
''')
    cell('code', '''# 01 · 코드 준비
import importlib.util, json, os, subprocess, sys
from pathlib import Path
PROJECT = Path('/content/moveBoxes_stage_compare')
PROJECT_REF = 'stage-act-chunk-compare'
if not PROJECT.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', PROJECT_REF, 'https://github.com/SongYunu/moveBoxes.git', str(PROJECT)], check=True)
print('Code commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip())
sys.path[:0] = [str(PROJECT/'ver2/stages'), str(PROJECT/'ver2'), str(PROJECT)]
''')
    cell('code', '''# 02 · 기존 프로젝트와 같은 simulator 버전. 설치되어 있으면 그대로 사용합니다.
from colab_layout import DEFAULT_CONFIG
UPSTREAM = Path('/content/berlin-marso-hackathon')
if not UPSTREAM.exists():
    subprocess.run(['git', 'clone', DEFAULT_CONFIG['repo_url'], str(UPSTREAM)], check=True)
    subprocess.run(['git', 'checkout', '--detach', DEFAULT_CONFIG['repo_commit']], cwd=UPSTREAM, check=True)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=UPSTREAM, text=True).strip()
if head != DEFAULT_CONFIG['repo_commit']:
    raise RuntimeError('Existing starter revision differs; set UPSTREAM to a fresh directory.')
packages = {'mani_skill': 'mani-skill==3.0.1', 'sapien': 'sapien==3.0.3',
            'hydra': 'hydra-core', 'omegaconf': 'omegaconf', 'gymnasium': 'gymnasium',
            'transforms3d': 'transforms3d', 'h5py': 'h5py'}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
os.environ['DISPLAY'] = ''
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
assert torch.cuda.is_available(), 'T4 GPU runtime required'
print('GPU:', torch.cuda.get_device_name(0))
''')
    cell('code', '''# 03 · 비교 설정 + 기존 GitHub Release에서 정확한 체크포인트만 복원
# 1 episode smoke 후 seeds를 8개로 늘릴 때 output도 새 폴더로 바꾸세요.
# Hard를 나중에 실행하려면 LEVELS에서 제외할 수 있습니다. Overall은 그때 N/A입니다.
LEVELS = ['easy', 'medium', 'hard']
SEEDS = list(range(61000, 61008))
OUTPUT = Path('/content/stage_compare_abc_v1')
from marso_github import github_token
from stage_compare_restore import restore_baselines
token = github_token()  # 기존 GH_TOKEN / Colab secret / 비공개 입력창. 복원은 read-only입니다.
specs = restore_baselines('/content/stage_compare_baselines', levels=LEVELS, token=token)
config = dict(output=str(OUTPUT), levels=specs, seeds=SEEDS, max_steps=200,
              stage_horizons=dict(pick=2, carry=6, place=2, done=1),
              gripper_margin=.5, gripper_confirm_steps=2)
OUTPUT.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = OUTPUT/'config.json'
CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding='utf-8')
print('Prepared:', CONFIG_PATH)
''')
    cell('code', '''# 04 · import / 체크포인트 / state shape / action / reset sanity + 동일 seed A/B/C
# 완료 episode마다 저장합니다. 같은 코드·설정으로 다시 실행하면 이어서 평가합니다.
subprocess.run([sys.executable, str(PROJECT/'ver2/stages/stage_compare.py'), str(CONFIG_PATH)], cwd=UPSTREAM, check=True)
''')
    cell('code', '''# 05 · 결과 확인 / 다운로드. 기존 best와 policy_config는 변경하지 않습니다.
from IPython.display import Markdown, display
display(Markdown((OUTPUT/'comparison.md').read_text()))
import shutil
archive = shutil.make_archive(str(OUTPUT)+'_results', 'zip', OUTPUT)
from google.colab import files
files.download(archive)
''')
    return dict(nbformat=4, nbformat_minor=5, cells=cells,
        metadata=dict(colab=dict(name='moveboxes_stage_compare_colab.ipynb'), accelerator='GPU',
            kernelspec=dict(display_name='Python 3', language='python', name='python3')))


if __name__ == '__main__':
    notebook = make_notebook()
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell-{i}', 'exec')
    path = Path(__file__).parent/'notebooks/moveboxes_stage_compare_colab.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(path)
