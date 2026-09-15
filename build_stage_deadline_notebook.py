"""Build the user-checkpoint-first Colab workflow requested for the deadline."""
import json
from pathlib import Path


def make_notebook():
    cells = []
    def add(kind, source):
        item = dict(cell_type=kind, id=f'deadline-{len(cells)}', metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    add('markdown', '''# State + Stage ACT deadline workflow

이 노트북은 현재 사용할 **State Stage ACT 체크포인트**를 그대로 A baseline으로 삼아 다음을 같은 seed에서 비교합니다.

- A: 기존 Stage ACT temporal ensemble
- B: stage별 action buffer + stage/recovery 즉시 replan
- C: B + 학습된 gripper logit의 짧은 hysteresis

과거 EasyLab/MediumLab이 실제 Stage ACT였다는 사실은 방향을 정하는 근거로만 사용합니다. 기본 설정은 과거 best를 자동 복원하지 않습니다. 모델 재학습, stage 재정의, dependency 전체 upgrade도 하지 않습니다.

Colab에서 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택한 뒤 순서대로 실행하세요.
''')

    add('code', '''# 01 · 실험 설정: 현재 사용할 checkpoint 경로만 입력하세요.
PROJECT_REF = 'stage-act-chunk-compare'
USE_DRIVE = True
RUN_NAME = 'stage_act_deadline_v1'
TEAM = 'my-team'

# Drive를 쓸 경우 /content/drive/MyDrive/... 절대 경로를 적습니다.
# 비워 둔 난이도는 검사·평가·패키징에서 제외됩니다.
CHECKPOINTS = {
    'easy': '',
    'medium': '',
    'hard': '',
}

# checkpoint 옆 policy_config.json을 신뢰하려면 True.
# 없거나 실험 설정을 명시하려면 False로 두고 아래 값을 사용합니다.
USE_SIDECARS = True
RUNTIME_POLICY = dict(ensemble_window=4, temporal_decay=.25,
                      gate_threshold=.65, stage_threshold=.60)

# 선택 사항: 원본 state HDF5를 적으면 실제 shape/stage/gripper 분포도 다시 검사합니다.
DATA_PATHS = {'easy':'', 'medium':'', 'hard':''}

SMOKE_SEEDS = [61000]
COMPARE_SEEDS = list(range(62000, 62008))
MAX_STEPS = 200
STAGE_HORIZONS = dict(pick=2, carry=6, place=2, done=1)
GRIPPER_MARGIN = .5
GRIPPER_CONFIRM_STEPS = 2

# 과거 hash-pinned baseline 복원은 명시적으로 켤 때만 사용합니다.
USE_KNOWN_BASELINES = False
''')

    add('code', '''# 02 · 공식 simulator 고정 환경 + 프로젝트 코드
import importlib.util, json, os, subprocess, sys
from pathlib import Path

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_ROOT = Path('/content/drive/MyDrive/moveboxes_stage_deadline')
else:
    OUTPUT_ROOT = Path('/content/moveboxes_stage_deadline')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

PROJECT = Path('/content/moveBoxes_stage_deadline')
PROJECT_URL = 'https://github.com/SongYunu/moveBoxes.git'
if not PROJECT.exists():
    subprocess.run(['git','clone','--no-checkout',PROJECT_URL,str(PROJECT)],check=True)
remote = subprocess.check_output(['git','remote','get-url','origin'],cwd=PROJECT,text=True).strip()
if remote != PROJECT_URL:
    raise RuntimeError('Dedicated project directory has a different Git remote.')
subprocess.run(['git','fetch','--depth','1','origin',PROJECT_REF],cwd=PROJECT,check=True)
subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=PROJECT,check=True)
PROJECT_COMMIT = subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
sys.path[:0] = [str(PROJECT/'ver2/stages'),str(PROJECT/'ver2'),str(PROJECT)]

from colab_layout import DEFAULT_CONFIG
UPSTREAM = Path('/content/berlin-marso-hackathon')
if not UPSTREAM.exists():
    subprocess.run(['git','clone',DEFAULT_CONFIG['repo_url'],str(UPSTREAM)],check=True)
    subprocess.run(['git','checkout','--detach',DEFAULT_CONFIG['repo_commit']],cwd=UPSTREAM,check=True)
UPSTREAM_COMMIT = subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip()
if UPSTREAM_COMMIT != DEFAULT_CONFIG['repo_commit']:
    raise RuntimeError('Existing official starter has another revision; use a fresh UPSTREAM path.')

required = {'mani_skill':'mani-skill==3.0.1','sapien':'sapien==3.0.3',
            'hydra':'hydra-core','omegaconf':'omegaconf','gymnasium':'gymnasium',
            'transforms3d':'transforms3d','h5py':'h5py'}
missing = [package for module,package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
subprocess.run([sys.executable,'-m','pip','install','-e',str(UPSTREAM),'--no-deps'],check=True)
sys.path.insert(0,str(UPSTREAM))
os.environ['DISPLAY'] = ''
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch, mani_skill, sapien, warehouse_sort
assert torch.cuda.is_available(), 'T4 GPU runtime required'
print('project / official:',PROJECT_COMMIT,UPSTREAM_COMMIT)
print('GPU:',torch.cuda.get_device_name(0))
print('torch / ManiSkill / SAPIEN:',torch.__version__,getattr(mani_skill,'__version__','unknown'),
      getattr(sapien,'__version__','unknown'))
''')

    add('code', '''# 03 · 사용자가 고른 checkpoint를 고정하고 config 생성
from stage_compare_restore import restore_baselines

if USE_KNOWN_BASELINES:
    chosen = restore_baselines('/content/stage_act_known_baselines')
else:
    chosen = {}
    for level,path in CHECKPOINTS.items():
        if not path:
            continue
        checkpoint = Path(path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'{level} checkpoint not found: {checkpoint}')
        spec = dict(checkpoint=str(checkpoint))
        sidecar = checkpoint.parent/'policy_config.json'
        if USE_SIDECARS:
            if not sidecar.is_file():
                raise FileNotFoundError(f'{level} sidecar not found: {sidecar}')
            spec['policy_config_path'] = str(sidecar)
        else:
            spec['policy_config'] = dict(RUNTIME_POLICY)
        chosen[level] = spec
if not chosen:
    raise ValueError('CHECKPOINTS에 최소 한 난이도의 현재 Stage ACT checkpoint 경로를 입력하세요.')
for level,path in DATA_PATHS.items():
    if level in chosen and path:
        data = Path(path).expanduser().resolve()
        if not data.is_file():
            raise FileNotFoundError(f'{level} state HDF5 not found: {data}')
        chosen[level]['data'] = str(data)

COMMON = dict(levels=chosen,max_steps=MAX_STEPS,stage_horizons=STAGE_HORIZONS,
              gripper_margin=GRIPPER_MARGIN,gripper_confirm_steps=GRIPPER_CONFIRM_STEPS)
def write_config(name,seeds):
    output = OUTPUT_ROOT/name
    output.mkdir(parents=True,exist_ok=True)
    value = dict(COMMON,output=str(output),seeds=seeds)
    path = output/'config.json'
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')
    return path,output

AUDIT_CONFIG,AUDIT_OUTPUT = write_config(RUN_NAME+'_audit',COMPARE_SEEDS)
SMOKE_CONFIG,SMOKE_OUTPUT = write_config(RUN_NAME+'_smoke',SMOKE_SEEDS)
COMPARE_CONFIG,COMPARE_OUTPUT = write_config(RUN_NAME+'_compare',COMPARE_SEEDS)
print('levels:',list(chosen),'output:',COMPARE_OUTPUT)
''')

    add('code', '''# 04 · checkpoint/state/action/stage/reset 정적 검사
subprocess.run([sys.executable,str(PROJECT/'ver2/stages/stage_compare.py'),str(AUDIT_CONFIG),
                '--audit-only','--device','cuda'],cwd=UPSTREAM,check=True)
audit = json.loads((AUDIT_OUTPUT/'audit.json').read_text())
for level,row in audit.items():
    print(level,row['format'],row['model_config'],'checkpoint',row['checkpoint_sha256'][:12])
    if 'stage_counts' in row:
        print(' stage:',row['stage_counts'],'gripper:',row['stage_gripper_counts'])
print('Static checks passed.')
''')

    add('code', '''# 05 · 실제 simulator 1-seed smoke: A/B/C 모두 실행
subprocess.run([sys.executable,str(PROJECT/'ver2/stages/stage_compare.py'),str(SMOKE_CONFIG)],
               cwd=UPSTREAM,check=True)
from IPython.display import Markdown,display
display(Markdown((SMOKE_OUTPUT/'comparison.md').read_text()))
''')

    add('code', '''# 06 · 같은 8 seeds로 A/B/C 본 비교
# episode마다 Drive에 저장하므로 끊긴 뒤 같은 셀을 재실행하면 이어집니다.
subprocess.run([sys.executable,str(PROJECT/'ver2/stages/stage_compare.py'),str(COMPARE_CONFIG)],
               cwd=UPSTREAM,check=True)
display(Markdown((COMPARE_OUTPUT/'comparison.md').read_text()))
''')

    add('code', '''# 07 · 비교 결과에서 난이도별 후보 선택 후 별도 제출 후보 생성
# 동점이면 기존 A를 우선합니다. 직접 정하려면 {'easy':'B',...}처럼 적으세요.
MANUAL_SELECTION = {}
import shutil
from stage_compare import load_spec,variant_config

comparison = json.loads((COMPARE_OUTPUT/'comparison.json').read_text())
preference = {'A':2,'B':1,'C':0}
selected = {}
for level in chosen:
    if level in MANUAL_SELECTION:
        variant = MANUAL_SELECTION[level]
    else:
        variant = max(('A','B','C'),key=lambda v:(comparison['results'][level][v]['sort_accuracy'],preference[v]))
    if variant not in ('A','B','C'):
        raise ValueError(f'Invalid selection for {level}: {variant}')
    selected[level] = variant

CANDIDATE = OUTPUT_ROOT/(RUN_NAME+'_candidate')
CANDIDATE.mkdir(parents=True,exist_ok=True)
for source in ('stage_chunk_policy.py','stage_policy.py','stage_model.py','stage_schema.py'):
    shutil.copy2(PROJECT/'ver2/stages'/source,CANDIDATE/source)
shutil.copy2(PROJECT/'ver2/act_v2_model.py',CANDIDATE/'act_v2_model.py')

level_lines = []
selection_report = dict(project_commit=PROJECT_COMMIT,official_commit=UPSTREAM_COMMIT,
                        comparison=str(COMPARE_OUTPUT),selected={},automatic=not bool(MANUAL_SELECTION))
for level,variant in selected.items():
    checkpoint,base,metadata = load_spec(chosen[level],{'easy':54,'medium':72,'hard':90}[level])
    folder = CANDIDATE/'checkpoints'/level
    folder.mkdir(parents=True,exist_ok=True)
    target = folder/'model.pt'
    shutil.copy2(checkpoint,target)
    policy = variant_config(base,variant,COMMON)
    (folder/'policy_config.json').write_text(json.dumps(policy,indent=2),encoding='utf-8')
    level_lines.append(f'    {level}: {{ checkpoint: checkpoints/{level}/model.pt }}')
    selection_report['selected'][level] = dict(variant=variant,
        development_sort_accuracy=comparison['results'][level][variant]['sort_accuracy'],
        checkpoint_sha256=metadata['checkpoint_sha256'],policy_config=policy)

submission = 'team: '+json.dumps(TEAM)+'\\nstate:\\n  policy: stage_chunk_policy:load_policy\\n  levels:\\n'+'\\n'.join(level_lines)+'\\n'
(CANDIDATE/'submission.yaml').write_text(submission,encoding='utf-8')
(CANDIDATE/'selection.json').write_text(json.dumps(selection_report,indent=2),encoding='utf-8')

# Exported flat module layout + each real checkpoint/sidecar must load independently.
check = """import json,sys,torch\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom stage_chunk_policy import load_policy\nroot=Path(sys.argv[1])\nselection=json.loads((root/'selection.json').read_text())\ndims={'easy':54,'medium':72,'hard':90}\nfor level in selection['selected']:\n p=root/'checkpoints'/level/'model.pt'\n agent=load_policy(p,torch.zeros(1,dims[level]),SimpleNamespace(shape=(4,)),'cpu')\n a=agent.act(torch.zeros(1,dims[level]))\n assert a.shape==(1,4) and torch.isfinite(a).all() and a.abs().max()<=1\n agent.reset()\n print(level,selection['selected'][level]['variant'],'OK')\n"""
subprocess.run([sys.executable,'-c',check,str(CANDIDATE)],cwd=CANDIDATE,check=True)
archive = shutil.make_archive(str(CANDIDATE),'zip',CANDIDATE)
print('selected:',selected,'candidate:',archive)
''')

    add('code', '''# 08 · 결과와 후보 ZIP 다운로드
display(Markdown((COMPARE_OUTPUT/'comparison.md').read_text()))
from google.colab import files
files.download(str(CANDIDATE)+'.zip')
''')

    return dict(nbformat=4,nbformat_minor=5,cells=cells,
        metadata=dict(colab=dict(name='moveboxes_stage_deadline_colab.ipynb'),accelerator='GPU',
            kernelspec=dict(display_name='Python 3',language='python',name='python3')))


if __name__ == '__main__':
    notebook = make_notebook()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']),cell['id'],'exec')
    path = Path(__file__).parent/'notebooks/moveboxes_stage_deadline_colab.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(path)
