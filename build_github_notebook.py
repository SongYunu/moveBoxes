"""Build the readable Colab entrypoint that loads code from moveBoxes."""
import copy
import json
from pathlib import Path
from colab_next_pick_layout import NEXT_PICK_CONFIG, make_notebook as next_pick_notebook

CONFIG = copy.deepcopy(NEXT_PICK_CONFIG)
CONFIG.update(
    run_name='moveboxes_next_pick_v02',
    output_root='/content/moveboxes_runs',
    data_source='/content/moveboxes_data_cache/marso_state_data.zip',
    project_dir='/content/moveBoxes',
    project_ref='main',
    github_repository='SongYunu/moveBoxes',
    download_cache='/content/moveboxes_data_cache',
)


def make_notebook(config=None):
    cfg = copy.deepcopy(CONFIG if config is None else config)
    nb = next_pick_notebook({}, cfg)
    config_text = ''.join(nb['cells'][0]['source'])
    config_text = config_text.replace('저장 경로 · 기존 결과를 불러오려면 같은 run_name',
        'GitHub 결과 · 같은 run_name은 저장 모델 재사용, 새 학습은 새 run_name')
    extra = ['    # GitHub · 토큰은 CONFIG에 쓰지 않고 Colab 보안 비밀 GH_TOKEN에 등록']
    for key in ('github_repository', 'project_dir', 'project_ref', 'download_cache'):
        extra.append(f'    "{key}": {cfg[key]!r},')
    config_text = config_text.rsplit('}', 1)[0]+'\n'+'\n'.join(extra)+'\n}\n'
    nb['cells'][0]['source'] = config_text.splitlines(keepends=True)
    bootstrap = '''# 02 · GitHub 코드 불러오기 (데이터·결과를 위해 Drive를 마운트하지 않습니다)
import importlib, os, subprocess, sys
from pathlib import Path

PROJECT = Path(CFG['project_dir'])
URL = 'https://github.com/'+CFG['github_repository']+'.git'
if not PROJECT.exists():
    subprocess.run(['git', 'clone', '--depth', '1', URL, str(PROJECT)], check=True)
else:
    remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=PROJECT, text=True).strip()
    if remote != URL:
        raise RuntimeError('기존 프로젝트 폴더가 다른 저장소입니다. project_dir를 새 경로로 바꾸세요.')
subprocess.run(['git', 'fetch', '--depth', '1', 'origin', CFG['project_ref']], cwd=PROJECT, check=True)
subprocess.run(['git', 'checkout', '--detach', 'FETCH_HEAD'], cwd=PROJECT, check=True)
CFG['project_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
sys.path.insert(0, str(PROJECT))
# A fresh notebook run should not retain a previously imported project module.
for name in ('marso_experiment', 'marso_train_test', 'next_pick_sampling', 'next_pick_diagnostics',
             'marso_next_pick', 'github_store', 'github_data', 'marso_github'):
    if name in sys.modules:
        importlib.reload(sys.modules[name])
from marso_github import GitHubExperiment, source_bundle
experiment = GitHubExperiment(CFG, source_bundle())
print('사용 코드:', CFG['project_commit'])
'''
    nb['cells'][1]['source'] = bootstrap.splitlines(keepends=True)
    nb['cells'][1]['metadata'] = {}
    connect = '''# 03 · GitHub 인증 / 저장된 결과 복원
# 왼쪽 열쇠 아이콘 → GH_TOKEN 추가 → 노트북 액세스 허용.
# Fine-grained token: SongYunu/moveBoxes → Contents: Read and write.
# 이 저장소는 공개이므로 여기에 올린 모델·로그·영상도 공개됩니다.
from google.colab import userdata
try:
    os.environ['GH_TOKEN'] = userdata.get('GH_TOKEN')
except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
    raise RuntimeError('Colab 보안 비밀에 GH_TOKEN을 등록하고 노트북 액세스를 허용하세요.') from None
experiment.connect()
experiment.show_results()
'''
    nb['cells'][2]['source'] = connect.splitlines(keepends=True)
    for cell in nb['cells'][3:]:
        text = ''.join(cell['source']).replace('Drive', 'GitHub')
        text = text.replace('데이터 준비', 'GitHub 데이터 다운로드·검증')
        cell['source'] = text.splitlines(keepends=True)
    nb['cells'][-1]['source'].append('\nprint("원격 결과:", "https://github.com/"+CFG["github_repository"]+"/releases")\n')
    nb['metadata']['colab']['name'] = 'moveboxes_colab.ipynb'
    return nb


if __name__ == '__main__':
    notebook = make_notebook()
    for i, cell in enumerate(notebook['cells']):
        compile(''.join(cell['source']), f'cell-{i}', 'exec')
    path = Path(__file__).parent/'notebooks/moveboxes_colab.ipynb'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    print(path.name, len(notebook['cells']), 'code cells')
