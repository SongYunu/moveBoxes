"""Build two backend notebooks and one combined code-only Colab notebook."""
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'foundation/notebooks'


def cell(source):
    return dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=[line+'\n' for line in source.rstrip().splitlines()])


def config(backend):
    if backend=='smol':
        specific="""    'backend': 'smol',
    'run_name': 'moveboxes_smolvla_rgb_v1',
    'micro_batch': 1,
    'accumulate': 3,
    'chunk': 8,
    'execute_steps': 2,
    'inference_steps': 5,
    'image_size': 256,
    'lr': 1e-4,
    'updates': 6000,
    'eval_interval': 2000,"""
    else:
        specific="""    'backend': 'octo',
    'run_name': 'moveboxes_octo_small_rgb_v1',
    'micro_batch': 3,
    'accumulate': 1,
    'chunk': 4,
    'execute_steps': 2,
    'inference_steps': 20,
    'image_size': 256,
    'lr': 3e-5,
    'updates': 6000,
    'eval_interval': 2000,"""
    return f"""# 01 · {backend.upper()} 공동 학습 설정 (Easy + Medium + Hard RGB)
CFG = {{
{specific}
    'smoke_updates': 3,
    'dev_episodes': 2,
    'test_episodes': 8,
    'max_steps': 200,
    'seed': 42,
    'device': 'cuda',
    'github_repository': 'SongYunu/moveBoxes',
    'project_ref': 'main',
    'project_dir': '/content/moveBoxes_{backend}',
    'output_root': '/content/moveboxes_results',
    'env_root': '/content/moveboxes_envs_{backend}',
    'data_dir': '/content/moveboxes_rgb_data',
    'download_cache': '/content/moveboxes_data_cache',
    'simulator_repo': '/content/berlin-marso-foundation-{backend}',
    'simulator_commit': '6048f33217f26ae39009a812f53c81171517f393',
}}

assert CFG['micro_batch'] * CFG['accumulate'] % 3 == 0
print('백엔드:', CFG['backend'], '· 실효 배치:', CFG['micro_batch'] * CFG['accumulate'])
print('한 모델에 Easy·Medium·Hard를 같은 비율로 넣습니다.')"""


BOOTSTRAP="""# 02 · GitHub 코드 로드 (기존 폴더가 있으면 최신 project_ref로 갱신)
import importlib, os, subprocess, sys
from pathlib import Path

PROJECT = Path(CFG['project_dir'])
if not (PROJECT/'.git').exists():
    subprocess.run(['git','clone','--filter=blob:none','https://github.com/SongYunu/moveBoxes.git',str(PROJECT)],check=True)
subprocess.run(['git','fetch','origin',CFG['project_ref']],cwd=PROJECT,check=True)
subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=PROJECT,check=True)
CFG['project_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
sys.path.insert(0,str(PROJECT))
importlib.invalidate_caches()
sys.modules.pop('foundation.code.foundation_experiment',None)
from foundation.code.foundation_experiment import FoundationExperiment
experiment=FoundationExperiment(CFG)
print('사용 코드:',CFG['project_commit'])"""


CONNECT="""# 03 · GitHub 결과 복원/백업 연결
import getpass, os
if not os.environ.get('GH_TOKEN'):
    try:
        from google.colab import userdata
        os.environ['GH_TOKEN']=userdata.get('GH_TOKEN')
    except Exception:
        token=getpass.getpass('GitHub fine-grained token (Contents: read/write): ').strip()
        if not token: raise RuntimeError('결과 복원/저장용 GitHub token이 필요합니다.')
        os.environ['GH_TOKEN']=token
# 개인 비공개 사본에서는 위 분기 대신 os.environ['GH_TOKEN']='토큰'도 가능하지만 공개 커밋은 금지합니다.
experiment.connect()
experiment.report()"""


def build(backend):
    cells=[config(backend),BOOTSTRAP,CONNECT,
        "# 04 · 모델/시뮬레이터 격리 환경 설치\nexperiment.install()",
        "# 05 · RGB 데이터 다운로드·해시 검증·600개 시연 분할\nexperiment.prepare()",
        "# 06 · T4 메모리/학습/정책/시뮬레이터 연결 실제 확인\n# 이 셀이 성공하기 전에는 긴 학습을 시작하지 않습니다.\nexperiment.check()",
        "# 07 · Easy + Medium + Hard 공동 학습 (중간 최고 모델은 GitHub Release에 백업)\nexperiment.train()",
        "# 08 · 같은 최고 모델로 Easy 별도 테스트\neasy_result=experiment.test('easy')",
        "# 09 · 같은 최고 모델로 Medium 별도 테스트\nmedium_result=experiment.test('medium')",
        "# 10 · 같은 최고 모델로 Hard 별도 테스트\nhard_result=experiment.test('hard')",
        "# 11 · 공동 학습 이력과 세 난이도 결과 확인\nexperiment.report()"]
    notebook=dict(cells=[cell(c) for c in cells],metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.12'),colab=dict(name=f'moveboxes_{backend}_colab.ipynb',provenance=[])),nbformat=4,nbformat_minor=5)
    name='moveboxes_smolvla_colab.ipynb' if backend=='smol' else 'moveboxes_octo_colab.ipynb'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(notebook,indent=1,ensure_ascii=False),encoding='utf-8')
    print(OUT/name)


def build_combined():
    combined_config="""# 01 · 통합 설정: 한 T4에서 두 공개 모델을 순서대로 비교
BACKENDS = ['smol', 'octo']
COMMON = {
    'smoke_updates': 3,
    'dev_episodes': 2,
    'test_episodes': 8,
    'max_steps': 200,
    'seed': 42,
    'device': 'cuda',
    'github_repository': 'SongYunu/moveBoxes',
    'project_ref': 'main',
    'project_dir': '/content/moveBoxes_foundation',
    'output_root': '/content/moveboxes_results',
    'env_root': '/content/moveboxes_envs_foundation',
    'data_dir': '/content/moveboxes_rgb_data',
    'download_cache': '/content/moveboxes_data_cache',
    'simulator_repo': '/content/berlin-marso-foundation',
    'simulator_commit': '6048f33217f26ae39009a812f53c81171517f393',
    'notebook_name': 'moveboxes_foundation_both_colab.ipynb',
}
MODEL = {
    'smol': dict(run_name='moveboxes_smolvla_rgb_v1',micro_batch=1,accumulate=3,
                 chunk=8,execute_steps=2,inference_steps=5,image_size=256,
                 lr=1e-4,updates=6000,eval_interval=2000),
    'octo': dict(run_name='moveboxes_octo_small_rgb_v1',micro_batch=3,accumulate=1,
                 chunk=4,execute_steps=2,inference_steps=20,image_size=256,
                 lr=3e-5,updates=6000,eval_interval=2000),
}
CFGS = {name: dict(COMMON, backend=name, **MODEL[name]) for name in BACKENDS}
for name,cfg in CFGS.items():
    assert cfg['micro_batch']*cfg['accumulate']%3==0
    print(name,'· 실효 배치',cfg['micro_batch']*cfg['accumulate'],'·',cfg['updates'],'updates')
print('두 모델을 섞어 평균내지 않고 각각 공동 학습한 뒤 0.2/0.3/0.5 점수로 비교합니다.')"""
    combined_bootstrap=BOOTSTRAP.replace("CFG['project_dir']","COMMON['project_dir']").replace("CFG['project_ref']","COMMON['project_ref']")
    combined_bootstrap=combined_bootstrap.replace("CFG['project_commit']=", "COMMON['project_commit']=")
    combined_bootstrap=combined_bootstrap.replace("FoundationExperiment(CFG)","{name:FoundationExperiment(dict(cfg,project_commit=COMMON['project_commit'])) for name,cfg in CFGS.items()}")
    combined_bootstrap=combined_bootstrap.replace("CFG['project_commit']","COMMON['project_commit']").replace("experiment=", "experiments=")
    combined_connect="""# 03 · 두 결과 Release 복원/백업 연결
import getpass, os
if not os.environ.get('GH_TOKEN'):
    try:
        from google.colab import userdata
        os.environ['GH_TOKEN']=userdata.get('GH_TOKEN')
    except Exception:
        token=getpass.getpass('GitHub fine-grained token (Contents: read/write): ').strip()
        if not token: raise RuntimeError('결과 복원/저장용 GitHub token이 필요합니다.')
        os.environ['GH_TOKEN']=token

def run_each(method,*args):
    results,errors={},{}
    for name in BACKENDS:
        print('\\n========',name.upper(),method,'========')
        try: results[name]=getattr(experiments[name],method)(*args)
        except Exception as exc:
            errors[name]=repr(exc);print(name,'실패:',repr(exc))
    if len(errors)==len(BACKENDS): raise RuntimeError(method+' 단계에서 두 백엔드가 모두 실패했습니다: '+repr(errors))
    return results,errors

run_each('connect')"""
    cells=[combined_config,combined_bootstrap,combined_connect,
        "# 04 · SmolVLA(Python 3.12), Octo(Python 3.10), 시뮬레이터 환경 설치\ninstall_results,install_errors=run_each('install')",
        "# 05 · RGB 데이터는 한 번 내려받고, 두 실험에서 같은 분할을 각각 검증\ndata_results,data_errors=run_each('prepare')",
        "# 06 · 각 공개 초기값으로 3 update + Easy 1회 실제 동작 확인\n# 한 모델이 T4에서 실패해도 다른 모델 확인은 계속합니다.\ncheck_results,check_errors=run_each('check')",
        "# 07 · 두 모델 순차 공동 학습; 2,000 update마다 각각 GitHub에 최고 adapter 저장\ntrain_results,train_errors=run_each('train')",
        "# 08 · 각 모델의 최고 adapter로 Easy 테스트\neasy_results,easy_errors=run_each('test','easy')",
        "# 09 · 각 모델의 최고 adapter로 Medium 테스트\nmedium_results,medium_errors=run_each('test','medium')",
        "# 10 · 각 모델의 최고 adapter로 Hard 테스트\nhard_results,hard_errors=run_each('test','hard')",
        """# 11 · 모델별 세 난이도 가중 개발 점수 비교
import json
comparison={}
for name,experiment in experiments.items():
    state=experiment.report();best=state.get('best')
    comparison[name]=None if not best else dict(weighted_score=best['score'],scores=best['scores'],folder=best['folder'])
valid={k:v for k,v in comparison.items() if v is not None}
print(json.dumps(comparison,indent=2,ensure_ascii=False))
if valid:
    winner=max(valid,key=lambda k:valid[k]['weighted_score'])
    print('현재 고정 개발 seed 기준 후보:',winner,valid[winner])
else:
    print('아직 비교할 완성 checkpoint가 없습니다.')"""]
    notebook=dict(cells=[cell(c) for c in cells],metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.12'),colab=dict(name='moveboxes_foundation_both_colab.ipynb',provenance=[])),nbformat=4,nbformat_minor=5)
    path=OUT/'moveboxes_foundation_both_colab.ipynb';OUT.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False),encoding='utf-8');print(path)


if __name__=='__main__':
    build('smol');build('octo');build_combined()
