import json
from pathlib import Path
from build_unified_notebook import CONFIG as BASE,make_notebook as base_notebook

CONFIG=dict(BASE,run_name='moveboxes_medium_block02_easy_probe',test_seed_start=330000,test_episodes=8,
    project_dir='/content/moveBoxes_probe',repo_dir='/content/berlin-marso-probe')

def make_notebook(config=None):
    cfg=dict(CONFIG,**(config or {}));nb=base_notebook(cfg)
    keys=('run_name','profile','github_repository','project_dir','project_ref','output_root','repo_dir','repo_url','repo_commit','packages','test_seed_start','test_episodes','test_record_video')
    nb['cells'][0]['source']=('# 01 · 기존 Medium block_02.pt 평가만 / 학습하지 않음\nCFG = {\n'+''.join(f'    {k!r}: {cfg[k]!r},\n' for k in keys)+'}\n').splitlines(keepends=True)
    code=''.join(nb['cells'][1]['source']).replace('from build_unified_notebook import CONFIG as UNIFIED_DEFAULTS','from build_medium_probe_notebook import CONFIG as UNIFIED_DEFAULTS')
    code=code.replace('from unified_experiment import UnifiedExperiment, source_bundle\nexperiment = UnifiedExperiment(CFG, source_bundle())',
        "for name in ('probe_medium_policy','probe_medium','build_medium_probe_notebook'):\n    if name in sys.modules: importlib.reload(sys.modules[name])\nfrom probe_medium import MediumProbe, source_bundle\nexperiment = MediumProbe(CFG, source_bundle())")
    nb['cells'][1]['source']=code.splitlines(keepends=True);nb['cells']=nb['cells'][:4]
    for i,(title,body) in enumerate([
        ('05 · 예전 Medium 40.6% 모델 원본 PT 복원 / 해시 검사','experiment.prepare()'),
        ('06 · 그 Medium 모델 그대로 Easy 8회 + 영상','_ = experiment.test("easy")'),
        ('07 · 같은 원본 모델로 Medium 8회 + 영상 / 비교용 선택 사항','_ = experiment.test("medium")')],5):
        nb['cells'].append(dict(cell_type='code',id=f'probe-{i}',metadata={},execution_count=None,outputs=[],source=(f'# {title}\n{body}\n').splitlines(keepends=True)))
    nb['metadata']['colab']['name']='moveboxes_medium_block02_easy_test.ipynb';return nb

if __name__=='__main__':
    nb=make_notebook()
    for c in nb['cells']:compile(''.join(c['source']),'cell','exec')
    path=Path(__file__).resolve().parents[1]/'notebooks/moveboxes_medium_block02_easy_test.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8');print(path)
