"""Separate notebook; preserve the first unified experiment and its results."""
import json
from pathlib import Path
from build_unified_notebook import CONFIG as BASE,make_notebook as base_notebook

CONFIG=dict(BASE,run_name='moveboxes_unified_focus_v2',easy_source_run_name='moveboxes_easy_lab_v1',
    focus_fraction=.7,plateau_blocks=3,max_blocks=dict(easy=6,medium=12,hard=12),pass_rate=dict(easy=1.,medium=.6,hard=.5))


def make_notebook(config=None):
    cfg=dict(CONFIG,**(config or {}));nb=base_notebook(cfg)
    first=''.join(nb['cells'][0]['source']).replace('저장 / 초기 Medium 모델','저장 / 초기 Easy 최고 모델')
    first=first.replace("    'medium_source_run_name': "+repr(cfg['medium_source_run_name'])+",", "    'easy_source_run_name': "+repr(cfg['easy_source_run_name'])+",")
    first=first.rsplit('}',1)[0]+f"    'focus_fraction': {cfg['focus_fraction']!r},\n}}\n"
    nb['cells'][0]['source']=first.splitlines(keepends=True)
    code=''.join(nb['cells'][1]['source'])
    code=code.replace("('build_unified_notebook','unified_policy','unified_data','unified_experiment')", "('build_unified_notebook','unified_policy','unified_data','unified_experiment','unified_v2','build_unified_v2_notebook')")
    code=code.replace('from build_unified_notebook import CONFIG as UNIFIED_DEFAULTS','from build_unified_v2_notebook import CONFIG as UNIFIED_DEFAULTS')
    code=code.replace('from unified_experiment import UnifiedExperiment, source_bundle\nexperiment = UnifiedExperiment(CFG, source_bundle())',
                      'from unified_v2 import UnifiedV2, source_bundle\nexperiment = UnifiedV2(CFG, source_bundle())')
    nb['cells'][1]['source']=code.splitlines(keepends=True)
    nb['cells'][4]['source']=['# 05 · 데이터 / 기존 Easy 최고 모델 / 공유 정책 복원\n','experiment.prepare()\n']
    nb['cells'][6]['source']=['# 07 · 원본 Easy와 변환 모델 비교 → 통과하면 재학습 생략 / 실패 부분 연속 보정\n','experiment.train_level("easy")\n']
    for i in (7,9,11):nb['cells'][i]['source']=[line.replace('experiment.test(', '_ = experiment.test(') for line in nb['cells'][i]['source']]
    nb['metadata']['colab']['name']='moveboxes_unified_v2_colab.ipynb'
    return nb


if __name__=='__main__':
    nb=make_notebook()
    for c in nb['cells']:compile(''.join(c['source']),'cell','exec')
    path=Path(__file__).resolve().parents[1]/'notebooks/moveboxes_unified_v2_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8');print(path)
