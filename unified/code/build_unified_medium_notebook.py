import json
from pathlib import Path
from build_unified_v2_notebook import CONFIG as BASE,make_notebook as base_notebook

CONFIG=dict(BASE,run_name='moveboxes_unified_medium_v3',unified_source_run_name='moveboxes_unified_focus_v2',
    demonstration_source_run_name='moveboxes_medium_zfocus_v22',easy_check_interval=3,
    block_iters=1000,max_blocks=dict(easy=1,medium=20,hard=1),replay_fraction=.3,focus_fraction=.7)

def make_notebook(config=None):
    cfg=dict(CONFIG,**(config or {}));nb=base_notebook(cfg)
    first=''.join(nb['cells'][0]['source'])
    first=first.replace('# 01 · T4 / 공유 모델 하나 / Easy → Medium → Hard','# 01 · T4 / v2 Easy PT에서 Medium 집중 학습')
    first=first.replace('단계별 통과와 반복 예산','반복 예산 / Medium 실행에서는 이전 단계 통과·plateau 조건 미사용')
    first=first.rsplit('}',1)[0]+''.join(f'    {k!r}: {cfg[k]!r},\n' for k in ('unified_source_run_name','demonstration_source_run_name','easy_check_interval'))+'}\n'
    nb['cells'][0]['source']=first.splitlines(keepends=True)
    code=''.join(nb['cells'][1]['source'])
    code=code.replace("'unified_v2','build_unified_v2_notebook')", "'unified_v2','build_unified_v2_notebook','unified_medium','build_unified_medium_notebook')")
    code=code.replace('from build_unified_v2_notebook import CONFIG as UNIFIED_DEFAULTS','from build_unified_medium_notebook import CONFIG as UNIFIED_DEFAULTS')
    code=code.replace('from unified_v2 import UnifiedV2, source_bundle\nexperiment = UnifiedV2(CFG, source_bundle())',
        'from unified_medium import UnifiedMedium, source_bundle\nexperiment = UnifiedMedium(CFG, source_bundle())')
    nb['cells'][1]['source']=code.splitlines(keepends=True);nb['cells']=nb['cells'][:4]
    tasks=[('05 · v2 Easy PT 그대로 복원 / 기존 Medium 성공·부분 성공 시연 가져오기','experiment.prepare()'),
        ('06 · 학습 전 Medium 개발 점수 / 시간 확인','_state, _checkpoint = experiment._current()\n_baseline = experiment._evaluate("medium", _checkpoint, "medium_initial")\nprint("Medium 초기 분류:", _baseline["sort_accuracy"], "평가 시간(초):", _baseline["elapsed_seconds"])'),
        (f"07 · Medium {cfg['block_iters']*cfg['max_blocks']['medium']:,}회 연속 학습 / {cfg['block_iters']:,}회마다 부분 보강 / Easy 점수로 중단하지 않음",'experiment.train_medium()'),
        ('08 · Medium 최고 모델 별도 테스트와 영상','_ = experiment.test("medium")'),
        ('09 · 같은 Medium 모델의 Easy 성능 확인 / 보관된 Easy 원본과는 별개','_ = experiment.test("easy")'),
        ('10 · Medium 선택 기록','_ = experiment.report()'),
        ('11 · Medium 최종 100회 평가 / 선택 사항','experiment.final_evaluation()'),
        ('12 · Medium 최고 모델 패키지','_ = experiment.package()')]
    for i,(title,body) in enumerate(tasks,5):
        nb['cells'].append(dict(cell_type='code',id=f'medium-v3-{i}',metadata={},execution_count=None,outputs=[],source=(f'# {title}\n{body}\n').splitlines(keepends=True)))
    nb['metadata']['colab']['name']='moveboxes_unified_medium_colab.ipynb';return nb

if __name__=='__main__':
    nb=make_notebook()
    for c in nb['cells']:compile(''.join(c['source']),'cell','exec')
    path=Path(__file__).resolve().parents[1]/'notebooks/moveboxes_unified_medium_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8');print(path)
