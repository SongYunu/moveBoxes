"""Hard-only T4 notebook using the shared stage/recovery implementation."""
import json
import sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from build_stage_notebook import CONFIG as STAGE_CONFIG, make_notebook as stage_notebook

CONFIG=dict(STAGE_CONFIG,run_name='moveboxes_hard_stage_v1',recovery_episodes=24,
    recovery_max_attempts=96,recovery_seed_start=140000,
    tuning_seed_start=58000,test_seed_start=48000,eval_seed_start=78000)


def make_notebook(config=None):
    cfg=dict(CONFIG,**(config or {}));nb=stage_notebook(cfg)
    first=''.join(nb['cells'][0]['source']).replace('단계 ACT CONFIG · 기존 ACT/DP와 별도 결과','HARD 전용 단계 ACT CONFIG')
    nb['cells'][0]['source']=first.splitlines(keepends=True)
    bootstrap=''.join(nb['cells'][1]['source']).replace(
        'from stage_experiment import StageExperiment, source_bundle\nexperiment = StageExperiment(CFG, source_bundle())',
        "sys.path.insert(0, str(PROJECT/'hard'/'code'))\n"
        "for name in ('build_hard_notebook','hard_lab'):\n"
        "    if name in sys.modules:\n        importlib.reload(sys.modules[name])\n"
        "from build_hard_notebook import CONFIG as HARD_DEFAULTS\n"
        "CFG = dict(HARD_DEFAULTS, **CFG)\n"
        "from hard_lab import HardLab, source_bundle\nexperiment = HardLab(CFG, source_bundle())")
    nb['cells'][1]['source']=bootstrap.splitlines(keepends=True)
    nb['cells']=nb['cells'][:5]
    tasks=[('06 · 100회 동작 확인', 'experiment.smoke("hard")'),
        ('07 · Hard 성공/복구 시연 수집 · 중단하면 같은 셀로 재개','experiment.collect("hard")'),
        ('08 · Hard 단계 ACT 학습 · 체크포인트 자동 저장','experiment.train("hard")'),
        ('09 · 8회 테스트 + 영상 + 2회 행동 기록','experiment.test("hard")\nexperiment.diagnose("hard")'),
        ('10 · 설정 비교 + 최종 100회 평가','experiment.evaluate("hard")'),
        ('11 · 저장 결과 보기','experiment.show_results()'),
        ('12 · 학습 정책 패키징','experiment.package()')]
    for number,(title,body) in enumerate(tasks,6):
        nb['cells'].append(dict(cell_type='code',execution_count=None,outputs=[],metadata={'id':f'hard-{number}'},
            source=(f'# {title}\n{body}\n').splitlines(keepends=True)))
    nb['metadata']['colab']['name']='moveboxes_hard_colab.ipynb'
    return nb


if __name__=='__main__':
    nb=make_notebook()
    for i,c in enumerate(nb['cells']):compile(''.join(c['source']),str(i),'exec')
    path=Path(__file__).parents[1]/'notebooks/moveboxes_hard_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path.name,len(nb['cells']),'code cells')
