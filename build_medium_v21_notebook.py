"""Separate Medium v2.1 notebook: learn partial successes and weight faster successes."""
import json
from pathlib import Path
from build_medium_v2_notebook import CONFIG as V2_CONFIG, make_notebook as v2_notebook

CONFIG=dict(V2_CONFIG,run_name='moveboxes_medium_curriculum_v21',minimum_correct=3,
    deadline_episodes=32,deadline_max_attempts=128,pilot_episodes=4,
    block_iters=1000,max_blocks=12,plateau_blocks=4,efficiency_bonus=.5,
    tuning_seed_start=54000,test_seed_start=44000,eval_seed_start=74000,
    pilot_seed_start=122000,deadline_seed_start=123000)


def make_notebook(config=None):
    cfg=dict(CONFIG,**(config or {}));nb=v2_notebook(cfg)
    first=''.join(nb['cells'][0]['source'])
    first=first.replace('MEDIUM v2 CONFIG','MEDIUM v2.1 CONFIG')
    first=first.replace('199행동 안에 네 상자가 성공한 실제 시연만 사용','199행동 안에 3개 이상 분류한 실제 시연 · 네 상자 완주와 구분')
    first=first.replace('500회마다 평가, 최대 3,000회, 2구간 개선 없으면 종료','1,000회마다 평가, 최대 12,000회, 4구간 개선 없으면 종료')
    first=first.rsplit('}',1)[0]+f"    # 네 상자 완주 시연끼리만 빠른 시연 가중치를 최대 1+bonus배 · 부분 성공은 1배\n    'minimum_correct': {cfg['minimum_correct']!r},\n    'efficiency_bonus': {cfg['efficiency_bonus']!r},\n}}\n"
    nb['cells'][0]['source']=first.splitlines(keepends=True)
    bootstrap=''.join(nb['cells'][1]['source'])
    bootstrap=bootstrap.replace('from medium_v2 import MediumV2, source_bundle\nexperiment = MediumV2(CFG, source_bundle())',
        "for name in ('build_medium_v21_notebook','medium_v21'):\n"
        "    if name in sys.modules:\n        importlib.reload(sys.modules[name])\n"
        "from build_medium_v21_notebook import CONFIG as V21_DEFAULTS\n"
        "CFG = dict(V21_DEFAULTS, **USER_CONFIG)\nCFG['project_commit'] = PROJECT_COMMIT\n"
        "from medium_v21 import MediumV21, source_bundle\nexperiment = MediumV21(CFG, source_bundle())")
    nb['cells'][1]['source']=bootstrap.splitlines(keepends=True)
    replacements={5:('06 · 시연 시험 8회 · 3개 이상 성공도 학습에 채택','_ = experiment.collect_deadline("pilot")'),
        6:('07 · 학습 시연 32개 수집 · 최대 128시도, 시간 예산 후 같은 셀로 재개','_ = experiment.collect_deadline("collect")'),
        7:('08 · 최대 12,000회 보정 · 시연 부족이면 수집을 이어가고 학습 대기','_ = experiment.run_blocks()')}
    for index,(title,code) in replacements.items():nb['cells'][index]['source']=(f'# {title}\n{code}\n').splitlines(keepends=True)
    for i,c in enumerate(nb['cells']):c['metadata']['id']=f'medium-v21-{i+1}'
    nb['metadata']['colab']['name']='moveboxes_medium_v21_colab.ipynb'
    return nb


if __name__=='__main__':
    nb=make_notebook()
    for i,c in enumerate(nb['cells']):compile(''.join(c['source']),str(i),'exec')
    path=Path(__file__).parent/'notebooks/moveboxes_medium_v21_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path.name,len(nb['cells']),'code cells')
