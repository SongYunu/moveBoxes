"""Separate T4 notebook for Medium pickup-height correction."""
import json
from pathlib import Path
from build_medium_v21_notebook import CONFIG as V21_CONFIG, make_notebook as v21_notebook

CONFIG=dict(V21_CONFIG,run_name='moveboxes_medium_zfocus_v22',source_run_name='moveboxes_medium_curriculum_v21',
    lr=1e-5,block_iters=500,max_blocks=8,plateau_blocks=4,ensemble_candidates=[1],ensemble_window=1,
    pick_z_weight=3.,contact_z_weight=6.,contact_gripper_weight=3.,
    tuning_seed_start=56000,test_seed_start=46000,eval_seed_start=76000)


def make_notebook(config=None):
    cfg=dict(CONFIG,**(config or {}));nb=v21_notebook(cfg)
    first=''.join(nb['cells'][0]['source']).replace('MEDIUM v2.1 CONFIG','MEDIUM v2.2 CONFIG')
    first=first.replace('1,000회마다 평가, 최대 12,000회, 4구간 개선 없으면 종료',
        '500회마다 평가, 최대 4,000회, 4구간 개선 없으면 종료')
    first=first.replace('시연 · 199행동 안에 3개 이상 분류한 실제 시연 · 네 상자 완주와 구분',
        'v2.1의 검증된 32개 시연을 가져오기 · 재수집하지 않음')
    first=first.rsplit('}',1)[0]+"    # 집기 높이 · 행동 디코더만 보정, 단계 판정은 고정\n"+''.join(
        f'    {k!r}: {cfg[k]!r},\n' for k in ('pick_z_weight','contact_z_weight','contact_gripper_weight'))+'}\n'
    nb['cells'][0]['source']=first.splitlines(keepends=True)
    bootstrap=''.join(nb['cells'][1]['source']).replace(
        'from medium_v21 import MediumV21, source_bundle\nexperiment = MediumV21(CFG, source_bundle())',
        "for name in ('build_medium_v22_notebook','medium_v22'):\n"
        "    if name in sys.modules:\n        importlib.reload(sys.modules[name])\n"
        "from build_medium_v22_notebook import CONFIG as V22_DEFAULTS\n"
        "CFG = dict(V22_DEFAULTS, **USER_CONFIG)\nCFG['project_commit'] = PROJECT_COMMIT\n"
        "from medium_v22 import MediumV22, source_bundle\nexperiment = MediumV22(CFG, source_bundle())")
    nb['cells'][1]['source']=bootstrap.splitlines(keepends=True)
    nb['cells']=nb['cells'][:4]
    tasks=[('05 · GPU 확인 + 기존 모델/시연 가져오기','experiment.check_runtime()\nexperiment.prepare()'),
        ('06 · 성공 시연의 실제 집기 높이 확인','experiment.audit_height()'),
        ('07 · Z축 보정 학습 · 500회마다 평가/저장 · 중단 후 같은 셀로 재개','_ = experiment.run_blocks()'),
        ('08 · 최고 모델 테스트 + 영상 + 궤적','experiment.test("medium")\nexperiment.diagnose("medium")'),
        ('09 · 학습 결과 확인','_ = experiment.report()'),
        ('10 · 선택 모델 최종 100회 평가','_ = experiment.final_evaluation()'),
        ('11 · 제출용 패키지 저장','_ = experiment.package()')]
    for i,(title,code) in enumerate(tasks,5):
        nb['cells'].append(dict(cell_type='code',execution_count=None,outputs=[],metadata={},
            source=(f'# {title}\n{code}\n').splitlines(keepends=True)))
    for i,c in enumerate(nb['cells']):c['metadata']['id']=f'medium-v22-{i+1}'
    nb['metadata']['colab']['name']='moveboxes_medium_v22_colab.ipynb'
    return nb


if __name__=='__main__':
    nb=make_notebook()
    for i,c in enumerate(nb['cells']):compile(''.join(c['source']),str(i),'exec')
    path=Path(__file__).parent/'notebooks/moveboxes_medium_v22_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path.name,len(nb['cells']),'code cells')
