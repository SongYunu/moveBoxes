"""Separate Hard v2 notebook: reuse data and train learned target-conditioned actions."""
import json
from pathlib import Path
from build_hard_notebook import CONFIG as V1_CONFIG,make_notebook as v1_notebook

CONFIG=dict(V1_CONFIG,run_name='moveboxes_hard_target_v2',source_run_name='moveboxes_hard_stage_v1',
    block_iters=1000,max_blocks=12,plateau_blocks=4,target_loss_weight=.5,batch_size=32,lr=1e-4,
    total_iters=dict(easy=1,medium=1,hard=12000),save_freq=1000,position_noise=0.,
    tuning_episodes=4,benchmark_episodes=100,tuning_seed_start=60000,test_seed_start=50000,eval_seed_start=80000)


def make_notebook(config=None):
    cfg=dict(CONFIG,**(config or {}));nb=v1_notebook(cfg)
    first=''.join(nb['cells'][0]['source']).replace('HARD 전용 단계 ACT CONFIG','HARD v2 · 학습한 상자 선택 + 정렬/복구')
    first=first.rsplit('}',1)[0]+"    # 실제 개발 평가 · 가장 좋은 모델만 백업\n"+''.join(
        f'    {k!r}: {cfg[k]!r},\n' for k in ('source_run_name','block_iters','max_blocks','plateau_blocks','target_loss_weight'))+'}\n'
    nb['cells'][0]['source']=first.splitlines(keepends=True)
    code=''.join(nb['cells'][1]['source']).replace(
        'from hard_lab import HardLab, source_bundle\nexperiment = HardLab(CFG, source_bundle())',
        "for name in ('build_hard_v2_notebook','hard_v2'):\n"
        "    if name in sys.modules:\n        importlib.reload(sys.modules[name])\n"
        "from build_hard_v2_notebook import CONFIG as V2_DEFAULTS\n"
        "CFG = dict(V2_DEFAULTS, **CFG)\n"
        "from hard_v2 import HardV2, source_bundle\nexperiment = HardV2(CFG, source_bundle())")
    nb['cells'][1]['source']=code.splitlines(keepends=True)
    nb['cells']=nb['cells'][:4]
    tasks=[('05 · 데이터 + 기존 Hard 복구 시연 24개 가져오기','experiment.prepare()'),
        ('06 · 입력/모델 준비 상태','print("새 모델: 상자 선택과 행동을 함께 학습 · 원본 200개 + 복구 24개")\n_ = experiment.report()'),
        ('07 · 최대 12,000회 · 1,000회마다 실제 평가 · 중단 뒤 재실행','_ = experiment.run_blocks()'),
        ('08 · 최고 모델 테스트 + 영상 + 궤적','experiment.test("hard")\nexperiment.diagnose("hard")'),
        ('09 · 구간별 성적','_ = experiment.report()'),
        ('10 · 별도 시드 최종 100회','_ = experiment.final_evaluation()'),
        ('11 · 제출용 패키지','_ = experiment.package()')]
    for number,(title,body) in enumerate(tasks,5):
        nb['cells'].append(dict(cell_type='code',execution_count=None,outputs=[],metadata={'id':f'hard-v2-{number}'},source=(f'# {title}\n{body}\n').splitlines(keepends=True)))
    nb['metadata']['colab']['name']='moveboxes_hard_v2_colab.ipynb';return nb


if __name__=='__main__':
    nb=make_notebook()
    for i,c in enumerate(nb['cells']):compile(''.join(c['source']),str(i),'exec')
    path=Path(__file__).parents[1]/'notebooks/moveboxes_hard_v2_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8');print(path.name,len(nb['cells']))
