"""Hard v3 Colab notebook: Easy/Medium transfer followed by Hard fine-tuning."""
import json
from pathlib import Path
from build_hard_v2_notebook import CONFIG as V2_CONFIG,make_notebook as v2_notebook

CONFIG=dict(V2_CONFIG,run_name='moveboxes_hard_transfer_v3',
    easy_source_run_name='moveboxes_easy_lab_v1',medium_source_run_name='moveboxes_medium_zfocus_v22',
    block_iters=750,max_blocks=8,plateau_blocks=3,lr=3e-5,
    total_iters=dict(easy=1,medium=1,hard=6000),save_freq=750,
    tuning_episodes=4,benchmark_episodes=100,tuning_seed_start=61000,test_seed_start=51000,eval_seed_start=81000)


def make_notebook(config=None):
    cfg=dict(CONFIG,**(config or {}));nb=v2_notebook(cfg)
    first=''.join(nb['cells'][0]['source']).replace('HARD v2 · 학습한 상자 선택 + 정렬/복구','HARD v3 · Easy/Medium 전이 + Hard 보정')
    first=first.rsplit('}',1)[0]+''.join(f'    {k!r}: {cfg[k]!r},\n' for k in ('easy_source_run_name','medium_source_run_name'))+'}\n'
    nb['cells'][0]['source']=first.splitlines(keepends=True)
    code=''.join(nb['cells'][1]['source']).replace(
        "for name in ('build_hard_v2_notebook','hard_v2'):",
        "for name in ('build_hard_v2_notebook','hard_v2','build_hard_v3_notebook','hard_v3'):")
    code=code.replace('from hard_v2 import HardV2, source_bundle\nexperiment = HardV2(CFG, source_bundle())',
        'from hard_v3 import HardV3, source_bundle\nexperiment = HardV3(CFG, source_bundle())')
    nb['cells'][1]['source']=code.splitlines(keepends=True)
    tasks=[('05 · 기존 데이터 + Easy/Medium 최고 모델 가져오기','experiment.prepare()'),
        ('06 · 복원 상태 확인','_ = experiment.report()'),
        ('07 · 두 전이 모델 비교 후 Hard 보정 학습','_ = experiment.run_blocks()'),
        ('08 · 현재 최고 Hard 모델 8회 시험','experiment.test("hard")\nexperiment.diagnose("hard")'),
        ('09 · 학습 구간별 성적','_ = experiment.report()'),
        ('10 · Hard 별도 시드 최종 100회','_ = experiment.final_evaluation()'),
        ('11 · Hard 현재 최고 모델 패키지','_ = experiment.package()')]
    nb['cells']=nb['cells'][:4]
    for number,(title,body) in enumerate(tasks,5):
        nb['cells'].append(dict(cell_type='code',execution_count=None,outputs=[],metadata={'id':f'hard-v3-{number}'},source=(f'# {title}\n{body}\n').splitlines(keepends=True)))
    nb['metadata']['colab']['name']='moveboxes_hard_v3_colab.ipynb';return nb


if __name__=='__main__':
    nb=make_notebook()
    for i,cell in enumerate(nb['cells']):compile(''.join(cell['source']),str(i),'exec')
    path=Path(__file__).parents[1]/'notebooks/moveboxes_hard_v3_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8');print(path.name,len(nb['cells']))
