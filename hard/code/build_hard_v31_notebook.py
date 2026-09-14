"""Medium-only direct transfer to Hard, with optional short fine-tuning."""
import json
from pathlib import Path
from build_hard_v3_notebook import CONFIG as V3_CONFIG,make_notebook as v3_notebook

CONFIG=dict(V3_CONFIG,run_name='moveboxes_hard_medium_v31',
    transfer_donors={'medium':'moveboxes_medium_zfocus_v22'},
    block_iters=500,max_blocks=3,plateau_blocks=2,lr=1e-5,
    total_iters=dict(easy=1,medium=1,hard=1500),save_freq=500,
    tuning_episodes=4,test_seed_start=52000,tuning_seed_start=62000,eval_seed_start=82000)


def make_notebook(config=None):
    cfg=dict(CONFIG,**(config or {}));nb=v3_notebook(cfg)
    first=''.join(nb['cells'][0]['source']).replace('HARD v3 · Easy/Medium 전이 + Hard 보정','HARD v3.1 · Medium 최고 모델 직접 전이')
    first=first.rsplit('}',1)[0]+f"    'transfer_donors': {cfg['transfer_donors']!r},\n"+'}\n'
    nb['cells'][0]['source']=first.splitlines(keepends=True)
    code=''.join(nb['cells'][1]['source']).replace(
        "for name in ('build_hard_v2_notebook','hard_v2','build_hard_v3_notebook','hard_v3'):",
        "for name in ('build_hard_v2_notebook','hard_v2','build_hard_v3_notebook','hard_v3','build_hard_v31_notebook','hard_v31'):")
    code=code.replace('from hard_v3 import HardV3, source_bundle\nexperiment = HardV3(CFG, source_bundle())',
        'from hard_v31 import HardV31, source_bundle\nexperiment = HardV31(CFG, source_bundle())')
    nb['cells'][1]['source']=code.splitlines(keepends=True)
    tasks=[('05 · Hard 데이터 + Medium 최고 모델 직접 변환','experiment.prepare()'),
        ('06 · 재학습 전 직접 전이 모델 시험','experiment.test_transfer()\nexperiment.diagnose("hard")'),
        ('07 · 선택 사항: 직접 전이가 괜찮을 때만 최대 1,500회 보정','_ = experiment.run_blocks()'),
        ('08 · 현재 최고 Hard 모델 시험','experiment.test("hard")\nexperiment.diagnose("hard")'),
        ('09 · 직접 전이/보정 성적 확인','_ = experiment.report()'),
        ('10 · 괜찮을 때만 Hard 최종 100회','_ = experiment.final_evaluation()'),
        ('11 · Hard 모델 패키지','_ = experiment.package()')]
    nb['cells']=nb['cells'][:4]
    for number,(title,body) in enumerate(tasks,5):
        nb['cells'].append(dict(cell_type='code',execution_count=None,outputs=[],metadata={'id':f'hard-v31-{number}'},source=(f'# {title}\n{body}\n').splitlines(keepends=True)))
    nb['metadata']['colab']['name']='moveboxes_hard_v31_colab.ipynb';return nb


if __name__=='__main__':
    nb=make_notebook()
    for i,cell in enumerate(nb['cells']):compile(''.join(cell['source']),str(i),'exec')
    path=Path(__file__).parents[1]/'notebooks/moveboxes_hard_v31_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8');print(path.name,len(nb['cells']))
