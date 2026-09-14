"""Code-cell-only T4 notebook for one shared policy and a guarded curriculum."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from build_stage_notebook import CONFIG as BASE,make_notebook as base_notebook

CONFIG=dict(BASE,run_name='moveboxes_unified_curriculum_v1',medium_source_run_name='moveboxes_medium_zfocus_v22',
    batch_size=32,lr=3e-5,width=128,heads=4,layers=2,history=16,chunk_size=16,latent_dim=16,
    block_iters=500,max_blocks=dict(easy=6,medium=6,hard=6),plateau_blocks=2,
    pass_rate=dict(easy=.8,medium=.6,hard=.5),replay_fraction=.3,speed_bonus=.5,
    warmup_steps=50,amp=True,position_noise=0.,validation_batches=4,save_freq=500,
    total_iters=dict(easy=3000,medium=3000,hard=3000),tuning_episodes=8,test_episodes=8,benchmark_episodes=100,
    tuning_seed_start=310000,confirmation_seed_start=320000,test_seed_start=330000,eval_seed_start=340000)


def make_notebook(config=None):
    cfg=dict(CONFIG,**(config or {}));nb=base_notebook(cfg)
    groups=[('저장 / 초기 Medium 모델',['run_name','profile','github_repository','project_dir','project_ref','output_root','medium_source_run_name']),
        ('T4 환경 / state 데이터',['repo_dir','repo_url','repo_commit','data_dir','data_source','download_cache','packages']),
        ('같은 모델 하나',['seed','num_demos','batch_size','lr','amp','history','chunk_size','width','heads','layers','latent_dim']),
        ('단계별 통과와 반복 예산',['block_iters','max_blocks','plateau_blocks','pass_rate','replay_fraction','speed_bonus']),
        ('전체 에피소드 평가 / 시드 분리',['tuning_episodes','tuning_seed_start','confirmation_seed_start','test_episodes','test_seed_start','benchmark_episodes','eval_seed_start','max_episode_steps']),
        ('학습 / 출력',['warmup_steps','validation_batches','position_noise','console_interval_seconds','team'])]
    lines=['# 01 · T4 / 공유 모델 하나 / Easy → Medium → Hard','CFG = {']
    for title,keys in groups:
        lines.append('    # '+title)
        lines += [f'    {key!r}: {cfg[key]!r},' for key in keys]
    lines.append('}');nb['cells'][0]['source']=('\n'.join(lines)+'\n').splitlines(keepends=True)
    code=''.join(nb['cells'][1]['source'])
    code=code.replace('from stage_experiment import StageExperiment, source_bundle\nexperiment = StageExperiment(CFG, source_bundle())',
        "sys.path.insert(0,str(PROJECT/'hard'/'code'))\nsys.path.insert(0,str(PROJECT/'unified'/'code'))\n"
        "for name in ('build_unified_notebook','unified_policy','unified_data','unified_experiment'):\n"
        "    if name in sys.modules: importlib.reload(sys.modules[name])\n"
        "from build_unified_notebook import CONFIG as UNIFIED_DEFAULTS\nCFG = dict(UNIFIED_DEFAULTS, **CFG)\n"
        "from unified_experiment import UnifiedExperiment, source_bundle\nexperiment = UnifiedExperiment(CFG, source_bundle())")
    nb['cells'][1]['source']=code.splitlines(keepends=True)
    nb['cells']=nb['cells'][:4]
    tasks=[('05 · 데이터 / Medium 초기 가중치 / 공유 정책 복원','experiment.prepare()'),
        ('06 · 100회 학습 + 2회 평가로 T4 메모리·예상 시간 측정','_ = experiment.calibrate()'),
        ('07 · EASY 전체 평가 → 전문가 시연 실패 관련 구간 보강 → 반복','experiment.train_level("easy")'),
        ('08 · EASY 별도 테스트와 영상','experiment.test("easy")'),
        ('09 · EASY 통과 후 MEDIUM / 이전 데이터 재사용 / 성공 시연 속도 보너스','experiment.train_level("medium")'),
        ('10 · MEDIUM 별도 테스트와 영상','experiment.test("medium")'),
        ('11 · 이전 단계 통과 후 HARD 배치 변화 학습','experiment.train_level("hard")'),
        ('12 · HARD 별도 테스트와 영상','experiment.test("hard")'),
        ('13 · 공유 모델 상태 / 단계별 채택 기록','_ = experiment.report()'),
        ('14 · 세 단계 통과 후 최종 각 100회 / 오래 걸리므로 마지막에 실행','experiment.final_evaluation()'),
        ('15 · 같은 체크포인트 하나를 쓰는 세 난이도 패키지','_ = experiment.package()')]
    for i,(title,body) in enumerate(tasks,5):
        nb['cells'].append(dict(cell_type='code',id=f'unified-{i}',execution_count=None,outputs=[],metadata={'id':f'unified-{i}'},source=(f'# {title}\n{body}\n').splitlines(keepends=True)))
    nb['metadata']['colab']['name']='moveboxes_unified_colab.ipynb';return nb


if __name__=='__main__':
    nb=make_notebook()
    for cell in nb['cells']:compile(''.join(cell['source']),'cell','exec')
    path=ROOT/'unified/notebooks/moveboxes_unified_colab.ipynb';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8');print(path.name,len(nb['cells']))
