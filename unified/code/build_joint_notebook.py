import json
from pathlib import Path
from build_unified_notebook import CONFIG as BASE,make_notebook as base_notebook

CONFIG=dict(BASE,run_name='moveboxes_joint_object_v1',project_dir='/content/moveBoxes_joint',repo_dir='/content/berlin-marso-joint',
    num_demos=None,history=8,chunk_size=8,spatial_layers=1,ensemble_window=2,ensemble_candidates=[1,2,4],
    kl_weight=0.,stage_loss_weight=0.,gate_loss_weight=0.,position_noise=0.,
    lr=1e-4,warmup_steps=500,joint_total_iters=30000,joint_eval_interval=2000,
    tuning_seed_start=410000,confirmation_seed_start=420000,test_seed_start=430000,eval_seed_start=440000)

def make_notebook(config=None):
    cfg=dict(CONFIG,**(config or {}));nb=base_notebook(cfg)
    groups=[('저장 / 다른 버전과 별도 실험',('run_name','profile','github_repository','project_dir','project_ref','output_root')),
        ('T4 환경',('repo_dir','repo_url','repo_commit','data_dir','data_source','download_cache','packages')),
        ('무작위 초기화 / 모든 상자 attention / 세 난이도 공동 학습',('seed','num_demos','batch_size','lr','amp','history','chunk_size','width','heads','layers','spatial_layers','joint_total_iters','joint_eval_interval','warmup_steps')),
        ('세 난이도 평가 / 별도 테스트',('tuning_episodes','tuning_seed_start','test_episodes','test_seed_start','benchmark_episodes','eval_seed_start','max_episode_steps','validation_batches','console_interval_seconds','team'))]
    lines=['# 01 · Easy/Medium/Hard 균등 혼합 / 기존 체크포인트 없이 처음부터','CFG = {']
    for title,keys in groups:
        lines.append('    # '+title);lines.extend(f'    {k!r}: {cfg[k]!r},' for k in keys)
    lines.append('}');nb['cells'][0]['source']=('\n'.join(lines)+'\n').splitlines(keepends=True)
    code=''.join(nb['cells'][1]['source']).replace('from build_unified_notebook import CONFIG as UNIFIED_DEFAULTS','from build_joint_notebook import CONFIG as UNIFIED_DEFAULTS')
    code=code.replace('from unified_experiment import UnifiedExperiment, source_bundle\nexperiment = UnifiedExperiment(CFG, source_bundle())',
        "for name in ('joint_data','joint_experiment','build_joint_notebook'):\n    if name in sys.modules: importlib.reload(sys.modules[name])\nfrom joint_experiment import JointExperiment, source_bundle\nexperiment = JointExperiment(CFG, source_bundle())")
    nb['cells'][1]['source']=code.splitlines(keepends=True);nb['cells']=nb['cells'][:4]
    tasks=[('05 · 원본 세 난이도 데이터 준비 / 기존 모델 다운로드 없음','experiment.prepare()'),
        ('06 · GPU 역전파 / 같은 모델 세 난이도 로딩 / 초기화 확인','experiment.check_runtime()\nprint("난이도 비율: Easy ≈ Medium ≈ Hard = 1:1:1")\nprint("학습 예산:", experiment.cfg["joint_total_iters"], "회 / 평가 간격:", experiment.cfg["joint_eval_interval"])\n_ = experiment.report()'),
        ('07 · 세 난이도 처음부터 공동 학습 / 같은 모델을 세 환경에서 평가','experiment.train_joint()'),
        ('08 · 같은 선택 모델 Easy 테스트','_ = experiment.test("easy")'),
        ('09 · 같은 선택 모델 Medium 테스트','_ = experiment.test("medium")'),
        ('10 · 같은 선택 모델 Hard 테스트','_ = experiment.test("hard")'),
        ('11 · 공동 학습 결과','_ = experiment.report()'),
        ('12 · 선택 사항 / 같은 모델로 각 난이도 최종 100회','experiment.final_evaluation()'),
        ('13 · 동일 체크포인트 하나로 세 난이도 패키지','_ = experiment.package()')]
    for i,(title,body) in enumerate(tasks,5):nb['cells'].append(dict(cell_type='code',id=f'joint-{i}',metadata={},execution_count=None,outputs=[],source=(f'# {title}\n{body}\n').splitlines(keepends=True)))
    nb['metadata']['colab']['name']='moveboxes_joint_scratch_colab.ipynb';return nb

if __name__=='__main__':
    nb=make_notebook()
    for c in nb['cells']:compile(''.join(c['source']),'cell','exec')
    path=Path(__file__).resolve().parents[1]/'notebooks/moveboxes_joint_scratch_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8');print(path)
