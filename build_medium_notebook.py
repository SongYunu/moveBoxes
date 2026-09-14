"""Medium only: repeated short training and actual simulator feedback."""
import json
from pathlib import Path
from build_stage_notebook import CONFIG as STAGE_CONFIG, make_notebook as stage_notebook

CONFIG = dict(STAGE_CONFIG,run_name='moveboxes_medium_lab_v1',source_run_name='moveboxes_stage_act_v1',
    warm_start=True,block_iters=2000,max_blocks=10,development_episodes=8,final_episodes=100,
    target_accuracy=.95,success_streak=2,plateau_blocks=4,first_pick_fraction=.30,
    warmup_steps=100,tuning_seed_start=50000,eval_seed_start=70000)


def make_notebook(config=None):
    cfg = dict(CONFIG,**(config or {}))
    nb = stage_notebook(cfg)
    groups = [
        ('GitHub 저장 · 기존 결과 보존, Medium 전용 Release',
         ['run_name','profile','github_repository','project_dir','project_ref','output_root','source_run_name','warm_start']),
        ('T4 / 설치 / 데이터', ['repo_dir','repo_url','repo_commit','data_dir','data_source','download_cache','packages']),
        ('반복 예산 · 2,000회마다 실제 평가, 최대 20,000회',
         ['block_iters','max_blocks','development_episodes','target_accuracy','success_streak','plateau_blocks']),
        ('작은 모델 · 실행 조건으로 학습, 첫 집기 구간 30% 별도 표집',
         ['seed','num_demos','batch_size','lr','amp','history','chunk_size','width','heads','layers','latent_dim','first_pick_fraction']),
        ('별도 시드와 동일한 200스텝 평가 제한',
         ['tuning_seed_start','eval_seed_start','final_episodes','test_seed_start','test_episodes','max_episode_steps']),
        ('실행 / 출력', ['ensemble_candidates','ensemble_window','temporal_decay','gate_threshold','stage_threshold','console_interval_seconds','team'])]
    lines = ['# 01 · MEDIUM 집중 실험 CONFIG','CFG = {']
    for title,keys in groups:
        lines.append('    # '+title)
        lines.extend(f'    {k!r}: {cfg[k]!r},' for k in keys)
        lines.append('')
    lines.append('}')
    nb['cells'][0]['source']=('\n'.join(lines)+'\n').splitlines(keepends=True)
    bootstrap=''.join(nb['cells'][1]['source'])
    bootstrap=bootstrap.replace('from stage_experiment import StageExperiment, source_bundle\nexperiment = StageExperiment(CFG, source_bundle())',
        "for name in ('build_medium_notebook','medium_lab'):\n"
        "    if name in sys.modules:\n        importlib.reload(sys.modules[name])\n"
        "from build_medium_notebook import CONFIG as MEDIUM_DEFAULTS\n"
        "CFG = dict(MEDIUM_DEFAULTS, **CFG)\n"
        'from medium_lab import MediumLab, source_bundle\nexperiment = MediumLab(CFG, source_bundle())')
    nb['cells'][1]['source']=bootstrap.splitlines(keepends=True)
    nb['cells'][2]['source']=['# 03 · GitHub 인증/복원\n','experiment.connect()\n','experiment.report()\n']
    nb['cells']=nb['cells'][:4]
    tasks = [
        ('05 · Medium GPU 확인 / 기존 성공 시연과 모델 가져오기', 'experiment.check_runtime()\nexperiment.prepare()'),
        ('06 · MEDIUM 반복 학습 · 중단 후 같은 셀을 다시 실행하면 이어서 진행', 'history = experiment.run_blocks()'),
        ('07 · 최고 모델 8회 별도 테스트 + 영상 + 2회 행동 기록', 'experiment.test("medium")\nexperiment.diagnose("medium")'),
        ('08 · 학습 곡선과 최고 결과', 'experiment.report()'),
        ('09 · 목표 달성 모델의 별도 시드 최종 평가', 'experiment.final_evaluation()'),
        ('10 · 평가 완료한 Medium 모델만 패키징', 'experiment.package()')]
    for i,(title,body) in enumerate(tasks,5):
        nb['cells'].append(dict(cell_type='code',execution_count=None,metadata={'id':f'medium-{i}'},outputs=[],
            source=(f'# {title}\n{body}\n').splitlines(keepends=True)))
    nb['metadata']['colab']['name']='moveboxes_medium_colab.ipynb'
    return nb


if __name__ == '__main__':
    nb=make_notebook()
    for i,c in enumerate(nb['cells']):
        compile(''.join(c['source']),str(i),'exec')
    path=Path(__file__).parent/'notebooks/moveboxes_medium_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path.name,len(nb['cells']),'code cells')
