"""Separate Colab stage/recovery experiment; no changes to existing notebooks."""
import json
from pathlib import Path
from build_act_v2_notebook import CONFIG as ACT_CONFIG, GROUPS, make_notebook as act_notebook

CONFIG = dict(ACT_CONFIG, run_name='moveboxes_stage_act_v2', gate_threshold=.65, stage_threshold=.6,
    stage_loss_weight=.3, gate_loss_weight=.3, recovery_episodes=16, recovery_max_attempts=48,
    recovery_seed_start=100000, collection_max_steps=dict(easy=500,medium=900,hard=1400),
    noise_probability=.08, action_noise_std=.12, drop_probability=.015, action_training_mode='prior',
    repair_iters=2000, allow_zero_success_evaluation=False)


def make_notebook(config=None):
    cfg = dict(CONFIG, **(config or {}))
    nb = act_notebook(cfg)
    groups = list(GROUPS)+[
        ('실행 조건으로 행동 학습 · 빠른 테스트가 0이면 긴 평가 생략',
         ['action_training_mode','repair_iters','allow_zero_success_evaluation']),
        ('단계 판단 · 학습된 완료/복구 확신이 낮으면 현재 단계 유지',
         ['gate_threshold','stage_threshold','stage_loss_weight','gate_loss_weight']),
        ('복구 시연 · 수집 전용 expert, 학습/제출은 학습된 정책',
         ['recovery_episodes','recovery_max_attempts','recovery_seed_start','collection_max_steps',
          'noise_probability','action_noise_std','drop_probability'])]
    lines = ['# 01 · 단계 ACT CONFIG · 기존 ACT/DP와 별도 결과', 'CFG = {']
    for title,keys in groups:
        lines.append('    # '+title)
        lines.extend(f'    {k!r}: {cfg[k]!r},' for k in keys)
        lines.append('')
    lines.append('}')
    nb['cells'][0]['source'] = ('\n'.join(lines)+'\n').splitlines(keepends=True)
    bootstrap = ''.join(nb['cells'][1]['source'])
    bootstrap = bootstrap.replace('from act_v2_experiment import ActV2Experiment, source_bundle\nexperiment = ActV2Experiment(CFG, source_bundle())',
        "sys.path.insert(0, str(PROJECT/'ver2'/'stages'))\n"
        "for name in ('stage_schema','stage_model','stage_policy','stage_labels','stage_data','stage_teacher',\n"
        "             'stage_collect','stage_eval','stage_experiment','build_stage_notebook'):\n"
        "    if name in sys.modules:\n        importlib.reload(sys.modules[name])\n"
        'from stage_experiment import StageExperiment, source_bundle\nexperiment = StageExperiment(CFG, source_bundle())')
    nb['cells'][1]['source'] = bootstrap.splitlines(keepends=True)
    cells = nb['cells'][:5]
    def code(text, ident):
        return dict(cell_type='code', execution_count=None, metadata={'id':ident}, outputs=[], source=text.splitlines(keepends=True))
    for level in ('easy','medium','hard'):
        for title,body in [
            ('짧은 동작 확인 · 100 iteration + 2회 실행, 별도 임시 모델', f'experiment.smoke("{level}")'),
            ('복구 시연 수집 · 성공 에피소드만 저장, 중단하면 완료 시점부터 재개', f'experiment.collect("{level}")'),
            ('단계 ACT 학습 · 기존 데이터 + 복구 시연, 체크포인트 자동 업로드', f'experiment.train("{level}")'),
            ('8회 테스트 + 영상 + 별도 2회 단계/행동 기록', f'experiment.test("{level}")\nexperiment.diagnose("{level}")'),
            ('설정 비교 + 최종 100회 평가 · 평가 한도 200스텝 유지', f'experiment.evaluate("{level}")')]:
            number = len(cells)+1
            cells.append(code(f'# {number:02d} · {level.upper()} · {title}\n{body}\n', f'stage-{number}'))
    cells.append(code('# 21 · 저장 결과와 영상\nexperiment.show_results()\n', 'stage-21'))
    cells.append(code('# 22 · 학습된 정책 패키징 · 수집 expert 제외\nexperiment.package()\n', 'stage-22'))
    nb['cells'] = cells
    nb['metadata']['colab']['name'] = 'moveboxes_stages_colab.ipynb'
    return nb


def make_repair_notebook():
    nb = make_notebook(dict(run_name='moveboxes_stage_act_v1'))
    nb['cells'] = nb['cells'][:5]
    nb['cells'][0]['source'].insert(0, '# 기존 실패 실험을 복원합니다. 보정 결과는 별도 _prior_fix_v1 Release에 저장됩니다.\n')
    for level in ('easy','medium','hard'):
        for title,body in [
            ('기존 모델/시연 재사용 → 보정 학습 → 8회 테스트/진단',f'fixed_{level} = experiment.repair("{level}")'),
            ('빠른 테스트 성공 확인 후 튜닝/최종 평가',f'fixed_{level}.evaluate("{level}")')]:
            number = len(nb['cells'])+1
            text = f'# {number:02d} · {level.upper()} · {title}\n{body}\n'
            nb['cells'].append(dict(cell_type='code',execution_count=None,metadata={'id':f'repair-{number}'},
                                    outputs=[],source=text.splitlines(keepends=True)))
    text = ('# 12 · 보정 결과 보기 / 평가 완료 모델 패키징\n'
            'completed = [globals()["fixed_"+level] for level in ("easy","medium","hard") if "fixed_"+level in globals()]\n'
            'if completed:\n    completed[-1].show_results()\n    completed[-1].package()\n')
    nb['cells'].append(dict(cell_type='code',execution_count=None,metadata={'id':'repair-12'},outputs=[],source=text.splitlines(keepends=True)))
    nb['metadata']['colab']['name'] = 'moveboxes_stage_repair_colab.ipynb'
    return nb


if __name__ == '__main__':
    for nb in (make_notebook(),make_repair_notebook()):
        for i,c in enumerate(nb['cells']):
            compile(''.join(c['source']), f'stage-cell-{i+1}', 'exec')
        path = Path(__file__).parent/'notebooks'/nb['metadata']['colab']['name']
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
        print(path.name, len(nb['cells']), 'code cells')
