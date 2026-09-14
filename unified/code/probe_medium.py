"""Evaluation-only experiment for the historic 40.625% Medium checkpoint."""
from pathlib import Path
from stage_experiment import StageExperiment,source_bundle as stage_sources
from unified_import import SnapshotSource
from marso_experiment import read_json,save_json,digest

EXPECTED='f7c0d5edf5b2bd6b93918d10af43b9c4ee023befd3162139e391b371e1541730'


class MediumProbe(StageExperiment):
    def notebook_snapshot(self):
        from build_medium_probe_notebook import make_notebook
        return make_notebook(dict(self.cfg,project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))

    def _stage_helpers(self):
        super()._stage_helpers()
        for name,source in self.sources.items():
            if name.startswith('probe_'):(self.repo/name).write_text(source,encoding='utf-8')

    def prepare(self):
        self._ready();self._stage_helpers()
        for level in ('easy','medium'):(self.run_dir/level).mkdir(exist_ok=True)
        target=self.run_dir/'medium/checkpoints/block_02.pt'
        with self.persist_operation('medium'):
            if not target.exists():
                source=SnapshotSource(self.cfg['github_repository'],'moveboxes_medium_lab_v1','medium',self.run_dir/'medium/source')
                selection=read_json(source.fetch('medium/lab_best.json'))
                if selection['checkpoint_sha256']!=EXPECTED:raise ValueError('Historic Medium selection changed; refusing to substitute another model')
                source.fetch('medium/checkpoints/block_02.pt',target)
                save_json(target.parent/'policy_config.json',selection['policy_config'])
                save_json(self.run_dir/'medium/probe_origin.json',dict(snapshot=source.name,checkpoint_sha256=EXPECTED))
            if digest(target)!=EXPECTED:raise ValueError('Historic Medium checkpoint hash mismatch')
        print('추가 학습 없음. 원본 Medium block_02.pt SHA-256:',EXPECTED)
        print('Easy에서는 없는 두 상자 슬롯만 0으로 채웁니다. 모델 파일은 변경하지 않습니다.')

    def _test_candidate(self,level):
        if level not in ('easy','medium'):raise ValueError('Probe supports Easy/Medium only')
        checkpoint=self.run_dir/'medium/checkpoints/block_02.pt'
        if not checkpoint.exists() or digest(checkpoint)!=EXPECTED:raise RuntimeError('05 셀에서 기존 Medium 모델을 먼저 복원하세요')
        return checkpoint,read_json(checkpoint.parent/'policy_config.json')


def source_bundle():
    bundle=stage_sources();root=Path(__file__).parent
    for name in ('probe_medium.py','probe_medium_policy.py','build_medium_probe_notebook.py','unified_import.py'):
        bundle[name]=(root/name).read_text(encoding='utf-8')
    evaluator=bundle['stage_eval.py'].replace('from stage_policy import load_stage','from probe_medium_policy import load_stage')
    bundle['stage_eval.py']=evaluator;bundle['colab_eval_modular.py']=evaluator
    bundle['colab_policy.py']=bundle['probe_medium_policy.py'];return bundle
