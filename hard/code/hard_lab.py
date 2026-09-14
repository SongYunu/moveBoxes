"""Hard-only facade over the shared stage/recovery experiment."""
from pathlib import Path
from stage_experiment import StageExperiment, source_bundle as stage_sources


class HardLab(StageExperiment):
    def notebook_snapshot(self):
        from build_hard_notebook import make_notebook
        return make_notebook(dict(self.cfg,project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))

    def _hard(self,level):
        if level!='hard':raise ValueError('Hard notebook only supports hard')

    def smoke(self,level):self._hard(level);return super().smoke(level)
    def collect(self,level):self._hard(level);return super().collect(level)
    def train(self,level):self._hard(level);return super().train(level)
    def test(self,level):self._hard(level);return super().test(level)
    def diagnose(self,level):self._hard(level);return super().diagnose(level)
    def evaluate(self,level):self._hard(level);return super().evaluate(level)


def source_bundle():
    bundle=stage_sources();root=Path(__file__).parent
    for name in ('hard_lab.py','build_hard_notebook.py'):
        bundle[name]=(root/name).read_text(encoding='utf-8')
    return bundle
