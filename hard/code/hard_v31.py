"""Medium-only Hard v3.1 facade."""
from pathlib import Path
from hard_v3 import HardV3,source_bundle as v3_sources


class HardV31(HardV3):
    def notebook_snapshot(self):
        from build_hard_v31_notebook import make_notebook
        return make_notebook(dict(self.cfg,project_ref=self.cfg.get('project_commit',self.cfg['project_ref'])))


def source_bundle():
    bundle=v3_sources();root=Path(__file__).parent
    for name in ('hard_v31.py','build_hard_v31_notebook.py'):
        bundle[name]=(root/name).read_text(encoding='utf-8')
    return bundle
