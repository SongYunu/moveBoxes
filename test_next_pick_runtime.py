# Embedded after policy tensor checks in Colab cell 05; no GPU training required.
import tempfile
import json
from pathlib import Path
from next_pick_sampling import make_sampler
from next_pick_diagnostics import NextPickObserver

with tempfile.TemporaryDirectory() as check_dir:
    config_path = Path(check_dir)/'sampling_config.json'
    focus_config = dict(enabled=True, weight=3.0, before=2, after=1, min_grasp=3, min_release=3)
    config_path.write_text(json.dumps(focus_config))
    observations = torch.zeros(21, 54)
    observations[3:7, 25] = 1
    observations[13:17, 25] = 1

    class SamplerDataset:
        obs_horizon = 2
        slices = [(0, t-1, t+15) for t in range(20)]
        trajectories = {'observations': [observations]}
        def __len__(self):
            return len(self.slices)

    first = make_sampler(SamplerDataset(), config_path, 42)
    second = make_sampler(SamplerDataset(), config_path, 42)
    assert list(first) == list(second), 'Weighted sampler seeds must be reproducible'
    assert first.weights[0] == 1 and first.weights[13] == 3
    audit = json.loads((Path(check_dir)/'sampling_audit.json').read_text())
    assert audit['focus_fraction_weighted'] > audit['focus_fraction_uniform']
    focus_config['enabled'] = False
    config_path.write_text(json.dumps(focus_config))
    uniform = make_sampler(SamplerDataset(), config_path, 42)
    assert sorted(list(uniform)) == list(range(20)), 'Disabled weighting must restore uniform sampling'

class ObserverCheckPolicy:
    def reset(self):
        pass
    def act(self, obs, deterministic=True):
        return observer_action

observer_action = torch.tensor([[0.1, -0.2, 0.3, -1.]])
observer = NextPickObserver(ObserverCheckPolicy())
for observation in observations:
    assert observer.act(observation.unsqueeze(0)) is observer_action, 'Observer changed policy action'
assert observer.report()['stable_grasp_cycles'] == 2
observer.reset()
assert observer.report()['observed_steps'] == 0
print('Next-pick sampler reproducibility, uniform fallback and observation-only diagnostics passed')
