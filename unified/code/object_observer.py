"""Simulator diagnostics only; no stage labels or privileged data enter actions."""
from next_pick_diagnostics import NextPickObserver
from hard_observer import instrument_observer


class BaseObserver(NextPickObserver):
    def reset(self):
        super().reset()
        self.frames, self.decisions = [], []

    def act(self, obs, deterministic=True):
        action = super().act(obs, deterministic)
        # The shared parcel-progress observer uses only the length of this list.
        self.decisions.append(dict(stage=-1))
        if getattr(self, 'trace', False):
            state = obs['state'] if isinstance(obs, dict) else obs
            self.frames.append(dict(step=len(self.frames), state=state[0].detach().cpu().tolist(),
                                    action=action[0].detach().cpu().tolist()))
        return action


Observer = instrument_observer(BaseObserver)
