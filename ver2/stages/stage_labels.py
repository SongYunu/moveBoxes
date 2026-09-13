"""OFFLINE labels from provided observations; never imported by the deployed policy.

Base demonstrations have no skill labels. These are weak geometric labels, not
ground-truth success annotations. Collection also checks official simulator success.
"""
import numpy as np
from stage_schema import PICK, CARRY, PLACE, DONE, HOLD, COMPLETE, RECOVER


def geometry(state):
    s = np.asarray(state, dtype=np.float32)
    if s.ndim != 1 or s.size not in (54, 72, 90) or not np.isfinite(s).all():
        raise ValueError('Expected one finite WarehouseSort state')
    n = (s.size-36)//9
    boxes = s[26:26+7*n].reshape(n, 7)[:, :3]
    tags = s[26+7*n:26+9*n].reshape(n, 2).argmax(-1)
    bins = s[26+9*n:32+9*n].reshape(2, 3)
    return s[18:21], boxes, tags, bins, bool(s[25] > .5)


class StageLabels:
    def __init__(self, n):
        self.n = n
        self.stage, self.target = PICK, 0
        self.placed = np.zeros(n, dtype=bool)
        self.inside_steps = np.zeros(n, dtype=np.int32)
        self.grasp_steps = self.released_steps = 0

    def observe(self, state, held_override=None):
        tcp, boxes, tags, bins, grasp = geometry(state)
        old = self.stage
        held = int(np.linalg.norm(boxes-tcp, axis=1).argmin()) if grasp else -1
        if held_override is not None:
            held = int(held_override)
            grasp = held >= 0
        inside = (np.abs(boxes[:, 0]-bins[tags, 0]) < .11) & (np.abs(boxes[:, 1]-bins[tags, 1]) < .13)
        inside &= (boxes[:, 2] > 0) & (boxes[:, 2] < .06)
        inside &= np.arange(self.n) != held
        self.inside_steps = np.where(inside, self.inside_steps+1, 0)
        self.placed |= self.inside_steps >= 2
        self.grasp_steps = self.grasp_steps+1 if grasp and held == self.target else 0
        self.released_steps = 0 if grasp else self.released_steps+1
        gate = HOLD
        if self.placed[self.target]:
            remaining = np.flatnonzero(~self.placed)
            if len(remaining):
                self.target, self.stage = int(remaining[0]), PICK
            else:
                self.stage = DONE
            if old != DONE:
                gate = COMPLETE
        elif old == PICK:
            if self.grasp_steps >= 2 and boxes[self.target, 2] > .10 and tcp[2] >= .23:
                self.stage, gate = CARRY, COMPLETE
        elif old in (CARRY, PLACE):
            goal = bins[tags[self.target]]
            above_bin = abs(tcp[0]-goal[0]) < .075 and abs(tcp[1]-goal[1]) < .105
            if old == CARRY and not grasp:
                self.stage, gate = PICK, RECOVER
            elif old == CARRY and above_bin and tcp[2] >= .20:
                self.stage, gate = PLACE, COMPLETE
            elif old == PLACE and not grasp and self.released_steps >= 12:
                self.stage, gate = PICK, RECOVER
            elif old == PLACE and grasp and not above_bin:
                self.stage, gate = CARRY, RECOVER
        return dict(previous_stage=old, stage=self.stage, gate=gate, target=self.target)


def annotate(obs):
    labeler = StageLabels((obs.shape[-1]-36)//9)
    rows = [labeler.observe(s) for s in obs[:-1]]
    return {key:np.asarray([r[key] for r in rows], dtype=np.int64) for key in rows[0]}
