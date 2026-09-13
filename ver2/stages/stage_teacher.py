"""COLLECTION ONLY: corrective expert adapted from the official scripted expert.

It is deliberately excluded from submission packages. No timeout counts as a
successful pick/place: retries keep the same parcel, and exhausted retries fail.
"""
import numpy as np
from stage_labels import geometry


class TeacherFailure(RuntimeError):
    pass


class CorrectiveTeacher:
    def __init__(self, n):
        self.n, self.target = n, 0
        self.phase, self.ticks, self.retries = 'open', 0, 0
        self.confirmed = np.zeros(n, dtype=bool)
        self.inside_steps = np.zeros(n, dtype=np.int32)

    def goto(self, phase):
        self.phase, self.ticks = phase, 0

    def retry(self):
        self.retries += 1
        if self.retries > 6:
            raise TeacherFailure(f'Parcel {self.target}: exhausted 6 corrective attempts')
        self.goto('retreat')

    def action(self, state, grasped):
        tcp, boxes, tags, bins, _ = geometry(state)
        grasped = np.asarray(grasped, dtype=bool)
        if grasped.shape != (self.n,):
            raise ValueError('Teacher needs one grasp flag per parcel')
        goal = bins[tags]
        inside = (np.abs(boxes[:, 0]-goal[:, 0]) < .11) & (np.abs(boxes[:, 1]-goal[:, 1]) < .13)
        inside &= (boxes[:, 2] > 0) & (boxes[:, 2] < .06) & ~grasped
        self.inside_steps = np.where(inside, self.inside_steps+1, 0)
        self.confirmed |= self.inside_steps >= 2
        if self.confirmed[self.target]:
            remaining = np.flatnonzero(~self.confirmed)
            if not len(remaining):
                self.goto('done')
                return np.asarray([0, 0, 0, 1], dtype=np.float32), False
            self.target, self.retries = int(remaining[0]), 0
            self.goto('retreat')
        self.ticks += 1
        j, phase = self.target, self.phase
        recovering = False
        held = bool(grasped[j])
        if phase in ('lift','carry','drop') and not held:
            self.retry()
            phase, recovering = self.phase, True
        # Timeouts recover; they never open the gate to the next task stage.
        timeout = dict(open=15,retreat=50,above=85,descend=65,grasp=18,lift=45,carry=65,drop=45,release=25)
        if self.ticks > timeout.get(phase, 1000):
            self.retry()
            phase, recovering = self.phase, True
        tag = tags[j]
        peers = np.flatnonzero(tags == tag)
        slot = int(np.flatnonzero(peers == j)[0])
        offset = (slot-(len(peers)-1)/2)*.07
        hover = np.array([*boxes[j, :2], .22])
        pickup = np.array([*boxes[j, :2], max(.06, float(boxes[j, 2])+.029)])
        carry = np.array([bins[tag, 0], bins[tag, 1]+offset, .26])
        drop = np.array([carry[0], carry[1], .08])

        def move(target, grip, limit=1.):
            return np.r_[((target-tcp)/.1).clip(-limit, limit), grip].astype(np.float32)

        if phase == 'open':
            action = np.array([0,0,0,1], dtype=np.float32)
            if self.ticks >= 4:
                self.goto('above')
        elif phase == 'retreat':
            # Clear the table vertically before moving laterally to another parcel.
            action = move(np.array([tcp[0], tcp[1], .26]), 1.)
            if tcp[2] >= .24 and self.ticks >= 4:
                self.goto('above')
        elif phase == 'above':
            action = move(hover, 1.)
            if np.linalg.norm(tcp[:2]-hover[:2]) < .005 and abs(tcp[2]-.22) < .035:
                self.goto('descend')
        elif phase == 'descend':
            action = move(pickup, 1., .6)
            if abs(tcp[2]-pickup[2]) < .008 and np.linalg.norm(tcp[:2]-pickup[:2]) < .008:
                self.goto('grasp')
        elif phase == 'grasp':
            action = np.array([0,0,0,-1], dtype=np.float32)
            if held and self.ticks >= 4:
                self.goto('lift')
            elif self.ticks >= 12:
                self.retry()
                recovering = True
        elif phase == 'lift':
            action = move(np.array([tcp[0], tcp[1], .26]), -1.)
            if held and boxes[j, 2] > .10 and tcp[2] >= .25:
                self.goto('carry')
        elif phase == 'carry':
            action = move(carry, -1.)
            if held and np.linalg.norm(tcp-carry) < .03:
                self.goto('drop')
        elif phase == 'drop':
            action = move(drop, -1., .7)
            if held and np.linalg.norm(tcp-drop) < .02:
                self.goto('release')
        elif phase == 'release':
            # Stay over the same bin until this exact parcel is confirmed inside.
            action = np.array([0,0,.1,1], dtype=np.float32)
        else:
            action = np.array([0,0,0,1], dtype=np.float32)
        return action, recovering
