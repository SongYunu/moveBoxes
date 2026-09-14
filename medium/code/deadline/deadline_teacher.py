"""Collection-only precision teacher. Never included in the submitted policy."""
import numpy as np
from stage_labels import geometry
from stage_teacher import TeacherFailure


class DeadlineTeacher:
    def __init__(self, n, gain=1.5):
        if not 1 <= gain <= 2:
            raise ValueError('Teacher gain must be in [1,2]')
        self.n, self.gain, self.target = n, gain, 0
        self.phase, self.ticks, self.retries, self.held_steps = 'above', 0, 0, 0
        self.inside = np.zeros(n, dtype=np.int32)
        self.confirmed = np.zeros(n, dtype=bool)

    def goto(self, phase):
        self.phase, self.ticks = phase, 0

    def retry(self):
        self.retries += 1
        if self.retries > 2:
            raise TeacherFailure('Two grasp recovery attempts exhausted')
        self.goto('retreat')

    def action(self, state, grasped):
        tcp, boxes, tags, bins, _ = geometry(state)
        grasped = np.asarray(grasped, dtype=bool)
        if grasped.shape != (self.n,):
            raise ValueError('One observed grasp flag per parcel required')
        goals = bins[tags]
        inside = (np.abs(boxes[:,0]-goals[:,0]) < .11) & (np.abs(boxes[:,1]-goals[:,1]) < .13)
        inside &= (boxes[:,2] > 0) & (boxes[:,2] < .06) & ~grasped
        self.inside = np.where(inside, self.inside+1, 0)
        self.confirmed |= self.inside >= 2
        if self.confirmed[self.target]:
            remaining = np.flatnonzero(~self.confirmed)
            if not len(remaining):
                self.goto('done')
                return np.array([0,0,0,1],np.float32), False
            self.target, self.retries = int(remaining[0]), 0
            self.goto('retreat')
        j = self.target
        self.held_steps = self.held_steps+1 if grasped[j] else 0
        self.ticks += 1
        recovering = False
        if self.phase in ('lift','carry','drop') and not grasped[j]:
            self.retry(); recovering = True
        if self.ticks > dict(above=35,descend=20,grasp=10,lift=18,carry=25,drop=18,release=12,retreat=18).get(self.phase,1000):
            self.retry(); recovering = True
        phase = self.phase
        peers = np.flatnonzero(tags == tags[j])
        slot = int(np.flatnonzero(peers == j)[0])
        offset = (slot-(len(peers)-1)/2)*.07
        hover = np.array([*boxes[j,:2], .24])
        pickup = np.array([*boxes[j,:2], max(.06, float(boxes[j,2])+.029)])
        carry = np.array([bins[tags[j],0], bins[tags[j],1]+offset, .24])
        drop = np.array([carry[0],carry[1],.08])

        def move(goal, grip, cap=1.):
            error = goal-tcp
            # Faster transit, original proportional gain for the final 2 cm.
            gains = np.where(np.abs(error) > .02, self.gain, 1.)
            return np.r_[(error*gains/.1).clip(-cap,cap),grip].astype(np.float32)

        if phase == 'retreat':
            action = move(np.array([tcp[0],tcp[1],.24]),1)
            if tcp[2] >= .21:
                self.goto('above')
        elif phase == 'above':
            action = move(hover,1)
            if np.linalg.norm(tcp[:2]-hover[:2]) < .005 and tcp[2] >= .21:
                self.goto('descend')
        elif phase == 'descend':
            action = move(pickup,1,.8)
            if abs(tcp[2]-pickup[2]) < .008 and np.linalg.norm(tcp[:2]-pickup[:2]) < .005:
                self.goto('grasp')
        elif phase == 'grasp':
            # Correct lateral drift while closing; never lift just because time passed.
            action = move(pickup,-1,.15)
            if self.held_steps >= 2:
                self.goto('lift')
        elif phase == 'lift':
            action = move(np.array([tcp[0],tcp[1],.24]),-1)
            if grasped[j] and boxes[j,2] > .10 and tcp[2] >= .23:
                self.goto('carry')
        elif phase == 'carry':
            action = move(carry,-1)
            if np.linalg.norm(tcp-carry) < .025:
                self.goto('drop')
        elif phase == 'drop':
            action = move(drop,-1,.8)
            if np.linalg.norm(tcp-drop) < .02:
                self.goto('release')
        elif phase == 'release':
            action = np.array([0,0,.1,1],np.float32)
        else:
            action = np.array([0,0,0,1],np.float32)
        return action, recovering
