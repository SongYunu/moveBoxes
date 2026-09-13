"""Identifiers only. Deployment imports no expert or geometry-based controller."""
PICK, CARRY, PLACE, DONE = range(4)
HOLD, COMPLETE, RECOVER = range(3)
STAGES = ('pick', 'carry', 'place', 'done')
GATES = ('hold', 'complete', 'recover')
