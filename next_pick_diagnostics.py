"""Read-only rollout diagnostics; observations/actions pass through unchanged."""
from next_pick_sampling import stable_grasp_events


class NextPickObserver:
    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        self.policy.reset()
        self.grasped, self.gripper = [], []

    def act(self, obs, deterministic=True):
        action = self.policy.act(obs, deterministic=deterministic)
        state = obs['state'] if isinstance(obs, dict) else obs
        if state.shape[0] != 1 or state.shape[-1] not in (54, 72, 90):
            raise ValueError('Next-pick diagnostics require one WarehouseSort state environment.')
        self.grasped.append(float(state[0, 25].item()))
        self.gripper.append(float(action[0, -1].item()))
        return action

    def report(self):
        events = stable_grasp_events(self.grasped)
        # Ignore commands near zero; count open<->close command sign reversals.
        signs = [1 if a > 0 else -1 for a in self.gripper if abs(a) >= 0.25]
        return dict(stable_grasp_onsets=events, stable_grasp_cycles=len(events),
                    gripper_sign_reversals=sum(a != b for a, b in zip(signs, signs[1:])),
                    observed_steps=len(self.grasped))


def progress_summary(rows):
    if not rows:
        return None
    first_sorted = sum(r['mean_sorted'] >= 1 for r in rows)
    second_sorted = sum(r['mean_sorted'] >= 2 for r in rows)
    result = dict(episodes=len(rows),
        at_least_one_sorted_rate=first_sorted/len(rows),
        at_least_two_sorted_rate=second_sorted/len(rows),
        exactly_one_sorted_rate=sum(r['mean_sorted'] == 1 for r in rows)/len(rows),
        two_given_one_sorted_rate=second_sorted/first_sorted if first_sorted else None,
        note='Correctly sorted parcel counts. Grasp cycles below may regrasp the same parcel; '
             'grasp observations are measured before each action, excluding the terminal observation.')
    observed = [r['next_pick'] for r in rows if 'next_pick' in r]
    if len(observed) == len(rows):
        first = sum(r['stable_grasp_cycles'] >= 1 for r in observed)
        second = sum(r['stable_grasp_cycles'] >= 2 for r in observed)
        result.update(at_least_one_grasp_cycle_rate=first/len(rows),
            at_least_two_grasp_cycles_rate=second/len(rows),
            second_cycle_given_first_rate=second/first if first else None,
            mean_gripper_sign_reversals=sum(r['gripper_sign_reversals'] for r in observed)/len(rows))
    return result
