"""Offline target-balanced sampling of every parcel's pick, including later cycles."""
from collections import defaultdict
from stage_data import StageWindows as OriginalWindows
from stage_schema import PICK


class StageWindows(OriginalWindows):
    def __init__(self,trajectories,ids,history,chunk,training=True,first_pick_fraction=0.):
        super().__init__(trajectories,ids,history,chunk,training,first_pick_fraction)
        self.pick_order_counts = {}
        if not training or not first_pick_fraction:
            return
        groups = defaultdict(list)
        episodes = defaultdict(set)
        for i in ids:
            item = trajectories[i]
            seen = []
            for t in range(len(item['actions'])):
                if int(item['stage'][t]) != PICK:
                    continue
                target = int(item['target'][t])
                if target not in seen:
                    seen.append(target)
                # A retry retains its original parcel order, rather than being
                # counted as a new "next box" merely because grasp flickered.
                order = seen.index(target)
                groups[order].append((i,t))
                episodes[order].add(i)
        self.pick_order_counts = {str(k):len(v) for k,v in sorted(groups.items())}
        self.pick_order_episodes = {str(k):len(episodes[k]) for k in sorted(groups)}
        if not groups:
            raise ValueError('No labelled pick frames for all-pick sampling')
        # Equal exposure per encountered parcel order. The remaining batch
        # fraction still uses the original full/recovery/transition mixture.
        longest = max(map(len,groups.values()))
        self.first_pick = [pool[j % len(pool)] for _,pool in sorted(groups.items())
                           for j in range(longest)]
        print('All-parcel pick audit:',dict(
            target_order_unique_episodes=self.pick_order_episodes,
            target_order_windows=self.pick_order_counts,
            total_pick_windows=sum(self.pick_order_counts.values()),
            recovery_windows=len(self.recovery),transition_windows=len(self.transitions),
            all_windows=len(self.indices),focus_fraction=first_pick_fraction),flush=True)
