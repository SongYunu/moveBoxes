"""Read-only parcel identity diagnostics and score-based comparison."""
import math


class ParcelProgress:
    def __init__(self):
        self.events=[];self.last_ids=();self.completed={};self.actions=0

    def record(self,grasp_ids,correct_ids,actions):
        ids=tuple(sorted(grasp_ids));self.actions=int(actions)
        for parcel in ids:
            if parcel not in self.last_ids:self.events.append(dict(parcel=int(parcel),action=int(actions)))
        self.last_ids=ids
        for parcel in correct_ids:self.completed.setdefault(str(parcel),int(actions))

    def report(self):
        unique=set(e['parcel'] for e in self.events)
        return dict(grasp_events=list(self.events),unique_grasped_parcels=len(unique),
            repeat_grasp_events=len(self.events)-len(unique),sorted_parcel_first_actions=dict(self.completed),
            observed_actions=self.actions,
            note='Contact onsets may include flicker; only official correct parcel counts determine score.')


def summarize(metrics,parcels=6):
    rows=metrics.get('episodes',[])
    if not rows:raise ValueError('No completed episode evidence')
    counts=[int(r['mean_sorted']) for r in rows]
    if any(not 0<=c<=parcels for c in counts):raise ValueError('Invalid correct parcel count')
    return dict(episodes=len(rows),sort_accuracy=sum(counts)/(parcels*len(rows)),
        correct_count_histogram={str(i):counts.count(i) for i in range(parcels+1)},
        all_correct_rate=counts.count(parcels)/len(rows),
        at_least_k_correct={str(i):sum(c>=i for c in counts)/len(rows) for i in range(1,parcels+1)},
        diagnostic_grasps=[dict(seed=r['seed'],correct=int(r['mean_sorted']),
            **r.get('next_pick',{}).get('parcel_progress',{})) for r in rows])


def paired_comparison(baseline,candidate):
    """Fixed-seed development selection; grip events never promote a checkpoint."""
    left,right=baseline['episodes'],candidate['episodes']
    if not baseline.get('complete') or not candidate.get('complete'):raise ValueError('Incomplete comparison')
    if not left or [r['seed'] for r in left]!=[r['seed'] for r in right]:raise ValueError('Paired seeds differ')
    for key in ('level','max_steps'):
        if baseline['protocol'][key]!=candidate['protocol'][key]:raise ValueError('Evaluation conditions differ')
    deltas=[float(b['sort_accuracy'])-float(a['sort_accuracy']) for a,b in zip(left,right)]
    if not all(math.isfinite(x) for x in deltas):raise ValueError('Non-finite comparison')
    wins=sum(d>1e-9 for d in deltas);losses=sum(d< -1e-9 for d in deltas)
    return dict(mean_score_delta=sum(deltas)/len(deltas),wins=wins,losses=losses,ties=len(deltas)-wins-losses,
        promote=sum(deltas)>1e-9 and wins>losses,
        rule='Higher official score and more improving than regressing development seeds; no grip-count tie-break.')
