"""Attach simulator-only diagnostics without changing policy observations/actions."""
import torch
from hard_progress import ParcelProgress


def instrument_observer(parent):
    class Observer(parent):
        def reset(self):
            super().reset();self.progress=ParcelProgress()

        def capture(self):
            env=getattr(self,'env',None)
            if env is None:return
            base=env.unwrapped
            grasp=torch.stack([base.agent.is_grasping(p)[0] for p in base.parcels]).detach().cpu().tolist()
            correct=base._placed_correct[0].detach().cpu().tolist()
            self.progress.record([i for i,v in enumerate(grasp) if v],[i for i,v in enumerate(correct) if v],len(self.decisions))

        def act(self,obs,deterministic=True):
            self.capture()
            return super().act(obs,deterministic)

        def report(self):
            return dict(super().report(),parcel_progress=self.progress.report(),
                final_stage=int(self.decisions[-1]['stage']) if self.decisions else 0)
    return Observer
