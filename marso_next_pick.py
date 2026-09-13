"""T4 next-pick experiment with weighted demonstration sampling and diagnostics."""
import hashlib
import json
from pathlib import Path

from marso_experiment import read_json, save_json
from marso_train_test import TrainTestExperiment
from next_pick_sampling import patched_trainer
from next_pick_diagnostics import progress_summary


class NextPickExperiment(TrainTestExperiment):
    def connect(self):
        from next_pick_sampling import sampling_weights
        # Validate focus settings before creating a run or starting GPU work.
        c = self.focus_config()
        sampling_weights([[0, 0]], [(0, 0, 1)], 1, weight=c['weight'],
                         before=c['before'], after=c['after'],
                         min_grasp=c['min_grasp'], min_release=c['min_release'])
        super().connect()
        print('새 실험: 다음 집기 구간 재표집 / 기본 4스텝마다 재계획 / T4 소형 모델')

    def notebook_snapshot(self):
        from colab_next_pick_layout import make_notebook
        return make_notebook(self.sources, self.cfg)

    def focus_config(self):
        return dict(enabled=self.cfg['focus_sampling'], weight=self.cfg['focus_weight'],
                    before=self.cfg['focus_before_steps'], after=self.cfg['focus_after_steps'],
                    min_grasp=self.cfg['focus_min_grasp_steps'],
                    min_release=self.cfg['focus_min_release_steps'])

    def _stage_helpers(self):
        super()._stage_helpers()
        for name in ('next_pick_sampling.py', 'next_pick_diagnostics.py'):
            (self.repo/name).write_text(self.sources[name], encoding='utf-8')

    def training_script(self, level, flags):
        folder = self.run_dir/level
        config_path = folder/'sampling_config.json'
        save_json(config_path, self.focus_config())
        source = (self.base/'train.py').read_text(encoding='utf-8')
        patched = patched_trainer(source, config_path)
        script = self.base/'train_next_pick.py'
        script.write_text(patched, encoding='utf-8')
        (self.base/'next_pick_sampling.py').write_text(self.sources['next_pick_sampling.py'], encoding='utf-8')
        # The original train.py is unchanged; retain exact generated training code on Drive.
        (folder/'train_next_pick.py').write_text(patched, encoding='utf-8')
        (folder/'next_pick_sampling.py').write_text(self.sources['next_pick_sampling.py'], encoding='utf-8')
        save_json(folder/'training_provenance.json', dict(
            repo_commit=self.cfg['repo_commit'], config=self.cfg,
            official_trainer_sha256=hashlib.sha256(source.encode()).hexdigest(),
            patched_trainer_sha256=hashlib.sha256(patched.encode()).hexdigest(),
            sampling_sha256=hashlib.sha256(self.sources['next_pick_sampling.py'].encode()).hexdigest()))
        return script

    def _job(self, level, checkpoint, policy, seeds, label):
        job = super()._job(level, checkpoint, policy, seeds, label)
        job['diagnostics_sources_sha256'] = hashlib.sha256(
            (self.sources['next_pick_sampling.py']+self.sources['next_pick_diagnostics.py']).encode()).hexdigest()
        identity = {k: v for k, v in job.items() if k not in ('fingerprint', 'output', 'video')}
        job['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return job

    def _trial(self, level, checkpoint, policy, seeds, label):
        result = super()._trial(level, checkpoint, policy, seeds, label)
        metrics = read_json(self.run_dir/level/(label+'.json'))
        progress = progress_summary(metrics['episodes'])
        save_json(self.run_dir/level/(label+'_progress.json'), dict(
            protocol=metrics['protocol'], complete=metrics['complete'], progress=progress))
        return result

    def _show_progress(self, level, label):
        metrics = read_json(self.run_dir/level/(label+'.json'))
        if not metrics or not metrics.get('episodes'):
            return
        p = progress_summary(metrics['episodes'])
        print(f"[{level}] {len(metrics['episodes'])}회 중 1개 이상 정답 분류 {p['at_least_one_sorted_rate']:.1%} / "
              f"2개 이상 {p['at_least_two_sorted_rate']:.1%} / 1개만 {p['exactly_one_sorted_rate']:.1%}")
        conditional = p.get('second_cycle_given_first_rate')
        if conditional is not None:
            print(f"첫 안정적 집기 이후 두 번째 집기 사이클 관측: {conditional:.1%}")
            print('집기 사이클은 같은 상자를 다시 잡는 경우도 포함합니다. 실제 분류 성과는 위 정답 개수로 확인하세요.')
        print('에피소드별 집기 시점·집게 명령 전환:', self.run_dir/level/(label+'.json'))

    def test(self, level):
        result = super().test(level)
        if result:
            self._show_progress(level, 'test_metrics')
        return result

    def evaluate(self, level):
        result = super().evaluate(level)
        self._show_progress(level, 'metrics')
        return result

    def show_tests(self, videos=False):
        super().show_tests(videos=videos)
        for level in ('easy', 'medium', 'hard'):
            self._show_progress(level, 'test_metrics')
