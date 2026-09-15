"""Separate stage ACT workflow; existing DP and ACT ver2 are preserved."""
import hashlib
import json
import sys
from pathlib import Path
from marso_github import GitHubExperiment, github_token
from act_v2_experiment import source_bundle as act_sources
from marso_train_test import TrainTestExperiment
from marso_experiment import LEVELS, read_json, save_json
from github_store import GitHubStore
from build_github_notebook import CONFIG as LEGACY_CONFIG


class StageExperiment(GitHubExperiment):
    def __init__(self, cfg, sources):
        defaults = dict(action_training_mode='prior', repair_iters=2000, allow_zero_success_evaluation=False)
        super().__init__(dict(LEGACY_CONFIG, **dict(defaults, **cfg)), sources)

    def connect(self):
        if self.connected:
            return
        c = self.cfg
        if c['width'] % c['heads'] or not 1 <= c['ensemble_window'] <= c['chunk_size']:
            raise ValueError('ACT width/heads or ensemble_window/chunk_size mismatch')
        if min(c['batch_size'],c['history'],c['chunk_size'],c['layers'],c['latent_dim'],
               c['save_freq'],c['validation_batches'],c['warmup_steps'],*c['total_iters'].values()) < 1:
            raise ValueError('Training sizes and intervals must be positive')
        if not c['ensemble_candidates'] or any(not 1 <= n <= c['chunk_size'] for n in c['ensemble_candidates']):
            raise ValueError('Invalid temporal ensemble candidates')
        if c['position_noise'] < 0 or c['temporal_decay'] < 0 or c['kl_weight'] < 0:
            raise ValueError('Noise and regularization must be nonnegative')
        if not 0 < c['gate_threshold'] < 1 or not 0 < c['stage_threshold'] < 1:
            raise ValueError('Gate thresholds must be in (0,1)')
        if not 0 <= c['noise_probability'] <= 1 or not 0 <= c['drop_probability'] <= 1 or c['action_noise_std'] < 0:
            raise ValueError('Invalid perturbation configuration')
        if self.remote_enabled:
            github_token()
            self.store = GitHubStore(c['github_repository'], 'run-'+self.run_dir.name)
            self.store.load()
            if not any(self.run_dir.glob('*')):
                print('GitHub 복원:', self.store.restore(self.run_dir), 'files')
        try:
            TrainTestExperiment.connect(self)
            self.sync_common()
        except BaseException:
            self.connected = False
            raise
        print('단계 ACT: 학습한 완료/복구 판단 + 집기/운반/놓기 행동, 별도 실험')

    def notebook_snapshot(self):
        from build_stage_notebook import make_notebook
        cfg = dict(self.cfg, project_ref=self.cfg.get('project_commit', self.cfg['project_ref']))
        return make_notebook(cfg)

    def _stage_helpers(self):
        super()._stage_helpers()
        for name, code in self.sources.items():
            if name.startswith(('act_v2_', 'stage_')):
                (self.repo/name).write_text(code, encoding='utf-8')

    def model_config(self, level):
        return dict(state_dim={'easy':54,'medium':72,'hard':90}[level],
                    **{k:self.cfg[k] for k in ('history','chunk_size','width','heads','layers','latent_dim')})

    def policy_config(self, level):
        return dict(model_config=self.model_config(level), temporal_decay=self.cfg['temporal_decay'],
                    ensemble_window=self.cfg['ensemble_window'], gate_threshold=self.cfg['gate_threshold'],
                    stage_threshold=self.cfg['stage_threshold'], act_horizon=1, num_inference_steps=1)

    def train(self, level):
        self._ready()
        assert level in LEVELS
        folder = self.run_dir/level
        data = self.data/level/'trajectory.state.pd_ee_delta_pos.physx_cuda.h5'
        if not data.exists():
            raise FileNotFoundError('05 셀에서 데이터 다운로드를 먼저 실행하세요.')
        script = self.repo/'stage_train.py'
        if not script.exists():
            raise FileNotFoundError('04 셀에서 단계 ACT 실행 코드를 설치하세요.')
        cfg = {k:self.cfg[k] for k in ('seed','batch_size','lr','save_freq','warmup_steps','kl_weight',
            'position_noise','validation_batches','amp','console_interval_seconds','stage_loss_weight','gate_loss_weight',
            'action_training_mode')}
        if 'first_pick_fraction' in self.cfg:
            cfg['first_pick_fraction'] = self.cfg['first_pick_fraction']
        smoke = self.cfg['profile']=='smoke'
        cfg['total_iters'] = 100 if smoke else self.cfg['total_iters'][level]
        if smoke:
            cfg.update(save_freq=100, validation_batches=2)
        job = dict(folder=str(folder), data=str(data), model_config=self.model_config(level), train_config=cfg,
            policy_config=self.policy_config(level), num_demos=4 if smoke else self.cfg['num_demos'],
            source_sha256=hashlib.sha256(''.join(self.sources[n] for n in
                ('act_v2_model.py','act_v2_data.py','stage_schema.py','stage_labels.py','stage_model.py','stage_data.py','stage_train.py')).encode()).hexdigest())
        recovery = folder/'collection/manifest.json'
        initial = folder/'initial_model.pt'
        if initial.exists():
            from marso_experiment import digest
            job.update(warm_start=str(initial), warm_start_sha256=digest(initial))
        if not smoke:
            manifest = read_json(recovery, {})
            if not manifest.get('complete'):
                raise RuntimeError(f'[{level}] 먼저 이 난이도의 복구 시연 수집 셀을 실행하세요.')
            job['recovery_manifest'] = str(recovery)
            from marso_experiment import digest
            job['recovery_manifest_sha256'] = digest(recovery)
        folder.mkdir(parents=True, exist_ok=True)
        previous = read_json(folder/'stage_train_job.json')
        if (folder/'checkpoints/latest.pt').exists():
            keys = ('model_config','train_config','num_demos','source_sha256','recovery_manifest_sha256','warm_start_sha256')
            if not previous or any(previous.get(k)!=job.get(k) for k in keys):
                raise ValueError('이 run_name의 저장된 학습 설정/코드와 다릅니다. 새 실험은 새 run_name을 사용하세요.')
            if (folder/'training_complete.json').exists():
                print(f'[{level}] 완료된 단계 ACT 모델 재사용. 다음 테스트/평가 셀을 실행하세요.')
                return
        with self.persist_operation(level):
            save_json(folder/'stage_train_job.json', job)
            try:
                self.run([sys.executable, str(script), str(folder/'stage_train_job.json')], cwd=self.repo,
                         log=folder/'train.log')
            except BaseException as error:
                save_json(folder/'train_status.json', dict(status='interrupted',error=str(error)))
                raise
        print(f'[{level}] 단계 ACT 학습 완료. 다음 테스트 셀을 실행하세요.')

    def _test_candidate(self, level):
        folder = self.run_dir/level
        selected = read_json(folder/'selection.json', {}).get('selected')
        if selected:
            checkpoint = folder/'checkpoints'/Path(selected['checkpoint']).name
            if checkpoint.exists() and self.checkpoint_hash(checkpoint)==selected.get('checkpoint_sha256'):
                return checkpoint, selected['policy_config']
        for name in ('best_val.pt','latest.pt'):
            checkpoint = folder/'checkpoints'/name
            if checkpoint.exists():
                job = read_json(folder/'stage_train_job.json')
                if not job:
                    raise ValueError('Saved ACT training architecture is missing')
                return checkpoint, dict(job['policy_config'])
        return None

    def _job(self, level, checkpoint, policy, seeds, label):
        job = super()._job(level, checkpoint, policy, seeds, label)
        job['model_sources_sha256'] = hashlib.sha256(''.join(self.sources[n] for n in ('act_v2_model.py','stage_model.py','stage_policy.py','stage_schema.py')).encode()).hexdigest()
        identity = {k:v for k,v in job.items() if k not in ('fingerprint','output','video')}
        job['fingerprint'] = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        return job

    def evaluate(self, level):
        self._ready()
        assert level in LEVELS
        with self.persist_operation(level):
            restored = self._saved_final(level)
            if restored:
                print(f'[{level}] 저장된 최종 평가 재사용: {restored[0]["sort_accuracy"]:.1%}')
                return restored[0]
            candidate = self._test_candidate(level)
            if candidate is None:
                print(f'[{level}] 먼저 이 난이도의 학습 셀을 실행하세요.')
                return None
            folder = self.run_dir/level
            if not self.cfg['allow_zero_success_evaluation']:
                quick = self.test(level)
                if quick['sort_accuracy'] <= 0:
                    save_json(folder/'evaluation_gate.json', dict(passed=False, reason='zero_correct_in_quick_test',
                        checkpoint=str(candidate[0]), checkpoint_sha256=self.checkpoint_hash(candidate[0])))
                    print(f'[{level}] 빠른 테스트 정답 분류 0: 긴 튜닝/100회 평가는 실행하지 않습니다. 진단을 확인하세요.')
                    return None
            trials, hashes = [], set()
            for name in ('best_val.pt','latest.pt'):
                ck = folder/'checkpoints'/name
                if not ck.exists():
                    continue
                digest = self.checkpoint_hash(ck)
                if digest in hashes:
                    continue
                hashes.add(digest)
                for window in self.cfg['ensemble_candidates']:
                    policy = dict(candidate[1], ensemble_window=window)
                    trials.append(self._trial(level, ck, policy, self.seeds(True), f'tune_{ck.stem}_w{window}'))
            winner = max(trials,key=lambda r:r['score'])
            if winner['score'] <= 0 and not self.cfg['allow_zero_success_evaluation']:
                save_json(folder/'evaluation_gate.json', dict(passed=False, reason='zero_correct_in_tuning', trials=trials))
                print(f'[{level}] 모든 튜닝 결과가 0: 최종 100회 평가는 실행하지 않습니다.')
                return None
            save_json(folder/'selection.json', dict(selected=winner,trials=trials,tuning_seeds=self.seeds(True)))
            save_json(folder/'checkpoints/policy_config.json', winner['policy_config'])
            self._trial(level, Path(winner['checkpoint']), winner['policy_config'], self.seeds(False), 'metrics')
            self._show_progress(level, 'metrics')
            if self.cfg['record_eval_video']:
                self.record_video(level)
            return read_json(folder/'metrics.json')

    def repair(self, level):
        """Reuse old successful collection and weights in a separate prior-only run."""
        import copy, shutil
        from marso_experiment import digest
        self._ready()
        assert level in LEVELS
        source = self.run_dir/level
        previous = read_json(source/'stage_train_job.json')
        manifest = read_json(source/'collection/manifest.json', {})
        checkpoint = source/'checkpoints/latest.pt'
        if not previous or not checkpoint.exists() or not manifest.get('complete'):
            raise RuntimeError(f'[{level}] 보정할 기존 모델/성공 시연이 없습니다. 기존 run_name으로 03 셀에서 복원하세요.')
        cfg = copy.deepcopy(self.cfg)
        cfg.update(run_name=self.cfg['run_name']+'_prior_fix_v1', action_training_mode='prior',
                   save_freq=500, warmup_steps=100)
        cfg.update({k:v for k,v in previous['model_config'].items() if k != 'state_dim'})
        cfg['total_iters'] = {k:self.cfg['repair_iters'] for k in LEVELS}
        child = type(self)(cfg, self.sources)
        child.connect()
        child._stage_helpers()
        target = child.run_dir/level
        target.mkdir(parents=True, exist_ok=True)
        with child.persist_operation(level):
            for entry in manifest['episodes']:
                path = source/'collection'/entry['file']
                if path.resolve().parent != (source/'collection').resolve() or digest(path) != entry['sha256']:
                    raise ValueError('Saved collection digest/path mismatch')
            pairs = [(checkpoint, target/'initial_model.pt')]
            pairs += [(p,target/'collection'/p.name) for p in (source/'collection').iterdir() if p.suffix in ('.json','.npz')]
            for src,dst in pairs:
                if dst.exists():
                    if digest(src) != digest(dst):
                        raise ValueError('Existing repair input differs; use a separate run_name')
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src,dst)
            save_json(target/'repair_origin.json', dict(run=str(self.run_dir), checkpoint=str(checkpoint),
                checkpoint_sha256=digest(checkpoint), objective='supervise zero-latent deployment actions'))
        print(f'[{level}] 기존 성공 시연 재사용 → 별도 보정 학습 {cfg["total_iters"][level]}회 → 빠른 테스트')
        child.train(level)
        child.test(level)
        child.diagnose(level)
        print('보정 결과:', child.run_dir, '/ 성공 확인 뒤 fixed.evaluate(level)로 최종 평가하세요.')
        return child


    def collect(self, level):
        self._ready()
        assert level in LEVELS
        c = self.cfg
        collection = dict(episodes=c['recovery_episodes'], max_attempts=c['recovery_max_attempts'],
            seed_start=c['recovery_seed_start']+LEVELS.index(level)*10000,
            max_steps=c['collection_max_steps'][level], noise_probability=c['noise_probability'],
            action_noise_std=c['action_noise_std'], drop_probability=c['drop_probability'])
        if collection['episodes'] < 2 or collection['max_attempts'] < collection['episodes']:
            raise ValueError('Collection needs at least two episodes and enough attempts')
        collection_seeds = set(range(collection['seed_start'], collection['seed_start']+collection['max_attempts']))
        if collection_seeds.intersection(self.test_seeds()+self.seeds(True)+self.seeds(False)):
            raise ValueError('Collection seeds must be separate from test/tuning/final evaluation')
        folder = self.run_dir/level
        job = dict(level=level, folder=str(folder/'collection'), collection_config=collection,
            repo_commit=c['repo_commit'], source_sha256=hashlib.sha256(''.join(self.sources[n] for n in
                ('stage_teacher.py','stage_labels.py','stage_collect.py','stage_schema.py')).encode()).hexdigest())
        with self.persist_operation(level):
            save_json(folder/'collection_job.json', job)
            self.run([sys.executable, 'stage_collect.py', str(folder/'collection_job.json')],
                     cwd=self.repo, log=folder/'collection.log')

    def package(self):
        # Data-generation code is not part of the submitted learned controller.
        import os, zipfile
        from marso_experiment import Experiment
        path = Experiment.package(self)
        if path is None:
            return None
        excluded = {'stage_teacher.py','stage_collect.py','stage_labels.py','stage_data.py','stage_train.py'}
        temporary = Path(str(path)+'.tmp')
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as dest:
            for entry in source.infolist():
                if entry.filename not in excluded:
                    dest.writestr(entry, source.read(entry.filename))
        os.replace(temporary, path)
        self.sync_common()
        return path


def source_bundle():
    root = Path(__file__).parent
    bundle = act_sources()
    for path in root.glob('stage_*.py'):
        bundle[path.name] = path.read_text(encoding='utf-8')
    bundle['build_stage_notebook.py'] = (root.parents[1]/'build_stage_notebook.py').read_text(encoding='utf-8')
    bundle['colab_policy.py'] = bundle['stage_policy.py']
    bundle['colab_eval_modular.py'] = bundle['stage_eval.py']
    bundle['colab_trace.py'] = "import json, sys\nfrom pathlib import Path\nfrom stage_eval import main\njob = json.loads(Path(sys.argv[1]).read_text())\njob['trace_only'] = True\nmain(job)\n"
    bundle['test_policy_runtime.py'] = bundle['stage_runtime_check.py']
    return bundle
