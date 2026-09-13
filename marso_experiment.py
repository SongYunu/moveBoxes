"""Drive-backed orchestration, independent of notebook cell execution history."""
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import deque
from pathlib import Path

LEVELS = ('easy', 'medium', 'hard')
WEIGHTS = {'easy': 0.2, 'medium': 0.3, 'hard': 0.5}
ARCH = ('obs_horizon', 'pred_horizon', 'diffusion_step_embed_dim', 'unet_dims', 'n_groups')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def completed_metrics(metrics, seeds=None):
    if not metrics or metrics.get('complete') is False:
        return False
    rows = metrics.get('episodes', [])
    expected = seeds if seeds is not None else metrics.get('protocol', {}).get('seeds')
    return (isinstance(expected, list) and len(expected) > 0
            and metrics.get('n_episodes') == len(expected)
            and [r.get('seed') for r in rows] == expected
            and isinstance(metrics.get('sort_accuracy'), (float, int))
            and 0 <= metrics['sort_accuracy'] <= 1)


class Experiment:
    def __init__(self, config, sources):
        self.cfg = json.loads(json.dumps(config))
        self.sources = sources
        self.repo = Path(config['repo_dir'])
        self.data = Path(config['data_dir'])
        self.run_dir = Path(config['output_root']) / (config['run_name']+'_'+config['profile'])
        self.base = self.repo/'il/baselines/diffusion_policy'
        self.connected = False
        self.hashes = {}

    def connect(self):
        c = self.cfg
        assert re.fullmatch(r'[A-Za-z0-9_-]+', c['run_name'])
        assert c['profile'] in ('smoke', 'benchmark')
        assert c['obs_horizon'] in (1, 2)
        assert len(c['unet_dims']) == 3 and c['pred_horizon'] % 8 == 0
        assert all(d % c['n_groups'] == 0 for d in c['unet_dims'])
        assert c['obs_horizon']+c['act_horizon']-1 <= c['pred_horizon']
        assert c['chunk_candidates'] and c['max_checkpoints'] > 0
        assert all(1 <= n <= 100 for n in c['denoising_candidates'])
        assert c['tuning_episodes'] > 0 and c['benchmark_episodes'] > 0
        assert not set(self.seeds(True)).intersection(self.seeds(False)), '검증·최종 시드가 겹칩니다.'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.session = self.run_dir/'sessions'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        self.session.mkdir(parents=True, exist_ok=True)
        save_json(self.session/'config.json', c)
        save_json(self.run_dir/'config_current.json', c)
        if not (self.run_dir/'config.json').exists():
            save_json(self.run_dir/'config.json', c)
        for name, source in self.sources.items():
            (self.session/name).write_text(source, encoding='utf-8')
        # Produce a complete notebook, with all helper sources embedded, on Drive.
        notebook = self.notebook_snapshot()
        notebook_name = notebook['metadata']['colab']['name']
        self.notebook_path = self.run_dir/notebook_name
        save_json(self.notebook_path, notebook)
        save_json(self.session/notebook_name, notebook)
        self.connected = True
        print('로컬 작업 경로:', self.run_dir)
        print('실행 노트북 사본:', self.notebook_path)
        print('새 런타임에서는 01~03 셀로 결과를 복원하고, 학습·평가는 04~05 셀 준비 후 실행합니다.')

    def notebook_snapshot(self):
        from colab_layout import make_notebook
        return make_notebook(self.sources, self.cfg)

    def _ready(self):
        if not self.connected:
            raise RuntimeError('먼저 03 · 저장소 연결 셀을 실행하세요.')

    def run(self, command, cwd=None, log=None):
        command = list(map(str, command))
        log = Path(log or self.session/'commands.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        print('실행:', ' '.join(command))
        print('전체 로그:', log)
        recent = deque(maxlen=30)
        last_print = last_flush = time.monotonic()
        with log.open('a', encoding='utf-8') as handle:
            handle.write('\nCOMMAND: '+' '.join(command)+'\n')
            handle.flush()
            with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors='replace', bufsize=1) as process:
                try:
                    for line in process.stdout:
                        handle.write(line)
                        line = line.strip()
                        if line:
                            recent.append(line)
                        now = time.monotonic()
                        if now-last_flush >= 5:
                            handle.flush()
                            last_flush = now
                        if line and now-last_print >= self.cfg['console_interval_seconds']:
                            print(line, flush=True)
                            last_print = now
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            if code:
                raise RuntimeError(f'종료 코드 {code}; 로그: {log}\n'+'\n'.join(recent))
        print('완료')

    def install(self):
        self._ready()
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError('Colab 런타임 2026.07 / Python 3.12를 선택하세요.')
        self.run(['nvidia-smi'], log=self.session/'gpu.log')
        if not self.repo.exists():
            self.run(['git', 'clone', self.cfg['repo_url'], self.repo])
        head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=self.repo, text=True).strip()
        if head != self.cfg['repo_commit']:
            self.run(['git','checkout','--detach',self.cfg['repo_commit']], cwd=self.repo)
        self.run([sys.executable,'-m','pip','install',*self.cfg['packages']], log=self.session/'install.log')
        self.run([sys.executable,'-m','pip','install','-e',self.repo], log=self.session/'install_repo.log')
        self._stage_helpers()
        freeze = subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True)
        (self.session/'pip-freeze.txt').write_text(freeze, encoding='utf-8')
        os.environ['DISPLAY'] = ''
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        print('설치 완료. 다음 데이터·GPU 확인 셀을 실행하세요.')

    def _stage_helpers(self):
        for name in ('colab_policy.py','colab_eval_modular.py','marso_experiment.py',
                     'download_dataset_colab.py','test_policy_runtime.py'):
            (self.repo/name).write_text(self.sources[name], encoding='utf-8')
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))

    def prepare_data(self):
        self._ready()
        from download_dataset_colab import inspect, stage
        expected = [self.data/l/'trajectory.state.pd_ee_delta_pos.physx_cuda.h5' for l in LEVELS]
        if all(p.exists() and p.with_suffix('.json').exists() for p in expected):
            manifest = inspect(self.data)
        else:
            manifest = stage(self.cfg['data_source'], self.data)
        save_json(self.run_dir/'dataset_manifest.json', manifest)
        for level in LEVELS:
            found = next((r for r in manifest if r['level']==level and '.state.' in r['file']), None)
            print(level, f"{found['episodes']} demos" if found else 'state 데이터 없음')

    def check_runtime(self):
        self._ready()
        self.run([sys.executable, str(self.repo/'test_policy_runtime.py')], cwd=self.repo,
                 log=self.session/'policy_check.log')
        code = '''import torch
from warehouse_sort.utils import compose_cfg, make_env
assert torch.cuda.is_available(), 'GPU runtime required'
for level, dims in [('easy',54),('medium',72),('hard',90)]:
    cfg = compose_cfg(['difficulty='+level, 'num_envs=1'])
    env, _ = make_env(cfg, 'state', cfg.randomization, num_envs=1, render_mode='rgb_array')
    try:
        obs, _ = env.reset(seed=42)
        assert tuple(obs.shape)==(1,dims)
        env.step(torch.zeros((1,4),device='cuda'))
        assert env.render() is not None
        print(level, 'GPU/render OK')
    finally:
        env.close()
'''
        self.run([sys.executable,'-c',code], cwd=self.repo, log=self.session/'gpu_check.log')

    def train_flags(self, level):
        c = self.cfg
        smoke = c['profile']=='smoke'
        flags = dict(exp_name=f"{c['run_name']}_{c['profile']}_{level}_s{c['seed']}",
            seed=c['seed'], env_id='WarehouseSort-v1',
            demo_path=str(self.data/level/'trajectory.state.pd_ee_delta_pos.physx_cuda.h5'),
            control_mode='pd_ee_delta_pos', sim_backend='gpu', max_episode_steps=c['max_episode_steps'][level],
            total_iters=100 if smoke else c['total_iters'][level],
            batch_size=min(32,c['batch_size']) if smoke else c['batch_size'], lr=c['lr'],
            obs_horizon=c['obs_horizon'], act_horizon=c['act_horizon'], pred_horizon=c['pred_horizon'],
            unet_dims=c['unet_dims'], diffusion_step_embed_dim=c['diffusion_step_embed_dim'], n_groups=c['n_groups'],
            num_eval_envs=1 if smoke else c['num_eval_envs'], num_eval_episodes=1 if smoke else c['train_eval_episodes'],
            eval_freq=100 if smoke else c['eval_freq'], save_freq=100 if smoke else min(c['save_freq'],c['total_iters'][level]),
            log_freq=50 if smoke else c['log_freq'], capture_video=False, num_dataload_workers=0, track=False)
        if smoke or c['num_demos'] is not None:
            flags['num_demos'] = min(4,c['num_demos'] or 4) if smoke else c['num_demos']
        return flags

    def training_script(self, level, flags):
        return self.base/'train.py'

    def train(self, level):
        self._ready()
        assert level in LEVELS
        folder = self.run_dir/level
        ckdir = folder/'checkpoints'
        checkpoints = list(ckdir.glob('*.pt'))
        if checkpoints:
            complete = (folder/'training_complete.json').exists()
            print(f'[{level}] 저장 모델 {len(checkpoints)}개 재사용; 완료 기록: {complete}. 학습 재실행 생략.')
            if not complete:
                print('완료 여부를 확인할 수 없는 기존 모델입니다. 평가 가능하며, 새 학습은 새 run_name을 사용하세요.')
            return
        flags = self.train_flags(level)
        data = Path(flags['demo_path'])
        if not data.exists() or not data.with_suffix('.json').exists():
            raise FileNotFoundError(f'{level} 데이터가 없습니다. 05 셀을 실행하세요: {data}')
        if not (self.base/'train.py').exists():
            raise FileNotFoundError('공식 trainer가 없습니다. 04 셀을 실행하세요.')
        script = self.training_script(level, flags)
        ckdir.mkdir(parents=True, exist_ok=True)
        link = self.base/'runs'/flags['exp_name']
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != folder.resolve():
                raise RuntimeError(f'학습 출력 링크가 다른 위치를 가리킵니다: {link}')
        elif link.exists():
            raise FileExistsError(f'기존 학습 출력 폴더를 확인하세요: {link}')
        else:
            link.symlink_to(folder, target_is_directory=True)
        policy = {k:flags[k] for k in ARCH}
        policy.update(act_horizon=flags['act_horizon'], num_inference_steps=16)
        save_json(ckdir/'policy_config.json',policy)
        save_json(folder/'train_args.json',flags)
        save_json(folder/'train_status.json',{'status':'running','session':str(self.session)})
        command = [sys.executable,str(script)]
        for key,value in flags.items():
            flag = '--'+key.replace('_','-')
            if isinstance(value,bool):
                command.append(flag if value else '--no-'+key.replace('_','-'))
            elif isinstance(value,list):
                command.extend([flag,*map(str,value)])
            else:
                command.extend([flag,str(value)])
        try:
            self.run(command,cwd=self.base,log=folder/'train.log')
            if not list(ckdir.glob('*.pt')):
                raise RuntimeError('trainer가 종료됐지만 checkpoint가 생성되지 않았습니다.')
        except BaseException as error:
            save_json(folder/'train_status.json',{'status':'interrupted','error':str(error)})
            raise
        save_json(folder/'training_complete.json',flags)
        save_json(folder/'train_status.json',{'status':'complete'})
        print(f'[{level}] 학습 완료 → {ckdir}')

    def seeds(self, tuning):
        key = 'tuning_seed_start' if tuning else 'eval_seed_start'
        count = 2 if self.cfg['profile']=='smoke' else self.cfg['tuning_episodes' if tuning else 'benchmark_episodes']
        return list(range(self.cfg[key],self.cfg[key]+count))

    def checkpoint_hash(self, path):
        path = Path(path)
        stat = path.stat()
        key = (str(path),stat.st_size,stat.st_mtime_ns)
        if key not in self.hashes:
            self.hashes[key] = digest(path)
        return self.hashes[key]

    def _job(self, level, checkpoint, policy, seeds, label):
        folder = self.run_dir/level
        identity = dict(level=level, checkpoint=str(checkpoint), checkpoint_sha256=self.checkpoint_hash(checkpoint),
                        policy_config=policy, seeds=seeds, max_steps=self.cfg['max_episode_steps'][level],
                        repo_commit=self.cfg['repo_commit'], policy_sha256=hashlib.sha256(self.sources['colab_policy.py'].encode()).hexdigest(),
                        evaluator_sha256=hashlib.sha256(self.sources['colab_eval_modular.py'].encode()).hexdigest())
        fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        return dict(identity,fingerprint=fingerprint,output=str(folder/(label+'.json')),
                    video=False)

    def _trial(self, level, checkpoint, policy, seeds, label):
        job = self._job(level,checkpoint,policy,seeds,label)
        previous = read_json(job['output'])
        if previous and previous.get('protocol',{}).get('fingerprint')==job['fingerprint'] and completed_metrics(previous,seeds):
            metrics = previous
            print(f'[{level}] 평가 재사용: {label} {metrics["sort_accuracy"]:.1%}')
        else:
            if (previous and previous.get('protocol', {}).get('fingerprint') == job['fingerprint']
                    and previous.get('episodes')):
                done = len(previous['episodes'])
                print(f'[{level}] {label}: 저장 {done}/{len(seeds)}회, 남은 {len(seeds)-done}회 이어 실행')
            job_path = self.run_dir/level/'eval_job.json'
            save_json(job_path,job)
            self.run([sys.executable,'colab_eval_modular.py',str(job_path)],cwd=self.repo,
                     log=self.run_dir/level/(label+'.log'))
            metrics = read_json(job['output'])
            if not completed_metrics(metrics,seeds):
                raise RuntimeError(f'평가가 완료되지 않았습니다: {job["output"]}. 이 난이도 평가 셀을 다시 실행하세요.')
        return dict(checkpoint=str(checkpoint),chunk=policy['act_horizon'],steps=policy['num_inference_steps'],
                    score=metrics['sort_accuracy'],policy_config=policy,
                    checkpoint_sha256=job['checkpoint_sha256'])

    def _saved_final(self, level):
        folder = self.run_dir/level
        metrics = read_json(folder/'metrics.json')
        selection = read_json(folder/'selection.json',{}).get('selected')
        if not selection or not completed_metrics(metrics,self.seeds(False)):
            return None
        checkpoint = Path(selection['checkpoint'])
        if not checkpoint.exists():
            checkpoint = folder/'checkpoints'/checkpoint.name
        if not checkpoint.exists():
            return None
        protocol = metrics.get('protocol',{})
        if protocol.get('level') != level or protocol.get('max_steps') != self.cfg['max_episode_steps'][level]:
            return None
        if Path(protocol.get('checkpoint','')).name != checkpoint.name:
            return None
        if protocol.get('fingerprint'):
            expected = self._job(level,checkpoint,selection['policy_config'],self.seeds(False),'metrics')
            if expected['fingerprint'] != protocol['fingerprint']:
                return None
        # Legacy t4_eval results contain no hashes. Preserve them with a visible provenance label.
        return metrics,selection,checkpoint

    def evaluate(self, level):
        self._ready()
        assert level in LEVELS
        restored = self._saved_final(level)
        if restored:
            metrics,selection,checkpoint = restored
            origin = '' if metrics.get('protocol',{}).get('fingerprint') else ' (기존 노트북 결과)'
            print(f'[{level}] 완료된 평가 복원{origin}: {metrics["sort_accuracy"]:.1%}; 재계산 생략')
            self.summary()
            return metrics
        folder = self.run_dir/level
        ckdir = folder/'checkpoints'
        checkpoints = list(ckdir.glob('*.pt'))
        if not checkpoints:
            print(f'[{level}] 모델 없음. 바로 위 {level.upper()} 학습 셀을 실행하세요. 다른 난이도 결과는 보존됩니다.')
            self.summary()
            return None
        record = read_json(folder/'train_args.json')
        if not record:
            record = read_json(ckdir/'policy_config.json')
        if not record or any(k not in record for k in ARCH):
            print(f'[{level}] 학습 시 모델 설정 기록을 찾을 수 없습니다: {ckdir}')
            return None
        architecture = {k:record[k] for k in ARCH}
        for chunk in self.cfg['chunk_candidates']:
            if not 1 <= chunk <= record['pred_horizon']-record['obs_horizon']+1:
                raise ValueError('저장 모델의 horizon과 chunk_candidates가 맞지 않습니다.')
        best = ckdir/'best_eval_sort_accuracy.pt'
        candidates = ([best] if best.exists() else []) + sorted(
            (p for p in checkpoints if p.stem.isdigit()),key=lambda p:int(p.stem),reverse=True)
        candidates = candidates[:self.cfg['max_checkpoints']]
        if not candidates:
            print(f'[{level}] 표준 checkpoint 후보가 없습니다.')
            return None
        trials = []
        for i,checkpoint in enumerate(candidates):
            for chunk in self.cfg['chunk_candidates']:
                policy = dict(architecture,act_horizon=chunk,num_inference_steps=16)
                trials.append(self._trial(level,checkpoint,policy,self.seeds(True),f'tune_c{i}_a{chunk}_d16'))
        best_base = max(trials,key=lambda x:(x['score'],-x['steps']))
        for steps in self.cfg['denoising_candidates']:
            if steps!=16:
                policy = dict(best_base['policy_config'],num_inference_steps=steps)
                trials.append(self._trial(level,Path(best_base['checkpoint']),policy,self.seeds(True),f'tune_finalist_d{steps}'))
        winner = max(trials,key=lambda x:(x['score'],-x['steps']))
        save_json(folder/'selection.json',dict(selected=winner,trials=trials,tuning_seeds=self.seeds(True)))
        # The saved deployment config always matches the selected checkpoint.
        save_json(ckdir/'policy_config.json',winner['policy_config'])
        self._trial(level,Path(winner['checkpoint']),winner['policy_config'],self.seeds(False),'metrics')
        metrics = read_json(folder/'metrics.json')
        print(f'[{level}] 최종 {metrics["n_episodes"]}회: {metrics["sort_accuracy"]:.1%} → {folder/"metrics.json"}')
        self.summary()
        if self.cfg['record_eval_video']:
            self.record_video(level)
        return metrics

    def record_video(self, level):
        restored = self._saved_final(level)
        if not restored:
            print(f'[{level}] 완료된 최종 평가가 없습니다.')
            return
        if list((self.run_dir/level/'videos').rglob('*.mp4')):
            return
        _,selection,checkpoint = restored
        job = self._job(level,checkpoint,selection['policy_config'],[self.seeds(False)[0]],'video')
        job['video_only'] = True
        path = self.run_dir/level/'video_job.json'
        save_json(path,job)
        try:
            self.run([sys.executable,'colab_eval_modular.py',str(path)],cwd=self.repo,
                     log=self.run_dir/level/'video.log')
        except Exception as error:
            print(f'[{level}] 점수는 저장되었습니다. 영상만 실패: {error}')

    def summary(self):
        self._ready()
        rows, scores, protocols = [],{},{}
        expected = self.seeds(False)
        for level in LEVELS:
            folder = self.run_dir/level
            metrics = read_json(folder/'metrics.json')
            policy = metrics.get('protocol',{}) if metrics else {}
            valid = (completed_metrics(metrics,expected)
                     and policy.get('level')==level
                     and policy.get('max_steps')==self.cfg['max_episode_steps'][level])
            if valid:
                scores[level] = metrics['sort_accuracy']
                protocols[level] = policy
            elif completed_metrics(metrics):
                status = '저장 점수의 평가 조건이 현재 CONFIG와 다름'
            checkpoints = list((folder/'checkpoints').glob('*.pt'))
            train_complete = (folder/'training_complete.json').exists()
            completed = len(metrics.get('episodes', [])) if metrics else 0
            same_protocol = (policy.get('level') == level
                             and policy.get('max_steps') == self.cfg['max_episode_steps'][level]
                             and policy.get('seeds') == expected)
            partial = (not valid and same_protocol and 0 < completed < len(expected)
                       and [r.get('seed') for r in metrics['episodes']] == expected[:completed])
            if valid:
                status = '평가 완료'
            elif partial:
                status = f'평가 미완료 {completed}/{len(expected)}회 · 해당 평가 셀에서 이어 실행'
            elif not completed_metrics(metrics):
                status = ('학습 완료 · 최종 평가 대기' if train_complete else '학습 완료 기록 없음 · 저장 모델 있음') if checkpoints else '모델 없음'
            rows.append(dict(level=level,train_complete=train_complete,
                             checkpoints=len(checkpoints),status=status,
                             evaluation_episodes=completed, expected_episodes=len(expected),
                             partial_sort_accuracy=metrics.get('sort_accuracy') if partial else None,
                             sort_accuracy=scores.get(level),weight=WEIGHTS[level]))
        summary = dict(profile=self.cfg['profile'],scores=scores,missing_levels=[l for l in LEVELS if l not in scores],
                       weighted_score=sum(WEIGHTS[l]*scores.get(l,0) for l in LEVELS),
                       evaluation='public-config self-evaluation; saved per-level protocols',
                       level_status=rows,protocols=protocols,config=self.cfg)
        save_json(self.run_dir/'summary.json',summary)
        data = io.StringIO()
        writer = csv.writer(data)
        writer.writerow(['level','sort_accuracy','weight','evaluated'])
        for level in LEVELS:
            writer.writerow([level,scores.get(level,0),WEIGHTS[level],level in scores])
        (self.run_dir/'scores.csv').write_text(data.getvalue(),encoding='utf-8')
        return summary

    def show_results(self, videos=False):
        result = self.summary()
        for row in result['level_status']:
            score = '--' if row['sort_accuracy'] is None else f"{row['sort_accuracy']:.2%}"
            if row['partial_sort_accuracy'] is not None:
                score = f"{row['partial_sort_accuracy']:.2%} (중간)"
            print(f"{row['level']:6} | {score:>7} | {row['status']} | 학습완료 기록 {row['train_complete']}")
        print(f"가중 점수: {result['weighted_score']:.4f} / 미평가: {result['missing_levels']}")
        if videos:
            from IPython.display import Video,display
            for level in LEVELS:
                paths = sorted((self.run_dir/level/'videos').rglob('*.mp4'))
                if paths:
                    print(level)
                    display(Video(str(paths[-1]),embed=True,width=640))
        return None

    def package(self):
        self._ready()
        result = self.summary()
        # Rebuild a fresh package so incomplete levels do not inherit stale old files.
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            submission = {'team':self.cfg['team'],'state':{'policy':'colab_policy:load_policy','levels':{}}}
            for level in LEVELS:
                restored = self._saved_final(level)
                if not restored:
                    continue
                _,selection,checkpoint = restored
                destination = target/'checkpoints'/level
                destination.mkdir(parents=True)
                shutil.copy2(checkpoint,destination/checkpoint.name)
                save_json(destination/'policy_config.json',selection['policy_config'])
                submission['state']['levels'][level] = {'checkpoint':f'checkpoints/{level}/{checkpoint.name}'}
                for name in ('metrics.json','selection.json'):
                    shutil.copy2(self.run_dir/level/name,target/f'{level}_{name}')
            if not submission['state']['levels']:
                print('패키징할 평가 완료 모델이 없습니다.')
                return None
            # JSON is valid YAML, and avoids an extra dependency for packaging.
            save_json(target/'submission.yaml',submission)
            save_json(target/'summary.json',result)
            save_json(target/'config.json',self.cfg)
            for name,source in self.sources.items():
                (target/name).write_text(source,encoding='utf-8')
            shutil.copy2(self.notebook_path,target/self.notebook_path.name)
            for name in ('dataset_manifest.json','scores.csv'):
                if (self.run_dir/name).exists():
                    shutil.copy2(self.run_dir/name,target/name)
            freeze_files = sorted((self.run_dir/'sessions').glob('*/pip-freeze.txt'))
            freeze = freeze_files[-1] if freeze_files else self.run_dir/'pip-freeze.txt'
            if freeze.exists():
                shutil.copy2(freeze,target/'pip-freeze.txt')
            (target/'REPRODUCE.txt').write_text(
                'Repository: '+self.cfg['repo_url']+'\nCommit: '+self.cfg['repo_commit']+
                '\nUse the included notebook to install dependencies.\n'
                'For external evaluation call policy.reset() after every environment reset.\n'
                'Incomplete training markers do not mean weights are unusable; consult per-level records.\n',encoding='utf-8')
            import zipfile
            local_zip = Path(temp).parent/('marso_'+uuid.uuid4().hex+'.zip')
            try:
                with zipfile.ZipFile(local_zip,'w',compression=zipfile.ZIP_DEFLATED) as zf:
                    for path in target.rglob('*'):
                        if path.is_file():
                            zf.write(path,path.relative_to(target))
                destination = self.run_dir/('checkpoint_package_'+time.strftime('%Y%m%d_%H%M%S')+'.zip')
                shutil.copy2(local_zip,destination)
            finally:
                local_zip.unlink(missing_ok=True)
        print('저장한 패키지:',destination)
        return str(destination)
