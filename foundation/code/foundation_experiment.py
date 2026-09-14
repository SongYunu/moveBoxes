"""Colab orchestration only: stdlib, isolated Python environments, compact Releases."""
import hashlib,json,os,secrets,shutil,subprocess,sys,time,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from marso_experiment import read_json as read,save_json as write,digest
from github_store import GitHubStore
from github_data import download_archive

LEVELS=('easy','medium','hard')
SCORE_WEIGHTS={'easy':.2,'medium':.3,'hard':.5}


class FoundationExperiment:
    def __init__(self,cfg):
        self.cfg=dict(cfg);self.backend=cfg['backend'];self.code=ROOT/'foundation/code'
        self.run_dir=Path(cfg['output_root'])/cfg['run_name'];self.run_dir.mkdir(parents=True,exist_ok=True)
        self.envroot=Path(cfg['env_root']);self.model_python=self.envroot/self.backend/'bin/python'
        self.sim_python=self.envroot/'simulator/bin/python';self.repo=Path(cfg['simulator_repo'])
        self.data=Path(cfg['data_dir']);self.store=None
        if self.backend not in ('smol','octo'):raise ValueError('Unknown backend')
        if cfg['micro_batch']*cfg['accumulate']%3:raise ValueError('Effective batch must include equal counts of all three levels')
        if self.backend=='octo' and (cfg['micro_batch']%3 or cfg['accumulate']!=1):raise ValueError('Octo uses micro_batch divisible by 3, accumulate=1')
        if not 1<=cfg['execute_steps']<=cfg['chunk']:raise ValueError('Invalid execution horizon')

    def env(self,model=False):
        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',DISPLAY='',PYOPENGL_PLATFORM='egl',
            TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1',TF_CPP_MIN_LOG_LEVEL='2',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.50',
            PYTHONPATH=os.pathsep.join([str(self.code),str(ROOT),str(self.repo)]))
        # Octo's language encoder is Flax; prevent Transformers from importing TF as a backend.
        if model and self.backend=='octo':env.update(USE_TORCH='0',USE_TF='0',USE_FLAX='1')
        return env

    def run(self,cmd,log,model=False,cwd=None):
        log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);print('실행:',Path(str(cmd[0])).name,*map(str,cmd[1:3]),'· 로그:',log,flush=True)
        with log.open('w',encoding='utf-8') as f:
            proc=subprocess.Popen(list(map(str,cmd)),cwd=cwd or ROOT,env=self.env(model),stdout=f,stderr=subprocess.STDOUT)
            position=0
            try:
                while proc.poll() is None:
                    time.sleep(1)
                    if time.monotonic()-getattr(self,'last_print',0)>30:
                        with log.open(encoding='utf-8',errors='replace') as reader:reader.seek(position);lines=reader.read().splitlines();position=reader.tell()
                        if lines:print('\n'.join(lines[-3:]),flush=True)
                        self.last_print=time.monotonic()
                f.flush()
                if proc.returncode:
                    tail='\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-18:])
                    raise RuntimeError(f'종료 코드 {proc.returncode}: {log}\n{tail}')
            except BaseException:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
        print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-3:]),flush=True)

    def connect(self):
        from marso_github import github_token
        github_token();self.store=GitHubStore(self.cfg['github_repository'],'run-'+self.cfg['run_name']);self.store.load()
        if not (self.run_dir/'state.json').exists():print('복원 파일:',self.store.restore(self.run_dir))
        saved=read(self.run_dir/'state.json',dict(completed=0,best=None,latest=None,history=[]))
        write(self.run_dir/'state.json',saved);write(self.run_dir/'config.json',self.cfg)
        notebook=ROOT/'foundation/notebooks'/self.cfg.get('notebook_name',f'moveboxes_{"smolvla" if self.backend=="smol" else "octo"}_colab.ipynb')
        if notebook.exists():shutil.copy2(notebook,self.run_dir/notebook.name)
        print('저장:',self.run_dir);self.sync()

    def sync(self):
        if not self.store:return
        files=list(self.run_dir.glob('*.json'))+list(self.run_dir.glob('*.ipynb'))
        files+=list((self.run_dir/'metrics').rglob('*.json'))+list((self.run_dir/'metrics').rglob('test*.mp4'))
        state=read(self.run_dir/'state.json',{})
        checkpoint_folders=[]
        if state.get('best'):checkpoint_folders.append(self.run_dir/state['best']['folder'])
        if state.get('pending'):
            checkpoint_folders.append(self.run_dir/state['pending']['folder'])
            if state['pending'].get('checkpoint'):checkpoint_folders.append(self.run_dir/state['pending']['checkpoint'])
        for folder in set(checkpoint_folders):
            if folder.exists():files+=[p for p in folder.iterdir() if p.name in ('adapter.pt','adapter.npz','metadata.json','resource.json')]
        self.store.sync(self.run_dir,'common',files)

    def install(self):
        if sys.platform!='linux':raise RuntimeError('이 설치 셀은 Linux Colab용입니다')
        if not shutil.which('nvidia-smi'):raise RuntimeError('Colab T4 GPU를 선택하세요')
        self.run(['nvidia-smi'],self.run_dir/'install_gpu.log')
        # A tiny tool install only; all model/simulator packages go into isolated environments.
        subprocess.run([sys.executable,'-m','pip','install','uv==0.8.22','-q'],check=True)
        uv=shutil.which('uv')
        if not uv:raise RuntimeError('uv installation failed')
        self.envroot.mkdir(parents=True,exist_ok=True)
        for name,version in ((self.backend,'3.10' if self.backend=='octo' else '3.12'),('simulator','3.12')):
            dest=self.envroot/name
            if not (dest/'bin/python').exists():self.run([uv,'venv','--python',version,'--seed',dest],self.run_dir/f'create_{name}.log')
            requirement=ROOT/'foundation/requirements'/({'smol':'smolvla'}.get(name,name)+'.lock')
            command=[uv,'pip','install','--python',dest/'bin/python','-r',requirement]
            if name=='octo':command+=['--find-links','https://storage.googleapis.com/jax-releases/jax_cuda_releases.html']
            self.run(command,self.run_dir/f'install_{name}.log')
            self.run([uv,'pip','check','--python',dest/'bin/python'],self.run_dir/f'check_{name}.log')
        if not self.repo.exists():self.run(['git','clone','https://github.com/marso-robotics/berlin-marso-hackathon',self.repo],self.run_dir/'clone_simulator.log')
        self.run(['git','checkout','--detach',self.cfg['simulator_commit']],self.run_dir/'simulator_commit.log',cwd=self.repo)
        self.run([uv,'pip','install','--python',self.sim_python,'--no-deps','-e',self.repo],self.run_dir/'simulator_editable.log')
        for name,python in (('model',self.model_python),('simulator',self.sim_python)):
            self.run([python,'-m','pip','freeze'],self.run_dir/f'{name}_freeze.log',model=name=='model')
        print('모델·시뮬레이터 환경 분리 설치 완료. 노트북 커널 재시작은 필요 없습니다.')

    def prepare(self):
        archive=download_archive(ROOT/'data/manifest.json','rgb',self.cfg['download_cache'])
        manifest=read(ROOT/'data/manifest.json')['archives']['rgb'];self.data.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for item in manifest['files']:
                relative=Path(item['path']);target=self.data/relative
                if target.is_file() and digest(target)==item['sha256']:continue
                matches=[n for n in z.namelist() if n.replace('\\','/').endswith(relative.as_posix())]
                if len(matches)!=1:raise ValueError('Ambiguous RGB archive member: '+str(relative))
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(matches[0]) as source,target.with_suffix(target.suffix+'.part').open('wb') as dest:shutil.copyfileobj(source,dest)
                partial=target.with_suffix(target.suffix+'.part')
                if digest(partial)!=item['sha256']:raise ValueError('RGB file hash mismatch')
                os.replace(partial,target)
        job=self.run_dir/'audit_job.json';write(job,dict(data=str(self.data),cfg=self.cfg,output=str(self.run_dir/'data_audit.json')))
        self.run([self.model_python,self.code/'foundation_worker.py','audit',job],self.run_dir/'audit.log',model=True)
        audit=read(self.run_dir/'data_audit.json')
        for level,groups in audit['episodes'].items():print(level,{k:len(v) for k,v in groups.items()})
        self.sync()

    def signature(self):
        audit=read(self.run_dir/'data_audit.json')
        if not audit:raise RuntimeError('데이터 준비 셀부터 실행하세요')
        settings={k:v for k,v in self.cfg.items() if k not in ('project_commit','project_dir','output_root','env_root','simulator_repo','data_dir','download_cache')}
        value=dict(cfg=settings,data=audit['hashes'],sources={p.name:digest(p) for p in sorted(self.code.glob('*.py'))})
        return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

    def train_job(self,checkpoint,destination,stop):
        return dict(cfg=self.cfg,data=str(self.data),checkpoint=str(checkpoint) if checkpoint else None,
            destination=str(destination),stop=stop,signature=self.signature())

    def check(self):
        dest=self.run_dir/'smoke';job=self.run_dir/'smoke_job.json'
        write(job,self.train_job(None,dest,self.cfg.get('smoke_updates',1)))
        self.run([self.model_python,self.code/'foundation_worker.py','train',job],self.run_dir/'smoke.log',model=True)
        self.evaluate(dest,'easy',[510000],'smoke',video=True)
        report=read(dest/'resource.json');print('연산·실제 환경 확인:',report)
        write(self.run_dir/'runtime_check.json',dict(signature=self.signature(),resource=report,complete=True));self.sync()

    def evaluate(self,checkpoint,level,seeds,label,video=False):
        output=self.run_dir/'metrics'/level/(label+'.json');output.parent.mkdir(parents=True,exist_ok=True)
        adapter=checkpoint/('adapter.pt' if self.backend=='smol' else 'adapter.npz')
        identity=dict(checkpoint_sha256=digest(adapter),metadata_sha256=digest(checkpoint/'metadata.json'),
            level=level,seeds=seeds,steps=self.cfg['max_steps'],signature=self.signature())
        fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();old=read(output,{})
        if old.get('complete') and old.get('fingerprint')==fp:return old
        ready=self.run_dir/'policy_ready.json';ready.unlink(missing_ok=True)
        job=dict(cfg=self.cfg,checkpoint=str(checkpoint),ready=str(ready),level=level,seeds=seeds,
            output=str(output),fingerprint=fp,max_steps=self.cfg['max_steps'],video=video)
        path=self.run_dir/'evaluation_job.json';write(path,job)
        # Only a local authenticated socket crosses the interpreter boundary.
        key=secrets.token_hex(32);model_env=self.env(True);model_env['MOVEBOXES_IPC_KEY']=key
        with (output.with_suffix('.policy.log')).open('w',encoding='utf-8') as log:
            worker=subprocess.Popen([str(self.model_python),str(self.code/'foundation_worker.py'),'serve',str(path)],
                cwd=ROOT,env=model_env,stdout=log,stderr=subprocess.STDOUT)
            try:
                start=time.monotonic();notice=start
                while not ready.exists():
                    if worker.poll() is not None:raise RuntimeError('정책 로딩 실패. 로그: '+str(output.with_suffix('.policy.log')))
                    if time.monotonic()-start>900:raise TimeoutError('Policy loading timed out')
                    if time.monotonic()-notice>30:print('사전학습 모델 로딩·컴파일 중:',level,flush=True);notice=time.monotonic()
                    time.sleep(1)
                previous=os.environ.get('MOVEBOXES_IPC_KEY');os.environ['MOVEBOXES_IPC_KEY']=key
                try:self.run([self.sim_python,self.code/'foundation_eval.py',path],output.with_suffix('.log'),cwd=self.repo)
                finally:
                    if previous is None:os.environ.pop('MOVEBOXES_IPC_KEY',None)
                    else:os.environ['MOVEBOXES_IPC_KEY']=previous
            finally:
                if worker.poll() is None:
                    worker.terminate()
                    try:worker.wait(10)
                    except subprocess.TimeoutExpired:worker.kill();worker.wait()
        result=read(output)
        if not result or not result['complete']:raise RuntimeError('Incomplete environment evaluation')
        return result

    def train(self):
        state=read(self.run_dir/'state.json')
        if not state:raise RuntimeError('GitHub 연결 셀을 먼저 실행하세요')
        if read(self.run_dir/'runtime_check.json',{}).get('signature')!=self.signature():raise RuntimeError('05 GPU/환경 확인 셀을 먼저 실행하세요')
        while state['completed']<self.cfg['updates']:
            pending=state.get('pending')
            if pending:
                dest=self.run_dir/pending['folder'];current=self.run_dir/pending['checkpoint'] if pending.get('checkpoint') else None
                start,stop,trained_updates=pending['start'],pending['stop'],pending['trained_updates']
            else:
                current=self.run_dir/state['latest'] if state.get('latest') else None
                if current and not (current/'metadata.json').exists():current=self.run_dir/state['best']['folder'] if state.get('best') else None
                start=read(current/'metadata.json')['step'] if current else 0
                # A new runtime continues from the compact best adapter with a fresh optimizer.
                trained_updates=min(self.cfg['eval_interval'],self.cfg['updates']-state['completed'])
                stop=start+trained_updates;dest=self.run_dir/'candidates'/f'update_{state["completed"]+trained_updates:06d}'
                pending=dict(folder=dest.relative_to(self.run_dir).as_posix(),checkpoint=current.relative_to(self.run_dir).as_posix() if current else None,
                    start=start,stop=stop,trained_updates=trained_updates)
                state['pending']=pending;write(self.run_dir/'state.json',state);self.sync()
            path=self.run_dir/'train_job.json';write(path,self.train_job(current,dest,stop))
            try:
                if not (dest/'metadata.json').exists():self.run([self.model_python,self.code/'foundation_worker.py','train',path],self.run_dir/'train.log',model=True)
                # Publish the closed adapter before the slower three-level environment evaluation.
                self.sync()
                scores={level:self.evaluate(dest,level,list(range(520000,520000+self.cfg['dev_episodes'])),dest.name)['sort_accuracy'] for level in LEVELS}
                score=sum(SCORE_WEIGHTS[level]*scores[level] for level in LEVELS)
                if state['best'] is None or score>state['best']['score']:state['best']=dict(folder=dest.relative_to(self.run_dir).as_posix(),score=score,scores=scores)
                state['completed']+=trained_updates;state['latest']=dest.relative_to(self.run_dir).as_posix();state['pending']=None
                state['history'].append(dict(updates=state['completed'],scores=scores,weighted_score=score))
                write(self.run_dir/'state.json',state);print('공동 학습',state['completed'],'/',self.cfg['updates'],scores,flush=True)
            finally:self.sync()

    def test(self,level):
        state=read(self.run_dir/'state.json');best=state.get('best') if state else None
        if not best:raise RuntimeError('선택 모델이 없습니다. 공동 학습 셀을 먼저 실행하세요')
        try:
            result=self.evaluate(self.run_dir/best['folder'],level,list(range(530000,530000+self.cfg['test_episodes'])),'test',video=True)
            print(level,'RGB 분류율:',f'{result["sort_accuracy"]:.1%}')
            video=self.run_dir/'metrics'/level/'test.mp4'
            if video.exists():
                from IPython.display import Video,display
                display(Video(str(video),embed=True))
            return result
        finally:self.sync()

    def report(self):
        state=read(self.run_dir/'state.json',{});print(json.dumps(state,indent=2,ensure_ascii=False));return state
