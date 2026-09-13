"""Extend the unchanged learning algorithm with per-level smoke and quick tests."""
import copy
import sys
from pathlib import Path
from marso_experiment import ARCH, LEVELS, Experiment, completed_metrics, read_json, save_json


class TrainTestExperiment(Experiment):
    def connect(self):
        if self.cfg['test_episodes']<1 or not 1<=self.cfg['test_inference_steps']<=100:
            raise ValueError('test_episodes와 test_inference_steps를 확인하세요.')
        seeds=set(self.test_seeds())
        if seeds.intersection(self.seeds(True)) or seeds.intersection(self.seeds(False)):
            raise ValueError('빠른 테스트 / 튜닝 / 최종 평가 seed를 각각 구분하세요.')
        super().connect()

    def notebook_snapshot(self):
        from colab_train_test_layout import make_notebook
        return make_notebook(self.sources,self.cfg)

    def test_seeds(self):
        count=2 if self.cfg['profile']=='smoke' else self.cfg['test_episodes']
        return list(range(self.cfg['test_seed_start'],self.cfg['test_seed_start']+count))

    def smoke(self,level):
        self._ready()
        assert level in LEVELS
        cfg=copy.deepcopy(self.cfg)
        cfg['output_root']=str(self.run_dir/'checks')
        cfg['run_name']='runtime_check'
        cfg['profile']='smoke'
        cfg['test_record_video']=False
        # This is not a training warm-start: smoke weights remain in checks/.
        child=type(self)(cfg,self.sources)
        child.connect()
        child.train(level)
        result=child.test(level)
        if result is None:
            raise RuntimeError(f'{level} 동작 확인 실패: 출력과 checks 폴더를 확인하세요.')
        print(f'[{level}] 학습·로딩·2회 실행 확인 완료. 다음 전체 학습 셀을 실행하세요.')

    def _test_candidate(self,level):
        folder=self.run_dir/level
        selection=read_json(folder/'selection.json',{}).get('selected')
        if selection:
            checkpoint=Path(selection['checkpoint'])
            if not checkpoint.exists():
                checkpoint=folder/'checkpoints'/checkpoint.name
            if checkpoint.exists() and (not selection.get('checkpoint_sha256')
                    or selection['checkpoint_sha256']==self.checkpoint_hash(checkpoint)):
                return checkpoint,dict(selection['policy_config'])
        ckdir=folder/'checkpoints'
        best=ckdir/'best_eval_sort_accuracy.pt'
        periodic=sorted((p for p in ckdir.glob('*.pt') if p.stem.isdigit()),
                        key=lambda p:int(p.stem),reverse=True)
        checkpoint=best if best.exists() else periodic[0] if periodic else None
        if checkpoint is None:
            return None
        recorded=read_json(folder/'train_args.json') or read_json(ckdir/'policy_config.json')
        if not recorded or any(k not in recorded for k in ARCH):
            raise ValueError(f'{level}: 저장된 모델 구조 설정이 없습니다.')
        cfg={k:recorded[k] for k in ARCH}
        cfg.update(act_horizon=recorded.get('act_horizon',self.cfg['act_horizon']),
                   num_inference_steps=self.cfg['test_inference_steps'])
        return checkpoint,cfg

    def test(self,level):
        self._ready()
        assert level in LEVELS
        selected=self._test_candidate(level)
        if selected is None:
            print(f'[{level}] 학습 모델 없음. 이 난이도의 전체 학습 셀을 먼저 실행하세요.')
            return None
        checkpoint,policy=selected
        self._trial(level,checkpoint,policy,self.test_seeds(),'test_metrics')
        path=self.run_dir/level/'test_metrics.json'
        metrics=read_json(path)
        print(f'[{level}] 빠른 테스트 {metrics["n_episodes"]}회: {metrics["sort_accuracy"]:.1%}')
        print('결과:',path,'/ 모델:',checkpoint.name)
        print('빠른 테스트는 최종 가중 점수에 합산하지 않습니다.')
        if self.cfg['test_record_video']:
            self.test_video(level,checkpoint,policy)
            self._display_test_video(level)
        return metrics

    def test_video(self,level,checkpoint,policy):
        folder=self.run_dir/level
        job=self._job(level,checkpoint,policy,[self.test_seeds()[0]],'test_video')
        job.update(video_only=True,video_dir=str(folder/'test_videos'/job['fingerprint'][:16]))
        destination=Path(job['video_dir'])
        if not list(destination.glob('**/*.mp4')):
            path=folder/'test_video_job.json'
            save_json(path,job)
            try:
                self.run([sys.executable,'colab_eval_modular.py',str(path)],cwd=self.repo,
                         log=folder/'test_video.log')
            except Exception as error:
                print(f'[{level}] 테스트 점수는 저장됐습니다. 영상 생성 실패: {error}')
                return
        save_json(folder/'test_video_index.json',{'directory':str(destination),'fingerprint':job['fingerprint'],
            'checkpoint_sha256':job['checkpoint_sha256'],'policy_config':policy})

    def _display_test_video(self,level):
        index=read_json(self.run_dir/level/'test_video_index.json')
        if not index:
            return
        metrics=read_json(self.run_dir/level/'test_metrics.json',{})
        protocol=metrics.get('protocol',{})
        if any(index.get(key)!=protocol.get(key) for key in ('checkpoint_sha256','policy_config')):
            return
        paths=sorted(Path(index['directory']).rglob('*.mp4'))
        if paths:
            from IPython.display import Video,display
            display(Video(str(paths[-1]),embed=True,width=640))

    def show_tests(self,videos=False):
        self._ready()
        print('빠른 테스트 (최종 점수와 별도)')
        for level in LEVELS:
            metrics=read_json(self.run_dir/level/'test_metrics.json')
            if completed_metrics(metrics):
                print(f'{level}: {metrics["sort_accuracy"]:.1%} / {metrics["n_episodes"]}회')
                if videos:
                    self._display_test_video(level)
            else:
                print(f'{level}: 테스트 미완료')
