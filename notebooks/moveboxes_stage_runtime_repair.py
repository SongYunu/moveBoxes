# 1 · 기존 run/candidate 연결 + 중단 후 남은 평가 프로세스 정리
import json, os, signal, subprocess, sys, time
from pathlib import Path

if 'experiment' not in globals() or 'CFG' not in globals():
    raise RuntimeError('학습한 노트북의 같은 런타임 아래에 이 셀을 추가하세요.')
RUN_DIR = Path(experiment.run_dir)
CANDIDATE = RUN_DIR/'integrated_candidate'
UPSTREAM = Path(CFG['repo_dir'])
manifest = json.loads((CANDIDATE/'manifest.json').read_text(encoding='utf-8'))
selected = {level:CANDIDATE/'checkpoints'/level/'model.pt'
            for level in manifest['levels']}
if not selected or any(not path.is_file() for path in selected.values()):
    raise FileNotFoundError(f'{CANDIDATE}의 기존 candidate checkpoint를 확인하세요.')
MAX_STEPS = 200
SMOKE_SEED = 61000

def stop_leftover_official_eval():
    # Linux /proc: match the exact evaluator and this run's output directory.
    # Training/collection commands and other runs cannot match these conditions.
    proc = Path('/proc')
    if not proc.is_dir():
        return
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command_bytes = (entry/'cmdline').read_bytes()
            args = command_bytes.decode(errors='replace').split('\0')
            evaluator = str((UPSTREAM/'eval.py').resolve())
            outputs = [a.split('=',1)[1] for a in args if a.startswith('hydra.run.dir=')]
            if evaluator not in args or not outputs:
                continue
            if not all(Path(output).resolve().is_relative_to(RUN_DIR.resolve()) for output in outputs):
                continue
            pid = int(entry.name)
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic()+5
            while entry.exists() and time.monotonic() < deadline:
                # Zombies have already released GPU resources; the parent reaps them.
                if (entry/'stat').read_text().split(') ',1)[1].startswith('Z'):
                    break
                time.sleep(.1)
            else:
                if entry.exists() and (entry/'cmdline').read_bytes() == command_bytes:
                    os.kill(pid, signal.SIGKILL)
            print('중단 후 남은 이 run의 공식 평가 프로세스 종료:', pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue

stop_leftover_official_eval()
print('학습 파일 유지:', RUN_DIR)
print('기존 candidate 그대로 사용:', list(selected))


# Colab inline 영상 player · 한 번만 실행
from IPython.display import Video, display

VIDEO_LEVEL = next(iter(selected))  # 'easy', 'medium', 'hard' 중 현재 생성된 난이도
VIDEO_WIDTH = 960
DOWNLOAD_VIDEO = False

def show_official_video(label, level=None):
    level = level or VIDEO_LEVEL
    if level not in selected:
        raise ValueError(f'{level} checkpoint가 없습니다. 가능한 난이도: {list(selected)}')
    folder = RUN_DIR/level/'integrated_official_eval'/label/'videos'
    videos = sorted(folder.rglob('*.mp4'), key=lambda path:path.stat().st_mtime)
    if not videos:
        raise FileNotFoundError(f'{folder}에 MP4가 없습니다. 바로 앞 평가 셀을 먼저 실행하세요.')
    video = videos[-1]
    print(f'[{level}/{label}] {video.name} · {video.stat().st_size/1024**2:.1f} MiB')
    # Older Colab IPython checks os.path.exists(data) before filename.
    display(Video(str(video), embed=True, width=VIDEO_WIDTH))
    if DOWNLOAD_VIDEO:
        from google.colab import files
        files.download(str(video))
    return video

def show_all_official_videos(label):
    for level in selected:
        try:
            show_official_video(label, level)
        except FileNotFoundError as error:
            print(error)

show_all_official_videos('smoke')


# 공식 eval.py · 1 episode smoke
import os, subprocess, sys
RESULTS = RUN_DIR
SMOKE_CONFIG = RUN_DIR/'integrated_smoke_eval.yaml'
SMOKE_CONFIG.write_text('eval:\n  n_episodes: 1\n  seeds: ['+str(SMOKE_SEED)+']\n', encoding='utf-8')
UPSTREAM = Path(CFG['repo_dir'])

def run_official(level, eval_config, label):
    output = RUN_DIR/level/'integrated_official_eval'/label
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(UPSTREAM/'eval.py'), 'difficulty='+level,
        'obs_mode=state', 'policy=stage_chunk_policy:load_policy',
        'checkpoint='+str(CANDIDATE/'checkpoints'/level/'model.pt'),
        'eval_config='+str(eval_config), 'max_episode_steps='+str(MAX_STEPS),
        'hydra.run.dir='+str(output)]
    child_env = dict(os.environ)
    child_env['PYTHONPATH'] = str(CANDIDATE)+os.pathsep+str(UPSTREAM)+os.pathsep+child_env.get('PYTHONPATH','')
    log = output/'official_eval.log'
    with log.open('w', encoding='utf-8') as handle:
        process = subprocess.Popen(command, cwd=UPSTREAM, env=child_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end='', flush=True)
            code = process.wait()
        finally:
            # Colab's stop button interrupts the kernel, not its GPU subprocess.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
    if code:
        raise RuntimeError(f'official eval failed ({level}); {log} 확인')
    return log


def run_missing_smoke():
    for level in selected:
        folder = RUN_DIR/level/'integrated_official_eval/smoke/videos'
        if any(path.stat().st_size > 0 for path in folder.rglob('*.mp4')):
            print(level, '기존 smoke 영상 유지')
            continue
        run_official(level, SMOKE_CONFIG, 'smoke')
    show_all_official_videos('smoke')

# 영상이 없는 난이도만 공식 평가합니다. GitHub 업로드를 호출하지 않습니다.
