from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

BASE = Path('/data/libero_mem_baseline')
OFFICIAL = Path('/home/nvidia/libero-mem-official')
PHASE = Path('/home/nvidia/zyx/PhaseVLA')
RUNNER_PYTHON = BASE / 'SANE/.venv/bin/python'
POLICY_PYTHON = PHASE / 'environments/futuremamba/.venv/bin/python'
SERVER = PHASE / 'scripts/serve_policy.py'
PORT = int(os.environ.get('BOWL_PORT', '8261'))
GPU = int(os.environ.get('BOWL_GPU', '0'))
SELECTED_TASK_ID = int(os.environ['BOWL_TASK_ID'])
CHECKPOINT_STEP = int(os.environ['BOWL_CHECKPOINT_STEP'])
TASK_IDS = [SELECTED_TASK_ID]
OUTPUT_ROOT = (
    BASE / 'formal_action_history_handoff04_mamba_bowl_step_sweep_v1'
    / f'task_{SELECTED_TASK_ID:02d}' / f'step_{CHECKPOINT_STEP:04d}'
)
OUTPUT = OUTPUT_ROOT
TRIALS = 20
TRAIN_SEED = 42
ROLLOUT_SEED = 10001
MAX_STEPS = 600
REPLAN_STEPS = 20
FPS = 20
STABILIZED_INIT_ROOT = BASE / 'stabilized_init_states_v3_mujoco322'
STABILIZED_INIT_AUDIT = STABILIZED_INIT_ROOT / 'audit.json'
TASK_NAMES = {
    0: 'pick up the bowl and place it back on the plate',
    2: 'lift the bowl and place it back on the plate 3 times',
}
POLICY_CONFIGS = {
    0: 'futuremamba_action_history_handoff04_libero_mem_bowl_t1',
    2: 'futuremamba_action_history_handoff04_libero_mem_bowl_t3',
}
POLICY_CONFIG = POLICY_CONFIGS[SELECTED_TASK_ID]
POLICY_DIR = BASE / 'checkpoints/action_history_handoff04_futuremamba' / POLICY_CONFIG / str(CHECKPOINT_STEP)
IDENTITY_FILE = POLICY_DIR / 'metadata.json'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return 'sha256:' + digest.hexdigest()


def wilson(successes: int, episodes: int) -> dict[str, float]:
    z = 1.959963984540054
    rate = successes / episodes
    denominator = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / episodes + z * z / (4 * episodes * episodes)) / denominator
    return {'success_rate': rate, 'ci95_low': center - radius, 'ci95_high': center + radius}


def wait_ready(process: subprocess.Popen, log: Path, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'policy server exited before readiness: {log.read_text(errors="replace")[-12000:]}')
        try:
            with socket.create_connection(('127.0.0.1', PORT), timeout=1):
                return
        except OSError:
            time.sleep(3)
    raise TimeoutError(f'policy server port {PORT} did not become ready')


def load_recorder_module():
    path = BASE / 'record_libero_mem_20_videos.py'
    spec = importlib.util.spec_from_file_location('libero_mem_video_recorder', path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if int(module.FPS) != FPS:
        raise ValueError(f'recorder FPS mismatch: {module.FPS}')
    return module


def load_dependencies():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    import examples.libero_mem.env_adapter as env_adapter
    import robosuite

    class CanonicalSuite:
        def __init__(self, suite):
            self._suite = suite
            self.n_tasks = suite.n_tasks

        def get_task(self, index):
            task = self._suite.get_task(index)
            language = task.language
            if language and language[0].isdigit():
                language = language.split(' ', 1)[1]
            return task._replace(language=language)

        def get_task_init_states(self, index):
            import torch

            task = self._suite.get_task(index)
            path = STABILIZED_INIT_ROOT / task.init_states_file
            if not path.is_file():
                raise FileNotFoundError(path)
            return torch.load(path, map_location='cpu', weights_only=False)

    class CanonicalBenchmark:
        def get_benchmark_dict(self):
            mapping = dict(benchmark.get_benchmark_dict())
            return {name: (lambda factory=factory: CanonicalSuite(factory())) for name, factory in mapping.items()}

    identity = {
        'libero_env_module': OffScreenRenderEnv.__module__,
        'libero_env_file': sys.modules[OffScreenRenderEnv.__module__].__file__,
        'robosuite_file': robosuite.__file__,
        'robosuite_version': getattr(robosuite, '__version__', None),
        'fixed_init_state_method': 'libero.libero.envs.OffScreenRenderEnv.set_init_state',
        'stabilized_init_root': str(STABILIZED_INIT_ROOT),
        'stabilized_init_audit': str(STABILIZED_INIT_AUDIT),
        'stabilized_init_audit_sha256': sha256(STABILIZED_INIT_AUDIT),
    }
    return {
        'benchmark': CanonicalBenchmark(),
        'get_libero_path': get_libero_path,
        'OffScreenRenderEnv': OffScreenRenderEnv,
        'websocket_client_policy': websocket_client_policy,
        'env_adapter': env_adapter,
        'identity': identity,
    }


def audit(raw_path: Path, summary_path: Path, dependencies: dict) -> dict:
    raw_bytes = raw_path.read_bytes()
    rows = [json.loads(line) for line in raw_bytes.decode('utf-8').splitlines() if line.strip()]
    expected_checksum = sha256(IDENTITY_FILE)
    grouped: dict[str, list[dict]] = defaultdict(list)
    violations = []
    for row in rows:
        grouped[str(row['task'])].append(row)
    expected_tasks = {TASK_NAMES[index] for index in TASK_IDS}
    for task, task_rows in grouped.items():
        episode_ids = [int(row['episode']) for row in task_rows]
        duplicates = sorted(episode for episode, count in Counter(episode_ids).items() if count > 1)
        if sorted(episode_ids) != list(range(TRIALS)) or duplicates:
            violations.append({'task': task, 'failure': 'episode_coverage', 'episode_ids': episode_ids, 'duplicates': duplicates})
        for row in task_rows:
            config = row.get('config', {})
            video = row.get('video', {})
            video_path = Path(video.get('path', ''))
            checks = {
                'task_family': row.get('task_family') == 'libero_mem',
                'train_seed': row.get('train_seed') == TRAIN_SEED,
                'rollout_seed': row.get('rollout_seed') == ROLLOUT_SEED,
                'checkpoint_checksum': row.get('checkpoint_checksum') == expected_checksum,
                'checkpoint_path': config.get('checkpoint_path') == str(IDENTITY_FILE),
                'task_ids': config.get('task_ids') == TASK_IDS,
                'num_trials_per_task': config.get('num_trials_per_task') == TRIALS,
                'max_steps': config.get('max_steps') == MAX_STEPS,
                'replan_steps': config.get('replan_steps') == REPLAN_STEPS,
                'num_steps_wait': config.get('num_steps_wait') == 10,
                'video_out_path': config.get('video_out_path') == str(OUTPUT / 'videos'),
                'video_path': row.get('video_path') == str(video_path),
                'video_exists': video_path.is_file(),
                'video_nonempty': video_path.is_file() and video_path.stat().st_size > 0,
                'video_sha256': video_path.is_file() and video.get('sha256') == sha256(video_path),
                'video_frames': int(video.get('frames', 0)) == int(row.get('steps', -1)) + 1,
                'video_fps': video.get('fps') == FPS,
                'video_resolution': video.get('width') == 256 and video.get('height') == 256,
            }
            failed = sorted(name for name, valid in checks.items() if not valid)
            if failed:
                violations.append({'task': task, 'episode': row.get('episode'), 'failure': 'identity', 'failed': failed})
    per_task = {}
    for task, task_rows in sorted(grouped.items()):
        successes = sum(bool(row.get('success')) for row in task_rows)
        per_task[task] = {
            'episodes': len(task_rows),
            'successes': successes,
            **wilson(successes, len(task_rows)),
            'successful_episode_ids': sorted(int(row['episode']) for row in task_rows if row.get('success')),
        }
    successes = sum(item['successes'] for item in per_task.values())
    valid = len(rows) == len(TASK_IDS) * TRIALS and set(grouped) == expected_tasks and not violations
    manifest = json.loads(IDENTITY_FILE.read_text(errors='strict'))
    report = {
        'schema': 'phasevla.libero-mem-action-history-handoff04-mamba-video.v1',
        'status': 'completed' if valid else 'invalid',
        'audited_at': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'model': {
            'config': POLICY_CONFIG,
            'checkpoint': str(POLICY_DIR),
            'checkpoint_metadata_sha256': expected_checksum,
            'architecture': manifest['architecture'],
            'memory_input_source': manifest['memory_input_source'],
            'action_history_encoding': manifest['action_history_encoding'],
            'uses_vlm_hidden_for_memory': manifest['uses_vlm_hidden_for_memory'],
            'uses_prefix_kv_for_progress': manifest['uses_prefix_kv_for_progress'],
            'handoff_ratio': manifest['handoff_ratio'],
            'denoising_order': manifest['denoising_order'],
            'progress_denoise_steps': manifest['progress_denoise_steps'],
            'action_history_chunk_size': manifest['action_history_chunk_size'],
            'training_query_stride': manifest['training_query_stride'],
            'step': manifest['step'],
        },
        'environment': dependencies['identity'],
        'protocol': {
            'suite': 'libero_mem',
            'task_ids': TASK_IDS,
            'task_names': [TASK_NAMES[index] for index in TASK_IDS],
            'episodes_per_task': TRIALS,
            'episode_ids': list(range(TRIALS)),
            'rollouts': len(TASK_IDS) * TRIALS,
            'train_seed': TRAIN_SEED,
            'rollout_seed': ROLLOUT_SEED,
            'max_steps': MAX_STEPS,
            'replan_steps': REPLAN_STEPS,
            'predicted_action_horizon': 20,
            'execution_horizon': 20,
            'memory_update_timing': 'append_previous_executed_actions_before_current_query',
            'partial_chunk_behavior': 'right_pad_and_mask_without_commit',
            'handoff_ratio': 0.4,
            'denoising_order': 'progress_expert_then_action_expert',
            'progress_denoise_steps': 4,
            'action_expert_denoise_steps': 6,
            'empty_history_behavior': 'learned_token_without_state_update',
            'stable_frames': 6,
            'num_steps_wait': 10,
            'fixed_init_state_required': True,
            'fixed_init_state_applied_via': dependencies['identity']['fixed_init_state_method'],
            'stabilized_init_root': str(STABILIZED_INIT_ROOT),
            'stabilized_init_audit_sha256': sha256(STABILIZED_INIT_AUDIT),
            'mujoco_version': '3.2.2',
            'video_fps': FPS,
            'video_out_path': str(OUTPUT / 'videos'),
        },
        'raw_results': {
            'path': str(raw_path),
            'records': len(rows),
            'sha256': 'sha256:' + hashlib.sha256(raw_bytes).hexdigest(),
        },
        'per_task': per_task,
        'overall': {'episodes': len(rows), 'successes': successes, **wilson(successes, len(rows))},
        'videos': {
            'count': sum(1 for row in rows if Path(row.get('video_path', '')).is_file()),
            'total_bytes': sum(Path(row['video_path']).stat().st_size for row in rows if Path(row.get('video_path', '')).is_file()),
            'manifest': [row['video'] for row in rows],
        },
        'audit': {'valid': valid, 'violations': violations},
    }
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    return report


def main() -> int:
    required = [
        POLICY_DIR / 'plugin.safetensors',
        IDENTITY_FILE,
        Path(json.loads(IDENTITY_FILE.read_text(errors='strict'))['assets_uri']) / 'libero-mem/LIBERO-Mem-Lerobot/norm_stats.json',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    init_audit = json.loads(STABILIZED_INIT_AUDIT.read_text(errors='strict'))
    if not init_audit.get('audit', {}).get('valid'):
        raise RuntimeError(f'invalid stabilized init audit: {init_audit.get("audit")}')
    if init_audit.get('mujoco_version') != '3.2.2':
        raise RuntimeError(f"stabilized init audit must use MuJoCo 3.2.2, got {init_audit.get('mujoco_version')}")
    OUTPUT.mkdir(parents=True)
    raw_path = OUTPUT / 'rollouts.jsonl'
    summary_path = OUTPUT / 'audited_summary.json'
    server_log = OUTPUT / 'server.log'
    runner_log = OUTPUT / 'runner.log'
    environment = os.environ.copy()
    environment.update({
        'CUDA_VISIBLE_DEVICES': str(GPU),
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'MUJOCO_GL': 'egl',
        'LIBERO_CONFIG_PATH': str(BASE / 'official_libero_config'),
        'PYTHONUNBUFFERED': '1',
        'PYTHONNOUSERSITE': '1',
        'PYTHONPATH': f"{PHASE / 'src'}:{PHASE / 'packages/openpi-client/src'}",
    })
    with server_log.open('w', buffering=1) as stream:
        server = subprocess.Popen([
            str(POLICY_PYTHON), str(SERVER), '--port', str(PORT), 'policy:checkpoint',
            '--policy.config', POLICY_CONFIG, '--policy.dir', str(POLICY_DIR),
        ], cwd=PHASE, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_ready(server, server_log)
        sys.path[:0] = [
            str(BASE / 'envs/mujoco-3.2.2-cp311'),
            str(PHASE), str(PHASE / 'examples/libero_mem'), str(BASE / 'official_site'),
            str(OFFICIAL), str(OFFICIAL / 'thirdparty/robosuite'), str(OFFICIAL / 'thirdparty/robomimic'),
            str(BASE / 'envs/libero-eval/lib/python3.11/site-packages'), str(PHASE / 'packages/openpi-client/src'),
        ]
        os.environ.update({'LIBERO_CONFIG_PATH': str(BASE / 'official_libero_config'), 'MUJOCO_GL': 'egl'})
        from openpi_client import websocket_client_policy
        import examples.libero_mem.main as eval_main

        recorder = load_recorder_module()
        dependencies = load_dependencies()
        expected_env_file = str(OFFICIAL / 'libero/libero/envs/env_wrapper.py')
        expected_robosuite_file = str(OFFICIAL / 'thirdparty/robosuite/robosuite/__init__.py')
        if dependencies['identity']['libero_env_file'] != expected_env_file or dependencies['identity']['robosuite_file'] != expected_robosuite_file:
            raise RuntimeError(dependencies['identity'])
        eval_main._load_real_dependencies = lambda: dependencies
        args = eval_main.Args(
            host='127.0.0.1', port=PORT, task_suite_name='libero_mem', replan_steps=REPLAN_STEPS,
            max_steps=MAX_STEPS, stable_frames=6, train_seed=TRAIN_SEED, rollout_seed=ROLLOUT_SEED,
            num_trials_per_task=TRIALS, num_steps_wait=10, task_ids=tuple(TASK_IDS),
            results_path=str(raw_path), checkpoint_path=str(IDENTITY_FILE),
            video_out_path=str(OUTPUT / 'videos'), handoff_ratio=0.4, history_condition='action_history_mamba_pe_ae_handoff',
        )
        suite = dependencies['benchmark'].get_benchmark_dict()['libero_mem']()
        task_lookup = {str(suite.get_task(index).language): index for index in TASK_IDS}
        print(json.dumps({'event': 'evaluation_ready', 'task_ids': TASK_IDS, 'gpu': GPU, 'port': PORT, 'output': str(OUTPUT)}, separators=(',', ':')), flush=True)

        def episode_runner(**kwargs):
            task = str(kwargs['task'])
            task_id = task_lookup[task]
            episode = int(kwargs['episode'])
            kwargs['video_path'] = str(
                OUTPUT / 'videos' / f'task_{task_id:02d}' /
                f'episode_{episode:02d}_{recorder.safe_task_name(task)}.mp4'
            )
            return recorder.run_recorded_episode(**kwargs)

        with runner_log.open('w', buffering=1) as stream:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = stream
            try:
                eval_main.eval_libero_mem(args, episode_runner=episode_runner)
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
        report = audit(raw_path, summary_path, dependencies)
        print(json.dumps({
            'status': report['status'], 'overall': report['overall'],
            'videos': {'count': report['videos']['count'], 'total_bytes': report['videos']['total_bytes']},
            'summary': str(summary_path),
        }, ensure_ascii=False, separators=(',', ':')))
        return 0 if report['status'] == 'completed' else 2
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == '__main__':
    raise SystemExit(main())
