"""Pack original PAE input tokens and provenance into one pickle-free NPZ."""
from pathlib import Path
import hashlib
import json
import numpy as np

SOURCE = Path(__file__).resolve().parent
OUTPUT = SOURCE.parent / 'mamba_pae_memory_all_tasks.npz'


def main():
    manifest = json.loads((SOURCE / 'extraction_manifest.json').read_text(encoding='utf-8'))
    tasks = sorted(manifest['tasks'], key=lambda t: t['task'])
    assert [t['task'] for t in tasks] == list(range(1, 11))
    rows = {k: [] for k in ('memory', 'task_id', 'episode_index', 'episode_id', 'query_step', 'fraction')}
    counts, hashes = [], {}
    for task in tasks:
        task_id = task['task']
        path = SOURCE / f'T{task_id:02d}_memory.npz'
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            memory, episode, query = data['memory'], data['episode'], data['query']
            fraction = data['fraction']
            n = len(memory)
            assert memory.shape == (n, 1024) and memory.dtype == np.float32
            assert list(memory.shape) == task['memory_shape']
            assert np.isfinite(memory).all()
            assert episode.shape == query.shape == fraction.shape == (n,)
            assert np.array_equal(np.unique(episode), np.arange(5))
            for index, info in enumerate(task['episodes']):
                mask = episode == index
                assert mask.sum() == info['queries']
                assert np.array_equal(query[mask], np.arange(20, info['length'], 20))
                assert np.allclose(fraction[mask], query[mask] / (info['length'] - 1))
            order = np.lexsort((query, episode))
            rows['memory'].append(memory[order])
            rows['task_id'].append(np.full(n, task_id, dtype=np.int64))
            rows['episode_index'].append(episode[order])
            rows['episode_id'].append(np.array([task['episodes'][int(i)]['id'] for i in episode[order]]))
            rows['query_step'].append(query[order])
            rows['fraction'].append(fraction[order])
            counts.append(n)
    packed = {key: np.concatenate(value) for key, value in rows.items()}
    packed.update(
        task_ids=np.arange(1, 11),
        task_names=np.array([t['task_name'] for t in tasks]),
        task_counts=np.array(counts),
        task_offsets=np.r_[0, np.cumsum(counts)],
        task_train_seeds=np.array([t['seed'] for t in tasks]),
        task_checkpoint_steps=np.array([t['step'] for t in tasks]),
        task_checkpoints=np.array([t.get('original_checkpoint', t['checkpoint']) for t in tasks]),
    )
    metadata = {
        'schema_version': 1,
        'feature': 'advance_history returned memory_token[0,0]: Mamba output after memory_token_projection, input to PAE',
        'processing': 'Original float32 features, sorted by task, episode, query; no normalization or dimensionality reduction.',
        'source_type': 'Offline causal replay of simulation demonstrations, not online policy rollouts.',
        'comparison_note': 'Tasks use separately trained task-specific checkpoints; joint t-SNE differences may reflect both task and model differences.',
        'row_fields': {
            'memory': 'N x 1024 float32 PAE input tokens',
            'task_id': 'Task number 1 through 10',
            'episode_index': 'Zero-based index within the five demonstrations of a task',
            'episode_id': 'Original demonstration key; unique only within each task',
            'query_step': 'Number of executed actions already in history; interval 20; excludes empty history',
            'fraction': 'query_step / (demonstration length - 1), normalized time, not annotated phase',
        },
        'task_fields': 'task_names/counts/train_seeds/checkpoint_steps/checkpoints align with task_ids; task_offsets delimit memory rows.',
        'source_npz_sha256': hashes,
        'extraction_manifest': manifest,
    }
    packed['metadata_json'] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(OUTPUT, **packed)
    with np.load(OUTPUT, allow_pickle=False) as checked:
        assert checked['memory'].shape == (838, 1024)
        assert all(checked[key].dtype.kind != 'O' for key in checked.files)
        for task_id in range(1, 11):
            with np.load(SOURCE / f'T{task_id:02d}_memory.npz', allow_pickle=False) as original:
                order = np.lexsort((original['query'], original['episode']))
                assert np.array_equal(checked['memory'][checked['task_id'] == task_id], original['memory'][order])
        assert len(set(zip(checked['task_id'], checked['episode_index'], checked['query_step']))) == 838
        json.loads(str(checked['metadata_json']))
    print(OUTPUT)
    print('Shape:', packed['memory'].shape, 'Counts:', counts)
    print('Verified: all tokens exactly equal to original task files; all metadata aligned.')
    print('Bytes:', OUTPUT.stat().st_size)


if __name__ == '__main__':
    main()
