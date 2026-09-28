# ActMem-VLA experiment and runtime release — 2026-09-28

This release preserves the actual experimental runtime entrypoints, audited episode records and plotting inputs. The repository model implementation includes the differentiable recurrent-history fix from commit `a0c4a5c`. Historical names (`PhaseVLA`, `futuremamba`, `progress_expert`, `handoff`) remain in code/config keys for checkpoint compatibility; the method name is **ActMem-VLA** and the lightweight expert is the **PreAction Expert (PAE)**.

## Files and scope

* `runtime_snapshots/`: immutable copies of the training/evaluation wrappers used on the servers. These are archival entrypoints, **not portable one-command launchers**: original absolute data paths, optional W&B key-file references and environment assumptions must be adapted. No credential values are distributed.
* `source_manifest.json`, `local_source_manifest.json`: source paths, SHA-256 hashes and provenance. Copied source is not silently rewritten.
* `data/a100_audited_evaluations.json.gz`, `data/lgd_audited_evaluations.json.gz`: fresh 2026-09-28 collections. Each record contains the audited summary and per-episode rollout records with original file hashes. Parse with Python's `gzip` and `json` modules.
* `data/paper_source_20260912.json.gz`, `data/paper_episodes.csv`: the separately preserved 978-job / 19,560-episode local analysis snapshot. This is **not a claim that all later ablations were complete on September 12**.
* `data/available_evaluation_index.csv`: all collected source records. `deduplicated_evaluation_index.csv` removes only identical job-ID/rollout-hash copies. IDs with differing records are listed in `collection_validation.json`; do not select one silently.
* `analysis/`: existing figure source tables, scripts, memory features and provenance. Plotting scripts may need their original path constants adapted. Figure-specific selection rules are preserved in their README/JSON files.

## Read raw results

```python
import gzip, json
from pathlib import Path
root = Path('releases/2026-09-28/data')
with gzip.open(root / 'a100_audited_evaluations.json.gz', 'rt') as f:
    records = json.load(f)
for r in records:
    episodes = r['rollouts']
    assert sum(x['success'] for x in episodes) == r['summary']['overall']['successes']
```

Keep model implementation, task, training seed, checkpoint, evaluation seed, protocol and episode budget as separate grouping keys. `steps` is an environment-step count, not wall-clock seconds. Completion-step boxplots condition on success and must be accompanied by success rates.

## Important interpretation boundaries

1. **Legacy vs corrected memory:** old forward recurrence did not imply full temporal gradients. Corrected runs use the differentiable recurrent update. Do not pool these runs as independent seeds of one implementation.
2. **Two no-PAE designs:** historical `no-PE` fine-tunes the original AE with memory; the later `frozen_memory_ae_early04_v1` keeps AE frozen, conditions only the first four denoising calls on memory, and leaves the remaining six unconditioned. They are different baselines. Loss/optimizer differences must also be disclosed.
3. **Training scope:** task-specific ActMem plugins and joint ten-task baseline models are different adaptation regimes. Evaluation seeds do not count as independently trained models.
4. **Checkpoint selection:** some figures select checkpoints using existing evaluation results, or use post-hoc step 1500. These are descriptive/exploratory comparisons, not an unbiased held-out selection protocol. Consult each figure's original selection file.
5. **Budgets:** preserve original 600-step rates separately from task-specific deadline rates. Do not relabel late successes as original-benchmark failures. The full demonstration duration is not an oracle optimal completion time.
6. **Memory plots:** offline demonstration replay of projected memory tokens does not establish effective memory horizon or a causal contribution to policy decisions.
7. **Coverage:** the available archive includes historical diagnostics and partial/nonstandard episode counts. It must not be interpreted as a complete Cartesian grid simply because a file exists. Missing/quarantined results are not imputed.

## Representative model release

Public model repository: [HITdongdong/ActMem-VLA](https://huggingface.co/HITdongdong/ActMem-VLA). All ten task-specific representative plugins use training seed42 and step1500. Their remote weight SHA-256 values have been verified against source files; see `model_publication_receipt.json`. These are representative checkpoints, not task-wise selected best models. Plugin weights require the matching fine-tuned base checkpoint, not an arbitrary pi0.5 model. The model card explicitly documents this dependency.

## Validation performed for this release

All included rollout counts and success sums were checked against their audited summaries. Source manifests record SHA-256 values. New Python snapshots were parsed and scanned for common credential literals. This release does **not** claim a new full GPU-training/simulation rerun or a clean repository-wide test suite. Earlier gradient-validation evidence is retained under `results/` in the repository.

Model weights are intentionally excluded from Git history. Original demonstration datasets, caches, private credentials and unrelated projects are not included. Upstream code/model/dataset license terms continue to apply; this release does not relicense third-party weights or data.
