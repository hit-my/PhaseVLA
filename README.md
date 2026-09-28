# ActMem-VLA

## Code and experiment archive (2026-09-28)

The [release archive](releases/2026-09-28/README.md) contains the actual corrected training wrappers, audited raw per-episode results from A100 and LGD, separate seed/checkpoint indices, plotting inputs and provenance hashes. Historical and corrected experiments are explicitly distinguished. See its limitations before aggregating results.


ActMem-VLA (formerly PhaseVLA) studies task progress in vision-language-action policies using recurrent action memory and a lightweight **PreAction Expert (PAE)**. The implementation builds on [OpenPI](https://github.com/Physical-Intelligence/openpi) and π0.5; the main PyTorch model is named **FutureMamba** in the source code.

## Method

The current LIBERO-Mem mainline encodes previously executed action chunks with Mamba-2. The resulting memory tokens condition a PreAction Expert through attention KV. The frozen VLM supplies observation/language context. PAE performs the first four of ten denoising steps (`handoff_ratio=0.4`); the frozen original Action Expert (AE) performs the remaining six. Only the memory plugin and PAE are trained.

Episode resets clear recurrent state. Training and deployment keep causal action history, action masks, and checkpoint identity explicit. Earlier RoboMME observation-memory experiments are a separate protocol and should not be mixed with the LIBERO-Mem action-history results.

| Variant | Memory | Trainable expert | Denoising |
|---|---|---|---|
| ActMem-VLA | Mamba-2 action history | PAE | PAE → frozen AE |
| no-memory | No recurrent history | PAE | PAE → frozen AE |
| no-PE / Memory-AE | Mamba-2 action history | Original AE, with memory KV | AE only |
| no-PAE / frozen AE (later experiment) | Mamba-2 action history | Memory branch only; AE frozen | First 4 AE calls with memory, last 6 without |
| Capacity variants | Configurable memory depth/width | Configurable PAE depth/tokens | PAE → frozen AE |

The no-memory path samples independent queries from frozen conditioning caches. The no-PE path samples queries with their full causal action history and computes the frozen VLM prefix online. Training steps and wall-clock costs across these paths are not interchangeable without accounting for query counts and caching.

## Code map

- [Mainline model](src/openpi/models_pytorch/futuremamba.py), [Mamba memory](src/openpi/models_pytorch/mamba_memory.py), and [model configuration](src/openpi/models_pytorch/futuremamba_config.py).
- [Mainline/no-memory trainer](scripts/train_futuremamba_pytorch.py) and [independent-query cache loader](src/openpi/training/cached_query_data_loader.py).
- [Memory-AE model](src/openpi/models_pytorch/memory_ae.py), [trainer](scripts/train_memory_ae_pytorch.py), and [causal query loader](src/openpi/training/memory_ae_data_loader.py).
- [Experiment configurations](src/openpi/training/config.py), [policy loading](src/openpi/policies/policy_config.py), and [policy server](scripts/serve_policy.py).
- [Plugin checkpoints](src/openpi/training/futuremamba_checkpoint.py) and [Memory-AE checkpoints](src/openpi/training/memory_ae_checkpoint.py).
- [Committed evaluation artifacts](results/).

## Setup and execution

Use the dedicated [FutureMamba environment](environments/futuremamba/pyproject.toml), which specifies Python 3.11, PyTorch 2.9.1, Triton 3.5.1, and a pinned Mamba source revision. Building Mamba requires a compatible CUDA toolkit. Simulation runs in a separate environment.

```bash
git clone https://github.com/hit-my/PhaseVLA.git
cd PhaseVLA
uv sync --project environments/futuremamba
export PYTHONPATH="$PWD/src"
```

RoboMME experiments additionally require access to the private policy submodule and its benchmark dependencies:

```bash
git submodule update --init --recursive
```

Before training, set dataset, normalization assets, converted π0.5 base weights, action-cache, and conditioning-cache paths in the selected configuration. The LIBERO-Mem configurations retain experiment-server paths under `/data/libero_mem_baseline`; adapt them to your installation. Weights, datasets, caches, and rollout videos are external artifacts and are not distributed in this repository.

Mainline T6 example (after preparing those assets):

```bash
environments/futuremamba/.venv/bin/python scripts/train_futuremamba_pytorch.py   futuremamba_action_history_handoff04_libero_mem_bowl_t6   --seed 42 --num-train-steps 3000 --save-interval 500   --checkpoint-root /path/to/output/phasevla-t6-seed42
```

No-PE example (uses the seed recorded in the selected configuration):

```bash
environments/futuremamba/.venv/bin/python scripts/train_memory_ae_pytorch.py   futuremamba_nope_memory_ae_libero_mem_bowl_t6   --steps 3000 --save-interval 500   --checkpoint-root /path/to/output/nope-t6-seed42   --log-dir /path/to/output/nope-t6-seed42-logs
```

Use `--resume` with the same configuration and checkpoint root to restore a run. Checkpoint bundles retain model/protocol metadata and training state; do not substitute checkpoints between variants.

```bash
environments/futuremamba/.venv/bin/python scripts/serve_policy.py policy:checkpoint   --policy.config=futuremamba_action_history_handoff04_libero_mem_bowl_t6   --policy.dir=/path/to/output/phasevla-t6-seed42/3000
```

The server is only the policy endpoint. Reproducing the results below also requires the LIBERO-Mem task assets, stabilized initial states, and the matching formal evaluator. The generic LIBERO example is not a replacement for that protocol. Experiment queue/service wrappers currently live outside this repository on the experiment server.

## Recurrent-gradient update (September 10, 2026)

Gradient-enabled Mamba-2 `step` calls now use functional convolution and SSM
state updates. This lets a later query's loss reach earlier committed history
blocks. Native inference cache updates previously carried history forward but
did not propagate this recurrent gradient on CUDA. `torch.no_grad()` inference
retains the existing kernel path; parameter names and checkpoint tensor shapes
are unchanged. The differentiable step supports the reference `ngroups=1`,
non-distributed configuration and rejects unsupported grouped/parallel calls.
Training graph storage can grow with history length; the constant-size state
claim applies to inference, not full backpropagation through history.

Use a separate run directory for training with this update. Loading old weights
does not retroactively change their training history; record the Git revision
and keep old and corrected results separate. The historical tables below refer
to the old training implementation and are not results of this update. This
change does not select a final paper checkpoint or establish higher success on
every task. Memory-AE's separate full-sequence training path is unchanged.

With the pinned runtime, run the recurrent regression tests on an available GPU:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu PYTHONPATH=src   environments/futuremamba/.venv/bin/python -m pytest -q   src/openpi/models_pytorch/mamba_recurrent_gradient_test.py   src/openpi/models_pytorch/mamba_memory_test.py
```

The integrated backend passes **40 targeted regression tests**. Independent
FP32 output/gradient comparisons and five-point finite-difference checks for
`A_log` and `dt_bias` also pass. See the [validation record](results/mamba_recurrent_gradient_validation_20260910.json)
and [`validate_mamba_recurrent_gradients.py`](scripts/validate_mamba_recurrent_gradients.py).
The older verification snapshot below documents a different, partially failing
suite; it is retained for transparency.

## Validation

Run targeted CPU regression tests without allocating a training GPU:

```bash
CUDA_VISIBLE_DEVICES="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1   environments/futuremamba/.venv/bin/python -m pytest -q   src/openpi/models_pytorch/cached_query_futuremamba_test.py   src/openpi/models_pytorch/action_only_futuremamba_test.py   src/openpi/training/futuremamba_checkpoint_test.py   src/openpi/policies/policy_config_futuremamba_test.py   scripts/train_futuremamba_pytorch_test.py
```

These tests check selected model, loader, checkpoint, and policy contracts; they do not establish closed-loop robot success.

### Verification snapshot (September 8, 2026)

The targeted command above reports **19 passed, 25 failed**. An isolated checkout of the previous commit (`1522909`) reports **15 passed, the same 25 failed**; the four additional cached-query tests pass. There are no new failing test IDs in this selected suite. Existing failures include obsolete action-history fixtures, missing checkpoint identity fields in fixtures, and removed legacy RoboMME CLI arguments. This is not a clean test suite or proof that all runtime paths are correct.

Modified Python sources parse successfully; the T6/T7/T8 Memory-AE configurations and model/checkpoint/loader imports pass CPU validation. GPU training and simulation were not rerun for this publication. See the [compact validation record](results/code_sync_validation_20260908.json).

## Experiment status

As of September 8, 2026, the seed-42 ten-task sweep below is complete. T6/T7/T8 no-memory and no-PE sweeps at steps 500–3000 have also completed formal evaluation on the experiment server. Additional seed-0/seed-1 runs are in progress; no final multi-seed aggregate is claimed here. Capacity sweeps are exploratory, and real-robot validation has not been completed.

## Attribution

ActMem-VLA extends OpenPI and uses π0.5, Mamba-2, LIBERO-Mem, and RoboMME in the corresponding experiments. Upstream source, examples, and license notices are retained; generic OpenPI tutorials are available in the upstream repository. See [LICENSE](LICENSE).

## Latest LIBERO-Mem ten-task FutureMamba results

This section records the completed ten-task LIBERO-Mem FutureMamba experiment. Large runtime checkpoints, videos, and datasets remain outside Git; compact audited exports are committed under `results/`.

- Completion event: `2026-08-30 00:37:57 UTC`.
- Protocol: MuJoCo `3.2.2`, fixed stabilized initialization, 10 tasks, 20 episodes per task (episode IDs `0-19`), rollout seed `10001`, FutureMamba train seed `42`, `max_steps=600`, `replan_steps=20`, handoff ratio `0.4`.
- Conditioning cache: `961/961`; action cache: `961/961`; complete checkpoints: `60/60`; formal evaluations: `60/60`; audited rollouts/videos: `1200/1200` / `1200/1200`; audit violations: `0`.

### Overall success rates

| Method / checkpoint | Successes | Success rate | Wilson 95% CI | Delta vs baseline |
|---|---:|---:|---:|---:|
| baseline step-49999 | 162/200 | 81.0% | [75.00-85.83%] | - |
| FutureMamba step-500 | 166/200 | 83.0% | [77.18-87.57%] | +2.0 pp |
| FutureMamba step-1000 | 171/200 | 85.5% | [79.95-89.71%] | +4.5 pp |
| FutureMamba step-1500 | 168/200 | 84.0% | [78.29-88.43%] | +3.0 pp |
| FutureMamba step-2000 | 177/200 | 88.5% | [83.34-92.21%] | +7.5 pp |
| FutureMamba step-2500 | 168/200 | 84.0% | [78.29-88.43%] | +3.0 pp |
| FutureMamba step-3000 | 171/200 | 85.5% | [79.95-89.71%] | +4.5 pp |

The best common checkpoint is FutureMamba step-2000: `177/200 = 88.5%` (`+7.5 pp` vs baseline). The unadjusted paired exact McNemar test is `p=0.023703`; because step-2000 was selected from the six-point sweep, the Bonferroni-adjusted value is `0.142216` and should not be treated as confirmatory significance.

### Per-task success counts

| Task | Baseline | FM-500 | FM-1000 | FM-1500 | FM-2000 | FM-2500 | FM-3000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 - pick up the bowl and place it back on the plate | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 20/20 | 20/20 |
| T2 - lift the bottle and put it down on the plate | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| T3 - lift the bowl and place it back on the plate 3 times | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 20/20 | 20/20 |
| T4 - pick up the bottle and put it down the plate 3 times | 19/20 | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 20/20 |
| T5 - lift the bowl and place it back on the plate 5 times | 20/20 | 19/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| T6 - pick up the bowl and place it on the plate 7 times | 6/20 | 11/20 | 13/20 | 12/20 | 16/20 | 13/20 | 15/20 |
| T7 - swap the 2 bowls on their plates using the empty plate | 6/20 | 8/20 | 8/20 | 7/20 | 10/20 | 6/20 | 6/20 |
| T8 - rotate the 3 bowls on their plates from left to right using the empty plate | 11/20 | 8/20 | 10/20 | 9/20 | 14/20 | 10/20 | 10/20 |
| T9 - put the cream cheese in the nearest basket and place that basket in the center | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 20/20 | 20/20 |
| T10 - put the cream cheese in the nearest basket and place the empty basket in the center | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |

The clearest gains at step-2000 are T6 (`6/20` to `16/20`, +50 pp), T7 (`6/20` to `10/20`, +20 pp), and T8 (`11/20` to `14/20`, +15 pp). T1-T3, T5, T9, and T10 have 100% baseline success and therefore no success-rate headroom.

### Per-episode completion steps

The complete 1,400-row episode table contains baseline plus all six FutureMamba checkpoints for every task and episode. `completion_steps` is the recorded task step. Unsuccessful rollouts are retained and normally equal the `max_steps=600` timeout.

- `results/libero_mem_all10_summary.csv`: per-method/checkpoint/task success rates and successful-episode completion-step statistics.
- `results/libero_mem_all10_episode_steps.csv`: every baseline/FutureMamba task, checkpoint, and episode outcome with completion steps.
- `results/libero_mem_all10_completion_steps.md`: human-readable table with all 1,400 episode completion-step sequences.
- `results/libero_mem_all10_final_audit.json`: machine-readable audit export; source audit SHA256 `sha256:c65b7322b0cf38b7a1de9e213d81eb8a2964eb9adb92aafc7ae0062716d90daf`.

Task order follows authoritative `meta/tasks.jsonl`; these are ten independent task models, not one shared model. Checkpoint selection by the same evaluation set is exploratory and optimistic; use the common step-2000 row for the primary comparison.
