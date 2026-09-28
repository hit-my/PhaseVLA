# Parameter ablations

Success rates (%); checkpoint 1500; training seed 42; three evaluation seeds, 60 episodes per task.

Default: handoff 0.4, Mamba width 1024 / depth 2, PAE depth 4. Each row changes only the stated configuration parameter.

| Configuration | T6 | T7 | T8 | Average |
|---|---:|---:|---:|---:|
| Default | 80.0 | 40.0 | 45.0 | 55.0 |
| Handoff ratio 0.3 | 65.0 | 36.7 | 43.3 | 48.3 |
| Handoff ratio 0.5 | 66.7 | 25.0 | 43.3 | 45.0 |
| Handoff ratio 0.6 | 71.7 | 28.3 | 41.7 | 47.2 |
| Mamba depth 4 | 58.3 | 25.0 | 46.7 | 43.3 |
| Mamba width 1536 | 76.7 | 40.0 | 50.0 | 55.6 |
| PAE depth 2 | 70.0 | 21.7 | 40.0 | 43.9 |
| PAE depth 6 | 71.7 | 35.0 | 55.0 | 53.9 |

Checkpoint 1500 was selected post hoc from evaluation results; this is an exploratory comparison.
