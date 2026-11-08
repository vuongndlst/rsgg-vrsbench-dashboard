# FAIR storage inventory — read-only 24/10/2026

Snapshot UTC: 2026-10-24T08:09:57.299253+00:00

FAIR allocated: **6.7764GiB (7.2761GB)**. RSGG filesystem free: **4.8819GiB**. Same filesystem: yes.

| Path | Allocated GiB |
|---|---:|
| `${REMOTE_HOME}/fair-project` | 6.7764 |
| `${REMOTE_HOME}/fair-project/output_sg2text` | 3.6835 |
| `${REMOTE_HOME}/fair-project/output_sg2text/trained` | 2.9385 |
| `${REMOTE_HOME}/fair-project/datasets` | 0.9659 |
| `${REMOTE_HOME}/fair-project/third_party` | 0.7534 |
| `${REMOTE_HOME}/fair-project/datasets/vg_para` | 0.6580 |
| `${REMOTE_HOME}/fair-project/output_sg2text/cache` | 0.5356 |
| `${REMOTE_HOME}/fair-project/third_party/T2I-CompBench-main` | 0.5223 |
| `${REMOTE_HOME}/fair-project/models` | 0.4692 |
| `${REMOTE_HOME}/fair-project/models/hf_cache` | 0.4692 |
| `${REMOTE_HOME}/fair-project/output_captions_v2` | 0.2842 |
| `${REMOTE_HOME}/fair-project/output_captions_v2/cache` | 0.2465 |

Nested subdirectories are included in parent sizes; do not sum overlapping rows.

Hypothetical free after fully reclaiming FAIR: **11.6583GiB**, still **4.3417GiB below16GiB reserve**, before output/staging costs. This is an upper bound pending hardlink/open-file checks.

No own process cwd/command references FAIR at this snapshot. Other users/openFDs/external hardlinks were not fully audited. Nothing deleted; user requested inventory to consider deletion.
