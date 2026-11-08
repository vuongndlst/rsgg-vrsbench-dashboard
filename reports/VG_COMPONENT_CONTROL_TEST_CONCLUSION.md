# Context component TEST outcome —20October2026

Completed07:51Vietnam, one pass on26446/26446canonical VG150 TEST images; no missing/empty-GT images excluded. Context seed1729, validation-selected checkpoint epoch10. Source/checkpoint/preregistration, test ledger, and both saved prediction artifact hashes verified PASS.

| K | R@K% | mR@K% |
|---|---:|---:|
|20|20.6987|11.7989|
|50|27.5137|16.9198|
|100|32.2812|20.4280|

Canonical validation gates passed on5000images, real-output postprocessor parity32images, independent float64 evaluator parity and legacy checkpoint validation reproduction. Canonical validation R50=24.1294%/mR50=16.8228%; the older selection values24.9358%/16.9299% use legacy targets/postprocessor and should not be conflated.

| Published reference | R50% | mR50% | Our minus published R/mR, pp |
|---|---:|---:|---|
|[Salience-SGG R50](https://arxiv.org/html/2601.08728v1)|28.80|18.00|-1.2863/-1.0802|
|[Dual-SGG+LA tau0.2 R50](https://arxiv.org/html/2607.06176v1)|30.20|18.70|-2.6863/-1.7802|
|[Dual-SGG+LA tau0.3 R50](https://arxiv.org/html/2607.06176v1)|25.30|22.20|+2.2137/-5.2802|

The Context result is numerically below Salience-SGG R50 in both metrics (1.2863pp R50,1.0802pp mR50). This is a descriptive gap, not a matched-recipe significance test. Dual-SGG tradeoffs are reported separately. The current component is a useful measured control, not demonstrated SOTA or evidence that regions help.

One selected seed, pretrained-head adaptation, fixed cached transforms, legacy training annotations and resize rounding differ from the native training recipe. Canonical targets/postprocessor/evaluator checks do not erase those training differences. Benchmark weights are not caption-clean and caption quality has not been assessed.

No additional test inference/scoring was performed during provenance verification. Keep this assessment closed; do not select models, thresholds, losses, stopping or calibration from it. Remaining work is the independent clean-data audit/release, audited fresh detector/SGG training and matched graph-to-text controls. Earlier confirmation remains preserved and human-superseded; no blind resume.
