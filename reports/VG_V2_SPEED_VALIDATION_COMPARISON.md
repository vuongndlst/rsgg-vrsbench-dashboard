# VG optimized screening validation comparison

Snapshot: 2026-10-19T22:33:44.348513+00:00. Validation5000 / SGDet GC / class and inclusive subject/object IoU>=0.5. All scores below are percentages.

| Arm | Seed | Status | Selected epoch | R@50 | mR@50 |
|---|---:|---|---:|---:|---:|
| Frozen reference | — | fixed | — | 25.7751 | 16.0156 |
| native |1729| complete |6| 26.0278 | 16.6467 |
| context |1729| complete |6| 26.2568 | 16.2430 |
| oneway |1729| complete |6| 26.0996 | 16.6954 |
| feedback |1729| complete |6| 26.0309 | 16.8343 |

Completed screening selections only. Single-seed validation does not establish benefit from regions, superiority or caption-clean performance. Confirmation and locked test require the separately registered gates.

The superseded native run uses a different dropout RNG schedule and is historical only; it is not a matched control for this version. Fresh caption-clean detector training and caption evaluation require the remaining audited release and resource gates.
