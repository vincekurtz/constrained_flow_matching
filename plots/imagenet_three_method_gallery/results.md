Inpainting study: existing_gallery references

Paired evaluation: 5/5 cases.

| Method | Success | FID full ↓ | FID hole ↓ | KID hole ×1000 ↓ | Known MAE ↓ | Hole RMSE ↓ | s/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| pcfm | 5/5 | 179.29 | 350.17 | 65.02 | 0.00000 | 0.3113 | 0.2 |
| pigdm | 5/5 | 132.73 | 382.86 | 42.19 | 0.00096 | 0.1979 | 160.4 |
| ldf | 5/5 | 94.26 | 274.17 | -4.41 | 0.00044 | 0.2205 | 51.0 |

- Synthetic sources measure model consistency, not real-data FID.
- Finite-sample FID is biased; compare identical sample counts.
- FID uses the common successful subset; failures are reported.
- Paired KID excludes matched cross pairs; may be negative.
- FID uses clipped uint8 images; pixel errors use raw floats.
- Hole FID uses the 192px missing crop with Clean-FID resizing.
- Runtime excludes compilation; budgets differ across methods.
- Original conditioning references also form the FID reference set.
- Small-sample diagnostic only; not a reliable FID ranking.
