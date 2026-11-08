Inpainting study: synthetic references

Paired evaluation: 251/256 cases.

| Method | Success | FID full ↓ | FID hole ↓ | KID hole ×1000 ↓ | Known MAE ↓ | Hole RMSE ↓ | s/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| pcfm | 256/256 | 83.24 | 181.19 | 74.25 | 0.00000 | 0.3221 | 0.1 |
| pigdm | 251/256 | 56.49 | 124.88 | 21.06 | 0.00107 | 0.1850 | 69.2 |
| ldf | 256/256 | 55.40 | 103.98 | 8.72 | 0.00044 | 0.2345 | 32.2 |

- Synthetic sources measure model consistency, not real-data FID.
- Finite-sample FID is biased; compare identical sample counts.
- FID uses the common successful subset; failures are reported.
- Paired KID excludes matched cross pairs; may be negative.
- FID uses clipped uint8 images; pixel errors use raw floats.
- Hole FID uses the 192px missing crop with Clean-FID resizing.
- Runtime excludes compilation; budgets differ across methods.
- Original conditioning references also form the FID reference set.
