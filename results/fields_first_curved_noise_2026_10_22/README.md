# Curved torus with measurement noise

The subsequent [correction and paired reruns](../fields_first_curved_noise_fix_2026_10_22/README.md)
pass all three levels. This report records the earlier configuration.

The successful clean configuration did not pass at 1%, 5% or 10% noise.
This is a controlled check of that configuration, not a completed exhaustive
search over longer windows.

The same four sensors, orbit, seed 31 and 2,400/320 train/query split are used.
Noise is independent Gaussian measurement error with per-channel standard
deviation equal to the stated fraction of the clean TRAIN signal deviation.
The same Gaussian realization is scaled across levels. Fits receive only
readings; phase number is not supplied. Conditioning, equations and acceptance
thresholds are unchanged. Jobs ran sequentially with one numerical thread.

## Same clean configuration

Use 24 neighbours and the lag search 4, 8, which contains the successful clean
configuration. All these attempts completed at every noise level.

| Noise | Estimated phases | Result at lag 8 | Tangent spectral gap, required > 0.1 |
|---|---:|---|---:|
| 0% | 3 | Passed, query phase MAE 0.00420 rad | 1.000 |
| 1% | 3 | Tangent geometry rejected | 0.00247 |
| 5% | 3 | Tangent geometry rejected | 0.00211 |
| 10% | Unresolved | No common topology scale with the required coverage buffer | - |

[Machine-readable comparison](baseline_summary.json) links the individual
records. The 1% and 5% baseline records were extracted from completed attempts
in the saved broader-search checkpoints. No failed-model phase or reading
errors are reported as successful predictions.

Exact quadratic geometry no longer validates at 1% and 5%; the local fallback
fails its tangent separation check. The retained numerical window span grows
from 13 to 32, although MSSA still chooses rank 13. Under the existing size
rule this also disables the global affine field block. Its contribution has
not been isolated from the tangent estimation error.

## Additional search and resource limits

At 1%, lag 8 with 48 and 96 neighbours reaches the field and period fit but
fails its training checks:

| Neighbours | Closed-integral correction, required < 0.05 | Phase/field error, required < 0.15 |
|---|---:|---:|
| 48 | 0.2767 | 0.3901 |
| 96 | 0.2646 | 0.3709 |

Those results are in [the checkpoint](noise01.full_search_checkpoint.json).
A lag-16 probe was also completed, but further probing was interrupted before
large field allocations. The 5% broader search was interrupted after its
baseline attempts. These searches are not claimed complete.

Longer noisy windows can create multi-GB arrays in the current implementation.
They were excluded from the final controlled comparison to respect the low-load
constraint. One seed gives a paired severity test, not a robustness rate.

## Reproduce the controlled comparison

Run each command after the previous one finishes:

```bash
nice -n 10 .venv/bin/python results/fields_first_curved_noise_2026_10_22/run_noise.py --noise 0.01 --baseline-only
nice -n 10 .venv/bin/python results/fields_first_curved_noise_2026_10_22/run_noise.py --noise 0.05 --baseline-only
nice -n 10 .venv/bin/python results/fields_first_curved_noise_2026_10_22/run_noise.py --noise 0.10 --baseline-only
```

The runner caches dimension estimates only for exactly identical source arrays
and parameters, returning separate copies of the diagnostics. It does not reuse
estimates between noise levels or change any fitting equation.
