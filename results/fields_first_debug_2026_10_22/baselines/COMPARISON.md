| Experiment (train/query) | r: ours / CC / TC | Reading NRMSE, %: ours / SSA / MSSA | Phase MAE, rad: ours / CC / TC |
|---|---|---|---|
| scalar_clean (1800/600) | 1 / 1 / 1 | 1.138 / 0.527 / 0.527 | 0.00430 / 0.37980 / 0.37980 |
| scalar_noise10 (1800/600) | 1 / 1 / 1 | 2.267 / 3.290 / 3.290 | 0.01070 / 0.37870 / 0.37870 |
| nonlinear_sensors (2400/320) | 2 / 2 / 2 | 2.705 / 4.060 / 1.941 | 0.01405 / 0.19191 / 0.18388 |
| optical (2400/320) | 2 / 2 / 2 | 1.041 / 1.586 / 0.550 | 0.00536 / 0.09363 / 0.06985 |
| scalar_clean (1800/320) | 1 / 1 / 1 | 1.150 / 0.527 / 0.527 | 0.00426 / 0.37945 / 0.37945 |
| scalar_noise10 (1800/320) | 1 / 1 / 1 | 2.548 / 3.753 / 3.753 | 0.01210 / 0.38152 / 0.38152 |
| three_phases_linear (2400/320) | 3 / 3 / 3 | 1.302 / 0.125 / 0.000 | 0.00387 / 0.11691 / 0.10072 |
| two_waves (2400/320) | 2 / 2 / 2 | 0.887 / 1.728 / 0.702 | 0.00498 / 0.11481 / 0.12429 |
| fundamental2waves_noise0p05 (2400/600) | 2 / 2 / 2 | 0.891 / 1.675 / 0.643 | 0.00478 / 0.11475 / 0.12460 |
| nonlinear3sensor_wide_noise0p05 (2400/600) | 2 / 2 / 2 | 2.730 / 4.097 / 2.050 | 0.01725 / 0.19255 / 0.18907 |
| opticalsteady1d_layered_noise0p05 (2400/600) | 2 / 2 / 2 | 1.132 / 1.578 / 0.633 | 0.00540 / 0.09444 / 0.07063 |

All methods receive the same raw measurements, train/query split and evaluation indices; tuning uses training data only. SSA/MSSA use lag 32; circular (CC) and toroidal (TC) coordinates share our training-selected lag in this controlled comparison. Phase MAE measures uniform dynamical-phase recovery, not topological correctness; this benchmark is conditional on accepted fields-first runs. NRMSE uses each clean oscillatory channel's training standard deviation; - means unavailable and failed means no valid complete phase basis.
