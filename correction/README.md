# Boundary correction

Reproduces the numbers of the boundary-correction experiment: the accuracy of spectrogram-to-waveform
corrections on synthetic stimuli annotated by eight annotators (A1-A8) at five FFT window sizes.

    python correction.py

needs numpy and scipy, runs in a few seconds, and writes `results.json`.

## Data

- `data/annotations/A1.csv` ... `A8.csv`: one row per annotation, with the stimulus file name, the visible
  onset and offset (s) and the display configuration stored by CallMark (`nfft`, `hopLength`, `samplingRate`,
  frequency range, ...). The true duration and the noise level are encoded in the stimulus file name; every
  stimulus starts 1 s into its file. The 34 stimuli presented to all annotators are scored (1,359 annotations).
- `data/noise_ratio.json`: noise-to-signal ratio rho of each stimulus (median RMS amplitude of 10-ms blocks
  divided by their maximum, bounded below by 0.001), used by the noise-aware correction.

## Corrections

Each boundary is moved inwards by a shift; `T = N / fs` is the window duration.

| name | shift |
|---|---|
| `raw` | 0 |
| `half_window` | T/2 |
| `callmark` | T/2 (1 - cbrt(6/N)) |
| `noise_aware` | T (1/2 - 1/4 cbrt(rho)) |

`bounded` limits the shift so that the corrected annotation spans at least one hop length. The CallMark
correction includes this bound.

## Expected output

Duration RMSE (ms), key `methods.<name>.duration_rmse_ms` and `duration_rmse_by_window_ms`:

| method | onset bias ± s.d. | onset RMSE | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---|---|---|---|---|---|---|---|
| `raw_unbounded` | -5.98 ± 6.40 | 8.76 | 2.37 | 4.16 | 8.08 | 16.99 | 35.41 | 18.06 |
| `half_window_unbounded` | 3.02 ± 2.78 | 4.10 | 2.46 | 3.48 | 5.07 | 7.69 | 13.89 | 7.69 |
| `half_window_bounded` | 2.40 ± 2.09 | 3.18 | 2.45 | 3.46 | 4.93 | 6.81 | 9.71 | 6.05 |
| `callmark_bounded` | 1.07 ± 1.68 | 1.99 | 1.99 | 2.59 | 3.29 | 4.13 | 6.01 | 3.86 |
| `noise_aware_unbounded` | 0.98 ± 1.16 | 1.52 | 2.14 | 2.73 | 3.25 | 3.21 | 3.61 | 3.03 |
| `noise_aware_bounded` | | | 2.14 | 2.73 | 3.24 | 3.18 | 3.08 | 2.90 |

| quantity | key | value |
|---|---|---|
| duration RMSE at 0 / 5 / 10 % noise, half-window | `methods.half_window_unbounded.duration_rmse_by_noise_ms` | 2.79 / 8.47 / 9.84 ms |
| same, CallMark | `methods.callmark_bounded.duration_rmse_by_noise_ms` | 2.63 / 3.62 / 4.99 ms |
| same, noise-aware | `methods.noise_aware_unbounded.duration_rmse_by_noise_ms` | 2.12 / 3.06 / 3.69 ms |
| durations <= 0, half-window (all; by noise) | `nonpositive_duration_pct`, `..._by_noise` | 17.7 %; 6.2 / 21.2 / 25.2 % |
| durations <= 0, noise-aware (all; by noise) | | 3.4 %; 1.6 / 4.2 / 4.3 % |
| durations <= 0, CallMark without / with bound | | 8.1 % / 0 % |
| duration error of harmonic stacks (raw, half-window, CallMark, noise-aware) | `duration_mape_harmonic_pct` | 76 / 33 / 17 / 13 % |
| CallMark vs half-window across annotators, Wilcoxon | `callmark_vs_half_window.wilcoxon_p` | 0.0078 |
| CallMark vs half-window, bootstrap at 0 / 5 / 10 % noise | `callmark_vs_half_window.bootstrap_p_by_noise` | 0.26 / 0.00025 / 0.00025 |
| fitted constant N0 | `n0_fit.n0` | 6.40 |
| five fitted fractions, duration RMSE | `fractions_fit.duration_rmse_ms` | 3.77 ms |
| CallMark shift at N = 128 and 2048 | `callmark_shift_over_T` | 0.32 T, 0.43 T |
| unbiased shift at N = 2048, clean / 10 % noise | `unbiased_shift_over_T_at_2048` | 0.46 T / 0.32 T |
| one-column bound applied | `one_column_bound` | 120 annotations (8.8 %), 1 not a click in noise, RMSE 1.0 ms |
| N0 refitted without each annotator | `leave_one_annotator_out.n0_range` | 5.29-6.88 |
| held-out CallMark duration RMSE (range over annotators) | `leave_one_annotator_out.callmark_heldout_*` | 3.85 ms (3.34-4.60) |
| held-out annotators improved over half-window, Wilcoxon | `leave_one_annotator_out.wilcoxon_p` | 8 of 8, 0.0078 |
| noise-aware k refitted without each annotator, held-out RMSE | `leave_one_annotator_out.k_mean`, `noise_aware_heldout_duration_rmse_ms` | 0.279, 2.97 ms |
| blinded vs other annotators, CallMark | `blinding` | 3.69 vs 3.92 ms, permutation P = 0.54 |
| blinded vs other annotators, half-window | `blinding` | 6.96 vs 7.92 ms |
| display settings per annotator | `display_settings` | hop lengths, frequency ranges, number of columns |
