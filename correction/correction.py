"""Boundary-correction experiment: accuracy of spectrogram-to-waveform corrections on
synthetic stimuli.

Eight annotators (A1-A8) marked the visible onset and offset of synthetic stimuli on
spectrograms computed with five FFT window sizes. Each correction moves both boundaries
inwards by a shift that depends on the window; the corrected onsets and durations are scored
against the known waveform times.

    python correction.py        writes results.json

Symbols used throughout, as in the README:
    N    FFT window size (samples)
    T    window duration N / fs (s)
    rho  noise-to-signal ratio of the stimulus
    N0   constant of the CallMark correction
    k    constant of the noise-aware correction
"""
import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ANNOTATORS = [f"A{i}" for i in range(1, 9)]
BLINDED = ("A1", "A3")              # annotated files with blinded names
WINDOWS = (128, 256, 512, 1024, 2048)
NOISE = (0, 5, 10)                  # noise amplitude in percent of the signal amplitude
TRUE_ONSET = 1.0                    # waveform onset of every stimulus (s)
METHODS = ("raw", "half_window", "callmark", "noise_aware")
CALLMARK_N0 = 6.0
NOISE_AWARE_K = 0.25
COLUMN_NAMES = (
    "annotator", "stimulus", "N", "T", "hop", "hop_samples", "marked_onset", "marked_duration",
    "true_duration", "noise", "click", "rho", "min_frequency", "max_frequency",
    "spectrogram_columns",
)


def read_annotation_table(annotator):
    """Rows of one annotator's CSV file, as a list of dicts."""
    path = HERE / "data" / "annotations" / f"{annotator}.csv"
    return list(csv.DictReader(open(path)))


def stimuli_presented_to_all(tables):
    """Set of stimulus file names that occur in every annotator's table."""
    common = None
    for table in tables.values():
        names = set()
        for row in table:
            names.add(row["filename"])
        if common is None:
            common = names
        else:
            common = common & names
    return common


def parse_display_configuration(text):
    """Display configuration stored as "key=value;key=value;...", as a dict of strings."""
    configuration = {}
    for item in text.split(";"):
        key, value = item.split("=")
        configuration[key] = value
    return configuration


def nearest_tested_window(nfft):
    """The tested window size closest to nfft."""
    distances = [abs(window - nfft) for window in WINDOWS]
    return WINDOWS[distances.index(min(distances))]


def load():
    """One entry per annotation, as a dict of arrays.

    Ground truth is encoded in the stimulus file name.
    """
    tables = {}
    for annotator in ANNOTATORS:
        tables[annotator] = read_annotation_table(annotator)
    noise_ratio = json.load(open(HERE / "data" / "noise_ratio.json"))
    # score the stimuli that were presented to every annotator
    scored = stimuli_presented_to_all(tables)

    columns = {}
    for name in COLUMN_NAMES:
        columns[name] = []
    for annotator, table in tables.items():
        for row in table:
            stimulus = row["filename"]
            if stimulus not in scored:
                continue
            configuration = parse_display_configuration(row["configSnapshot"])
            sampling_rate = float(configuration["samplingRate"])
            # window sizes that differ slightly from a tested size are assigned to it
            window = nearest_tested_window(int(configuration["nfft"]))
            hop_length = float(configuration["hopLength"])
            marked_onset = float(row["onset"])
            marked_offset = float(row["offset"])
            true_duration_ms = float(re.search(r"dur([\d.]+)ms", stimulus).group(1))
            noise_percent = int(re.search(r"noise(\d+)", stimulus).group(1))

            columns["annotator"].append(annotator)
            columns["stimulus"].append(stimulus)
            columns["N"].append(window)
            columns["T"].append(window / sampling_rate)
            columns["hop"].append(hop_length / sampling_rate)
            columns["hop_samples"].append(int(hop_length))
            columns["marked_onset"].append(marked_onset)
            columns["marked_duration"].append(marked_offset - marked_onset)
            columns["true_duration"].append(true_duration_ms / 1000)
            columns["noise"].append(noise_percent)
            columns["click"].append("click" in stimulus)
            columns["rho"].append(noise_ratio[stimulus])
            columns["min_frequency"].append(int(configuration["minFreq"]))
            columns["max_frequency"].append(int(configuration["maxFreq"]))
            columns["spectrogram_columns"].append(int(configuration["numSpecColumns"]))

    data = {}
    for name, values in columns.items():
        data[name] = np.array(values)
    return data


def subset(data, mask):
    """The annotations selected by a boolean mask."""
    selected = {}
    for name, values in data.items():
        selected[name] = values[mask]
    return selected


def rmse(errors):
    """Root mean square of the errors."""
    return float(np.sqrt(np.mean(np.square(errors))))


def rmse_by_group(errors, labels, groups):
    """RMSE of the errors within each group, as a dict keyed by group."""
    result = {}
    for group in groups:
        result[group] = rmse(errors[labels == group])
    return result


def largest_shift_leaving_one_column(data):
    """Shift (s) after which the annotation spans exactly one hop length."""
    return (data["marked_duration"] - data["hop"]) / 2


def shift(data, method, bounded, n0=CALLMARK_N0, k=NOISE_AWARE_K):
    """Inward shift of each boundary (s).

    bounded: limit the shift so that the corrected annotation spans at least one spectrogram
    column (one hop length), centred on the annotation.
    """
    T = data["T"]
    if method == "raw":
        shifts = np.zeros_like(T)
    elif method == "half_window":
        shifts = T / 2
    elif method == "callmark":
        shifts = T / 2 * (1 - np.cbrt(n0 / data["N"]))
    elif method == "noise_aware":
        shifts = T * (0.5 - k * np.cbrt(data["rho"]))
    if bounded:
        shifts = np.minimum(shifts, largest_shift_leaving_one_column(data))
    return shifts


def duration(data, method, bounded, n0=CALLMARK_N0, k=NOISE_AWARE_K):
    """Corrected duration (s)."""
    return data["marked_duration"] - 2 * shift(data, method, bounded, n0=n0, k=k)


def duration_error(data, method, bounded, n0=CALLMARK_N0, k=NOISE_AWARE_K):
    """Corrected minus true duration (ms)."""
    corrected = duration(data, method, bounded, n0=n0, k=k)
    return (corrected - data["true_duration"]) * 1000


def onset_error(data, method, bounded):
    """Corrected minus true onset (ms)."""
    corrected = data["marked_onset"] + shift(data, method, bounded)
    return (corrected - TRUE_ONSET) * 1000


def summary(data, method, bounded):
    """Onset and duration accuracy of one correction, pooled and by window, noise, annotator."""
    duration_errors = duration_error(data, method, bounded)
    onset_errors = onset_error(data, method, bounded)
    nonpositive = duration(data, method, bounded) <= 0
    harmonic = ~data["click"]
    true_harmonic_ms = data["true_duration"][harmonic] * 1000
    relative_errors = np.abs(duration_errors[harmonic]) / true_harmonic_ms

    nonpositive_pct_by_noise = {}
    for noise in NOISE:
        at_this_noise = data["noise"] == noise
        nonpositive_pct_by_noise[noise] = float(100 * nonpositive[at_this_noise].mean())

    return {
        "onset_bias_ms": float(onset_errors.mean()),
        "onset_sd_ms": float(onset_errors.std(ddof=1)),
        "onset_rmse_ms": rmse(onset_errors),
        "duration_rmse_ms": rmse(duration_errors),
        "duration_rmse_by_window_ms": rmse_by_group(duration_errors, data["N"], WINDOWS),
        "duration_rmse_by_noise_ms": rmse_by_group(duration_errors, data["noise"], NOISE),
        "duration_rmse_by_annotator_ms": rmse_by_group(
            duration_errors, data["annotator"], ANNOTATORS),
        "duration_mape_harmonic_pct": float(100 * np.mean(relative_errors)),
        "nonpositive_duration_pct": float(100 * nonpositive.mean()),
        "nonpositive_duration_pct_by_noise": nonpositive_pct_by_noise,
    }


def within_half_window_limit(data, squared_errors, by_window):
    """Which candidates are at least as accurate as the unbounded half-window correction.

    squared_errors has one row per candidate and one column per annotation. A candidate
    passes if its RMSE is no larger than that of the half-window correction at every noise
    level (and, with by_window, at every window size).
    """
    half_window_squared = np.square(duration_error(data, "half_window", False))
    if by_window:
        window_masks = [data["N"] == window for window in WINDOWS]
    else:
        window_masks = [np.ones(len(data["N"]), bool)]

    passes = np.ones(len(squared_errors), bool)
    for window_mask in window_masks:
        for noise in NOISE:
            cell = (data["noise"] == noise) & window_mask
            candidate_rmse = np.sqrt(squared_errors[:, cell].mean(axis=1))
            half_window_rmse = np.sqrt(half_window_squared[cell].mean())
            # the tolerance absorbs rounding when the two are equal
            passes &= candidate_rmse <= half_window_rmse + 1e-12
    return passes


def best_admissible(scores, admissible):
    """Index of the lowest score among the admissible candidates."""
    admissible_indices = np.flatnonzero(admissible)
    return admissible_indices[np.argmin(scores[admissible])]


def fit_n0(data):
    """Constant N0 of the bounded duration correction T (1 - cbrt(N0 / N)).

    Returns the value that minimises the pooled duration RMSE, among the values that are at
    least as accurate as the half-window correction at every window size and noise level
    (grid search, step 0.01).
    """
    candidates = np.round(np.arange(1.0, 20.0001, 0.01), 2)
    # one row per candidate N0, one column per annotation
    shifts = data["T"] / 2 * (1 - np.cbrt(candidates[:, None] / data["N"]))
    shifts = np.minimum(shifts, largest_shift_leaving_one_column(data))
    corrected = data["marked_duration"] - 2 * shifts
    squared_errors = np.square((corrected - data["true_duration"]) * 1000)
    admissible = within_half_window_limit(data, squared_errors, by_window=True)
    pooled_rmse = np.sqrt(squared_errors.mean(axis=1))
    best = best_admissible(pooled_rmse, admissible)
    lowest_admissible = float(candidates[admissible].min())
    highest_admissible = float(candidates[admissible].max())
    return {
        "n0": float(candidates[best]),
        "duration_rmse_ms": float(pooled_rmse[best]),
        "admissible_range": [lowest_admissible, highest_admissible],
    }


def fit_fractions(data):
    """One fraction f per window size, for the bounded duration correction f T.

    Each fraction minimises the duration RMSE among the fractions at least as accurate as the
    half-window correction at every noise level.
    """
    candidates = np.linspace(0.3, 1.0, 7001)
    fractions = {}
    sum_of_squared_errors = 0.0
    for window in WINDOWS:
        at_window = subset(data, data["N"] == window)
        # one row per candidate fraction, one column per annotation
        unbounded = at_window["marked_duration"] - candidates[:, None] * at_window["T"]
        corrected = np.maximum(unbounded, at_window["hop"])
        squared_errors = np.square((corrected - at_window["true_duration"]) * 1000)
        admissible = within_half_window_limit(at_window, squared_errors, by_window=False)
        best = best_admissible(squared_errors.mean(axis=1), admissible)
        fractions[window] = float(candidates[best])
        sum_of_squared_errors += squared_errors[best].sum()
    pooled_rmse = np.sqrt(sum_of_squared_errors / len(data["N"]))
    return {"fractions_of_T": fractions, "duration_rmse_ms": float(pooled_rmse)}


def fit_k(data):
    """Least-squares k of the noise-aware shift T (1/2 - k cbrt(rho)), fitted on durations.

    The corrected duration W - T + 2 k T cbrt(rho), with W the marked duration, is set equal
    to the true duration.
    """
    regressor = 2 * data["T"] * np.cbrt(data["rho"])
    target = data["true_duration"] - data["marked_duration"] + data["T"]
    return float(regressor @ target / (regressor @ regressor))


def unbiased_shift_over_T(data):
    """Shift, as a fraction of T, that makes the durations unbiased (least squares)."""
    excess_duration = data["marked_duration"] - data["true_duration"]
    numerator = np.sum(excess_duration * data["T"])
    denominator = np.sum(2 * data["T"] ** 2)
    return float(numerator / denominator)


def bootstrap_p(data, resamples=4000, seed=0):
    """Two-sided paired bootstrap over annotations: half-window versus CallMark duration RMSE, both with the
    one-column bound.

    The P value cannot be smaller than 1 / resamples.
    """
    half_window_squared = np.square(duration_error(data, "half_window", True))
    callmark_squared = np.square(duration_error(data, "callmark", True))
    n_annotations = len(half_window_squared)
    random = np.random.default_rng(seed)
    drawn = random.integers(n_annotations, size=(resamples, n_annotations))
    half_window_rmse = np.sqrt(half_window_squared[drawn].mean(axis=1))
    callmark_rmse = np.sqrt(callmark_squared[drawn].mean(axis=1))
    difference = half_window_rmse - callmark_rmse
    fraction_not_above_zero = (difference <= 0).mean()
    fraction_not_below_zero = (difference >= 0).mean()
    two_sided = 2 * min(fraction_not_above_zero, fraction_not_below_zero)
    two_sided = min(1, two_sided)
    return float(max(1 / resamples, two_sided))


def rmse_difference(squared_errors, in_group):
    """RMSE of the annotations in the group minus RMSE of the other annotations."""
    rmse_inside = np.sqrt(squared_errors[in_group].mean())
    rmse_outside = np.sqrt(squared_errors[~in_group].mean())
    return rmse_inside - rmse_outside


def group_accuracy(data, in_group):
    """Number of annotations of a group and its CallMark and half-window duration RMSEs."""
    group = subset(data, in_group)
    return {
        "n": int(in_group.sum()),
        "callmark_duration_rmse_ms": rmse(duration_error(group, "callmark", True)),
        "half_window_duration_rmse_ms": rmse(duration_error(group, "half_window", True)),
    }


def blinding(data):
    """Blinded versus other annotators, with a permutation test.

    The test runs over the 28 ways to choose two of eight annotators as the blinded pair
    (statistic: difference of the two groups' CallMark duration RMSEs).
    """
    is_blinded = np.isin(data["annotator"], BLINDED)
    result = {
        "blinded": group_accuracy(data, is_blinded),
        "others": group_accuracy(data, ~is_blinded),
    }

    squared_errors = np.square(duration_error(data, "callmark", True))
    observed = rmse_difference(squared_errors, is_blinded)
    permuted = []
    for pair in itertools.combinations(ANNOTATORS, 2):
        in_pair = np.isin(data["annotator"], pair)
        permuted.append(rmse_difference(squared_errors, in_pair))
    permuted = np.array(permuted)
    # the tolerance keeps the observed pair itself among the permutations counted
    at_least_as_extreme = np.abs(permuted) >= abs(observed) - 1e-12
    result["permutation_p"] = float(np.mean(at_least_as_extreme))
    return result


def count_smaller(first, second):
    """Number of positions at which the value in first is smaller than the value in second."""
    count = 0
    for first_value, second_value in zip(first, second):
        if first_value < second_value:
            count += 1
    return count


def leave_one_annotator_out(data):
    """Refit N0 (CallMark) and k (noise-aware) without each annotator and score that annotator."""
    n0_by_annotator = {}
    k_by_annotator = {}
    callmark_squared_errors = []
    noise_aware_squared_errors = []
    callmark_rmse = []
    half_window_rmse = []
    for annotator in ANNOTATORS:
        is_heldout = data["annotator"] == annotator
        train = subset(data, ~is_heldout)
        test = subset(data, is_heldout)
        n0 = fit_n0(train)["n0"]
        k = fit_k(train)
        n0_by_annotator[annotator] = n0
        k_by_annotator[annotator] = k
        callmark_squared = np.square(duration_error(test, "callmark", True, n0=n0))
        noise_aware_squared = np.square(duration_error(test, "noise_aware", True, k=k))
        callmark_squared_errors.append(callmark_squared)
        noise_aware_squared_errors.append(noise_aware_squared)
        callmark_rmse.append(float(np.sqrt(callmark_squared.mean())))
        half_window_rmse.append(rmse(duration_error(test, "half_window", True)))

    pooled_callmark_rmse = np.sqrt(np.concatenate(callmark_squared_errors).mean())
    pooled_noise_aware_rmse = np.sqrt(np.concatenate(noise_aware_squared_errors).mean())
    n0_values = n0_by_annotator.values()
    k_values = list(k_by_annotator.values())
    return {
        "n0": n0_by_annotator,
        "n0_range": [min(n0_values), max(n0_values)],
        "callmark_heldout_duration_rmse_ms": float(pooled_callmark_rmse),
        "callmark_heldout_duration_rmse_by_annotator_ms": dict(zip(ANNOTATORS, callmark_rmse)),
        "callmark_heldout_range_ms": [min(callmark_rmse), max(callmark_rmse)],
        "annotators_improved_over_half_window": count_smaller(callmark_rmse, half_window_rmse),
        "wilcoxon_p": float(wilcoxon(callmark_rmse, half_window_rmse).pvalue),
        "k": k_by_annotator,
        "k_mean": float(np.mean(k_values)),
        "noise_aware_heldout_duration_rmse_ms": float(pooled_noise_aware_rmse),
    }


def one_column_bound(data):
    """Annotations for which the bound of the CallMark correction applies."""
    bound_applies = duration(data, "callmark", False) < data["hop"]
    click_in_noise = data["click"] & (data["noise"] > 0)
    bounded_errors = duration_error(subset(data, bound_applies), "callmark", True)
    return {
        "n": int(bound_applies.sum()),
        "pct": float(100 * bound_applies.mean()),
        "n_not_clicks_in_noise": int(np.sum(bound_applies & ~click_in_noise)),
        "duration_rmse_ms": rmse(bounded_errors),
    }


def sorted_unique(values):
    """Distinct values of an array, as a sorted list."""
    return sorted(set(values.tolist()))


def display_settings(data):
    """Spectrogram display used by each annotator, read from the stored display configuration."""
    settings = {}
    for annotator in ANNOTATORS:
        own = data["annotator"] == annotator

        frequency_ranges = set()
        for low, high in zip(data["min_frequency"][own], data["max_frequency"][own]):
            frequency_ranges.add((int(low), int(high)))

        hop_lengths_by_window = {}
        for window in WINDOWS:
            own_at_window = own & (data["N"] == window)
            hop_lengths_by_window[window] = sorted_unique(data["hop_samples"][own_at_window])

        settings[annotator] = {
            "spectrogram_columns": sorted_unique(data["spectrogram_columns"][own]),
            "frequency_ranges_hz": sorted(frequency_ranges),
            "hop_lengths_by_window": hop_lengths_by_window,
        }
    return settings


def main():
    data = load()

    n_by_annotator = {}
    for annotator in ANNOTATORS:
        n_by_annotator[annotator] = int(np.sum(data["annotator"] == annotator))
    results = {
        "n_annotations": len(data["N"]),
        "n_stimuli": len(set(data["stimulus"])),
        "n_by_annotator": n_by_annotator,
        "display_settings": display_settings(data),
    }

    results["methods"] = {}
    for method in METHODS:
        for bounded in (False, True):
            if method == "raw" and bounded:
                # no shift, so there is nothing to bound
                continue
            label = "bounded" if bounded else "unbounded"
            results["methods"][f"{method}_{label}"] = summary(data, method, bounded)

    rmse_by_annotator = {}
    for name in ("half_window_bounded", "callmark_bounded", "noise_aware_bounded"):
        by_annotator = results["methods"][name]["duration_rmse_by_annotator_ms"]
        rmse_by_annotator[name] = list(by_annotator.values())
    half_window_rmse = rmse_by_annotator["half_window_bounded"]
    callmark_rmse = rmse_by_annotator["callmark_bounded"]
    noise_aware_rmse = rmse_by_annotator["noise_aware_bounded"]

    bootstrap_p_by_noise = {}
    for noise in NOISE:
        bootstrap_p_by_noise[noise] = bootstrap_p(subset(data, data["noise"] == noise))
    results["callmark_vs_half_window"] = {
        "wilcoxon_p": float(wilcoxon(half_window_rmse, callmark_rmse).pvalue),
        "bootstrap_p_by_noise": bootstrap_p_by_noise,
    }
    results["annotators_improved_by_noise_aware_over_callmark"] = count_smaller(
        noise_aware_rmse, callmark_rmse)

    results["n0_fit"] = fit_n0(data)
    results["fractions_fit"] = fit_fractions(data)

    callmark_shift_over_T = {}
    for window in (WINDOWS[0], WINDOWS[-1]):
        callmark_shift_over_T[window] = 0.5 * (1 - np.cbrt(CALLMARK_N0 / window))
    results["callmark_shift_over_T"] = callmark_shift_over_T

    unbiased_at_2048 = {}
    for noise in (0, 10):
        selected = subset(data, (data["N"] == 2048) & (data["noise"] == noise))
        unbiased_at_2048[noise] = unbiased_shift_over_T(selected)
    results["unbiased_shift_over_T_at_2048"] = unbiased_at_2048

    results["one_column_bound"] = one_column_bound(data)
    results["k_fit"] = fit_k(data)
    results["leave_one_annotator_out"] = leave_one_annotator_out(data)
    results["blinding"] = blinding(data)

    text = json.dumps(results, indent=1)
    (HERE / "results.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
