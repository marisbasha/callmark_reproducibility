"""Statistical tests of the benchmark: paired bootstraps and the Wilcoxon signed-rank test across
corpora."""
import numpy as np
from scipy.stats import wilcoxon

from benchmark import config, metrics

REPLICATES = config.BOOTSTRAP_REPLICATES

# Number of replicates drawn by one call of the random generator.
BATCH = 50


def p_value(replicates):
    """Return the two-sided bootstrap P value.

    It is twice the smaller fraction of replicate differences on either side of zero, and at least
    1 / number of replicates."""
    fraction_below = np.mean(replicates <= 0)
    fraction_above = np.mean(replicates >= 0)
    tail = min(fraction_below, fraction_above)
    return float(min(1, max(1 / len(replicates), 2 * tail)))


def summary(difference, replicates):
    """Return the observed difference, its 95% percentile interval and the P value."""
    lower, upper = np.percentile(replicates, [2.5, 97.5])
    return {
        "difference": float(difference),
        "interval": [float(lower), float(upper)],
        "p": p_value(replicates),
    }


def wilcoxon_corpora(a, b):
    """Return the two-sided Wilcoxon signed-rank P value for two systems paired across corpora.

    `a` and `b` are {corpus: value}; the corpora present in both are used."""
    corpora = sorted(set(a) & set(b))
    values_a = [a[corpus] for corpus in corpora]
    values_b = [b[corpus] for corpus in corpora]
    return float(wilcoxon(values_a, values_b).pvalue)


def bootstrap_macro_f1(truth, pred_a, pred_b, taxa, seed=42):
    """Return the paired bootstrap over test examples of the macro-F1 difference a - b.

    Both systems are evaluated on the same resampled examples and the macro-F1 over `taxa` is
    recomputed in every replicate."""
    counts_a = metrics.taxon_counts(truth, pred_a, taxa)
    counts_b = metrics.taxon_counts(truth, pred_b, taxa)
    n_examples = len(truth)
    flat_a = counts_a.reshape(n_examples, -1)
    flat_b = counts_b.reshape(n_examples, -1)
    batch_shape = (-1, len(taxa), 3)
    uniform = np.full(n_examples, 1 / n_examples)
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(REPLICATES // BATCH):
        # Each row of weights counts how often every example is drawn in one replicate.
        weights = rng.multinomial(n_examples, uniform, size=BATCH)
        resampled_a = (weights @ flat_a).reshape(batch_shape)
        resampled_b = (weights @ flat_b).reshape(batch_shape)
        macro_f1_a = metrics.taxon_f1(resampled_a).mean(axis=1)
        macro_f1_b = metrics.taxon_f1(resampled_b).mean(axis=1)
        replicates.extend(macro_f1_a - macro_f1_b)
    observed_a = metrics.taxon_f1(counts_a.sum(axis=0)).mean()
    observed_b = metrics.taxon_f1(counts_b.sum(axis=0)).mean()
    return summary(observed_a - observed_b, np.array(replicates))


def bootstrap_within_corpora(a, b, corpora, rng):
    """Return the paired bootstrap of the difference a - b in the pooled mean of 0/1 outcomes.

    `a` and `b` hold one entry per vocalization or pair; the entries are resampled within each
    corpus."""
    difference = a.astype(float) - b
    replicates = np.zeros(REPLICATES)
    for corpus in np.unique(corpora):
        corpus_difference = difference[corpora == corpus]
        n_entries = len(corpus_difference)
        uniform = np.full(n_entries, 1 / n_entries)
        batches = []
        for _ in range(REPLICATES // BATCH):
            weights = rng.multinomial(n_entries, uniform, size=BATCH)
            batches.append(weights @ corpus_difference)
        replicates += np.concatenate(batches)
    return summary(difference.mean(), replicates / len(difference))


def bootstrap_chunks(hits_a, hits_b, reference_counts, seed=42):
    """Return one P value per column for the recall difference a - b, bootstrapped over chunks.

    `hits_a` and `hits_b` are per-chunk hit counts [chunk, column] and `reference_counts` the
    number of reference vocalizations [chunk]: the vocalizations of a chunk are resampled
    together."""
    rng = np.random.default_rng(seed)
    n_chunks = len(reference_counts)
    uniform = np.full(n_chunks, 1 / n_chunks)
    weights = rng.multinomial(n_chunks, uniform, size=REPLICATES).astype(float)
    hit_difference = weights @ hits_a - weights @ hits_b
    replicates = hit_difference / (weights @ reference_counts)[:, None]
    p_values = []
    for column in range(replicates.shape[1]):
        p_values.append(p_value(replicates[:, column]))
    return p_values
