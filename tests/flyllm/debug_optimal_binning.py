
import numpy as np
import sys
import logging

from flyllm.optimal_binning import (assign_bins, compute_transition_matrix, compute_log_likelihood,
                                    _compute_log_likelihood_from_counts, _update_bins_and_counts_for_edge_k,
                                    set_boundary_edges, optimize_bin_edges, split_sequences, subsample_sequences)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def compute_transition_matrix0(binned_sequences, K):
    """
    Original version: Compute transition count matrix and probability matrix.

    Parameters:
    -----------
    binned_sequences : list of ndarrays
        List of binned sequences
    K : int
        Number of bins

    Returns:
    --------
    counts : ndarray of shape (K, K)
        Transition count matrix
    prob_matrix : ndarray of shape (K, K)
        Transition probability matrix
    """
    counts = np.zeros((K, K), dtype=np.float64)

    for seq in binned_sequences:
        for t in range(len(seq) - 1):
            i, j = seq[t], seq[t+1]
            counts[i, j] += 1

    # Compute probabilities
    prob_matrix = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        total = counts[i, :].sum()
        if total > 0:
            prob_matrix[i, :] = counts[i, :] / total

    return counts, prob_matrix


def compute_log_likelihood0(X, edges, prob_matrix):
    """
    Original version: Compute log-likelihood of data given edges and transition probabilities.

    Log-likelihood = sum_t [log p(b(x_t)|b(x_{t-1})) - log width(b(x_t))]

    Parameters:
    -----------
    X : list of ndarrays
        List of sequences
    edges : ndarray
        Bin edges
    prob_matrix : ndarray of shape (K, K)
        Transition probability matrix

    Returns:
    --------
    log_likelihood : float
        Total log-likelihood
    """
    K = len(edges) - 1
    widths = np.diff(edges)

    log_likelihood = 0.0

    for seq in X:
        bins = np.searchsorted(edges, seq, side='right') - 1
        bins = np.clip(bins, 0, K - 1)

        for t in range(len(seq) - 1):
            i, j = bins[t], bins[t+1]
            p_ij = prob_matrix[i, j]

            if p_ij > 0:  # Only include terms where transition occurs
                log_likelihood += np.log(p_ij) - np.log(widths[j])

    return log_likelihood


def _compute_log_likelihood_from_counts0(counts, widths, K):
    """
    Original version: Compute log-likelihood from transition counts and bin widths.

    This is faster than recomputing from raw data when we have counts.
    """
    log_lik = 0.0

    for i in range(K):
        row_sum = counts[i, :].sum()
        if row_sum > 0:
            for j in range(K):
                if counts[i, j] > 0:
                    p_ij = counts[i, j] / row_sum
                    log_lik += counts[i, j] * (np.log(p_ij) - np.log(widths[j]))

    return log_lik


def load_debug_data(filepath, feature_idx=2):
    """
    Load zscored_velocity data

    Args:
        filepath (str): Path to the .npz file
        feature_idx (int, optional): Index of the feature to extract. Defaults to 2.

    Returns:
        feature_data: ndarray: Extracted feature data of shape (nflies, T)
        useoutputmask: ndarray: Corresponding useoutputmask of shape (nflies, T)
    """
    # Load data
    LOG.info(f"Loading data from {filepath}...")
    data = np.load(filepath)

    LOG.info(f"Extracting feature index {feature_idx}")
    feature_data = data['zscored_velocity'][:, :, feature_idx]  # (nflies, T, nfeatures)
    useoutputmask = data['useoutputmask'].T  # (nflies,T)

    LOG.info(f"Data shape: {feature_data.shape}")
    LOG.info(f"  nflies={feature_data.shape[0]}")
    LOG.info(f"  T={feature_data.shape[1]}")

    return feature_data, useoutputmask


def generate_synthetic_data(n_sequences=10, seq_length=1000, K=20,
                            data_range=(-10, 10), sparsity=0.8, seed=42):
    """
    Generate synthetic Markovian data matching the assumed model structure.

    Creates K bins with random widths, a random transition probability matrix,
    and generates sequences by sampling uniformly within bins according to
    the transition probabilities.

    Parameters:
    -----------
    n_sequences : int
        Number of sequences to generate
    seq_length : int
        Length of each sequence
    K : int
        Number of bins
    data_range : tuple
        (min, max) range for data
    sparsity : float
        Sparsity of transition matrix (fraction of zero entries)
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    sequences : list of ndarrays
        Generated sequences
    true_edges : ndarray
        Ground truth bin edges
    true_prob_matrix : ndarray
        Ground truth transition probability matrix
    """
    np.random.seed(seed)

    # Generate random bin widths that sum to the total range
    total_width = data_range[1] - data_range[0]
    # Sample from Dirichlet to get random proportions
    width_proportions = np.random.dirichlet(np.ones(K) * 2)  # alpha=2 for moderate variation
    widths = width_proportions * total_width

    # Create edges from widths
    true_edges = np.zeros(K + 1)
    true_edges[0] = data_range[0]
    true_edges[1:] = data_range[0] + np.cumsum(widths)

    LOG.info("GROUND TRUTH")
    LOG.info(f"True bin edges:")
    for i in range(0, len(true_edges), 10):
        LOG.info("  " + ", ".join(f"{edge:.6f}" for edge in true_edges[i:i + 10]))
    LOG.info(f"True bin widths - min: {widths.min():.6f}, max: {widths.max():.6f}, mean: {widths.mean():.6f}")

    # Generate random transition probability matrix
    true_prob_matrix = np.random.rand(K, K)

    # Apply sparsity
    mask = np.random.rand(K, K) > sparsity
    true_prob_matrix = true_prob_matrix * mask

    # Ensure each row sums to 1 (each state must transition somewhere)
    # If a row is all zeros, make it uniform or self-transition
    for i in range(K):
        if true_prob_matrix[i, :].sum() == 0:
            true_prob_matrix[i, i] = 1.0  # Self-transition if isolated
        else:
            true_prob_matrix[i, :] = true_prob_matrix[i, :] / true_prob_matrix[i, :].sum()

    LOG.info(f"True transition matrix sparsity: {(true_prob_matrix > 0).sum() / true_prob_matrix.size:.2f}% non-zero")

    # Generate sequences
    def generate_sequence(n):
        seq = np.zeros(n)
        # Start in a random bin
        current_bin = np.random.randint(K)

        for t in range(n):
            # Sample uniformly within current bin
            seq[t] = np.random.uniform(true_edges[current_bin], true_edges[current_bin + 1])

            # Transition to next bin according to transition probabilities
            current_bin = np.random.choice(K, p=true_prob_matrix[current_bin, :])

        return seq

    sequences = [generate_sequence(seq_length) for _ in range(n_sequences)]

    LOG.info(f"Generated {len(sequences)} sequences")
    LOG.info(f"Total timepoints: {sum(len(s) for s in sequences)}")
    LOG.info(f"Sequence lengths - min: {min(len(s) for s in sequences)}, "
             f"max: {max(len(s) for s in sequences)}, "
             f"mean: {np.mean([len(s) for s in sequences]):.1f}\n")

    return sequences, true_edges, true_prob_matrix


def test_incremental_updates(sequences=None, K=10):
    """
    Sanity check: verify incremental updates match full recomputation.
    Also verify vectorized versions match original loop-based versions.
    """
    LOG.info("SANITY CHECK: Testing incremental vs full computation")

    # Generate small test data
    if sequences is None:
        np.random.seed(123)
        sequences = [np.random.randn(100) * 5 for _ in range(5)]

    # Create initial edges
    all_data = np.concatenate(sequences)
    boundary_edges = (all_data.min() - 0.1, all_data.max() + 0.1)
    edges = np.linspace(boundary_edges[0], boundary_edges[1], K + 1)

    # Compute initial bins and counts
    current_bins = assign_bins(sequences, edges)

    # Test 1: compare compute_transition_matrix versions
    LOG.info("[Test 1] Comparing compute_transition_matrix versions...")
    counts_old, prob_old = compute_transition_matrix0(current_bins, K)
    counts_new, prob_new = compute_transition_matrix(current_bins, K)
    counts_match = np.allclose(counts_old, counts_new)
    prob_match = np.allclose(prob_old, prob_new)
    LOG.info(f"  Counts match: {counts_match}")
    LOG.info(f"  Probabilities match: {prob_match}")
    if not (counts_match and prob_match):
        LOG.info("  ✗ FAILED!")
        return False

    counts = counts_new

    # Test 2: compare compute_log_likelihood versions
    LOG.info("[Test 2] Comparing compute_log_likelihood versions...")
    lik_old = compute_log_likelihood0(sequences, edges, prob_new)
    lik_new = compute_log_likelihood(sequences, edges, prob_new)
    lik_match = np.isclose(lik_old, lik_new)
    LOG.info(f"  Log-likelihood (old): {lik_old:.6f}")
    LOG.info(f"  Log-likelihood (new): {lik_new:.6f}")
    LOG.info(f"  Match: {lik_match}")
    if not lik_match:
        LOG.info("  ✗ FAILED!")
        return False

    # Test 3: compare _compute_log_likelihood_from_counts versions
    LOG.info("[Test 3] Comparing _compute_log_likelihood_from_counts versions...")
    widths = np.diff(edges)
    lik_from_counts_old = _compute_log_likelihood_from_counts0(counts, widths, K)
    lik_from_counts_new = _compute_log_likelihood_from_counts(counts, widths, K)
    lik_from_counts_match = np.isclose(lik_from_counts_old, lik_from_counts_new)
    LOG.info(f"  Log-likelihood from counts (old): {lik_from_counts_old:.6f}")
    LOG.info(f"  Log-likelihood from counts (new): {lik_from_counts_new:.6f}")
    LOG.info(f"  Match: {lik_from_counts_match}")
    if not lik_from_counts_match:
        LOG.info("  ✗ FAILED!")
        return False

    # Test 4: incremental updates for edge k
    k = 5
    new_edge_k = edges[k] + 0.5

    LOG.info(f"[Test 4] Testing edge {k} move from {edges[k]:.3f} to {new_edge_k:.3f}")

    # Method 1: Incremental update
    new_bins_incr, new_counts_incr = _update_bins_and_counts_for_edge_k(
        sequences, current_bins, counts, edges, k, new_edge_k, K
    )

    # Method 2: Full recomputation
    edges_full = edges.copy()
    edges_full[k] = new_edge_k
    new_bins_full = assign_bins(sequences, edges_full)
    new_counts_full, _ = compute_transition_matrix(new_bins_full, K)

    # Compare bins
    bins_match = all(np.array_equal(b1, b2) for b1, b2 in zip(new_bins_incr, new_bins_full))
    LOG.info(f"  Bins match: {bins_match}")

    # Compare counts
    counts_match = np.allclose(new_counts_incr, new_counts_full)
    LOG.info(f"  Counts match: {counts_match}")
    LOG.info(f"  Max count difference: {np.abs(new_counts_incr - new_counts_full).max()}")

    # Compare likelihoods
    widths_incr = np.diff(edges_full)
    lik_incr = _compute_log_likelihood_from_counts(new_counts_incr, widths_incr, K)
    lik_full = _compute_log_likelihood_from_counts(new_counts_full, widths_incr, K)

    LOG.info(f"  Likelihood (incremental): {lik_incr:.6f}")
    LOG.info(f"  Likelihood (full):        {lik_full:.6f}")
    LOG.info(f"  Difference:               {abs(lik_incr - lik_full):.2e}")

    if bins_match and counts_match and abs(lik_incr - lik_full) < 1e-10:
        LOG.info("✓ ALL SANITY CHECKS PASSED!")
    else:
        LOG.info("✗ SANITY CHECK FAILED!")
        return False

    return True


def _synthetic_comparison_report(sequences, true_edges, true_prob_matrix,
                                 edges, prob_matrix, log_lik,
                                 edges_true_init, prob_matrix_true_init, log_lik_true_init):
    # Compare with ground truth if synthetic
    LOG.info("COMPARISON WITH GROUND TRUTH")

    # Compute likelihood under true model
    true_log_lik = compute_log_likelihood(sequences, true_edges, true_prob_matrix)

    LOG.info(f"Log-likelihood comparison:")
    LOG.info(f"  True model:              {true_log_lik:.6f}")
    LOG.info(f"  Fitted (uniform init):  {log_lik:.6f}")
    LOG.info(f"  Difference:              {log_lik - true_log_lik:.6f}")
    if log_lik > true_log_lik:
        LOG.info(f"  (Fitted model has HIGHER likelihood)")
    else:
        LOG.info(f"  (Fitted model has LOWER likelihood)")

    LOG.info(f"  Fitted (true init):      {log_lik_true_init:.6f}")
    LOG.info(f"  Difference:              {log_lik_true_init - true_log_lik:.6f}")
    if log_lik_true_init > true_log_lik:
        LOG.info(f"  (Fitted model has HIGHER likelihood)")
    else:
        LOG.info(f"  (Fitted model has LOWER likelihood)")

    # Edge comparison for quantile init
    edge_error = np.abs(edges - true_edges)
    LOG.info(f"Edge errors (quantile init):")
    LOG.info(f"  Mean: {edge_error.mean():.6f}")
    LOG.info(f"  Max: {edge_error.max():.6f}")
    LOG.info(f"  RMS: {np.sqrt((edge_error ** 2).mean()):.6f}")

    # Edge comparison for true init
    edge_error_true = np.abs(edges_true_init - true_edges)
    LOG.info(f"Edge errors (true init):")
    LOG.info(f"  Mean: {edge_error_true.mean():.6f}")
    LOG.info(f"  Max: {edge_error_true.max():.6f}")
    LOG.info(f"  RMS: {np.sqrt((edge_error_true ** 2).mean()):.6f}")

    # Transition matrix comparison for quantile init
    prob_error = np.abs(prob_matrix - true_prob_matrix)
    LOG.info(f"Transition matrix errors (quantile init):")
    LOG.info(f"  Mean: {prob_error.mean():.6f}")
    LOG.info(f"  Max: {prob_error.max():.6f}")
    LOG.info(f"  RMS: {np.sqrt((prob_error ** 2).mean()):.6f}")
    frobenius_norm = np.linalg.norm(prob_matrix - true_prob_matrix, 'fro')
    LOG.info(f"  Frobenius norm: {frobenius_norm:.6f}")

    # Transition matrix comparison for true init
    prob_error_true = np.abs(prob_matrix_true_init - true_prob_matrix)
    LOG.info(f"Transition matrix errors (true init):")
    LOG.info(f"  Mean: {prob_error_true.mean():.6f}")
    LOG.info(f"  Max: {prob_error_true.max():.6f}")
    LOG.info(f"  RMS: {np.sqrt((prob_error_true ** 2).mean()):.6f}")
    frobenius_norm_true = np.linalg.norm(prob_matrix_true_init - true_prob_matrix, 'fro')
    LOG.info(f"  Frobenius norm: {frobenius_norm_true:.6f}")


def print_summary_statistics(edges, prob_matrix):
    # Print summary statistics
    LOG.info("Summary Statistics")
    bin_widths = np.diff(edges)
    LOG.info(f"Bin widths - min: {bin_widths.min():.6f}, max: {bin_widths.max():.6f}, "
             f"mean: {bin_widths.mean():.6f}")

    # Transition matrix sparsity
    nonzero_count = (prob_matrix > 0).sum()
    total_entries = prob_matrix.size
    LOG.info(f"Transition matrix sparsity: {100 * nonzero_count / total_entries:.2f}% non-zero")

    # print bin edges, format for readability
    LOG.info("Optimal bin edges:")
    for i in range(0, len(edges), 10):
        LOG.info("  " + ", ".join(f"{edge:.6f}" for edge in edges[i: i +10]))


def debug_zscored_velocity():

    # maximum optimization iterations
    max_iter = 200
    # optimization method
    method = 'coordinate_descent'  # or 'lbfgsb'

    # number of bins
    K = 50
    # target number of timepoints for subsampling (None = use all data)
    nsamples = None # 1_000_000

    filepath = 'notebooks/zscored_velocity.npz'
    feature_idx = 2
    zstd = 0.03285892
    outlier_thresh = 135 * np.pi / 180 / zstd * np.array([-1 ,1]) # force one bin to cover outliers
    min_width = .1 * np.pi / 180 / zstd # .1 deg in zscores
    feature_data, useoutputmask = load_debug_data(filepath ,feature_idx=feature_idx)
    feature_data[~useoutputmask] = np.nan  # Apply mask
    sequences = split_sequences(feature_data)

    # Subsample sequences if needed
    sequences = subsample_sequences(sequences, nsamples=nsamples)

    edges_fixed, boundary_edges = set_boundary_edges(sequences ,K ,outlier_thresh)

    # Optimize binning
    LOG.info(f"Optimizing bin edges with K={K} using method={method}...")
    edges, prob_matrix, log_lik = optimize_bin_edges(
        sequences,
        K=K,
        boundary_edges=boundary_edges,
        min_width=min_width,
        max_iter=max_iter,
        method=method,
        edges_fixed=edges_fixed
    )

    output_file = f'optimal_binning_results_{feature_idx}_v2.npz'

    if output_file is not None:
        # Save results
        LOG.info("Saving results...")
        np.savez(output_file,
                 edges=edges,
                 prob_matrix=prob_matrix,
                 log_likelihood=log_lik,
                 boundary_edges=boundary_edges)
        LOG.info(f"Saved to: {output_file}")


def debug_synthetic():

    # maximum optimization iterations
    max_iter = 200
    # optimization method
    method = 'coordinate_descent'  # or 'lbfgsb'

    # number of bins
    K = 10

    # minimum bin width
    min_width = 1e-6

    data_range = (-10. ,10.)
    sequences, true_edges, true_prob_matrix = generate_synthetic_data(K=K ,data_range=data_range)
    edges_fixed, boundary_edges = set_boundary_edges(sequences ,K ,boundary_edges=data_range)

    # Run sanity check first
    if not test_incremental_updates(sequences=sequences, K=K):
        LOG.info("ERROR: Sanity check failed! Exiting.")
        sys.exit(1)

        # Optimize binning
    LOG.info(f"Optimizing bin edges with K={K} using method={method}...")
    edges, prob_matrix, log_lik = optimize_bin_edges(
        sequences,
        K=K,
        boundary_edges=boundary_edges,
        min_width=min_width,
        max_iter=max_iter,
        method=method,
        edges_fixed=edges_fixed
    )

    # try with true initialization
    LOG.info(f"Re-optimizing with TRUE EDGES as initialization (method={method})...")
    edges_true_init, prob_matrix_true_init, log_lik_true_init = optimize_bin_edges(
        sequences,
        K=K,
        boundary_edges=boundary_edges,
        max_iter=max_iter,
        initial_edges=true_edges,
        method=method
    )
    _synthetic_comparison_report(sequences ,true_edges ,true_prob_matrix, edges, prob_matrix, log_lik,
                                 edges_true_init, prob_matrix_true_init, log_lik_true_init)

    print_summary_statistics(edges, prob_matrix)


if __name__ == "__main__":
    debug_synthetic()
