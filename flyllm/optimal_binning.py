"""
Optimal binning for 1D Markovian time series.

Given sequences of continuous 1D time series, find bin edges that maximize
the likelihood under a binned Markov model:
  p(x_t | x_{t-1}) = p(b(x_t) | b(x_{t-1})) / width(b(x_t))
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

def compute_boundary_edges(X,epsilon=1e-6):
    """
    Compute boundary edges from data based on percentile.

    Parameters:
    -----------
    X : list of ndarrays
        List of sequences
    percentile : float
        Percentile for boundary (default 0.001 for 0.001%)

    Returns:
    --------
    tuple (e_min, e_max)
        Minimum and maximum edge values
    """
    all_data = np.concatenate(X)
    e_min = np.min(all_data)
    e_max = np.max(all_data)

    # offset by a tiny amount
    d = e_max - e_min
    
    return (e_min-d*epsilon, e_max+d*epsilon)


def initialize_edges_quantile(X, K, boundary_edges, edges_fixed=None):
    """
    Initialize K bin edges using quantiles (uniform over data distribution).

    Parameters:
    -----------
    X : list of ndarrays
        List of sequences
    K : int
        Number of bins
    boundary_edges : tuple (e_min, e_max)
        Fixed first and last bin edges

    Returns:
    --------
    edges : ndarray of shape (K+1,)
        Initial bin edges
    """
    
    # Concatenate all sequenceslik:.6f}\n")
    all_data = np.concatenate(X)

    # Clip data to boundary edges
    all_data = np.clip(all_data, boundary_edges[0], boundary_edges[1])

    # Compute K-1 interior edges using quantiles
    LOG.info(f"Computing {K-1} interior edges using quantiles on {len(all_data)} points...")
    
    if edges_fixed is None or np.all(np.isnan(edges_fixed)):
    
        percentiles = np.linspace(0, 100, K + 1)[1:-1]  # Exclude 0 and 100
        interior_edges = np.percentile(all_data, percentiles)

        # Create full edges array
        edges = np.zeros(K + 1)
        edges[0] = boundary_edges[0]
        edges[-1] = boundary_edges[1]
        edges[1:-1] = interior_edges
        
    else:
        # for now, this only works if edges at the ends are fixed
        isfree = np.isnan(edges_fixed)
        idx = np.nonzero(isfree)[0]
        i0 = idx[0]
        i1 = idx[-1]
        assert np.any(isfree[i0:i1]), "Only fixed edges at the beginning and end are supported"        
        nfree = np.count_nonzero(isfree)
        frac0 = np.count_nonzero(all_data <= edges_fixed[i0-1]) / len(all_data)
        frac1 = np.count_nonzero(all_data <= edges_fixed[i1+1]) / len(all_data)
        percentiles = np.linspace(frac0*100, frac1*100, nfree + 2)[1:-1]  # Exclude ends
        interior_edges = np.percentile(all_data, percentiles)
        edges = edges_fixed.copy()
        edges[i0:i1+1] = interior_edges        

    return edges


def initialize_edges_uniform(K, boundary_edges, edges_fixed=None):
    """
    Initialize K bin edges with uniform bin widths.

    Parameters:
    -----------
    K : int
        Number of bins
    boundary_edges : tuple (e_min, e_max)
        Fixed first and last bin edges

    Returns:
    --------
    edges : ndarray of shape (K+1,)
        Initial bin edges with uniform spacing
    """
    LOG.info(f"Creating {K} bins with uniform widths...")
    if edges_fixed is None or np.all(np.isnan(edges_fixed)):
        edges = np.linspace(boundary_edges[0], boundary_edges[1], K + 1)
    else:
        # for now, this only works if edges at the ends are fixed
        isfree = np.isnan(edges_fixed)
        idx = np.nonzero(isfree)[0]
        i0 = idx[0]
        i1 = idx[-1]
        assert np.any(isfree[i0:i1]), "Only fixed edges at the beginning and end are supported"
        nfree = np.count_nonzero(isfree)
        edges = edges_fixed.copy()
        edges[i0:i1+1] = np.linspace(edges_fixed[i0-1], edges_fixed[i1+1], nfree + 2)[1:-1]
    return edges


def initialize_edges_kmeans(X, K, boundary_edges, edges_fixed=None):
    """
    Initialize K bin edges using k-means clustering.lik:.6f}\n")
    return initial_log_lik, binned_sequences, counts, prob_matrix


    Note: This function requires sklearn. 

    Parameters:
    -----------
    X : list of ndarrays
        List of sequences
    K : int
        Number of bins
    boundary_edges : tuple (e_min, e_max)
        Fixed first and last bin edges

    Returns:
    --------
    edges : ndarray of shape (K+1,)
        Initial bin edges
    """
    from sklearn.cluster import KMeans

    assert edges_fixed is None, 'edges_fixed parameter not supported for k-means initialization'

    # Concatenate all sequences
    all_data = np.concatenate(X)

    # Clip data to boundary edges for k-means
    all_data = np.clip(all_data, boundary_edges[0], boundary_edges[1])

    # Run k-means with K-1 clusters (for K-1 interior edges)
    LOG.info(f"Running k-means with {K-1} clusters on {len(all_data)} points...")
    kmeans = KMeans(n_clusters=K-1, random_state=42, n_init=10)
    kmeans.fit(all_data.reshape(-1, 1))

    # Sort cluster centers to get interior edges
    interior_edges = np.sort(kmeans.cluster_centers_.flatten())

    # Create full edges array
    edges = np.zeros(K + 1)
    edges[0] = boundary_edges[0]
    edges[-1] = boundary_edges[1]
    edges[1:-1] = interior_edges

    return edges


def assign_bins(X, edges):
    """
    Assign each data point to a bin.

    Parameters:
    -----------
    X : list of ndarrays
        List of sequences
    edges : ndarray
        Bin edges

    Returns:
    --------
    binned_sequences : list of ndarrays
        List of binned sequences (integer bin indices)
    """
    K = len(edges) - 1
    binned_sequences = []
    for seq in X:
        bins = np.searchsorted(edges, seq, side='right') - 1
        # Clip to [0, K-1]
        bins = np.clip(bins, 0, K - 1)
        binned_sequences.append(bins)
    return binned_sequences


def compute_transition_matrix(binned_sequences, K):
    """
    Compute transition count matrix and probability matrix.

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

    # Vectorized counting of transitions
    for seq in binned_sequences:
        if len(seq) < 2:
            continue
        from_bins = seq[:-1]
        to_bins = seq[1:]
        np.add.at(counts, (from_bins, to_bins), 1)

    # Compute probabilities (vectorized)
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    prob_matrix = counts / row_sums

    return counts, prob_matrix


def compute_log_likelihood(X, edges, prob_matrix):
    """
    Compute log-likelihood of data given edges and transition probabilities.

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
    log_widths = np.log(widths)

    log_likelihood = 0.0

    for seq in X:
        if len(seq) < 2:
            continue

        bins = np.searchsorted(edges, seq, side='right') - 1
        bins = np.clip(bins, 0, K - 1)

        # Vectorized computation of transitions
        from_bins = bins[:-1]
        to_bins = bins[1:]

        # Get transition probabilities
        probs = prob_matrix[from_bins, to_bins]

        # Only include terms where transition probability > 0
        valid = probs > 0
        if valid.any():
            log_likelihood += np.sum(np.log(probs[valid]) - log_widths[to_bins[valid]])

    return log_likelihood


def _initialize_edges(X, K, boundary_edges, initial_edges, init_method, edges_fixed=None):
    """
    Helper function to initialize edges.

    Parameters:
    -----------
    init_method : str
        Initialization method: 'uniform', 'quantile', or 'kmeans'
    """
    if initial_edges is not None:
        LOG.info("Using provided initial edges...")
        return initial_edges.copy()
    else:
        LOG.info(f"Initializing edges with {init_method}...")

        if init_method == 'uniform':
            return initialize_edges_uniform(K, boundary_edges, edges_fixed)
        elif init_method == 'quantile':
            return initialize_edges_quantile(X, K, boundary_edges)
        elif init_method == 'kmeans':
            return initialize_edges_kmeans(X, K, boundary_edges)
        else:
            raise ValueError(f"Unknown init_method: {init_method}. Choose 'uniform', 'quantile', or 'kmeans'")


def _compute_and_report_initial(X, edges, K):
    """Helper function to compute and report initial likelihood."""
    binned_sequences = assign_bins(X, edges)
    counts, prob_matrix = compute_transition_matrix(binned_sequences, K)
    initial_log_lik = compute_log_likelihood(X, edges, prob_matrix)
    LOG.info(f"Initial log-likelihood: {initial_log_lik:.6f}\n")
    return initial_log_lik, binned_sequences, counts, prob_matrix


def _finalize_and_report(X, edges, K, initial_log_lik, iteration_info):
    """Helper function to finalize and report results."""
    binned_sequences = assign_bins(X, edges)
    counts, prob_matrix = compute_transition_matrix(binned_sequences, K)
    final_log_lik = compute_log_likelihood(X, edges, prob_matrix)

    LOG.info("Optimization complete!")
    LOG.info(f"Initial log-likelihood: {initial_log_lik:.6f}")
    LOG.info(f"Final log-likelihood:   {final_log_lik:.6f}")
    LOG.info(f"Improvement:            {final_log_lik - initial_log_lik:.6f}")
    for key, value in iteration_info.items():
        LOG.info(f"{key}: {value}")

    return edges, prob_matrix, final_log_lik


def _optimize_lbfgsb_iteration(X, K, boundary_edges, initial_edges, max_iter, min_width):
    """
    Run L-BFGS-B optimization iteration.

    Returns:
    --------
    final_edges : ndarray
        Optimized edges
    iteration_info : dict
        Information about the optimization
    """

    # Convert initial edges to widths
    initial_widths = np.diff(initial_edges)
    total_width = boundary_edges[1] - boundary_edges[0]
    initial_widths_params = initial_widths[:-1].copy()

    # Objective function: negative log-likelihood
    iteration_count = [0]
    def objective(widths_params):
        iteration_count[0] += 1
        sum_widths = np.sum(widths_params)
        last_width = total_width - sum_widths

        if last_width < min_width or sum_widths >= total_width:
            return 1e10

        widths = np.zeros(K)
        widths[:-1] = widths_params
        widths[-1] = last_width

        edges = np.zeros(K + 1)
        edges[0] = boundary_edges[0]
        edges[1:] = boundary_edges[0] + np.cumsum(widths)

        binned_sequences = assign_bins(X, edges)
        counts, prob_matrix = compute_transition_matrix(binned_sequences, K)
        log_lik = compute_log_likelihood(X, edges, prob_matrix)

        if iteration_count[0] % 10 == 0:
            LOG.info(f"  Iteration {iteration_count[0]:4d}: log-likelihood = {log_lik:.6f}")

        return -log_lik

    bounds = [(min_width, total_width - (K-1)*min_width) for _ in range(K - 1)]

    LOG.info("Running L-BFGS-B optimization...")

    result = minimize(
        objective,
        initial_widths_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-8}
    )

    # Reconstruct final edges
    final_widths_params = result.x
    last_width = total_width - np.sum(final_widths_params)
    final_widths = np.zeros(K)
    final_widths[:-1] = final_widths_params
    final_widths[-1] = last_width

    final_edges = np.zeros(K + 1)
    final_edges[0] = boundary_edges[0]
    final_edges[1:] = boundary_edges[0] + np.cumsum(final_widths)

    iteration_info = {
        "Number of iterations": iteration_count[0],
        "Optimizer message": result.message
    }
    return final_edges, iteration_info


def _compute_log_likelihood_from_counts(counts, widths, K):
    """
    Compute log-likelihood from transition counts and bin widths.

    This is faster than recomputing from raw data when we have counts.
    """
    # Vectorized computation
    row_sums = counts.sum(axis=1, keepdims=True)

    # Create mask for valid transitions (non-zero counts)
    valid_mask = counts > 0

    # Compute probabilities only where needed
    probs = np.where(valid_mask, counts / np.maximum(row_sums, 1), 0)

    # Compute log-likelihood
    # Only include terms where counts > 0
    log_widths = np.log(widths)
    log_probs = np.zeros_like(probs)
    log_probs[valid_mask] = np.log(probs[valid_mask])

    # Broadcast log_widths to match counts shape
    log_widths_matrix = np.broadcast_to(log_widths[np.newaxis, :], counts.shape)

    log_lik = np.sum(counts * (log_probs - log_widths_matrix))

    return log_lik


def _update_bins_and_counts_for_edge_k(X, current_bins, current_counts, current_edges, k, new_edge_k, K):
    """
    Incrementally update bin assignments and transition counts when edge k moves.

    Only points in range [edges[k-1], edges[k+1]] can change bins,
    and they can only move between bins k-1 and k.

    Only transitions involving bins k-1 and k need to be updated.

    Returns updated bins and counts.
    """
    # Copy current state
    new_bins = [b.copy() for b in current_bins]
    new_counts = current_counts.copy()

    # Only check points that could be affected
    lower = current_edges[k-1]
    upper = current_edges[k+1]

    # Update bins for all sequences (vectorized where possible)
    for seq_idx, seq in enumerate(X):
        # Find points in affected range
        affected = (seq >= lower) & (seq <= upper)

        if not affected.any():
            continue

        # Recompute bins for affected points (vectorized)
        # Points < new_edge_k go to bin k-1, points >= new_edge_k go to bin k
        new_bins[seq_idx][affected] = np.where(
            seq[affected] < new_edge_k, k - 1, k
        )

    # Incrementally update only the affected rows/columns of the transition matrix
    # Zero out the rows and columns for bins k-1 and k
    new_counts[k-1, :] = 0
    new_counts[k, :] = 0
    new_counts[:, k-1] = 0
    new_counts[:, k] = 0

    # Recount only transitions involving bins k-1 and k from new bins
    # We process all sequences, but only count transitions to/from k-1 and k
    for seq_bins in new_bins:
        if len(seq_bins) < 2:
            continue

        # Get all transitions
        from_bins = seq_bins[:-1]
        to_bins = seq_bins[1:]

        # Find transitions involving bins k-1 or k
        mask = (from_bins == k-1) | (from_bins == k) | (to_bins == k-1) | (to_bins == k)

        if not mask.any():
            continue

        # Count these transitions using vectorized operations
        from_bins_masked = from_bins[mask]
        to_bins_masked = to_bins[mask]

        # Use np.add.at for efficient counting
        np.add.at(new_counts, (from_bins_masked, to_bins_masked), 1)

    return new_bins, new_counts


def _optimize_coordinate_descent_iteration(X, K, boundary_edges, initial_edges, max_iter, min_width, tol=1e-6, edges_fixed=None):
    """
    Run coordinate descent optimization iteration with incremental updates.

    Returns:
    --------
    final_edges : ndarray
        Optimized edges
    iteration_info : dict
        Information about the optimization
    """
    current_edges = initial_edges.copy()

    # Compute initial binning and counts (do this ONCE)
    current_bins = assign_bins(X, current_edges)
    counts, _ = compute_transition_matrix(current_bins, K)
    widths = np.diff(current_edges)
    current_log_lik = _compute_log_likelihood_from_counts(counts, widths, K)

    LOG.info("Running Coordinate Descent optimization (with incremental updates)...")

    iteration_count = 0
    if edges_fixed is None:
        edge_indices = np.arange(1, K)  # All interior edges
    else:
        edge_indices = np.nonzero(np.isnan(edges_fixed))[0]

    for outer_iter in range(max_iter):
        prev_log_lik = current_log_lik

        # Adaptive edge ordering based on data density
        priorities = np.zeros(len(edge_indices))

        if outer_iter == 0:
            # First iteration: compute from raw data
            for idx, k in enumerate(edge_indices):
                lower = current_edges[k-1]
                upper = current_edges[k+1]
                count = sum(((seq >= lower) & (seq <= upper)).sum() for seq in X)
                priorities[idx] = count
        else:
            # Subsequent iterations: use counts matrix (much faster)
            for idx, k in enumerate(edge_indices):
                # Bins k-1 and k are adjacent to edge k
                # Sum of transitions from these bins estimates data density
                priorities[idx] = counts[k-1, :].sum() + counts[k, :].sum()

        # Optimize edges in order of priority (highest first)
        sorted_indices = np.argsort(-priorities)
        ordered_edge_indices = edge_indices[sorted_indices]

        for k in ordered_edge_indices:
            # Define bounds for this edge
            lower_bound = current_edges[k-1] + min_width
            upper_bound = current_edges[k+1] - min_width

            if lower_bound >= upper_bound:
                continue  # Skip if no room to move

            # Define objective for this single edge using incremental updates
            def objective_single_edge(edge_k_pos):
                # Incrementally update bins and counts
                new_bins, new_counts = _update_bins_and_counts_for_edge_k(
                    X, current_bins, counts, current_edges, k, edge_k_pos, K
                )

                # Compute new widths
                new_edges = current_edges.copy()
                new_edges[k] = edge_k_pos
                new_widths = np.diff(new_edges)

                # Compute likelihood from counts
                log_lik = _compute_log_likelihood_from_counts(new_counts, new_widths, K)

                return -log_lik  # Minimize negative

            # Optimize this edge
            result = minimize_scalar(
                objective_single_edge,
                bounds=(lower_bound, upper_bound),
                method='bounded',
                options={'xatol': min_width}
            )

            # Update edge if improvement
            if result.success:
                new_edge_pos = result.x

                # Apply the update
                new_bins, new_counts = _update_bins_and_counts_for_edge_k(
                    X, current_bins, counts, current_edges, k, new_edge_pos, K
                )

                new_edges = current_edges.copy()
                new_edges[k] = new_edge_pos
                new_widths = np.diff(new_edges)
                new_log_lik = _compute_log_likelihood_from_counts(new_counts, new_widths, K)
                assert not np.isnan(new_log_lik), "New log-likelihood is NaN. Check data and edges."

                if new_log_lik > current_log_lik:
                    current_edges = new_edges
                    current_bins = new_bins
                    counts = new_counts
                    widths = new_widths
                    current_log_lik = new_log_lik

            iteration_count += 1

        # Check convergence
        improvement = current_log_lik - prev_log_lik

        if (outer_iter + 1) % 1 == 0:
            LOG.info(f"  Pass {outer_iter+1:4d}: log-likelihood = {current_log_lik:.6f}, improvement = {improvement:.6f}")
            LOG.info(f"    Edges: " + ", ".join(f"{edge:.6f}" for edge in current_edges))

        if improvement < tol:
            LOG.info(f"  Converged (improvement < {tol})")
            break

    iteration_info = {
        "Number of passes": outer_iter + 1,
        "Total edge updates": iteration_count
    }
    return current_edges, iteration_info


def optimize_bin_edges(X, K, boundary_edges, max_iter=100, min_width=1e-6,
                       initial_edges=None, init_method='uniform', method='coordinate_descent', tol=1e-6,
                       edges_fixed=None, offset_epsilon=1e-6):
    """
    Optimize bin edges to maximize likelihood of Markovian time series.

    Main function that handles initialization, optimization, and reporting.

    Parameters:
    -----------
    X : list of ndarrays
        List of sequences
    K : int
        Number of bins
    boundary_edges : tuple (e_min, e_max)
        Fixed first and last bin edges
    max_iter : int
        Maximum optimization iterations
    min_width : float
        Minimum allowed bin width
    initial_edges : ndarray, optional
        Initial bin edges (if provided, init_method is ignored)
    init_method : str
        Initialization method: 'uniform', 'quantile', or 'kmeans' (default: 'uniform')
    method : str
        Optimization method: 'coordinate_descent' or 'lbfgsb'
    tol : float
        Convergence tolerance (for coordinate_descent)
    edges_fixed : ndarray of floats, shape (K+1)
        Values to fix certain edges to. Use NaN for edges that are free to optimize. If None,
        no edges are fixed. Default: None

    Returns:
    --------
    edges : ndarray
        Optimal bin edges
    prob_matrix : ndarray
        Transition probability matrix
    log_likelihood : float
        Final log-likelihood
    """
    
    # offset by a tiny amount so that everything is in bounds
    if offset_epsilon > 0:
        d = boundary_edges[1] - boundary_edges[0]
        boundary_edges = (boundary_edges[0] - d*offset_epsilon,boundary_edges[1] + d*offset_epsilon)
    
    # Initialize edges
    edges = _initialize_edges(X, K, boundary_edges, initial_edges, init_method, edges_fixed)
    assert np.all(np.diff(edges) > 0), "Initial edges must be strictly increasing"

    # Compute and report initial likelihood
    initial_log_lik, _, _, _ = _compute_and_report_initial(X, edges, K)
    assert not np.isnan(initial_log_lik), "Initial log-likelihood is NaN. Check data and initial edges."

    # Run optimization iteration
    if method == 'lbfgsb':
        final_edges, iteration_info = _optimize_lbfgsb_iteration(
            X, K, boundary_edges, edges, max_iter, min_width
        )
    elif method == 'coordinate_descent':
        final_edges, iteration_info = _optimize_coordinate_descent_iteration(
            X, K, boundary_edges, edges, max_iter, min_width, tol, edges_fixed
        )
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'lbfgsb' or 'coordinate_descent'")

    # Finalize and report
    return _finalize_and_report(X, final_edges, K, initial_log_lik, iteration_info)


def set_boundary_edges(sequences, K, outlier_thresh=None, boundary_edges=None):
    """
    Set boundary edges and fixed outlier edges. Caps the data at boundary edges.
    Parameters:
    -----------
    sequences : list of ndarrays
        List of sequences
    K : int
        Number of bins
    outlier_thresh : tuple or None
        (min, max) thresholds to fix edges for outliers. If None, no outlier edges are fixed. Default: None
    boundary_edges : tuple or None
        (min, max) boundary edges. If None, computed from min,max of data. Default: None 
    Returns:
    --------
    edges_fixed : ndarray
        Array of shape (K+1) with fixed edge values or NaN for free edges
    boundary_edges : tuple
        (min, max) boundary edges used
    """
    if boundary_edges is None:
        boundary_edges = compute_boundary_edges(sequences)

    # Threshold data at boundary edges
    for seq in sequences:
        seq[seq < boundary_edges[0]] = boundary_edges[0]
        seq[seq > boundary_edges[1]] = boundary_edges[1]

    edges_fixed = np.zeros(K+1) + np.nan
    edges_fixed[0] = boundary_edges[0]
    edges_fixed[-1] = boundary_edges[1]
    if outlier_thresh is not None:
        if outlier_thresh[0] > boundary_edges[0]:
            edges_fixed[1] = outlier_thresh[0]
        if outlier_thresh[1] < boundary_edges[1]:
            edges_fixed[-2] = outlier_thresh[1]

    return edges_fixed, boundary_edges


def split_sequences(feature_data):
    """
    Split data into sequences based on NaNs.

    Parameters:
    -----------
    feature_data : ndarray of shape (nflies, T)
        Input feature data with NaNs indicating missing data

    Returns:
    --------
    sequences : list of ndarrays
        List of continuous (non-NaN) sequences
    """

    nan_count = np.count_nonzero(np.isnan(feature_data))
    total_count = feature_data.size
    LOG.info(f"NaN count: {nan_count} / {total_count} ({100 * nan_count / total_count:.2f}%)")

    # Split into sequences for each fly
    sequences = []
    total_timepoints = 0

    LOG.info("Splitting into sequences based on NaNs...")
    for fly_idx in range(feature_data.shape[0]):
        fly_data = feature_data[fly_idx, :]

        # Find continuous non-NaN segments
        is_valid = ~np.isnan(fly_data)

        # Find transitions: valid -> invalid and invalid -> valid
        transitions = np.diff(np.concatenate([[False], is_valid, [False]]).astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]

        # Extract sequences
        for start, end in zip(starts, ends):
            seq = fly_data[start:end]
            if len(seq) > 1:  # Only include sequences with at least 2 points (for transitions)
                sequences.append(seq)
                total_timepoints += len(seq)

    LOG.info(f"Found {len(sequences)} sequences")
    LOG.info(f"Total timepoints: {total_timepoints}")
    LOG.info(f"Sequence lengths - min: {min(len(s) for s in sequences)}, "
             f"max: {max(len(s) for s in sequences)}, "
             f"mean: {np.mean([len(s) for s in sequences]):.1f}\n")

    return sequences


def subsample_sequences(sequences, nsamples=1_000_000, seed=42):
    """
    Subsample sequences to reach approximately target total timepoints.

    Selects complete sequences (in random order) until the cumulative
    number of timepoints reaches or exceeds the target.

    Parameters:
    -----------
    sequences : list of ndarrays
        List of sequences to subsample from
    nsamples : int or None
        Target total number of timepoints (default: 1M)
        If None, returns all sequences without subsampling
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    subsampled_sequences : list of ndarrays
        Selected sequences
    """
    # If no target specified, return all sequences
    if nsamples is None:
        total_available = sum(len(s) for s in sequences)
        LOG.info("No subsampling (nsamples=None)")
        LOG.info(f"Using all {len(sequences)} sequences")
        LOG.info(f"Total timepoints: {total_available:,}")
        return sequences

    np.random.seed(seed)

    # Calculate total available timepoints
    total_available = sum(len(s) for s in sequences)

    LOG.info("Subsampling sequences...")
    LOG.info(f"Total available timepoints: {total_available:,}")
    LOG.info(f"Target timepoints: {nsamples:,}")

    if total_available <= nsamples:
        LOG.info(f"Using all {len(sequences)} sequences (no subsampling needed)")
        return sequences

    # Shuffle sequences
    shuffled_indices = np.random.permutation(len(sequences))

    # Select sequences until we reach target
    selected_sequences = []
    cumulative_timepoints = 0

    for idx in shuffled_indices:
        seq = sequences[idx]
        selected_sequences.append(seq)
        cumulative_timepoints += len(seq)

        if cumulative_timepoints >= nsamples:
            break

    LOG.info(f"Selected {len(selected_sequences)} sequences")
    LOG.info(f"Total timepoints: {cumulative_timepoints:,}")
    LOG.info(f"Sequence lengths - min: {min(len(s) for s in selected_sequences)}, "
             f"max: {max(len(s) for s in selected_sequences)}, "
             f"mean: {np.mean([len(s) for s in selected_sequences]):.1f}")

    return selected_sequences
