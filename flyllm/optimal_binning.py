"""
Optimal binning for 1D Markovian time series.

Given sequences of continuous 1D time series, find bin edges that maximize
the likelihood under a binned Markov model:
  p(x_t | x_{t-1}) = p(b(x_t) | b(x_{t-1})) / width(b(x_t))
"""

import numpy as np
from scipy.optimize import minimize
import sys
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

def load_debug_data(filepath,feature_idx=2):
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
    feature_data = data['zscored_velocity'][:,:,feature_idx]  # (nflies, T, nfeatures)
    useoutputmask = data['useoutputmask'].T  # (nflies,T)

    LOG.info(f"Data shape: {feature_data.shape}")
    LOG.info(f"  nflies={feature_data.shape[0]}")
    LOG.info(f"  T={feature_data.shape[1]}")

    return feature_data, useoutputmask    

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
    LOG.info(f"NaN count: {nan_count} / {total_count} ({100*nan_count/total_count:.2f}%)")

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
        LOG.info("  " + ", ".join(f"{edge:.6f}" for edge in true_edges[i:i+10]))
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

def _synthetic_comparison_report(sequences,true_edges,true_prob_matrix,
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
    LOG.info(f"  RMS: {np.sqrt((edge_error**2).mean()):.6f}")

    # Edge comparison for true init
    edge_error_true = np.abs(edges_true_init - true_edges)
    LOG.info(f"Edge errors (true init):")
    LOG.info(f"  Mean: {edge_error_true.mean():.6f}")
    LOG.info(f"  Max: {edge_error_true.max():.6f}")
    LOG.info(f"  RMS: {np.sqrt((edge_error_true**2).mean()):.6f}")

    # Transition matrix comparison for quantile init
    prob_error = np.abs(prob_matrix - true_prob_matrix)
    LOG.info(f"Transition matrix errors (quantile init):")
    LOG.info(f"  Mean: {prob_error.mean():.6f}")
    LOG.info(f"  Max: {prob_error.max():.6f}")
    LOG.info(f"  RMS: {np.sqrt((prob_error**2).mean()):.6f}")
    frobenius_norm = np.linalg.norm(prob_matrix - true_prob_matrix, 'fro')
    LOG.info(f"  Frobenius norm: {frobenius_norm:.6f}")

    # Transition matrix comparison for true init
    prob_error_true = np.abs(prob_matrix_true_init - true_prob_matrix)
    LOG.info(f"Transition matrix errors (true init):")
    LOG.info(f"  Mean: {prob_error_true.mean():.6f}")
    LOG.info(f"  Max: {prob_error_true.max():.6f}")
    LOG.info(f"  RMS: {np.sqrt((prob_error_true**2).mean()):.6f}")
    frobenius_norm_true = np.linalg.norm(prob_matrix_true_init - true_prob_matrix, 'fro')
    LOG.info(f"  Frobenius norm: {frobenius_norm_true:.6f}")


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
        LOG.info("  " + ", ".join(f"{edge:.6f}" for edge in edges[i:i+10]))
        
        
def debug_zscored_velocity():
    
    # maximum optimization iterations
    max_iter = 200
    # optimization method
    method = 'coordinate_descent'  # or 'lbfgsb'

    # number of bins
    K = 50
    # target number of timepoints for subsampling (None = use all data)
    nsamples = None #1_000_000

    filepath = 'notebooks/zscored_velocity.npz'
    feature_idx = 2
    zstd = 0.03285892
    outlier_thresh = 135*np.pi/180/zstd*np.array([-1,1]) # force one bin to cover outliers
    min_width=.1*np.pi/180/zstd # .1 deg in zscores
    feature_data, useoutputmask = load_debug_data(filepath,feature_idx=feature_idx)
    feature_data[~useoutputmask] = np.nan  # Apply mask
    sequences = split_sequences(feature_data)

    # Subsample sequences if needed
    sequences = subsample_sequences(sequences, nsamples=nsamples)

    edges_fixed, boundary_edges = set_boundary_edges(sequences,K,outlier_thresh)

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

    data_range = (-10.,10.)
    sequences, true_edges, true_prob_matrix = generate_synthetic_data(K=K,data_range=data_range)
    edges_fixed, boundary_edges = set_boundary_edges(sequences,K,boundary_edges=data_range)

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
    _synthetic_comparison_report(sequences,true_edges,true_prob_matrix, edges, prob_matrix, log_lik,
                                    edges_true_init, prob_matrix_true_init, log_lik_true_init)

    print_summary_statistics(edges, prob_matrix)

if __name__ == "__main__":
    debug_synthetic()