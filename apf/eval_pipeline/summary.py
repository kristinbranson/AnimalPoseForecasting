"""Summary tables over evaluation runs. Used by the report task and summarize_eval.py."""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ── helpers ────────────────────────────────────────────────────────────────────

def wasserstein_from_distribution(dist_dict):
    """
    Compute mean Wasserstein distance and mean KL divergence over all features.

    Each value in dist_dict is (bin_edges, train_counts, sim_counts, kl_div).
    Returns (mean_wasserstein, mean_kl).
    """
    ws_vals = []
    kl_vals = []
    for feat_key, val in dist_dict.items():
        bin_edges, train_counts, sim_counts, kl_div = val
        bin_edges    = np.asarray(bin_edges,    dtype=float)
        train_counts = np.asarray(train_counts, dtype=float)
        sim_counts   = np.asarray(sim_counts,   dtype=float)
        centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        kl_vals.append(float(kl_div))
        # Skip degenerate features where one distribution is all-zero
        if train_counts.sum() == 0 or sim_counts.sum() == 0:
            continue
        ws = wasserstein_distance(
            u_values=centres, v_values=centres,
            u_weights=train_counts, v_weights=sim_counts,
        )
        ws_vals.append(ws)
    mean_ws = np.mean(ws_vals) if ws_vals else np.nan
    mean_kl = np.mean(kl_vals) if kl_vals else np.nan
    return mean_ws, mean_kl


def extract_wasserstein(all_distributions):
    """
    Returns two dicts, each {data_name: {frame_label: mean_value}}:
        ws_result  – mean Wasserstein distance per (data_name, frame_label)
        kl_result  – mean KL divergence per (data_name, frame_label)
    data_name ∈ {'pose', 'velocity'};  frame_label ∈ {'first', 'last', 'all'}
    """
    ws_result = {}
    kl_result = {}
    for data_name, frame_dists in all_distributions.items():
        ws_result[data_name] = {}
        kl_result[data_name] = {}
        for frame_label, dist_dict in frame_dists.items():
            ws, kl = wasserstein_from_distribution(dist_dict)
            ws_result[data_name][frame_label] = ws
            kl_result[data_name][frame_label] = kl
    return ws_result, kl_result


def extract_discrimination(real_sim_discrimination):
    """
    real_sim_discrimination structure:
        'discr_result'  : {input_type: class_results}
            class_results shape: (n_mlp_types, n_frame_labels, 2)  last dim = [accuracy, control_acc]
        'first_dim'     : layer_types   e.g. [[128], [512,256,128]]
        'second_dim'    : frame_labels  e.g. ['first', 'last', 'all']
        'third_dim'     : ['accuracy', 'control accuracy']

    Returns a nested dict:
        {input_type: {mlp_label: {frame_label: accuracy}}}
    """
    discr   = real_sim_discrimination['discr_result']
    mlp_types   = real_sim_discrimination['first_dim']
    frame_labels = real_sim_discrimination['second_dim']

    result = {}
    for input_type, class_results in discr.items():
        result[input_type] = {}
        for j, mlp_layers in enumerate(mlp_types):
            mlp_label = f"mlp_{'_'.join(str(x) for x in mlp_layers)}"
            result[input_type][mlp_label] = {}
            for i, frame_label in enumerate(frame_labels):
                result[input_type][mlp_label][frame_label] = float(class_results[j, i, 0])
    return result


def extract_action_frequency(action_classification_results):
    """
    action_classification_results structure (saved per max_dist_to_prompt):
        'frame_counts'  shape (2, n_actions);  row 0 = real, row 1 = sim
        'bout_counts'   same shape
        'actions'       list of action names
        'datanames'     ['train_new', 'sim_new']

    Returns
        overall_ratio  – scalar:  mean over actions of |sim_freq - real_freq| / real_freq
        per_action     – dict action → ratio
    """
    frame_counts = action_classification_results['frame_counts']  # (2, n_actions)
    actions      = action_classification_results['actions']

    real_counts = frame_counts[0].astype(float)
    sim_counts  = frame_counts[1].astype(float)

    total_real = real_counts.sum()
    total_sim  = sim_counts.sum()

    real_freq = real_counts / total_real if total_real > 0 else real_counts
    sim_freq  = sim_counts  / total_sim  if total_sim  > 0 else sim_counts

    per_action = {}
    ratios = []
    for i, action in enumerate(actions):
        if real_freq[i] > 0:
            ratio = abs(sim_freq[i] - real_freq[i]) / real_freq[i]
        else:
            ratio = np.nan
        per_action[action] = ratio
        if not np.isnan(ratio):
            ratios.append(ratio)

    overall = np.mean(ratios) if ratios else np.nan
    return overall, per_action


# ── convenience display / export helpers ──────────────────────────────────────

def print_summary(summary_df: pd.DataFrame, float_fmt: str = '.4f'):
    """Pretty-print the summary table (transposed so experiments are columns)."""
    with pd.option_context(
        'display.max_columns', None,
        'display.width', 200,
        'display.float_format', f'{{:{float_fmt}}}'.format,
    ):
        print(summary_df.T.to_string())


def to_latex(
    summary_df: pd.DataFrame,
    float_fmt: str = '.4f',
    column_groups: dict | None = None,
    higher_is_better: set | None = None,
    save_path: str | None = None,
) -> str:
    """
    Render summary_df as a LaTeX table string (booktabs style).
    The best value in each column is highlighted in bold.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of summarize_experiments().  Rows = experiments, columns = metrics.
    float_fmt : str
        Python format string for floats.
    column_groups : dict, optional
        Maps group label -> list of column names.  When provided, a \\cmidrule
        multi-header row is added above the column names.
    higher_is_better : set of column names, optional
        Columns where a higher value is better (e.g. F1, BA, accuracy).
        All other numeric columns are treated as lower-is-better.
        Defaults to all columns whose name contains 'F1', 'BA', or 'accuracy'.
    save_path : str, optional
        If given, write the LaTeX string to this file.

    Returns
    -------
    str  – complete LaTeX tabular environment.
    """
    df = summary_df.copy()
    cols = list(df.columns)
    fmt = f'{{:{float_fmt}}}'.format

    # Determine which columns are higher-is-better
    if higher_is_better is None:
        higher_is_better = {
            c for c in cols
            if any(kw in c.lower() for kw in ('f1', 'ba', 'accuracy'))
        }

    # Find the best row index per column
    best_idx = {}
    for c in cols:
        col_vals = df[c]
        numeric = col_vals.dropna()
        if numeric.empty or not np.issubdtype(numeric.dtype, np.number):
            continue
        best_idx[c] = numeric.idxmax() if c in higher_is_better else numeric.idxmin()

    # Build formatted cells, wrapping the best value in \textbf
    def format_cell(col, idx, val):
        if isinstance(val, (float, np.floating)):
            s = fmt(val)
        else:
            s = str(val)
        if col in best_idx and idx == best_idx[col]:
            s = rf'\textbf{{{s}}}'
        return s

    formatted = {
        c: {idx: format_cell(c, idx, val) for idx, val in df[c].items()}
        for c in cols
    }

    col_spec = 'l' + 'r' * len(cols)

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(rf'\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'\toprule')

    # Optional grouped header
    if column_groups is not None:
        group_header_parts = ['']  # empty cell for index column
        cmidrule_parts = []
        col_pos = 2  # 1-based; column 1 is the index
        for group_label, group_cols in column_groups.items():
            n = len(group_cols)
            group_header_parts.append(rf'\multicolumn{{{n}}}{{c}}{{{group_label}}}')
            cmidrule_parts.append(rf'\cmidrule(lr){{{col_pos}-{col_pos + n - 1}}}')
            col_pos += n
        lines.append(' & '.join(group_header_parts) + r' \\')
        lines.append(' '.join(cmidrule_parts))

    # Column name header
    escaped_cols = [c.replace('_', r'\_') for c in cols]
    header = r'\textbf{experiment} & ' + ' & '.join(
        rf'\textbf{{{c}}}' for c in escaped_cols
    ) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    # Data rows
    for exp_name in df.index:
        esc_name = str(exp_name).replace('_', r'\_')
        cells = ' & '.join(formatted[c][exp_name] for c in cols)
        lines.append(f'{esc_name} & {cells} ' + r'\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{Experiment summary}')
    lines.append(r'\label{tab:experiment_summary}')
    lines.append(r'\end{table}')

    latex_str = '\n'.join(lines)

    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(latex_str)
        print(f"LaTeX table written to {save_path}")

    return latex_str

# ── reading a run ─────────────────────────────────────────────────────────────

def summarize_run(results_dir, max_dist_to_prompt=64):
    """Returns (metrics of one run, per-action frequency deviation). Metrics of unfinished analyses are left out."""
    row = {}
    per_action = {}

    path = os.path.join(results_dir, 'all_distributions.pkl')
    if os.path.exists(path):
        ws, kl = extract_wasserstein(load_pickle(path))
        for data_name in ws:
            for frame_label in ws[data_name]:
                row[f'wasserstein_{data_name}_{frame_label}'] = ws[data_name][frame_label]
                row[f'kl_{data_name}_{frame_label}'] = kl[data_name][frame_label]

    path = os.path.join(results_dir, 'probe_metrics.pkl')
    if os.path.exists(path):
        metrics = load_pickle(path)
        for split in ['train', 'test']:
            row[f'{split}_F1_mean'] = float(np.nanmean([m[split]['f1'] for m in metrics.values()]))
            row[f'{split}_BA_mean'] = float(np.nanmean([m[split]['balanced_accuracy'] for m in metrics.values()]))

    path = os.path.join(results_dir, 'real_sim_discrimination.pkl')
    if os.path.exists(path):
        discr = extract_discrimination(load_pickle(path))
        for input_type, mlp_dict in discr.items():
            for mlp_label, frame_dict in mlp_dict.items():
                for frame_label, accuracy in frame_dict.items():
                    row[f'discr_{input_type}_{mlp_label}_{frame_label}'] = accuracy

    path = os.path.join(results_dir, f'action_classification_results_max_dist_{max_dist_to_prompt}.pkl')
    if os.path.exists(path):
        row['action_freq_deviation_mean'], per_action = extract_action_frequency(load_pickle(path))
        row['action_freq_deviation_mean'] = float(row['action_freq_deviation_mean'])

    return row, per_action


def summarize_runs(run_dirs, max_dist_to_prompt=64):
    """Returns a table with one row per run (named by its directory), and the per-action frequency deviations."""
    rows = {}
    per_action = {}
    for run_dir in run_dirs:
        name = os.path.basename(os.path.normpath(run_dir))
        rows[name], per_action[name] = summarize_run(os.path.join(run_dir, 'results'), max_dist_to_prompt)
    summary = pd.DataFrame(rows).T
    summary.index.name = 'experiment'
    breakdown = pd.DataFrame(per_action).T
    breakdown.index.name = 'experiment'
    return summary, breakdown
