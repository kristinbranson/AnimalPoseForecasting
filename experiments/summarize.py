import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_resultsdir(configfile, modelfile):
    configname = os.path.split(configfile)[-1].replace('.json', '')
    modelname = os.path.split(modelfile)[-1].replace('.pth', '')
    base = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/results"
    return os.path.join(base, f"{configname}_{modelname}")


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


def extract_action_classifier_metrics(action_stats, use_layer_id=5):
    """
    Returns a dict keyed by action with (train_f1, val_f1, train_ba, val_ba).

    Stats structure (from train_binary_classifier in ml.py):
        'F1s'        – np.ndarray shape (n_epochs, 2);  [:, 0] = train F1,  [:, 1] = val F1
        'accuracies' – np.ndarray shape (n_epochs, 2);  [:, 0] = train BA,  [:, 1] = val BA
    We take the values at the last epoch (index -1).
    """
    metrics = {}
    for action, layer_dict in action_stats.items():
        if use_layer_id not in layer_dict:
            continue
        stats = layer_dict[use_layer_id]

        F1s         = stats.get('F1s')
        accuracies  = stats.get('accuracies')

        train_f1 = float(F1s[-1, 0])        if F1s        is not None else np.nan
        val_f1   = float(F1s[-1, 1])        if F1s        is not None else np.nan
        train_ba = float(accuracies[-1, 0]) if accuracies is not None else np.nan
        val_ba   = float(accuracies[-1, 1]) if accuracies is not None else np.nan

        metrics[action] = dict(train_f1=train_f1, val_f1=val_f1,
                               train_ba=train_ba, val_ba=val_ba)
    return metrics


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


# ── main summarisation function ────────────────────────────────────────────────

def summarize_experiments(
    experiments: dict,
    use_layer_id: int = 5,
    max_dist_to_prompt: int = 64,
    return_action_breakdown: bool = True,
):
    """
    Parameters
    ----------
    experiments : dict
        Keys are experiment nicknames; values are [configfile, modelfile].
    use_layer_id : int
        Which transformer layer's classifier stats to use (default 5, matching evaluate_model).
    max_dist_to_prompt : int
        Which action_classification_results file to use (64 or 512).
    return_action_breakdown : bool
        If True also return a second DataFrame with per-action frequency ratios.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per experiment, columns described below.
    action_breakdown_df : pd.DataFrame  (only if return_action_breakdown=True)
        Rows = experiments, columns = actions, values = per-action frequency ratio.
    """

    rows = []
    action_breakdown = {}

    for nickname, (configfile, modelfile) in experiments.items():
        resultsdir = get_resultsdir(configfile, modelfile)
        row = {'experiment': nickname}

        # ── 1. Wasserstein distances ───────────────────────────────────────────
        dist_path = os.path.join(resultsdir, 'all_distributions.pkl')
        if os.path.exists(dist_path):
            all_distributions = load_pickle(dist_path)
            ws, kl = extract_wasserstein(all_distributions)

            for data_name in ['pose', 'velocity']:
                for frame_label in ['first', 'last', 'all']:
                    row[f'wasserstein_{data_name}_{frame_label}'] = ws.get(data_name, {}).get(frame_label, np.nan)
                    row[f'kl_{data_name}_{frame_label}']          = kl.get(data_name, {}).get(frame_label, np.nan)
        else:
            print(f"[{nickname}] WARNING: all_distributions.pkl not found at {dist_path}")
            for data_name in ['pose', 'velocity']:
                for frame_label in ['first', 'last', 'all']:
                    row[f'wasserstein_{data_name}_{frame_label}'] = np.nan
                    row[f'kl_{data_name}_{frame_label}']          = np.nan

        # ── 2. Action classifier metrics (F1, BA) ──────────────────────────────
        stats_path = os.path.join(resultsdir, 'action_stats.pkl')
        if os.path.exists(stats_path):
            action_stats = load_pickle(stats_path)
            action_metrics = extract_action_classifier_metrics(action_stats, use_layer_id)

            if action_metrics:
                row['train_F1_mean']  = np.nanmean([m['train_f1'] for m in action_metrics.values()])
                row['val_F1_mean']    = np.nanmean([m['val_f1']   for m in action_metrics.values()])
                row['train_BA_mean']  = np.nanmean([m['train_ba'] for m in action_metrics.values()])
                row['val_BA_mean']    = np.nanmean([m['val_ba']   for m in action_metrics.values()])
            else:
                for k in ['train_F1_mean', 'val_F1_mean', 'train_BA_mean', 'val_BA_mean']:
                    row[k] = np.nan
        else:
            print(f"[{nickname}] WARNING: action_stats.pkl not found at {stats_path}")
            for k in ['train_F1_mean', 'val_F1_mean', 'train_BA_mean', 'val_BA_mean']:
                row[k] = np.nan

        # ── 3. Real-vs-sim discrimination ──────────────────────────────────────
        discr_path = os.path.join(resultsdir, 'real_sim_discrimination.pkl')
        if os.path.exists(discr_path):
            rsd = load_pickle(discr_path)
            discr = extract_discrimination(rsd)
            for input_type, mlp_dict in discr.items():
                for mlp_label, frame_dict in mlp_dict.items():
                    for frame_label, acc in frame_dict.items():
                        col = f'discr_{input_type}_{mlp_label}_{frame_label}'
                        row[col] = acc
        else:
            print(f"[{nickname}] WARNING: real_sim_discrimination.pkl not found at {discr_path}")

        # ── 4. Action frequency match ──────────────────────────────────────────
        freq_path = os.path.join(
            resultsdir,
            f'action_classification_results_max_dist_{max_dist_to_prompt}.pkl'
        )
        if os.path.exists(freq_path):
            acr = load_pickle(freq_path)
            overall_ratio, per_action = extract_action_frequency(acr)
            row['action_freq_deviation_mean'] = overall_ratio
            action_breakdown[nickname] = per_action
        else:
            print(f"[{nickname}] WARNING: action_classification_results not found at {freq_path}")
            row['action_freq_deviation_mean'] = np.nan

        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index('experiment')

    if return_action_breakdown and action_breakdown:
        action_breakdown_df = pd.DataFrame(action_breakdown).T
        action_breakdown_df.index.name = 'experiment'
        return summary_df, action_breakdown_df

    return summary_df


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


# ── example usage ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    experiments = {
        'ref': [
            "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_optimalbinning_20251113.json",
            "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_20251114T194024_bestepoch200.pth",
        ],
        'short': [
            "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_optimalbinning_short_20260430.json",
            "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/flyllm_models/flypredvel_optimal_binning_64_20260502T050431_epoch235.pth",
        ],
    }

    summary, breakdown = summarize_experiments(experiments)
    print("=== Summary table ===")
    print_summary(summary)
    print("\n=== Per-action frequency deviation ===")
    print_summary(breakdown)

    # --- LaTeX output ---
    # Optionally define column groups for a multi-level header
    column_groups = {
        r'Wasserstein -- pose':      [c for c in summary.columns if c.startswith('wasserstein_pose')],
        r'KL -- pose':               [c for c in summary.columns if c.startswith('kl_pose')],
        r'Wasserstein -- velocity':  [c for c in summary.columns if c.startswith('wasserstein_velocity')],
        r'KL -- velocity':           [c for c in summary.columns if c.startswith('kl_velocity')],
        r'Action classif.\\ (F1)': ['train_F1_mean', 'val_F1_mean'],
        r'Action classif.\\ (BA)': ['train_BA_mean', 'val_BA_mean'],
        r'Real-sim discrim.':        [c for c in summary.columns if c.startswith('discr_')],
        r'Action freq.':             ['action_freq_deviation_mean'],
    }
    # Remove groups whose columns are absent
    column_groups = {k: v for k, v in column_groups.items() if v}

    latex = to_latex(summary, column_groups=column_groups, save_path='experiment_summary.tex')
    print("\n=== LaTeX ===")
    print(latex)

    latex_bd = to_latex(breakdown, save_path='action_breakdown.tex')
    print("\n=== Action breakdown LaTeX ===")
    print(latex_bd)