"""Compares finished evaluation runs.

    python summarize_eval.py RUN_DIR [RUN_DIR ...] [--latex OUTPUT.tex] [--max-dist 64]

RUN_DIR is the results directory printed by eval_model.py.
"""
import argparse

from apf.eval_pipeline.summary import print_summary, summarize_runs, to_latex


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dirs', nargs='+')
    parser.add_argument('--latex', default=None, help='Also write the summary table to this LaTeX file')
    parser.add_argument('--max-dist', type=int, default=64, help='Frames from the prompt for action frequencies')
    args = parser.parse_args(argv)

    summary, breakdown = summarize_runs(args.run_dirs, args.max_dist)
    print("=== Summary ===")
    print_summary(summary)
    if not breakdown.empty:
        print("\n=== Per-action frequency deviation ===")
        print_summary(breakdown)
    if args.latex is not None:
        to_latex(summary, save_path=args.latex)


if __name__ == '__main__':
    main()
