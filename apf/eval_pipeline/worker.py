"""Runs one element of one task. This is what the LSF jobs call."""
import argparse
import sys

from apf.eval_pipeline.context import RunContext
from apf.eval_pipeline.graph import build_graph, mark_done, missing_dependencies


def task_functions():
    from apf.eval_pipeline import analyses, probes, simulate
    return {
        'prepare': probes.run_prepare,
        'simulate': simulate.run_simulate,
        'train_features': probes.run_train_features,
        'stitch': simulate.run_stitch,
        'probes': probes.run_probes,
        'logits': probes.run_logits,
        'merge_logits': probes.run_merge_logits,
        'probe_metrics': analyses.run_probe_metrics,
        'action_stats': analyses.run_action_stats,
        'action_videos': analyses.run_action_videos,
        'distributions': analyses.run_distributions,
        'discrimination': analyses.run_discrimination,
        'sim_videos': analyses.run_sim_videos,
        'report': analyses.run_report,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('run_dir')
    parser.add_argument('--shard', type=int, default=1)
    args = parser.parse_args(argv)

    ctx = RunContext(args.run_dir)
    graph = build_graph(ctx.params)
    task = next(t for t in graph if t.name == args.task)
    missing = missing_dependencies(ctx.paths, task, graph)
    # The report still summarizes whatever finished, so that a partial failure leaves something to look at
    if missing and task.name != 'report':
        print(f"Not running {task.name}: these tasks did not finish: {missing}. "
              f"Check them with eval_model.py --status, then run eval_model.py again.", file=sys.stderr)
        return 1

    print(f"[{task.name} {args.shard}/{task.n_jobs}] {ctx.modelfile}", flush=True)
    task_functions()[task.name](ctx, args.shard)
    if missing:
        print(f"Report is partial, these tasks did not finish: {missing}", file=sys.stderr)
        return 1
    mark_done(ctx.paths, task.name, args.shard)
    print(f"[{task.name} {args.shard}/{task.n_jobs}] done", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
