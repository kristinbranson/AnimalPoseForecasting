"""Evaluates a trained model on the test set, on the cluster.

    python eval_model.py MODEL_PATH CONFIG_PATH             # submits everything and exits
    python eval_model.py MODEL_PATH CONFIG_PATH --status    # progress, and logs of anything unfinished

Running the first command again skips what has finished, leaves running jobs alone and resubmits the rest.

Other options:
    --dry-run                  print the bsub commands without submitting
    --set key=value            change a parameter in apf/eval_pipeline/params.py:EvalParams (repeatable)
    --resources key=value      queue, cpu_queue, gpu_slots, cpu_slots, walltime (repeatable)
    --keep-features            keep the simulated hidden states after use
    --clean-features           delete the stored train hidden states of this run
    --force                    delete this run's results and start over, needed if parameters or the model changed
"""
import argparse
import dataclasses
import json
import os
import shutil
import sys

from apf.eval_pipeline import lsf
from apf.eval_pipeline.graph import build_graph
from apf.eval_pipeline.params import (DEFAULT_CACHE_ROOT, DEFAULT_RESULTS_ROOT, EvalParams, Resources, RunPaths,
                                      model_fingerprint, read_json, run_name, write_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model_path')
    parser.add_argument('config_path')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--keep-features', action='store_true')
    parser.add_argument('--clean-features', action='store_true')
    parser.add_argument('--set', action='append', default=[], dest='overrides', metavar='KEY=VALUE')
    parser.add_argument('--resources', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--results-root', default=DEFAULT_RESULTS_ROOT)
    parser.add_argument('--cache-root', default=DEFAULT_CACHE_ROOT)
    return parser.parse_args(argv)


def parse_resources(assignments):
    names = {f.name for f in dataclasses.fields(Resources)}
    values = {}
    for assignment in assignments:
        key, sep, raw = assignment.partition('=')
        if not sep or key not in names:
            raise SystemExit(f"Invalid --resources {assignment!r}. Valid keys: {sorted(names)}")
        try:
            values[key] = json.loads(raw)
        except json.JSONDecodeError:
            values[key] = raw
    return Resources(**values)


def clear_run(paths: RunPaths):
    for path in [paths.run_dir, paths.cache_dir]:
        if os.path.isdir(path):
            print(f"Deleting {path}")
            shutil.rmtree(path)


def setup_run(paths, run, force, keep_features):
    """Creates the run, or checks that an existing run has the same settings so results don't get mixed."""
    if os.path.exists(paths.run_file):
        existing = read_json(paths.run_file)
        differences = [key for key in ['configfile', 'modelfile', 'fingerprint', 'params', 'cache_dir']
                       if existing[key] != run[key]]
        if differences and not force:
            raise SystemExit(f"{paths.run_dir} was started with different {differences}. "
                             "Use --force to delete its results and start over, or restore the original settings.")
        if differences:
            clear_run(paths)
    paths.make_dirs()
    write_json(paths.run_file, run)


def main(argv=None):
    args = parse_args(argv)
    model_path = os.path.abspath(args.model_path)
    config_path = os.path.abspath(args.config_path)
    for path in [model_path, config_path]:
        if not os.path.exists(path):
            raise SystemExit(f"File not found: {path}")

    name = run_name(config_path, model_path)
    paths = RunPaths(os.path.join(args.results_root, name), os.path.join(args.cache_root, name))
    params = EvalParams().with_overrides(args.overrides)
    graph = build_graph(params)

    if args.status:
        if not os.path.exists(paths.run_file):
            raise SystemExit(f"No run found at {paths.run_dir}")
        print(f"Results: {paths.run_dir}\nCache:   {paths.cache_dir}\n")
        graph = build_graph(EvalParams.from_dict(read_json(paths.run_file)['params']))
        print('\n'.join(lsf.status_lines(paths, graph, lsf.query_jobs(paths))))
        return 0

    if args.clean_features:
        print(f"Deleting {paths.feature_dir}")
        shutil.rmtree(paths.feature_dir, ignore_errors=True)
        return 0

    run = {
        'configfile': config_path,
        'modelfile': model_path,
        'fingerprint': model_fingerprint(model_path),
        'params': params.to_dict(),
        'cache_dir': paths.cache_dir,
        'keep_features': args.keep_features,
    }
    if args.dry_run and not os.path.exists(paths.run_file):
        print("Dry run: nothing has been created")
        for task in graph:
            print(lsf.bsub_command(paths, task, list(range(1, task.n_jobs + 1)), parse_resources(args.resources),
                                   list(task.deps)))
        return 0
    if not args.dry_run:
        setup_run(paths, run, args.force, args.keep_features)

    submitted = lsf.submit(paths, graph, parse_resources(args.resources), dry_run=args.dry_run)
    if not submitted:
        print("Nothing to submit: everything has finished or is running.")
    else:
        print(f"\n{'Would submit' if args.dry_run else 'Submitted'}: "
              + ', '.join(f"{name} ({len(shards)})" for name, shards in submitted))
    print(f"Results: {paths.run_dir}")
    print(f"Check progress with: python {sys.argv[0]} {args.model_path} {args.config_path} --status")
    return 0


if __name__ == '__main__':
    sys.exit(main())
