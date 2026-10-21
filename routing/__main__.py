"""Command line entry point for the local routing experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from routing.core import MODELS, STRATEGIES, Unavailable, execute, fit, load_tasks, summarize
from routing.evidence import SCENARIOS, digest, verify_dev, verify_eval, write


def environment():
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        packages={
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "huggingface-hub", "numpy")
        },
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        utc=datetime.now(UTC).isoformat(),
        threads=2,
        seed=1729,
        billed_api_usd=0,
        energy_measured=False,
    )


def unavailable(prompt):
    raise Unavailable("simulated outage of owned local adapter")


def run(args):
    from routing.local import load

    start = perf_counter()
    tasks = load_tasks(args.tasks)
    policy = verify_dev(args.dev, args.tasks) if args.split == "eval" else {}
    args.output.mkdir(parents=True, exist_ok=False)
    adapters, models = load(args.cache)
    metadata = environment()
    rows = []
    split_tasks = [t for t in tasks if t["split"] == args.split]
    with (args.output / "trace.jsonl").open("w") as trace:

        def save(row):
            rows.append(row)
            trace.write(json.dumps(row, allow_nan=False) + "\n")
            trace.flush()
            print(f"{args.split}: {len(rows)} observations", flush=True)

        if args.split == "dev":
            for i, task in enumerate(split_tasks):
                for model in MODELS if i % 2 == 0 else MODELS[::-1]:
                    tick = perf_counter()
                    reply = adapters[model](task["prompt"])
                    save(
                        dict(
                            id=task["id"],
                            model=model,
                            output=reply.text,
                            tokens=reply.tokens,
                            seconds=perf_counter() - tick,
                        )
                    )
            tick = perf_counter()
            policy = fit(tasks, rows)
            fit_seconds = perf_counter() - tick
            write(args.output / "policy.json", policy)
            extra = dict(
                policy=policy,
                fit_seconds=fit_seconds,
                inference_seconds=sum(r["seconds"] for r in rows),
            )
        else:
            for scenario, down in SCENARIOS.items():
                current = {m: unavailable if m in down else adapters[m] for m in MODELS}
                for i, task in enumerate(split_tasks):
                    strategies = STRATEGIES[i % 4 :] + STRATEGIES[: i % 4]
                    for strategy in strategies if scenario == "normal" else ("fitted",):
                        row = execute(strategy, task["kind"], task["prompt"], policy, current)
                        save(dict(id=task["id"], scenario=scenario, strategy=strategy, **row))
            extra = dict(
                policy_sha256=digest(args.dev / "policy.json"),
                dev_records_sha256=digest(args.dev / "records.json"),
                summary=summarize(rows, tasks),
            )
    write(args.output / "records.json", rows)
    write(
        args.output / "report.json",
        dict(
            schema_version=1,
            split=args.split,
            tasks_sha256=digest(args.tasks),
            environment=metadata,
            models=models,
            experiment_seconds=perf_counter() - start,
            **extra,
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--cache", type=Path, default=Path(".cache/models"))
    for name in ("run", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--split", choices=("dev", "eval"), required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--tasks", type=Path, default=Path("data/tasks.json"))
        command.add_argument("--dev", type=Path, default=Path("results/dev"))
        if name == "run":
            command.add_argument("--cache", type=Path, default=Path(".cache/models"))
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            from routing.local import download as fetch

            fetch(args.cache)
        elif args.command == "run":
            run(args)
        elif args.split == "dev":
            verify_dev(args.output, args.tasks)
        else:
            verify_eval(args.output, args.dev, args.tasks)
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "command": args.command}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
