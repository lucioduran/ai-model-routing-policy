"""Evidence validation without neural runtime dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from routing.core import MODELS, STRATEGIES, finite, fit, load_tasks, select, summarize

SCENARIOS = {
    "normal": [],
    "small_down": ["small"],
    "base_down": ["base"],
    "both_down": ["small", "base"],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_dev(folder: Path, tasks_path: Path) -> dict[str, Any]:
    report = read(folder / "report.json")
    tasks = load_tasks(tasks_path)
    if report["tasks_sha256"] != digest(tasks_path) or report["split"] != "dev":
        raise ValueError("development provenance mismatch")
    records = read(folder / "records.json")
    policy = read(folder / "policy.json")
    if fit(tasks, records) != policy or report["policy"] != policy:
        raise ValueError("development policy mismatch")
    if report["inference_seconds"] != sum(r["seconds"] for r in records):
        raise ValueError("development cost mismatch")
    return policy


def verify_eval(folder: Path, dev: Path, tasks_path: Path) -> dict[str, Any]:
    policy = verify_dev(dev, tasks_path)
    tasks = load_tasks(tasks_path)
    evaluation = {t["id"]: t for t in tasks if t["split"] == "eval"}
    report = read(folder / "report.json")
    if (
        report["tasks_sha256"] != digest(tasks_path)
        or report["split"] != "eval"
        or report["policy_sha256"] != digest(dev / "policy.json")
        or report["dev_records_sha256"] != digest(dev / "records.json")
    ):
        raise ValueError("evaluation provenance mismatch")
    rows = read(folder / "records.json")
    expected = {(tid, "normal", strategy) for tid in evaluation for strategy in STRATEGIES}
    expected |= {
        (tid, scenario, "fitted")
        for tid in evaluation
        for scenario in SCENARIOS
        if scenario != "normal"
    }
    seen = set()
    for row in rows:
        key = row["id"], row["scenario"], row["strategy"]
        if key not in expected or key in seen:
            raise ValueError("unexpected or duplicate evaluation row")
        seen.add(key)
        task = evaluation[row["id"]]
        selected = select(row["strategy"], task["kind"], task["prompt"], policy)
        if row["selected"] != selected:
            raise ValueError("route mismatch")
        order = [selected, next(m for m in MODELS if m != selected)]
        down = SCENARIOS[row["scenario"]]
        expected_attempts = []
        served = None
        for model in order:
            status = "unavailable" if model in down else "ok"
            expected_attempts.append((model, status))
            if status == "ok":
                served = model
                break
        attempts = row["attempts"]
        if [(a["model"], a["status"]) for a in attempts] != expected_attempts:
            raise ValueError("fallback mismatch")
        if row["served"] != served or row["status"] != ("ok" if served else "unavailable"):
            raise ValueError("response status mismatch")
        if (
            not isinstance(row["output"], str)
            or type(row["tokens"]) is not int
            or row["tokens"] < 0
        ):
            raise ValueError("invalid output")
        if served is None and (row["output"] != "" or row["tokens"] != 0):
            raise ValueError("unavailable response contains output")
        timings = [row["router_seconds"], row["total_seconds"]] + [a["seconds"] for a in attempts]
        if not all(finite(t) for t in timings):
            raise ValueError("invalid timing")
        if row["total_seconds"] + 1e-9 < row["router_seconds"] + sum(
            a["seconds"] for a in attempts
        ):
            raise ValueError("cost excludes routing or failed attempts")
    if seen != expected:
        raise ValueError("incomplete evaluation matrix")
    summary = summarize(rows, tasks)
    if report["summary"] != summary:
        raise ValueError("summary mismatch")
    return summary
