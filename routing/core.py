"""Validated fixtures, routing, fallback execution and metrics."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

MODELS = ("small", "base")
STRATEGIES = ("small", "base", "length", "fitted")
KINDS = ("sentiment", "extraction", "arithmetic", "logic")
PENALTY = 0.02


def normalize(text: str) -> str:
    return text.strip().lower().rstrip(".!?")


def load_tasks(path: Path) -> list[dict[str, str]]:
    tasks = json.loads(path.read_text())
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a nonempty list")
    ids: set[str] = set()
    prompts: set[str] = set()
    groups: dict[str, str] = {}
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {
            "id",
            "split",
            "kind",
            "group",
            "prompt",
            "answer",
        }:
            raise ValueError("invalid task fields")
        if any(not isinstance(v, str) or not v.strip() for v in task.values()):
            raise ValueError("task values must be nonempty strings")
        if task["split"] not in ("dev", "eval") or task["kind"] not in KINDS:
            raise ValueError("invalid split or kind")
        prompt = " ".join(task["prompt"].lower().split())
        if task["id"] in ids or prompt in prompts:
            raise ValueError("duplicate id or prompt")
        if groups.get(task["group"], task["split"]) != task["split"]:
            raise ValueError("template group overlaps splits")
        ids.add(task["id"])
        prompts.add(prompt)
        groups[task["group"]] = task["split"]
    if {(t["split"], t["kind"]) for t in tasks} != {(s, k) for s in ("dev", "eval") for k in KINDS}:
        raise ValueError("every split must contain every kind")
    return tasks


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def fit(tasks: list[dict[str, str]], records: list[dict[str, Any]]) -> dict[str, str]:
    dev = {t["id"]: t for t in tasks if t["split"] == "dev"}
    expected = {(tid, m) for tid in dev for m in MODELS}
    seen = set()
    utilities: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        key = (row["id"], row["model"])
        if key not in expected or key in seen or not finite(row["seconds"]):
            raise ValueError("invalid development observation")
        if not isinstance(row["output"], str):
            raise ValueError("invalid model output")
        seen.add(key)
        task = dev[row["id"]]
        score = int(normalize(row["output"]) == normalize(task["answer"]))
        utilities[task["kind"], row["model"]].append(score - PENALTY * row["seconds"])
    if seen != expected:
        raise ValueError("incomplete development observations")
    return {
        kind: max(MODELS, key=lambda m: sum(utilities[kind, m]) / len(utilities[kind, m]))
        for kind in KINDS
    }


def select(strategy: str, kind: str, prompt: str, policy: dict[str, str]) -> str:
    if strategy in MODELS:
        return strategy
    if strategy == "length":
        return "small" if len(prompt) <= 120 else "base"
    if strategy == "fitted":
        choice = policy.get(kind, "base")
        if choice not in MODELS:
            raise ValueError("invalid policy model")
        return choice
    raise ValueError("unknown strategy")


class Unavailable(Exception):
    """The owned local adapter is unavailable."""


@dataclass
class Reply:
    text: str
    tokens: int


def execute(
    strategy: str,
    kind: str,
    prompt: str,
    policy: dict[str, str],
    adapters: dict[str, Callable[[str], Reply]],
) -> dict[str, Any]:
    if set(adapters) != set(MODELS):
        raise ValueError("exactly two adapters required")
    start = perf_counter()
    chosen = select(strategy, kind, prompt, policy)
    router_seconds = perf_counter() - start
    attempts = []
    output = ""
    tokens = 0
    status = "unavailable"
    served = None
    for model in (chosen, next(m for m in MODELS if m != chosen)):
        attempt_start = perf_counter()
        try:
            reply = adapters[model](prompt)
        except Unavailable:
            attempts.append(
                dict(model=model, status="unavailable", seconds=perf_counter() - attempt_start)
            )
            continue
        if not isinstance(reply.text, str) or type(reply.tokens) is not int or reply.tokens < 0:
            raise ValueError("invalid adapter reply")
        attempts.append(dict(model=model, status="ok", seconds=perf_counter() - attempt_start))
        output, tokens, served, status = reply.text, reply.tokens, model, "ok"
        break
    return dict(
        selected=chosen,
        served=served,
        output=output,
        tokens=tokens,
        status=status,
        router_seconds=router_seconds,
        attempts=attempts,
        total_seconds=perf_counter() - start,
    )


def summarize(rows: list[dict[str, Any]], tasks: list[dict[str, str]]) -> dict[str, Any]:
    gold = {t["id"]: t for t in tasks}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[f"{row['scenario']}/{row['strategy']}/all"].append(row)
        buckets[f"{row['scenario']}/{row['strategy']}/{gold[row['id']]['kind']}"].append(row)
    result = {}
    for name, items in sorted(buckets.items()):
        correct = sum(
            r["status"] == "ok" and normalize(r["output"]) == normalize(gold[r["id"]]["answer"])
            for r in items
        )
        result[name] = dict(
            n=len(items),
            correct=correct,
            accuracy=correct / len(items),
            unavailable=sum(r["status"] == "unavailable" for r in items),
            total_seconds=sum(r["total_seconds"] for r in items),
            router_seconds=sum(r["router_seconds"] for r in items),
            attempt_seconds=sum(a["seconds"] for r in items for a in r["attempts"]),
            tokens=sum(r["tokens"] for r in items),
        )
    return result
