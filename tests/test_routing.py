import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from routing import __main__ as cli
from routing.core import (
    KINDS,
    MODELS,
    Reply,
    Unavailable,
    execute,
    finite,
    fit,
    load_tasks,
    normalize,
    select,
    summarize,
)
from routing.evidence import verify_dev, verify_eval, write

TASKS = Path("data/tasks.json")


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tasks = load_tasks(TASKS)
        self.records = [
            dict(id=t["id"], model=m, output=t["answer"] if m == "base" else "wrong", seconds=1.0)
            for t in self.tasks
            if t["split"] == "dev"
            for m in MODELS
        ]

    def test_normalization_and_numbers(self):
        self.assertEqual(normalize(" YES! "), "yes")
        self.assertEqual(normalize("New York."), "new york")
        for value in (-1, float("nan"), float("inf"), True, "1", None):
            self.assertFalse(finite(value))
        self.assertTrue(finite(0))

    def test_data_contract(self):
        mutations = [
            None,
            [],
            [{}],
            [dict(self.tasks[0], answer=1)],
            [dict(self.tasks[0], answer=" ")],
            [dict(self.tasks[0], split="test")],
            [dict(self.tasks[0], kind="new")],
            self.tasks + [self.tasks[0]],
            [self.tasks[0]],
        ]
        overlap = copy.deepcopy(self.tasks)
        overlap[-1]["group"] = overlap[0]["group"]
        mutations.append(overlap)
        duplicate = copy.deepcopy(self.tasks)
        duplicate[-1]["prompt"] = duplicate[0]["prompt"].upper()
        mutations.append(duplicate)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.json"
            for data in mutations:
                with self.subTest(data=str(data)[:60]):
                    write(path, data)
                    with self.assertRaises(ValueError):
                        load_tasks(path)

    def test_fit_uses_dev_only_and_penalizes_cost(self):
        self.assertEqual(fit(self.tasks, self.records), dict.fromkeys(KINDS, "base"))
        slow = [dict(r, seconds=100) if r["model"] == "base" else r for r in self.records]
        self.assertEqual(fit(self.tasks, slow), dict.fromkeys(KINDS, "small"))
        ties = [dict(r, output="wrong") for r in self.records]
        self.assertEqual(fit(self.tasks, ties), dict.fromkeys(KINDS, "small"))
        mutations = [
            [],
            self.records[:-1],
            self.records + [self.records[0]],
            [dict(self.records[0], id="eval-logic-1")] + self.records[1:],
            [dict(self.records[0], seconds=float("nan"))] + self.records[1:],
            [dict(self.records[0], output=None)] + self.records[1:],
        ]
        for rows in mutations:
            with self.assertRaises(ValueError):
                fit(self.tasks, rows)

    def test_select_boundary_and_unknown(self):
        self.assertEqual(select("length", "logic", "x" * 120, {}), "small")
        self.assertEqual(select("length", "logic", "x" * 121, {}), "base")
        self.assertEqual(select("fitted", "unknown", "a", {}), "base")
        self.assertEqual(select("fitted", "logic", "a", {"logic": "small"}), "small")
        with self.assertRaises(ValueError):
            select("fitted", "logic", "", {"logic": "other"})
        with self.assertRaises(ValueError):
            select("other", "logic", "", {})

    def test_fallback_and_fail_closed(self):
        def down(prompt):
            raise Unavailable("test")

        for first in MODELS:
            adapters = {m: (down if m == first else lambda p: Reply("yes", 2)) for m in MODELS}
            row = execute(first, "logic", "p", {}, adapters)
            self.assertEqual(row["selected"], first)
            self.assertNotEqual(row["served"], first)
            self.assertEqual([a["status"] for a in row["attempts"]], ["unavailable", "ok"])
            self.assertGreaterEqual(
                row["total_seconds"],
                row["router_seconds"] + sum(a["seconds"] for a in row["attempts"]),
            )
        row = execute("base", "logic", "", {}, dict.fromkeys(MODELS, down))
        self.assertEqual(
            (row["status"], row["served"], row["output"], row["tokens"]),
            ("unavailable", None, "", 0),
        )
        with self.assertRaises(ValueError):
            execute("small", "", "", {}, {})
        with self.assertRaises(ValueError):
            execute("small", "", "", {}, dict.fromkeys(MODELS, lambda p: Reply("x", -1)))

        def bug(prompt):
            raise RuntimeError("bug")

        with self.assertRaisesRegex(RuntimeError, "bug"):
            execute("small", "", "", {}, dict.fromkeys(MODELS, bug))

    def test_scoring_includes_unavailable(self):
        rows = [
            dict(
                id="dev-logic-1",
                scenario="normal",
                strategy="small",
                status="ok",
                output="yes",
                total_seconds=2.0,
                router_seconds=0.1,
                tokens=1,
                attempts=[dict(seconds=1.0)],
            )
        ]
        result = summarize(rows, self.tasks)["normal/small/all"]
        self.assertEqual(result["correct"], 1)
        rows[0]["status"] = "unavailable"
        self.assertEqual(summarize(rows, self.tasks)["normal/small/all"]["accuracy"], 0)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dev = self.root / "dev"
        self.evaluation = self.root / "eval"
        self.adapters = dict.fromkeys(MODELS, lambda p: Reply("yes", 2))
        self.patch_load = patch("routing.local.load", return_value=(self.adapters, {}))
        self.patch_env = patch.object(cli, "environment", return_value={"test": True})
        self.patch_load.start()
        self.patch_env.start()
        self.addCleanup(self.patch_load.stop)
        self.addCleanup(self.patch_env.stop)
        with patch("builtins.print"):
            self.assertEqual(cli.main(["run", "--split", "dev", "--output", str(self.dev)]), 0)
            self.assertEqual(
                cli.main(
                    [
                        "run",
                        "--split",
                        "eval",
                        "--output",
                        str(self.evaluation),
                        "--dev",
                        str(self.dev),
                    ]
                ),
                0,
            )

    def test_complete_replay(self):
        policy = verify_dev(self.dev, TASKS)
        self.assertEqual(set(policy), set(KINDS))
        result = verify_eval(self.evaluation, self.dev, TASKS)
        self.assertEqual(result["both_down/fitted/all"]["unavailable"], 24)
        self.assertEqual(result["normal/base/all"]["n"], 24)
        self.assertEqual(len(result), 35)

    def test_eval_mutations(self):
        path = self.evaluation / "records.json"
        original = json.loads(path.read_text())
        edits = {
            "selected": "other",
            "served": None,
            "status": "unavailable",
            "output": None,
            "tokens": -1,
            "total_seconds": 0,
            "router_seconds": -1,
            "attempts": [],
        }
        for key, value in edits.items():
            rows = copy.deepcopy(original)
            rows[0][key] = value
            write(path, rows)
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_eval(self.evaluation, self.dev, TASKS)
        for rows in (original[:-1], original + [original[0]]):
            write(path, rows)
            with self.assertRaises(ValueError):
                verify_eval(self.evaluation, self.dev, TASKS)
        rows = copy.deepcopy(original)
        rows[-1]["output"] = "invented"
        write(path, rows)
        with self.assertRaises(ValueError):
            verify_eval(self.evaluation, self.dev, TASKS)
        write(path, original)
        report_path = self.evaluation / "report.json"
        report = json.loads(report_path.read_text())
        for field in ("tasks_sha256", "policy_sha256", "dev_records_sha256", "summary"):
            write(report_path, dict(report, **{field: "changed"}))
            with self.assertRaises(ValueError):
                verify_eval(self.evaluation, self.dev, TASKS)
        write(report_path, report)

    def test_dev_mutations(self):
        path = self.dev / "report.json"
        report = json.loads(path.read_text())
        for field in ("tasks_sha256", "policy", "inference_seconds"):
            write(path, dict(report, **{field: "changed"}))
            with self.assertRaises(ValueError):
                verify_dev(self.dev, TASKS)

    def test_cli_errors_and_verification(self):
        with patch("builtins.print"):
            for split, path in [("dev", self.dev), ("eval", self.evaluation)]:
                self.assertEqual(
                    cli.main(
                        ["verify", "--split", split, "--output", str(path), "--dev", str(self.dev)]
                    ),
                    0,
                )
            self.assertEqual(cli.main(["verify", "--split", "dev", "--output", "missing"]), 2)
            self.assertEqual(cli.main(["run", "--split", "dev", "--output", str(self.dev)]), 2)
            with patch("routing.local.download") as download:
                self.assertEqual(cli.main(["download", "--cache", str(self.root)]), 0)
                download.assert_called_once_with(self.root)
        with self.assertRaises(Unavailable):
            cli.unavailable("p")

    def test_environment(self):
        self.patch_env.stop()
        with (
            patch.object(cli, "platform") as host,
            patch("importlib.metadata.version", return_value="1.0"),
            patch("subprocess.check_output", side_effect=["abc\n", ""]),
        ):
            host.python_version.return_value = "3.11.9"
            host.platform.return_value = "test OS"
            host.machine.return_value = "test machine"
            host.processor.return_value = "test CPU"
            info = cli.environment()
        self.assertEqual(info["git_commit"], "abc")
        self.assertFalse(info["git_dirty"])
        self.assertEqual(info["billed_api_usd"], 0)


if __name__ == "__main__":
    unittest.main()
