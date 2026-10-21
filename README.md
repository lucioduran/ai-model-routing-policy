# AI Model Routing Policy

[![CI](https://github.com/lucioduran/ai-model-routing-policy/actions/workflows/ci.yml/badge.svg)](https://github.com/lucioduran/ai-model-routing-policy/actions/workflows/ci.yml)

Measure when routing between two local models helps, including failed attempts and router cost.

## Problem

A cheaper model can lose quality on specific tasks. A router can add overhead or
fail when its selected provider is unavailable. Aggregate accuracy alone hides
these tradeoffs. This experiment publishes requests, model outputs, decisions,
latencies and failure outcomes so those tradeoffs can be checked.

## Scope

Project 14 of the engineering evidence series. Two pinned local FLAN-T5 models,
16 labeled development tasks, 24 held-out tasks, four task types, four strategies,
and three simulated outage scenarios. No hosted inference or paid services.

## Quick start

Python 3.11 or newer is enough to replay saved evidence without model packages:

```sh
git clone https://github.com/lucioduran/ai-model-routing-policy.git
cd ai-model-routing-policy
python -m routing verify --split dev --output results/dev
python -m routing verify --split eval --output results/eval
```

## Usage

`download` explicitly fetches pinned weights. `run` performs actual inference into
a new directory. `verify` checks recorded evidence without inference. Successful
commands exit 0; invalid inputs, missing files and execution failures exit 2.
A failed run may leave a partial trace and must be retried in a new directory.

```sh
python -m routing download --cache .cache/models
python -m routing run --split dev --output build/dev
python -m routing run --split eval --dev build/dev --output build/eval
python -m routing verify --split eval --dev build/dev --output build/eval
```

In the shared project environment, run every `run` command through the exclusive
wrapper shown below. The standalone commands above are for independent machines.

## Architecture

Validated tasks feed two resident CPU model adapters. Development observations fit
a task-type lookup policy. Evaluation compares always-small, always-base, a fixed
120-character selector, and the fitted policy. A request tries its selected adapter,
then the other adapter only on a typed unavailable error. Both unavailable means
an explicit failure with no answer. Other exceptions propagate.

The fitted objective is exact-match accuracy minus 0.02 times mean inference seconds,
per task type. This latency penalty is fixed before evaluation. Unknown types use
base. Router decisions never receive gold labels. See [design](docs/design.md),
[data provenance](data/README.md), and [model notices](docs/model-notices.md).

## Reproduction

Use Python 3.11.13 on Intel macOS for the recorded dependency environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-local.lock
python -m routing download
python /Users/luchoduran/Sites/open-source-projects/.coordination/benchmark_lock.py \
  .venv/bin/python -m routing run --split dev --output build/dev
python /Users/luchoduran/Sites/open-source-projects/.coordination/benchmark_lock.py \
  .venv/bin/python -m routing run --split eval --dev build/dev --output build/eval
python -m routing verify --split eval --dev build/dev --output build/eval
```

The wrapper belongs to the surrounding project series and is not a portable
repository dependency. On an independent machine, execute the same commands
without the wrapper and run one experiment at a time. PyTorch 2.2.2 availability
varies by Python/platform; the recorded run used Intel macOS, CPU float32, two
threads, greedy decoding, seed 1729, and at most 24 new tokens. Inference only reads
local model files. `requirements-models.txt` pins direct model dependencies;
`requirements-local.lock` records all installed versions for the measured host.

Reports contain model revisions and file hashes, code revision, runtime versions,
loading and warmup durations, training costs and full experiment wall time. Request
latency includes routing, all failed attempts, successful inference and dispatch.
Generated token counts include EOS but exclude the decoder start token. There is
no conversion from CPU seconds to dollars. Billed API cost is zero; energy is unmeasured.

## Results

Results will be recorded after the frozen experiment completes.

## Testing

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m coverage run -m unittest discover -v
python -m coverage report
```

Tests exercise invalid fixtures, split leakage, decision boundaries, cost penalties,
missing or duplicate development observations, fallback ordering, both-provider
failure, propagation of programming errors, timing accounting, scoring, corrupted
reports, CLI errors and a complete mocked experiment. The real local adapter is
excluded from coverage; actual inference is covered by the recorded experiment.
CI replays evidence on Linux, macOS and Windows, with an 85% coverage gate. It does
not download models or claim cross-platform inference equivalence.

## Limitations

- Forty synthetic tasks and one measured run do not establish production savings.
- Held-out templates differ, but task families overlap and pretraining overlap is unknown.
- Caller-supplied task types must be accurate. There is no task-type classifier cost.
- Exact matching can reject verbose but correct text. There was no independent human review.
- Both models remain resident. Per-request latency excludes loading and warmup,
  which are reported separately. Cold-start and memory-pressure deployment costs differ.
- CPU latency is measured, not energy or a bill. Scheduling noise and order effects remain.
- Outages are immediate injected exceptions in owned adapters, not network failures.
- Replay verifies internal consistency, not the authenticity of saved outputs or timings.
- The latency-weighted policy has no hard accuracy constraint or production SLO guarantee.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Preserve evaluation boundaries and include
regression tests. See [SECURITY.md](SECURITY.md) for security scope.

## License

Code, original fixtures, documentation and evidence use the [MIT license](LICENSE).
Downloaded models and dependencies retain their upstream licenses.
