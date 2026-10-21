# Experimental contract

Freeze fixtures and all decision rules before running either model. Use 16 development
and 24 held-out English tasks, balanced across sentiment, extraction, arithmetic and
logic. The task type is caller-provided metadata, not inferred using gold answers.
Development and evaluation use disjoint IDs, prompts and template groups. Templates
still share skills, so this small synthetic suite does not estimate generalization to
natural traffic. Model pretraining contamination cannot be excluded.

Fit one model choice per type on development results only: maximize exact-match
accuracy minus 0.02 times mean inference seconds. Ties prefer small. This coefficient
is a predeclared preference, not a monetary price or a guaranteed quality constraint.
Unknown task types conservatively route to base. Baselines always select small,
always select base, or select small when prompt length is at most 120 characters.

Run every strategy against actual CPU adapters, separately, in rotating order per
task. Decode greedily with at most 24 new tokens, no sampling, two CPU threads.
Both models stay resident. Record cold loading and warmup separately. Each request
records decision time, adapter attempts, end-to-end time, generated tokens and output.
Failed attempts are included in request cost. No billed APIs are used. Dollars,
electricity and energy are not inferred from latency. Training and full experimental
costs are reported separately from deployment request totals.

Outage experiments affect only owned in-process adapters. After normal evaluation,
run the fitted policy with small unavailable, base unavailable, and both unavailable.
An unavailable adapter raises a typed error before inference. Try each provider once;
fail closed with an explicit unavailable result if both fail. Programming errors
propagate instead of being mistaken for provider failures. This is fault injection,
not evidence about remote providers, networking, retry delays or real outages.

Saved evidence verification recomputes split integrity, policy fitting, routes,
fallback order, scores, aggregates and input hashes without model dependencies.
It checks timing consistency but cannot authenticate timing measurements or prove
saved text came from neural inference. Full reproduction reruns local models.
