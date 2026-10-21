# Fixture provenance

Forty original synthetic English tasks authored for this repository, licensed MIT.
Sixteen development tasks and twenty-four held-out tasks have disjoint template
groups and prompts. Four balanced task types: sentiment, extraction, arithmetic,
and deductive logic. Labels were specified with the prompts before inference.
There was no independent human review. This tiny, deliberately simple benchmark
is a reproducible routing experiment, not a production capability assessment.

Scoring lowercases, strips surrounding whitespace and terminal punctuation, then
compares with the one gold string. Verbose correct answers may fail exact match.
Do not adjust labels, prompts, thresholds or normalization after seeing evaluation.
