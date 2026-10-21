# Contributing

Use Python 3.11 and focused commits. Run all checks in the README before submitting
changes. Add regression tests for changed behavior. Keep prompts, labels and policy
rules frozen for an evaluation series; create a new versioned experiment to change
them. Report negative results and measurement limits. Do not commit weights or caches.
Heavy inference in the project series must use the shared exclusive benchmark wrapper.
