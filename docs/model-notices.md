# Model provenance

Evaluation uses Google's [FLAN-T5-small](https://huggingface.co/google/flan-t5-small)
and [FLAN-T5-base](https://huggingface.co/google/flan-t5-base), whose model cards list
Apache-2.0. Weights are downloaded by explicit command and not redistributed here.
Model IDs, immutable Hub revisions, parameter counts and local file SHA-256 hashes
are saved in both reports. No remote model code is loaded. Inference uses only
local safetensors files, CPU float32, greedy decoding and two threads.

The model cards describe instruction-tuned language models and their limitations.
This repository does not claim frontier performance or general safety. Inputs and
labels here are synthetic. Upstream training data overlap with task patterns is
unknown. Dependency licenses remain with their authors.
