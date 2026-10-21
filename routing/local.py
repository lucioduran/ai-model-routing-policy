"""Pinned CPU-only model adapters. Imported only for download or inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

from routing.core import Reply

MODELS = {
    "small": {"id": "google/flan-t5-small", "revision": "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"},
    "base": {"id": "google/flan-t5-base", "revision": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"},
}
FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
]


def download(cache: Path) -> None:
    from huggingface_hub import snapshot_download

    for name, spec in MODELS.items():
        snapshot_download(
            repo_id=spec["id"],
            revision=spec["revision"],
            local_dir=cache / name,
            allow_patterns=FILES,
            max_workers=2,
        )


def load(cache: Path):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1729)
    adapters = {}
    metadata = {}
    for name, spec in MODELS.items():
        folder = cache / name
        hashes = {}
        for path in sorted(folder.glob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        start = perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                folder, local_files_only=True, use_safetensors=True
            )
            .to("cpu")
            .eval()
        )
        load_seconds = perf_counter() - start

        def generate(prompt, tokenizer=tokenizer, model=model):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            if inputs["input_ids"].shape[1] > 384:
                raise ValueError("prompt exceeds 384 tokens")
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=24, do_sample=False, num_beams=1)
            return Reply(
                tokenizer.decode(output[0], skip_special_tokens=True), int(output.shape[1]) - 1
            )

        start = perf_counter()
        generate("Answer yes or no: Is water wet?")
        warmup_seconds = perf_counter() - start
        adapters[name] = generate
        metadata[name] = dict(
            **spec,
            file_sha256=hashes,
            load_seconds=load_seconds,
            warmup_seconds=warmup_seconds,
            parameters=sum(p.numel() for p in model.parameters()),
        )
    return adapters, metadata
