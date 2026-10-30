"""Exercise 1: compare two Hugging Face text-generation pipelines."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from transformers import pipeline


PROMPTS = [
    "Explain retrieval augmented generation in one sentence.",
    "What is a vector database?",
    "Define machine learning.",
    "Why are embeddings useful?",
    "What does fine tuning change?",
    "Explain a Python list.",
    "What is a tokenizer?",
    "Name one benefit of local models.",
    "What is an API?",
    "Explain semantic search.",
]


@dataclass(frozen=True)
class BenchmarkRow:
    model_name: str
    prompt_index: int
    latency_ms: float
    output_length: int


class ModelBenchmark:
    def __init__(
        self,
        model_names: tuple[str, str] = ("distilgpt2", "sshleifer/tiny-gpt2"),
        device: int = -1,
    ) -> None:
        if len(model_names) != 2 or model_names[0] == model_names[1]:
            raise ValueError("model_names harus berisi tepat dua model yang berbeda.")
        self.model_names = model_names
        self.device = device

    def run(self, prompts: list[str] | None = None) -> list[BenchmarkRow]:
        inputs = PROMPTS if prompts is None else prompts
        if len(inputs) != 10 or any(not prompt.strip() for prompt in inputs):
            raise ValueError("prompts harus berisi tepat 10 teks yang tidak kosong.")

        rows: list[BenchmarkRow] = []
        for model_name in self.model_names:
            generator = pipeline(
                "text-generation", model=model_name, device=self.device
            )
            for prompt_index, prompt in enumerate(inputs, start=1):
                started = time.perf_counter()
                result: list[dict[str, Any]] = generator(
                    prompt,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=generator.tokenizer.eos_token_id,
                )
                latency_ms = (time.perf_counter() - started) * 1000
                generated = result[0]["generated_text"]
                output = (
                    generated[len(prompt) :]
                    if generated.startswith(prompt)
                    else generated
                )
                rows.append(
                    BenchmarkRow(
                        model_name=model_name,
                        prompt_index=prompt_index,
                        latency_ms=latency_ms,
                        output_length=len(output),
                    )
                )
        return rows

    @staticmethod
    def print_table(rows: list[BenchmarkRow]) -> None:
        print(f"{'Model':30} {'Prompt':>6} {'Latency ms':>12} {'Output chars':>13}")
        print("-" * 67)
        for row in rows:
            print(
                f"{row.model_name:30} {row.prompt_index:>6} "
                f"{row.latency_ms:>12.2f} {row.output_length:>13}"
            )


if __name__ == "__main__":
    names = (
        os.environ.get("HF_BENCHMARK_MODEL_A", "distilgpt2"),
        os.environ.get("HF_BENCHMARK_MODEL_B", "sshleifer/tiny-gpt2"),
    )
    benchmark = ModelBenchmark(names)
    benchmark.print_table(benchmark.run())
