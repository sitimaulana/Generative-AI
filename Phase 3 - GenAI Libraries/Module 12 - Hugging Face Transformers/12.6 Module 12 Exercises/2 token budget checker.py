"""Exercise 2: filter texts that fit a configurable tokenizer budget."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer


@dataclass(frozen=True)
class AcceptedText:
    text: str
    token_count: int


class TokenBudgetChecker:
    def __init__(self, model_name: str, token_limit: int) -> None:
        if token_limit < 1:
            raise ValueError("token_limit harus lebih besar dari 0.")
        self.model_name = model_name
        self.token_limit = token_limit
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def filter_texts(self, texts: list[str]) -> list[AcceptedText]:
        accepted: list[AcceptedText] = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Semua item texts harus berupa string.")
            token_count = len(self.tokenizer.encode(text, add_special_tokens=True))
            if token_count <= self.token_limit:
                accepted.append(AcceptedText(text=text, token_count=token_count))
        return accepted


if __name__ == "__main__":
    checker = TokenBudgetChecker("gpt2", token_limit=32)
    texts = ["Short text.", "This is a longer example " * 20]
    for item in checker.filter_texts(texts):
        print(item)
