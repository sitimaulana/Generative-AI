"""Exercise 4: LoRA fine-tuning on 20 RAG Q&A pairs and inference verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


QA_PAIRS = [
    ("What does RAG stand for?", "RAG stands for Retrieval-Augmented Generation."),
    ("What does RAG retrieve?", "RAG retrieves relevant external context."),
    ("When does RAG retrieve knowledge?", "RAG retrieves knowledge at inference time."),
    ("What do embeddings represent?", "Embeddings represent text as numerical vectors."),
    ("What is a vector database?", "A vector database stores embeddings for similarity search."),
    ("What is semantic search?", "Semantic search finds text by meaning rather than exact words."),
    ("What is chunking?", "Chunking splits a long document into smaller passages."),
    ("Why use chunking?", "Chunking makes relevant passages easier to retrieve."),
    ("What is a retriever?", "A retriever selects relevant documents for a query."),
    ("What is a generator in RAG?", "The generator writes an answer using retrieved context."),
    ("What is groundedness?", "Groundedness measures whether an answer is supported by context."),
    ("What is hallucination?", "A hallucination is an unsupported or fabricated answer."),
    ("What is fine-tuning?", "Fine-tuning trains a pretrained model on task-specific examples."),
    ("Does RAG change model weights?", "RAG does not change model weights."),
    ("What does an index organize?", "An index organizes data for efficient retrieval."),
    ("What is cosine similarity?", "Cosine similarity compares the direction of vectors."),
    ("Why cite sources?", "Citations show which passages support an answer."),
    ("What is a prompt?", "A prompt is input provided to a language model."),
    ("What does LoRA train?", "LoRA trains a small set of low-rank adapter parameters."),
    ("What is a QA pair?", "A QA pair contains one question and its answer."),
]


def build_training_dataset(tokenizer: Any) -> Dataset:
    texts = [f"Question: {question}\nAnswer: {answer}" for question, answer in QA_PAIRS]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        encoded["labels"] = [
            [
                token_id if mask == 1 else -100
                for token_id, mask in zip(input_ids, attention_mask)
            ]
            for input_ids, attention_mask in zip(
                encoded["input_ids"], encoded["attention_mask"]
            )
        ]
        return encoded

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def fine_tune_lora(
    model_name: str = "distilgpt2",
    output_dir: str | Path = "lora_rag_qa",
    epochs: int = 3,
) -> tuple[Any, Any]:
    """Train and save a LoRA adapter, returning the model and tokenizer."""
    if epochs < 1:
        raise ValueError("epochs harus lebih besar dari 0.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=build_training_dataset(tokenizer),
    )
    trainer.train()
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return peft_model, tokenizer


def verify_answers(
    adapter_dir: str | Path, model_name: str = "distilgpt2"
) -> list[dict[str, str | bool]]:
    """Load the adapter and verify that generated answers contain expected facts."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    results = []
    for question, expected in QA_PAIRS:
        inputs = tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(
            {"question": question, "expected": expected, "answer": answer, "verified": expected in answer}
        )
    return results


if __name__ == "__main__":
    adapter_path = Path("lora_rag_qa")
    fine_tune_lora(output_dir=adapter_path)
    results = verify_answers(adapter_path)
    for item in results:
        status = "PASS" if item["verified"] else "FAIL"
        print(f"[{status}] {item['question']}\n  {item['answer']}")
    print(f"Verified {sum(item['verified'] for item in results)}/{len(results)} answers.")
