"""Exercise 3: extract unique PERSON, ORG, and LOC entities from a corpus."""

from __future__ import annotations

from collections import defaultdict

from transformers import pipeline


ENTITY_TYPES = ("PERSON", "ORG", "LOC")


def extract_entities_from_corpus(
    texts: list[str], model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
) -> dict[str, list[str]]:
    """Run NER over all texts and return unique values grouped by entity type."""
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("texts harus berisi string yang tidak kosong.")

    ner = pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
        device=-1,
    )
    entities: dict[str, set[str]] = defaultdict(set)
    for result in ner(texts):
        for entity in result:
            entity_type = str(entity.get("entity_group", "")).upper()
            normalized_type = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}.get(
                entity_type, entity_type
            )
            value = str(entity.get("word", "")).strip()
            if normalized_type in ENTITY_TYPES and value:
                entities[normalized_type].add(value)
    return {entity_type: sorted(entities[entity_type]) for entity_type in ENTITY_TYPES}


if __name__ == "__main__":
    corpus = [
        "Anthropic was founded in San Francisco by Dario Amodei.",
        "Microsoft is headquartered in Redmond.",
    ]
    print(extract_entities_from_corpus(corpus))
