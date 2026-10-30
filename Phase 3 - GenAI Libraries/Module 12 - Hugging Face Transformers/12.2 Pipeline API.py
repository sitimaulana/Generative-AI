from transformers import pipeline

# Text classification (sentiment analysis)
classifier = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,  # -1 = CPU; 0 = first GPU
)

texts = [
    "This LLM API is incredibly fast and easy to use.",
    "The documentation is confusing and the latency is terrible.",
    "The model works fine for most tasks.",
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(f"[{result['label']} |{result['score']:.3f}]{text[:60]}")


from transformers import pipeline

# Named entity recognition
ner = pipeline(
    task="ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
    device=-1,
)

text = "Anthropic was founded in San Francisco by Dario Amodei and Daniela Amodei."

entities = ner(text)

for ent in entities:
    print(f"{ent['word']!r:25s} →{ent['entity_group']} ({ent['score']:.3f})")


from transformers import pipeline

# Text generation (small local model)
generator = pipeline(
    task="text-generation",
    model="distilgpt2",  # small model - runs on CPU
    device=-1,
)

output = generator(
    "The future of AI agents is",
    max_new_tokens=50,
    num_return_sequences=2,
    temperature=0.8,
    do_sample=True,
    pad_token_id=50256,  # suppress warning for GPT-2
)

for i, seq in enumerate(output):
    print(f"Sequence{i+1}:{seq['generated_text']}")


from transformers import pipeline

# Summarisation
summariser = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    device=-1,
)

article = """
Large language models have fundamentally changed how software is built.

Developers now reach for AI APIs as naturally as they reach for database clients.
The shift from rule-based systems to neural language models has accelerated
since the release of GPT-3 in 2020, and the pace has only increased with
instruction-tuned models like ChatGPT, Claude, and Gemini. These models can
write code, explain concepts, extract information, and reason through complex
problems - tasks that previously required specialised rule-based systems or
significant human effort.
"""

summary = summariser(
    article,
    max_length=60,
    min_length=20,
    do_sample=False,
)

print(summary[0]["summary_text"])