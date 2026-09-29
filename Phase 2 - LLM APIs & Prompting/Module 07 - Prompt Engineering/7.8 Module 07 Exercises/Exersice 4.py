import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = "openrouter/free"

def evaluate_models_cot(scores_text: str) -> str:
    """Send scores to the model to rank using CoT."""
    
    system_prompt = """You are an AI evaluation expert.
You will be given a list of LLM evaluation scores across 3 tasks.
Perform the following steps:
1. Think step-by-step to calculate the average score for each model.
2. Rank the models from highest average to lowest.
3. Write a final 2-sentence recommendation on which model to choose.

Provide your output in the following format:
THINKING:
(Your step-by-step calculations)

RANKING:
1. [Model Name] (Average: [Score])
...

RECOMMENDATION:
(2 sentences)"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here are the scores:\n\n{scores_text}"}
            ],
        )
        if not hasattr(response, 'choices') or not response.choices:
            return "ERROR: OpenRouter returned empty response."
        return response.choices[0].message.content or "ERROR: Empty content."
    except Exception as e:
        return f"ERROR: {str(e)}"

def run_cot_demo():
    print("\n" + "=" * 70)
    print("EXERCISE 4 - CHAIN OF THOUGHT EVALUATION")
    print("=" * 70)
    
    test_inputs = [
        """
        Model A: Task1: 80, Task2: 90, Task3: 85
        Model B: Task1: 85, Task2: 85, Task3: 85
        Model C: Task1: 90, Task2: 95, Task3: 92
        """,
        """
        GPT-3.5: Task1: 70, Task2: 75, Task3: 80
        GPT-4: Task1: 95, Task2: 92, Task3: 98
        Claude-3: Task1: 94, Task2: 90, Task3: 96
        """,
        """
        Model X: Task1: 50, Task2: 60, Task3: 55
        Model Y: Task1: 55, Task2: 55, Task3: 65
        Model Z: Task1: 60, Task2: 50, Task3: 60
        """
    ]
    
    for i, scores in enumerate(test_inputs, start=1):
        print(f"\n--- Test Input {i} ---")
        print(scores.strip())
        result = evaluate_models_cot(scores.strip())
        print("\nModel Output:")
        print(result)

if __name__ == "__main__":
    run_cot_demo()
