import os
import asyncio
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

async def call_model(client, model: str, prompt: str):
    start_time = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "model": model,
            "response_text": response.choices[0].message.content[:50] + "...", # truncate long string
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "latency_ms": round(latency_ms, 2)
        }
    except Exception as error:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "model": model,
            "response_text": f"ERROR: {error}",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": round(latency_ms, 2)
        }

async def compare_models_async(client, prompt: str, models: list[str]) -> pd.DataFrame:
    tasks = [call_model(client, model, prompt) for model in models]
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)

def compare_models(prompt: str, models: list[str]) -> pd.DataFrame:
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )
    return asyncio.run(compare_models_async(client, prompt, models))


# Example Exercise 3
if __name__ == "__main__":
    print("Comparing model performance asynchronously...")
    # You can add other models if you have OpenRouter credits
    models = ["openrouter/free"] 
    results = compare_models("Explain what a vector database is.", models)
    
    print("\nExercise 3 result:")
    print(results.to_string(index=False))
