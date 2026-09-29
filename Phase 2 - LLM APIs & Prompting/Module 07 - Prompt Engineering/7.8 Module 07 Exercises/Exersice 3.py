import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = "openrouter/free"

def safe_json_parse(text: str) -> dict:
    """
    Parse JSON safely.
    Step 1: Try json.loads()
    Step 2: Remove markdown fences and try again.
    Step 3: Ask the model to repair the JSON.
    """
    # Step 1 - Direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 2 - Strip markdown fences
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 3 - Ask model to repair JSON
    repair_prompt = f"""Repair the following invalid JSON.
Return ONLY valid JSON.
Do not use markdown.
Do not provide explanations.

Invalid JSON:
{cleaned}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[
                {"role": "user", "content": repair_prompt}
            ],
        )
        if not hasattr(response, 'choices') or not response.choices:
            return {"error": "API returned empty choices."}
            
        repaired = response.choices[0].message.content or ""
        repaired = repaired.strip()
        
        repaired = re.sub(r"^```json\s*", "", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"^```\s*", "", repaired)
        repaired = re.sub(r"\s*```$", "", repaired)
        repaired = repaired.strip()

        return json.loads(repaired)
    except Exception as e:
        return {"error": f"Failed to repair JSON: {str(e)}"}


def run_json_repair_demo():
    print("\n" + "=" * 70)
    print("EXERCISE 3 - AUTOMATIC JSON REPAIR")
    print("=" * 70)

    invalid_json = (
        "```json\n"
        "{\n"
        '    "name": "Model",\n'
        '    "version": "1.0",\n'
        '    "provider": "OpenAI",\n'
        "}\n" # Trailing comma causing error
        "```"
    )

    print("Attempting to parse invalid JSON:")
    print(invalid_json)
    
    result = safe_json_parse(invalid_json)
    
    print("\nResult after safe_json_parse:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_json_repair_demo()
