import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = "openrouter/free"

BASIC_SYSTEM = """You are a code review assistant.
Review the provided code and identify bugs or problems.
Give a short explanation and suggest improvements."""

INTERMEDIATE_SYSTEM = """You are an experienced code reviewer.
Review the provided code for:
- Bugs
- Security issues
- Performance problems
- Readability issues

For each issue, explain the problem and provide a suggested fix.
Be concise and practical."""

EXPERT_SYSTEM = """You are a senior Python engineer reviewing
production code for an AI software system.

Perform a comprehensive code review covering:
1. Correctness and potential bugs
2. Security vulnerabilities
3. Performance and scalability
4. Error handling
5. Maintainability and readability
6. Python best practices

For every issue, use this format:
Issue:
Impact:
Recommendation:
Corrected code:

Do not give unnecessary compliments.
If no issue exists in a category, state that briefly.
Prioritize issues by severity: Critical, High, Medium, Low."""

code_snippets = [
    """
def divide(a, b):
    return a / b
""",
    """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return database.execute(query)
""",
    """
def calculate_average(numbers):
    total = 0
    for number in numbers:
        total += number
    return total / len(numbers)
""",
    """
def load_file(path):
    file = open(path, "r")
    data = file.read()
    return data
""",
    """
def find_user(users, name):
    for user in users:
        if user["name"] == name:
            return user
    return None
"""
]

def review_code(system_prompt: str, code: str) -> str:
    """Send code to the model for review."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Review this code:\n\n{code}"}
            ],
        )
        if not hasattr(response, 'choices') or not response.choices:
            return "ERROR: OpenRouter returned empty response (rate limit or down)."
        return response.choices[0].message.content or "ERROR: Empty content."
    except Exception as e:
        return f"ERROR: {str(e)}"

def evaluate_review_prompts():
    """Evaluate Basic, Intermediate, and Expert prompts."""
    prompts = {
        "Basic": BASIC_SYSTEM,
        "Intermediate": INTERMEDIATE_SYSTEM,
        "Expert": EXPERT_SYSTEM,
    }

    for level, prompt in prompts.items():
        print("\n" + "=" * 70)
        print(f"{level.upper()} CODE REVIEW")
        print("=" * 70)

        for index, code in enumerate(code_snippets, start=1):
            print(f"\n--- Code Snippet {index} ---")
            print(code.strip())
            result = review_code(prompt, code)
            print("\nReview:")
            print(result)

if __name__ == "__main__":
    evaluate_review_prompts()
