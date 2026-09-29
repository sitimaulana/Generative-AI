import openai
import os
import time
from dotenv import load_dotenv

load_dotenv()

def retry_on_rate_limit(client, messages, max_retries=5, system="", max_tokens=512):
    """
    Retry an API call when RateLimitError occurs using OpenRouter/OpenAI.
    """
    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages += messages

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                max_tokens=max_tokens,
                messages=api_messages,
            )
            return response

        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt
            print(f"Rate limit reached. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise RuntimeError("Maximum retries exceeded.")


# Mock client for testing
class MockRateLimitClient:
    def __init__(self, failures_before_success=3):
        self.failures_before_success = failures_before_success
        self.attempts = 0

    class Chat:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self.Completions(parent)

        class Completions:
            def __init__(self, parent):
                self.parent = parent

            def create(self, **kwargs):
                self.parent.attempts += 1
                if self.parent.attempts <= self.parent.failures_before_success:
                    import httpx
                    err_response = httpx.Response(429, request=httpx.Request("POST", "http://test"))
                    raise openai.RateLimitError(message="Mock rate limit", response=err_response, body=None)

                return MockResponse()

    @property
    def chat(self):
        return self.Chat(self)

class MockResponse:
    class Choice:
        class Message:
            content = "Mock response after retry."
        message = Message()
    choices = [Choice()]
    
    class Usage:
        prompt_tokens = 10
        completion_tokens = 20
    usage = Usage()


# Test Exercise 1
if __name__ == "__main__":
    print("Testing API simulation with Rate Limit error...")
    mock_client = MockRateLimitClient(failures_before_success=3)
    response = retry_on_rate_limit(mock_client, messages=[{"role": "user", "content": "Hello"}])
    print("\nExercise 1 result:")
    print(response.choices[0].message.content)