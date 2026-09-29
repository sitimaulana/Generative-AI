class BudgetExceeded(Exception):
    pass

class TokenBudgetManager:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def total_tokens(self):
        return self.total_input_tokens + self.total_output_tokens

    def add_usage(self, input_tokens: int, output_tokens: int):
        new_total = self.total_tokens + input_tokens + output_tokens
        if new_total > self.max_tokens:
            raise BudgetExceeded(f"Budget exceeded. Limit: {self.max_tokens}, requested: {new_total}")

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def remaining_tokens(self):
        return self.max_tokens - self.total_tokens


# Test Exercise 2
if __name__ == "__main__":
    budget = TokenBudgetManager(max_tokens=1000)
    print(f"Initial total budget: 1000")
    
    print("\nUsing 200 input, 150 output...")
    budget.add_usage(200, 150)
    print(f"Remaining tokens: {budget.remaining_tokens()}")
    
    print("\nUsing 300 input, 200 output...")
    budget.add_usage(300, 200)
    print(f"Remaining tokens: {budget.remaining_tokens()}")
    
    print("\nAttempting to use tokens exceeding budget (add 200 in, 100 out)...")
    try:
        budget.add_usage(200, 100)
    except BudgetExceeded as e:
        print(f"Error successfully caught: {e}")
