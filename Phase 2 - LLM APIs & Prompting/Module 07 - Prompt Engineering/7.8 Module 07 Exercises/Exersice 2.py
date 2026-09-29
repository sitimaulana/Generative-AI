import json
from dataclasses import dataclass, asdict

@dataclass
class PromptTemplate:
    """A reusable, versioned prompt template."""
    name: str
    system: str
    user: str
    version: str = "1.0"

class PromptLibrary:
    """Store named PromptTemplate instances."""

    def __init__(self):
        self.templates: dict[str, PromptTemplate] = {}
        self.last_evaluation: dict[str, str] = {}

    def add(self, template: PromptTemplate):
        """Add or replace a prompt template."""
        self.templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        """Get a prompt template by name."""
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found.")
        return self.templates[name]

    def save(self, path: str):
        """Save the prompt library to JSON."""
        data = {
            "templates": {
                name: asdict(template)
                for name, template in self.templates.items()
            },
            "last_evaluation": self.last_evaluation,
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load the prompt library from JSON."""
        library = cls()
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            
        for name, template_data in data["templates"].items():
            library.templates[name] = PromptTemplate(**template_data)
            
        library.last_evaluation = data.get("last_evaluation", {})
        return library

    def record_evaluation(self, template_name: str, version: str):
        """Record the version used for the last evaluation."""
        self.last_evaluation[template_name] = version

def run_prompt_library_demo():
    print("\n" + "=" * 70)
    print("EXERCISE 2 - PROMPT LIBRARY")
    print("=" * 70)

    library = PromptLibrary()
    
    library.add(PromptTemplate(
        name="code_review",
        version="1.0",
        system="You are a code review assistant.",
        user="Review the following code:\n\n{code}"
    ))
    library.add(PromptTemplate(
        name="code_review_intermediate",
        version="1.1",
        system="You are an experienced code reviewer.",
        user="Review the following code:\n\n{code}"
    ))

    library.record_evaluation("code_review", "1.0")
    library.record_evaluation("code_review_intermediate", "1.1")

    library.save("prompt_library.json")
    print("\nPrompt library saved to: prompt_library.json")
    
    print("\nLast evaluation versions:")
    for name, version in library.last_evaluation.items():
        print(f"- {name}: version {version}")

    loaded_library = PromptLibrary.load("prompt_library.json")
    print("\nLoaded templates:")
    for name, template in loaded_library.templates.items():
        print(f"- {name} v{template.version}")

if __name__ == "__main__":
    run_prompt_library_demo()
