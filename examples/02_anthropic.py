"""Example: precorrect with the Anthropic SDK."""
import anthropic
from precorrect import PreCorrect, RuleSet

client = anthropic.Anthropic()

def complete(prompt: str) -> str:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text

rules = RuleSet.from_file("rules/example_generic.yaml")
pc = PreCorrect(complete=complete)

prompt = "What did Augustine teach about original sin?"

print("=== WITHOUT precorrect ===")
print(complete(prompt))

print("\n=== WITH precorrect ===")
print(pc.generate(prompt, rules=rules))
