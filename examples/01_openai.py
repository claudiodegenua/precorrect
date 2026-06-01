"""Example: precorrect with the OpenAI SDK."""
import openai
from precorrect import PreCorrect, RuleSet

client = openai.OpenAI()

def complete(prompt: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content

rules = RuleSet.from_file("rules/example_generic.yaml")
pc = PreCorrect(complete=complete)

prompt = "What did Augustine teach about original sin?"

print("=== WITHOUT precorrect ===")
print(complete(prompt))

print("\n=== WITH precorrect ===")
print(pc.generate(prompt, rules=rules))
