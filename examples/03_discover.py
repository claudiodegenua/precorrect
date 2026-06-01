"""Example: auto-discover candidate bias rules for a topic.

The model surfaces its own likely failure modes. You review them,
curate what's accurate, and those become your rule set.
The curated content is your moat — it never ships with precorrect.
"""
import anthropic, json
from precorrect import PreCorrect, RuleSet

client = anthropic.Anthropic()

def complete(prompt: str) -> str:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text

pc = PreCorrect(complete=complete)

topic = "Second Temple Judaism and the halakhic authority of rabbis"
print(f"Discovering candidate rules for topic: {topic!r}\n")

candidates = pc.discover(topic=topic, n=6)
print("=== CANDIDATE RULES (review before using!) ===")
for r in candidates.rules:
    print(f"[{r.severity}] {r.text}")
    if r.triggers:
        print(f"    triggers: {r.triggers}")
