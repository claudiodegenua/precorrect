"""Example: build a VERTICAL BENCHMARK from your own KB, then use it two ways.

`build_evaluator_from_kb` turns your reviewed corpus into an answer-key — the field's correct
positions on the contested points — and hands you an Evaluator. The mechanism is generic; the
key is derived from YOUR KB, so the benchmark is specific to your vertical without writing a
single test by hand.

Two uses of the same Evaluator:
  USE 1 — test any LLM on your vertical: score an existing model's output, compare models,
          track regressions when you swap models.
  USE 2 — gate your own generation: score what you generate; if it contradicts the key,
          regenerate or inject the relevant correction rules.
"""
import anthropic
from precorrect import build_evaluator_from_kb

client = anthropic.Anthropic()


def complete(prompt: str) -> str:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text


# Your reviewed, titled source documents — {name: text}. This KB is your field's authority.
# (Two tiny inline samples so this runs as-is; in practice read your real corpus files.)
kb = {
    "venus.txt": "Venus rotates so slowly that a Venusian day (243 Earth days) is LONGER than its "
                 "year (225 Earth days). The morning star and the evening star are the same planet.",
    "wall.txt": "The Great Wall of China is NOT visible to the naked eye from low Earth orbit — a persistent myth.",
}

# --- PREP (once): derive the vertical answer-key from your KB ---
# If you already built lenses (your moat), pass lenses=... ; otherwise they bootstrap from the KB.
evaluator = build_evaluator_from_kb(complete, kb, topic="your topic")
print(f"answer-key: {len(evaluator)} positions extracted from the KB (REVIEW before trusting)")
evaluator.save("my_vertical_key.json")        # extract once, reuse forever

# ... later: evaluator = Evaluator.load("my_vertical_key.json", complete)

# --- USE 1: test an existing model on your vertical ---
model_output = complete("Explain <a question in your domain>.")
result = evaluator.evaluate(model_output)
print(f"\nvertical score: {result.score}  "
      f"({result.affirmed} affirmed / {result.contradicted_n} contradicted of {result.addressed} addressed)")
for c in result.contradicted:
    print(f"  CONTRADICTS the field: {c['question']}  (source: {c['source_ref']})")

# --- USE 2: gate your own generation ---
draft = complete("Write a paragraph about <your topic>.")
verdict = evaluator.evaluate(draft)
if verdict.score is not None and verdict.score < 0.8:
    print(f"\ndraft rejected (score {verdict.score}) — regenerate or inject corrective rules")
else:
    print(f"\ndraft accepted (score {verdict.score})")
