"""Example: build your own tuned rule set from your reviewed corpus (the moat).

Flow:
  1. discover_lenses  — let the model propose analytical perspectives from a KB sample
  2. discover_from_corpus — Mode-A sweep over the WHOLE corpus (mandatory once, upfront):
     per (doc x lens) generate grounded trap questions, probe the model zero-context with
     adaptive staged-N, keep the deviations, synthesize correction RULES.
  3. review + keep the accurate rules — that curated set is YOUR moat (never shipped).

Lazy per-query probing alone is blind to biases in never-retrieved chunks, so the full-corpus
sweep is the obligatory bootstrap; incremental top-ups come after.
"""
import anthropic
from precorrect import discover_lenses, discover_from_corpus

client = anthropic.Anthropic()


def complete(prompt: str) -> str:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text


# Your reviewed, titled source documents — {name: text}. NOT model-derived articles.
# (Two tiny inline samples so this runs as-is; in practice read your real corpus files.)
corpus = {
    "venus.txt": "Venus rotates so slowly that a Venusian day (243 Earth days) is LONGER than its "
                 "year (225 Earth days). The morning star and the evening star are the same planet.",
    "wall.txt": "The Great Wall of China is NOT visible to the naked eye from low Earth orbit — a persistent myth.",
}

# 1. let the corpus suggest its own lenses (or hand-write a LensRegistry — that's your moat)
lenses = discover_lenses(next(iter(corpus.values())), complete, n=6)
print(f"discovered {len(lenses)} lenses:", [l.name for l in lenses])

# 2. Mode-A bootstrap over the whole corpus. precision=0.15 -> precise bias-rate (slower, one-time).
rules = discover_from_corpus(
    complete, corpus, topic="your topic", lenses=lenses, precision=0.15,
)
print(f"\nsynthesized {len(rules.rules)} candidate rules (REVIEW before trusting):")
for r in rules.rules:
    print(f"  [{r.severity}] {r.text}")

# 3. keep the accurate ones, save as your private rule file (the moat):
#    RuleSet(kept).to ... -> your own my_rules.yaml, injected at generation time.
