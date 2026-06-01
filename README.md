# precorrect

**Debias an LLM *before* you generate, not after.**

A lightweight, model-agnostic Python library that injects correction rules into a prompt *before* generation — instead of detecting or filtering the output afterward.

```python
from precorrect import PreCorrect, RuleSet

rules = RuleSet.from_file("my_rules.yaml")   # bring your own
pc = PreCorrect(complete=my_llm_function)

output = pc.generate(prompt, rules=rules)    # rules injected before generation
```

30 seconds to first run. Zero hard dependencies. Works with any LLM.

---

## Why before, not after?

Most bias-mitigation tools detect or block *after* the model has already generated the distorted output.

```
┌──────────┐     ┌──────────────┐     ┌──────────┐
│  prompt  │────▶│     LLM      │────▶│  detect  │  ← too late
└──────────┘     └──────────────┘     └──────────┘
                  bias inserted here
```

**precorrect** moves the correction *upstream*, into the prompt:

```
┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐
│  prompt  │──▶│ inject rules │──▶│     LLM      │──▶│  output  │  ✓
└──────────┘   └──────────────┘   └──────────────┘   └──────────┘
                                   less bias here
```

This matters because the model's distortions come from its training-data patterns, not from what you put in the context — so cleaning the context alone doesn't fix them. Correcting *before* generation shrinks the window where bias is inserted.

---

## Installation

```bash
pip install precorrect
# optional: YAML rule files
pip install "precorrect[yaml]"
```

Or install from source:

```bash
git clone https://github.com/OWNER/precorrect
cd precorrect
pip install -e .
```

---

## Quickstart

**Step 1 — Define your LLM function** (any model, any API):

```python
import anthropic

client = anthropic.Anthropic()

def my_llm(prompt: str) -> str:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text
```

**Step 2 — Write correction rules** (`my_rules.yaml`):

```yaml
rules:
  - text: >
      Do not treat a secondary synthesis as a primary source.
      Cite the original or flag the claim as derived.
    severity: warn
  - text: >
      Popularity of an interpretation reflects training-data frequency,
      not historical accuracy. Note when these diverge.
    severity: critical
    triggers: [popular, mainstream, widely]
```

*Your tuned, domain-specific rules are your moat — they never ship with this library.*

**Step 3 — Generate with corrections:**

```python
from precorrect import PreCorrect, RuleSet

rules = RuleSet.from_file("my_rules.yaml")
pc = PreCorrect(complete=my_llm)

output = pc.generate(
    prompt="Explain the halakhic meaning of 'taking the yoke'.",
    rules=rules,
)
print(output)
```

---

## Benchmark

```bash
ANTHROPIC_API_KEY=sk-... python benchmark/run_benchmark.py
# works with any model — see the Ollama comment in the script
```

Sample result (3 targeted probes, Claude Haiku):

| | Baseline | With precorrect |
|---|---|---|
| Drift markers hit | 6 | 1 |
| Reduction | — | **83%** |

The `benchmark/last_run.json` contains a documented sample run. Run it yourself to reproduce with your model.

---

## Auto-discover candidate rules

If you don't know where to start, let the model surface its own likely failure modes for a topic:

```python
candidates = pc.discover(topic="Second Temple Judaism", n=6)
for r in candidates.rules:
    print(f"[{r.severity}] {r.text}")
```

**Review the output before using.** The discover step gives you candidates; curation is your job.

---

## How it works

1. **RuleSet.applicable(prompt)** — filters rules whose triggers match the prompt
2. **build_preamble(rules)** — formats applicable rules as an imperative instruction block (critical-first)
3. **complete(preamble + prompt)** — calls your LLM with the augmented prompt
4. Returns the model's output

The preamble looks like:

```
Before answering, apply these correction rules. They exist to counter systematic
biases a model tends to introduce on this topic. Follow them strictly.
- Popularity of an interpretation reflects training-data frequency, not historical accuracy...
- Do not conflate minority scholarly positions with the mainstream consensus...

[your original prompt here]
```

---

## API reference

```python
PreCorrect(complete: Callable[[str], str])
    .generate(prompt, rules=None) -> str
    .discover(topic, n=6)         -> RuleSet   # candidates; review before using

RuleSet.from_file(path)           # .yaml or .json
RuleSet.from_list(items)          # list of str or dict
RuleSet.applicable(prompt)        -> list[Rule]

Rule(text, triggers=[], severity="info")
```

---

## promptfoo integration

Use `precorrect` as a custom assertion in your [promptfoo](https://promptfoo.dev) test suite:

```yaml
# promptfoo.yaml
providers:
  - id: anthropic:messages:claude-haiku-4-5-20251001

tests:
  - vars:
      prompt: "Explain the meaning of 'take my yoke' in Matthew 11:29"
    assert:
      - type: javascript
        value: |
          const { PreCorrect, RuleSet } = require('./node_adapter');
          // assert correction rules were applied — check for absence of known distortions
          return !output.toLowerCase().includes("symbol of burden");
```

See `examples/promptfoo/` for a working integration.

---

## Background

`precorrect` was extracted from a private research pipeline that applies it to theology — a domain where:
- the model's training data is dominated by the most-popular (Western, post-Nicene) reading
- every distortion is checkable against the Hebrew and Greek source
- the stakes of a wrong answer are high

A test battery of 6,597 trap questions across 55 articles showed that a standard model **gets ~23% of targeted domain probes wrong** — systematically, not randomly. Moving the correction upstream reduced that failure rate measurably.

The method is domain-agnostic: anywhere the model inherits a dominant-but-distorted reading *and* claims are checkable against a primary source, the same pattern applies. Theology is the proving ground; the library is the extracted tool.

Read the full case study: [teocentro.com/about/methodology/llm-bias-removal](https://teocentro.com/about/methodology/llm-bias-removal/)

---

## Contributing

Bug reports and PRs welcome. For new features, open an issue first.
Domain-specific rule sets (the tuned content) are intentionally kept out of this repo — see the `rules/example_generic.yaml` for the expected format.

---

MIT License · [Claudio De Genua](https://teocentro.com/chi-siamo/claudio-de-genua/)
