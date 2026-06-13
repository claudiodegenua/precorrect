# precorrect

[![test](https://github.com/claudiodegenua/precorrect/actions/workflows/test.yml/badge.svg)](https://github.com/claudiodegenua/precorrect/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

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
# optional: load .yaml rule files
pip install "precorrect[yaml]"
```

Latest from source:

```bash
pip install git+https://github.com/claudiodegenua/precorrect
# or clone for an editable install (also gets the example rule files + benchmark):
git clone https://github.com/claudiodegenua/precorrect
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

**Step 2 — Pick your rules.** Build a `RuleSet` inline — zero dependencies, runs anywhere (incl. a
fresh `pip install`). Bring your own, or start from a couple of generic ones:

```python
from precorrect import PreCorrect, RuleSet

rules = RuleSet.from_list([
    "Popularity of an interpretation reflects training-data frequency, not accuracy.",
    {"text": "Do not conflate a minority scholarly position with the mainstream consensus.",
     "triggers": ["consensus", "scholars"], "severity": "warn"},
])
# The generic rules nudge the model toward caution; a tuned rule carries the specific
# fact that overturns a confident mistake — that's where the leverage is.
```

The cloned repo also ships ready-made rule files: `RuleSet.from_file("rules/critical_reasoning.yaml")`
(12 generic corrections; loading `.yaml` needs `pip install pyyaml`) and `rules/template.yaml` for the format.

*Your tuned, domain-specific rules are your moat — they never ship with this library.*

**Step 3 — Generate with corrections:**

```python
pc = PreCorrect(complete=my_llm)

output = pc.generate(
    prompt="What is the origin of the word 'posh'?",
    rules=rules,
)
print(output)
```

---

## See it work (reproduce it yourself)

`benchmark/run_benchmark.py` runs neutral, everyday-knowledge probes — once bare, once
with the rules injected — and prints both, so you can judge the difference directly.
No cherry-picked numbers.

```bash
# local model, free, no API key (recommended — works offline)
PRECORRECT_BACKEND=ollama PRECORRECT_MODEL=llama3.2 python benchmark/run_benchmark.py

# or a frontier model
PRECORRECT_BACKEND=anthropic ANTHROPIC_API_KEY=sk-... python benchmark/run_benchmark.py
```

**A real run** on a small local model (`llama3.2`). Asking *"is the origin of the word
'posh' known?"*, the bare model **invents a confident false etymology**; with the generic
rules it correctly answers *"the origin is uncertain — there are several theories."*

The value gradient is clearest on a confident misconception. Asking *"did Einstein fail
mathematics as a student?"* (a well-documented myth):

| | Output |
|---|---|
| **Bare** | "Yes, Einstein **failed** mathematics twice…" — repeats the myth as fact |
| **+ generic rules** (`critical_reasoning.yaml`) | "There is **no direct attestation**… sources are not always clear" — the model now doubts it |
| **+ one tuned rule** (a specific corrective fact) | "There is **no evidence**… Einstein **excelled** at mathematics, mastering calculus by ~15" — clean correction |

This is the whole point: **generic rules buy caution; a tuned rule buys the correction.**
Writing the tuned rules for your domain is the work — and the moat. The large reductions
in the theology case study come from a private, domain-tuned rule set, not shipped here.

---

## The full arc — prepare once, use daily

precorrect gives you three legs of a vertical-correction pipeline, in two phases.

**Prepare (once, from your KB)** — turn your reviewed corpus into your vertical assets:
- `discover_lenses` — the analytical perspectives your material calls for
- `discover_from_corpus` — the correction **rules** (where the model drifts vs your sources)
- `build_evaluator_from_kb` — the **answer-key**: the field's correct positions, as a benchmark

**Use (daily, plug-in)** — apply those assets per query:
- **correct** — `generate` / `generate_gated` inject the rules *before* the model writes a token
- **evaluate** — `Evaluator.evaluate` scores any output against your answer-key: test an existing
  model on your vertical, or gate your own generation (regenerate when it contradicts the key)

```
                 ┌────────── PREPARE (once, from your KB) ──────────┐
  [ your KB ] ──▶ discover_lenses · discover_from_corpus · build_evaluator_from_kb
                 └─────────────┬───────────────────────────┬────────┘
                        rules  │                           │  answer-key
                               ▼                           ▼
  prompt ──▶ [ inject rules ] ─▶ LLM ─▶ output ─▶ [ evaluate vs key ] ─▶ score + flags
               └──── correct ────┘              └──────── evaluate ───────┘
                          └──────────────── USE (daily) ───────────────┘
```

The KB and its curation stay yours; the mechanism ships. Retrieval and your thresholds are yours to wire. The *evaluate* leg is detailed in the next section.

## Build a vertical benchmark from your KB

Most eval sets are written by hand. `build_evaluator_from_kb` derives one from your corpus: it
probes the KB through lenses, and each contested point — with the stance your sources take —
becomes a benchmark item. You get a domain-specific evaluator without authoring a single test.

```python
from precorrect import build_evaluator_from_kb, Evaluator

ev = build_evaluator_from_kb(complete, kb_docs={"doc1": text1, "doc2": text2}, topic="your field")
ev.save("my_key.json")                 # extract once, reuse (Evaluator.load to restore)

# USE 1 — test ANY model on your vertical:
res = ev.evaluate(some_model_output)
print(res.score, res.contradicted)     # affirmed/(affirmed+contradicted); positions it got wrong

# USE 2 — gate your own generation:
if res.score is not None and res.score < 0.8:
    ...                                # regenerate, or inject the relevant correction rules
```

The answer-key is candidates — review/curate as you would discovered rules; a stronger judge
model reduces false flags. See `examples/05_evaluate.py`.

**Honest caveat:** effectiveness scales with two things you control — **rule quality** and
**model capability**. The shipped `rules/template.yaml` is a neutral *template*; the
large, measurable reductions reported in the [theology case study](https://teocentro.com/about/methodology/llm-bias-removal/)
come from a tuned, domain-specific rule set that is **not** shipped here.
precorrect gives you the *mechanism*; the rules are yours to write.

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
    .generate_gated(prompt, rules=None, probe_n=3, consistency_threshold=7.0) -> str
                                  # inject rules ONLY where the model is internally
                                  # inconsistent (no answer-key needed for the gate)
    .discover(topic, n=6)         -> RuleSet   # candidates; review before using
    .build_preamble(rules)        -> str       # the injected instruction block (see "How it works")

RuleSet.from_file(path)           # .yaml or .json
RuleSet.from_list(items)          # list of str or dict
RuleSet.applicable(prompt)        -> list[Rule]

Rule(text, triggers=[], severity="info")

# PREPARE (once, from your KB) — review all output before trusting:
discover_lenses(kb_sample, complete, n=6)             -> LensRegistry
discover_from_corpus(complete, kb_docs, topic, lenses, **kw)  -> RuleSet  # Mode A: whole-corpus sweep
discover_from_kb(complete, kb_text, topic, lenses, **kw)                  -> RuleSet  # per-doc (language=, staged=, precision=, ...)
build_evaluator_from_kb(complete, kb_docs, topic, lenses=None, n_lenses=6) -> Evaluator
extract_answer_key(complete, kb_docs, topic, lenses, **kw)               -> list[KeyEntry]

# USE (daily) — score any output against your KB-derived answer-key:
Evaluator.evaluate(output, max_check=40)              -> EvalResult   # .score .affirmed .contradicted
Evaluator.save(path)  ·  Evaluator.load(path, complete)
```

> **Roadmap (v0.2):** claim-level faithfulness — decompose an output into individual claims and
> check each against the retrieved KB span (supported / contradicted / unsupported), to complement
> the answer-key score. Normative correctness beyond what any single source literally says stays a
> domain plugin you own.

---

## promptfoo integration

Wrap `precorrect` in a [promptfoo](https://promptfoo.dev) **Python provider** to A/B the correction
layer against the bare model in your test suite. The shipped `examples/promptfoo/provider.py` calls
`PreCorrect.generate` when `use_precorrect="true"`, and the bare model when `"false"`:

```yaml
# examples/promptfoo/promptfoo.yaml
providers:
  - id: python:provider.py            # wraps precorrect around your completion
    label: "precorrect + claude-haiku"

prompts:
  - raw: "What does 'take my yoke upon you' mean in Matthew 11:29? Brief explanation."

tests:
  - vars: { use_precorrect: "true" }
    assert:
      - type: not-contains
        value: symbol of burden         # the popular-but-distorted reading
  - vars: { use_precorrect: "false" }    # baseline, recorded for side-by-side comparison
    assert:
      - type: javascript
        value: "true"
```

Run it: `cd examples/promptfoo && npx promptfoo@latest eval`. The full provider is in
`examples/promptfoo/provider.py`.

---

## Background

`precorrect` was extracted from a private research pipeline that applies it to theology — a domain where:
- the model's training data is dominated by the most-popular (Western, post-Nicene) reading
- every distortion is checkable against the Hebrew and Greek source
- the stakes of a wrong answer are high

A test battery of over 6,500 trap questions showed that a general-purpose model **gets ~23% of targeted domain probes wrong** — systematically, not randomly. Moving the correction upstream reduced that failure rate measurably.

The method is domain-agnostic: anywhere the model inherits a dominant-but-distorted reading *and* claims are checkable against a primary source, the same pattern applies. Theology is the proving ground; the library is the extracted tool.

Read the full case study: [teocentro.com/about/methodology/llm-bias-removal](https://teocentro.com/about/methodology/llm-bias-removal/)

---

## Contributing

Bug reports and PRs welcome. For new features, open an issue first.
Domain-specific rule sets (the tuned content) are intentionally kept out of this repo — see the `rules/template.yaml` for the expected format.

---

MIT License · [Claudio De Genua](https://teocentro.com/chi-siamo/claudio-de-genua/)
