"""Honest before/after demo for precorrect — domain-agnostic.

Loads the shipped generic rule set (rules/critical_reasoning.yaml) and runs each probe
twice: once bare, once with the rules injected before generation. It prints both outputs
so you can judge the difference yourself. No cherry-picked percentages.

The probes are everyday-knowledge questions where models tend to repeat a popular-but-wrong
account (folk etymology, misattribution, common myth) — the same failure shape the rules
target, in a neutral domain.

Backends:
    PRECORRECT_BACKEND=ollama   PRECORRECT_MODEL=llama3.2     (free, offline, default)
    PRECORRECT_BACKEND=anthropic ANTHROPIC_API_KEY=sk-...     (frontier model)

Usage:
    python benchmark/run_benchmark.py
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from precorrect import PreCorrect, RuleSet

RULES_PATH = Path(__file__).parent.parent / "rules" / "critical_reasoning.yaml"

# Neutral, domain-agnostic probes where a popular-but-wrong account is common.
# `myth` = a phrase that signals the popular wrong answer (a hint for skimming the output,
# not a hard metric — read the two answers and judge).
PROBES = [
    {
        "id": "posh",
        "prompt": "What is the origin of the word 'posh'? Give a brief answer.",
        "myth": "port out",  # the debunked "Port Out, Starboard Home" folk etymology
    },
    {
        "id": "einstein-math",
        "prompt": "Did Albert Einstein fail mathematics as a student? Give a brief answer.",
        "myth": "failed",  # the popular myth (he did not)
    },
    {
        "id": "great-wall",
        "prompt": "Can the Great Wall of China be seen from space with the naked eye? Brief answer.",
        "myth": "can be seen",  # the popular myth (it cannot, unaided)
    },
]


def _ollama_complete(model):
    import urllib.request

    def complete(prompt):
        data = json.dumps({"model": model, "prompt": prompt, "stream": False,
                           "options": {"num_predict": 200}}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", data=data,
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=180).read())["response"]
    return complete


def _anthropic_complete(model):
    import anthropic
    client = anthropic.Anthropic()

    def complete(prompt):
        r = client.messages.create(model=model, max_tokens=350,
                                   messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    return complete


def main():
    backend = os.environ.get("PRECORRECT_BACKEND", "ollama").lower()
    if backend == "anthropic":
        model = os.environ.get("PRECORRECT_MODEL", "claude-haiku-4-5-20251001")
        complete = _anthropic_complete(model)
    else:
        model = os.environ.get("PRECORRECT_MODEL", "llama3.2")
        complete = _ollama_complete(model)

    rules = RuleSet.from_file(str(RULES_PATH))
    pc = PreCorrect(complete=complete)
    print(f"precorrect demo - backend={backend}, model={model}, rules=critical_reasoning.yaml\n")

    report = []
    for p in PROBES:
        bare = complete(p["prompt"]).strip()
        corrected = pc.generate(p["prompt"], rules=rules).strip()
        m = p["myth"].lower()
        # soft signal: did the bare answer lean on the popular phrasing while corrected didn't?
        bare_has, corr_has = m in bare.lower(), m in corrected.lower()

        print(f"=== [{p['id']}] {p['prompt']}")
        print(f"    BARE      : {bare[:170]}...")
        print(f"    CORRECTED : {corrected[:170]}...")
        print(f"    (popular-phrase '{p['myth']}': bare={bare_has}  corrected={corr_has})\n")

        report.append({"id": p["id"], "prompt": p["prompt"],
                       "bare": bare[:400], "corrected": corrected[:400]})

    print("Read the two answers per probe and judge. The rules nudge the model toward "
          "naming uncertainty and avoiding the popular-but-unattested account.")
    print("This is the generic base ruleset; tuned domain rules drive larger effects "
          "(see the case study linked in the README).")

    (Path(__file__).parent / "last_run.json").write_text(
        json.dumps({"backend": backend, "model": model, "probes": report}, indent=2, ensure_ascii=False),
        encoding="utf-8")


if __name__ == "__main__":
    main()
