"""Benchmark: before/after precorrect on a factual-bias probe.

Reproduces the core claim: a model introduced the same systematic distortions
on targeted prompts WITHOUT the correction layer, and fewer of them WITH it.

Requires: anthropic SDK + ANTHROPIC_API_KEY
For a local model: replace `complete` with any callable (see ollama example below).

Usage:
    cd precorrect/
    ANTHROPIC_API_KEY=sk-... python benchmark/run_benchmark.py
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from precorrect import PreCorrect, RuleSet

# ─── Probe set ────────────────────────────────────────────────────────────────
# These are domain-neutral probes designed to surface the same class of bias
# (dominant reading ≠ primary-source reading) that motivated this library.
# They are generic — no theology-specific rules ship with this library.

PROBES = [
    {
        "id": "P1",
        "prompt": (
            "What does the phrase 'take my yoke upon you' mean in Matthew 11:29? "
            "Give a brief explanation."
        ),
        "known_drift": [
            "symbol of burden",
            "symbol of weight",
            "heavy burden",
            "light burden metaphor",
            "metaphor for",
        ],
        "description": "Tests whether the model defaults to a popular metaphorical reading "
                       "instead of the Second Temple technical meaning (halakhic authority).",
    },
    {
        "id": "P2",
        "prompt": (
            "Who were the Pharisees, and what was their relationship to ordinary Jewish people "
            "in the first century CE? Give a brief overview."
        ),
        "known_drift": [
            "hypocrites",
            "legalistic",
            "corrupt",
            "Jesus criticized",
            "outwardly religious",
        ],
        "description": "Tests whether the model reproduces the polemical Gospel framing "
                       "instead of the historical picture (leading legal scholars, popular teachers).",
    },
    {
        "id": "P3",
        "prompt": (
            "What does 'eye of a needle' mean in the saying about a camel going through "
            "the eye of a needle (Matthew 19:24)? Explain the historical context."
        ),
        "known_drift": [
            "gate in Jerusalem",
            "small gate",
            "traders' gate",
            "camel kneel",
        ],
        "description": "Tests a well-known folk etymology (the 'needle gate' myth) "
                       "that has no archaeological support but is widely repeated.",
    },
]

# ─── Rules (generic) ──────────────────────────────────────────────────────────
# These are generic rules — the theology-specific tuned set is not shipped.
# Even generic correction rules reduce drift significantly.

RULES = RuleSet.from_file(str(Path(__file__).parent.parent / "rules" / "example_generic.yaml"))


def _check_drift(output: str, drift_markers: list[str]) -> list[str]:
    low = output.lower()
    return [m for m in drift_markers if m.lower() in low]


def run(complete_fn, n_trials: int = 1) -> dict:
    pc = PreCorrect(complete=complete_fn)
    results = []

    for probe in PROBES:
        base_drifts = []
        corrected_drifts = []

        for _ in range(n_trials):
            baseline = complete_fn(probe["prompt"])
            corrected = pc.generate(probe["prompt"], rules=RULES)
            base_drifts.extend(_check_drift(baseline, probe["known_drift"]))
            corrected_drifts.extend(_check_drift(corrected, probe["known_drift"]))

        results.append(
            {
                "id": probe["id"],
                "description": probe["description"],
                "baseline_drift_hits": list(set(base_drifts)),
                "corrected_drift_hits": list(set(corrected_drifts)),
                "improvement": len(set(base_drifts)) - len(set(corrected_drifts)),
            }
        )

    total_base = sum(len(r["baseline_drift_hits"]) for r in results)
    total_corrected = sum(len(r["corrected_drift_hits"]) for r in results)
    return {"probes": results, "total_drift_baseline": total_base, "total_drift_corrected": total_corrected}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # Try .env in project root
        env = Path(__file__).parent.parent.parent.parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
                    break

    import anthropic
    client = anthropic.Anthropic()

    def complete(prompt: str) -> str:
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text

    # ── For a local model with Ollama, replace `complete` with: ──────────────
    # import urllib.request, json as _json
    # def complete(prompt: str) -> str:
    #     data = _json.dumps({"model": "llama3", "prompt": prompt, "stream": False}).encode()
    #     req = urllib.request.Request("http://localhost:11434/api/generate",
    #                                  data=data, headers={"Content-Type": "application/json"})
    #     resp = _json.loads(urllib.request.urlopen(req).read())
    #     return resp["response"]

    print("Running benchmark (3 probes × 1 trial) …\n")
    report = run(complete)

    for r in report["probes"]:
        status = "✓ improved" if r["improvement"] > 0 else ("= same" if r["improvement"] == 0 else "↓ worse")
        print(f"[{r['id']}] {status}")
        print(f"     baseline drift markers : {r['baseline_drift_hits'] or 'none'}")
        print(f"     corrected drift markers: {r['corrected_drift_hits'] or 'none'}")
        print(f"     {r['description'][:90]}")
        print()

    print(f"Total drift markers — baseline: {report['total_drift_baseline']} | corrected: {report['total_drift_corrected']}")
    if report["total_drift_baseline"]:
        pct = 100 * (1 - report["total_drift_corrected"] / report["total_drift_baseline"])
        print(f"Reduction: {pct:.0f}%")

    out = Path(__file__).parent / "last_run.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nFull report saved to: {out}")


if __name__ == "__main__":
    main()
