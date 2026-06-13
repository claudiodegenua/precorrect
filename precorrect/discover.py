"""Generic KB → correction-rule discovery (the moat-builder).

`discover_from_kb(complete, kb_text, topic, lenses, ...)`:
  1. per lens: generate trap yes/no questions grounded in the KB (source = ground truth)
  2. execute each question zero-context with adaptive staged-N (Wilson-gated) — cheap where
     the model is reliably right/wrong, full precision only where the verdict is contested
  3. keep only deviations (model answers against the source) and synthesize correction RULES,
     each classified into one of 12 generic reasoning-failure clusters
  4. return a RuleSet ready to inject before generation (precorrect.PreCorrect.generate)

`complete: Callable[[str], str]` is the user's own LLM backend (any vendor). No dependency
on Anthropic or any private pipeline. The lens identities + the curated KB are the user's moat;
this mechanism is the shipped, domain-agnostic engine.
"""
from __future__ import annotations

import ast
import json
import math
import re
from typing import Callable, Dict, List, Optional

from .core import RuleSet
from .lens import LensRegistry

Complete = Callable[[str], str]

# The 12 generic reasoning-failure clusters (= rules/critical_reasoning.yaml).
CLUSTERS = [
    "false_attribution", "anachronism", "popularity_not_accuracy", "technical_term_flattening",
    "over_hedging", "out_of_frame_rejection", "distinction_collapse", "conflation",
    "polemic_as_neutral", "source_type_confusion", "divergence_honesty", "coverage_omission",
]

DEFAULT_SYSTEM_PROMPT = (
    "Answer the following yes/no question plainly, as a general assistant, from broad general "
    "knowledge. Reply ONLY JSON: {\"answer\": true|false, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}"
)


def _parse(raw, fallback):
    if not raw:
        return fallback
    m = re.search(r"(\[.*\]|\{.*\})", raw, re.S)
    if m:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(m.group(1))
            except Exception:
                pass
    return fallback


def _wilson(w: int, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 1.0
    p = w / n
    z2 = z * z
    d = 1.0 + z2 / n
    c = (p + z2 / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def _ask_staged(complete: Complete, system_prompt: str, question: dict,
                stages=(5, 10, 20), z: float = 1.96, precision: Optional[float] = None):
    """A1 staged-N. Returns (bias_rate, n_used, last_confidence).

    Two stop rules:
      - verdict (precision=None, default/cheap): settle when the Wilson 95% interval is
        one-sided around 0.5 (the >50% verdict can't flip).
      - R3 precise-rate (precision set, e.g. 0.15): escalate until the interval WIDTH < ε,
        i.e. the bias-RATE itself is pinned — a stable 1/5 (~20%) is a real guessing signal,
        not "clean", so it must not be dropped just because it sits under 0.5.
    """
    expected = bool(question.get("expected_answer", True))
    answers, wrong, n, conf = [], 0, 0, 0.5
    for target in stages:
        while n < target:
            out = _parse(complete(system_prompt + "\n\nQ: " + question.get("question", "")), {})
            ans = bool(out.get("answer", True))
            conf = float(out.get("confidence", 0.5) or 0.5)
            answers.append(ans)
            if ans != expected:
                wrong += 1
            n += 1
        lo, hi = _wilson(wrong, n, z)
        if precision is not None:
            if (hi - lo) < precision or (wrong == 0 and hi < precision):
                break
        else:
            if wrong == 0 or hi < 0.5 or lo > 0.5:
                break
    return (wrong / n if n else 0.0), n, conf


def discover_from_kb(
    complete: Complete,
    kb_text: str,
    topic: str,
    lenses: LensRegistry,
    system_prompt: Optional[str] = None,
    forbidden_bias_types: Optional[List[str]] = None,
    staged: bool = True,
    precision: Optional[float] = None,
    language: Optional[str] = None,
    max_kb_chars: int = 6000,
) -> RuleSet:
    """Discover correction rules from a single reviewed KB document. See module docstring.

    language: output language for the synthesized rule text + triggers (e.g. "Italian"). Default
    None = English (the synthesis prompt's language). Set it when your generation pipeline runs in
    another language, so the rules' keywords actually match your prompts.
    """
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    kb = (kb_text or "")[:max_kb_chars]

    # 1. generate trap questions per lens (KB = ground truth)
    questions = []
    for lens in lenses:
        prompt = lens.render(topic, kb)
        qs = _parse(complete(prompt), [])
        if isinstance(qs, list):
            questions += [q for q in qs if isinstance(q, dict) and "question" in q]

    # 2. execute (staged-N) → keep deviations (model answers AGAINST the source).
    # precise mode (R3): extend the ladder so a tight ε-width is reachable for low rates,
    # and keep STABLE sub-50% rates as partial-reliability/guessing signals (severity=info).
    stages = (5, 10, 20, 40, 80, 160) if precision is not None else (5, 10, 20)
    deviations = []
    for q in questions:
        if staged:
            rate, _n, conf = _ask_staged(complete, system_prompt, q, stages=stages, precision=precision)
        else:
            out = _parse(complete(system_prompt + "\n\nQ: " + q.get("question", "")), {})
            ans = bool(out.get("answer", True))
            conf = float(out.get("confidence", 0.5) or 0.5)
            rate = 0.0 if ans == bool(q.get("expected_answer", True)) else 1.0
        if rate > 0.5:
            sev = "warn"
        elif precision is not None and rate > precision:
            sev = "info"  # stable partial-reliability (model guesses) — record, don't discard
        else:
            continue
        deviations.append({**q, "bias_rate": round(rate, 3), "confidence": conf, "severity": sev})

    if not deviations:
        return RuleSet([])

    # 3. synthesize correction rules (carry the source fact + classify into 12 clusters)
    forb = ", ".join(forbidden_bias_types or [])
    payload = [{"question": d.get("question", ""), "source_ref": d.get("source_ref", ""),
                "source_answer": d.get("expected_answer")} for d in deviations]
    det = (
        "A model deviated from a REVIEWED source on these questions (source_answer = the truth). "
        f"For EACH, write ONE imperative correction RULE that carries the source's fact, and "
        f"classify into exactly one cluster from {CLUSTERS}. "
        + (f"Do NOT reuse these existing bias types: {forb}. " if forb else "")
        + (f"Write the 'text' and 'triggers' in {language}. " if language else "")
        + "Reply ONLY JSON: [{\"text\":\"...\",\"triggers\":[\"keyword\"],\"severity\":\"warn\",\"cluster\":\"...\"}]\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    rules_raw = _parse(complete(det), [])
    return RuleSet.from_list(rules_raw if isinstance(rules_raw, list) else [])


def discover_from_corpus(
    complete: Complete,
    kb_docs: Dict[str, str],
    topic: str,
    lenses: LensRegistry,
    **kwargs,
) -> RuleSet:
    """R1 — Mode A bootstrap (MANDATORY, run once upfront over the WHOLE corpus).

    Lazy per-query probing (Mode B) only sees the chunks a query retrieves, so it is BLIND to
    biases living in KB material that is never retrieved — a good-but-flawed setup. Sweeping the
    full corpus once with this function intercepts those up front; Mode B then tops up incrementally.

    kb_docs = {doc_name: doc_text}. Returns the merged RuleSet (deduped by rule text). Pass the same
    kwargs as `discover_from_kb` (system_prompt, forbidden_bias_types, staged, precision, max_kb_chars).
    """
    merged: List = []
    seen_text = set()
    for _name, text in kb_docs.items():
        rs = discover_from_kb(complete, text, topic, lenses, **kwargs)
        for r in rs.rules:
            key = (r.text or "").strip().lower()
            if key and key not in seen_text:
                seen_text.add(key)
                merged.append(r)
    return RuleSet(merged)
