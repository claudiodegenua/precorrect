"""precorrect — a generation-time correction layer for LLMs.

Debias *before* you generate, not after. Bring your own rules; the engine ships empty.

The pattern: probe a model for the systematic biases it would introduce on a topic,
turn those into correction rules, and inject them into the prompt *before* generation —
instead of detecting/filtering the output afterward.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

# Any model backend: a callable that maps a prompt string to a completion string.
# Works with OpenAI, Anthropic, a local model, anything — you supply it.
Complete = Callable[[str], str]


@dataclass
class Rule:
    """A single correction instruction injected into the prompt before generation."""

    text: str
    triggers: list[str] = field(default_factory=list)  # keywords; empty = always apply
    severity: str = "info"  # info | warn | critical (controls ordering/emphasis)

    def applies_to(self, prompt: str) -> bool:
        if not self.triggers:
            return True
        p = prompt.lower()
        return any(t.lower() in p for t in self.triggers)


@dataclass
class RuleSet:
    """A bag of correction rules. Bring your own — the tuned content is your moat."""

    rules: list[Rule] = field(default_factory=list)

    @classmethod
    def from_list(cls, items: Iterable) -> "RuleSet":
        out: list[Rule] = []
        for it in items:
            if isinstance(it, str):
                out.append(Rule(text=it))
            elif isinstance(it, dict) and it.get("text"):
                out.append(
                    Rule(text=it["text"], triggers=it.get("triggers", []), severity=it.get("severity", "info"))
                )
        return cls(out)

    @classmethod
    def from_file(cls, path: str) -> "RuleSet":
        raw = open(path, encoding="utf-8").read()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml  # optional dependency
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("Install pyyaml to load .yaml rules, or use .json") from e
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        items = data["rules"] if isinstance(data, dict) else data
        return cls.from_list(items)

    def applicable(self, prompt: str) -> list[Rule]:
        return [r for r in self.rules if r.applies_to(prompt)]


_PREAMBLE_HEADER = (
    "Before answering, apply these correction rules. They exist to counter systematic "
    "biases a model tends to introduce on this topic. Follow them strictly.\n"
)

_DISCOVER_PROMPT = (
    "You are auditing your own likely failure modes. For the topic below, list the "
    "{n} most likely systematic biases, distortions, or oversimplifications you might "
    "introduce when answering — the recurring kind, not random slips. For each, write a "
    "one-sentence corrective instruction (imperative) that would prevent it.\n"
    'Return JSON only: [{{"text": "...", "triggers": ["keyword"], "severity": "warn"}}]\n\n'
    "TOPIC: {topic}"
)


class PreCorrect:
    """Inject correction rules into a prompt *before* generation.

    Domain-agnostic. You bring the rules (``RuleSet``) or surface candidate rules with
    :meth:`discover`. Your tuned rule content is your own — it never ships with this engine.
    """

    def __init__(self, complete: Complete):
        self.complete = complete

    def build_preamble(self, rules: Sequence[Rule]) -> str:
        if not rules:
            return ""
        order = {"critical": 0, "warn": 1, "info": 2}
        ordered = sorted(rules, key=lambda r: order.get(r.severity, 2))
        body = "\n".join(f"- {r.text}" for r in ordered)
        return _PREAMBLE_HEADER + body + "\n\n"

    def generate(self, prompt: str, rules: RuleSet | None = None) -> str:
        """Inject the applicable rules ahead of the prompt, then generate."""
        applicable = rules.applicable(prompt) if rules else []
        return self.complete(self.build_preamble(applicable) + prompt)

    def discover(self, topic: str, n: int = 6) -> RuleSet:
        """Use the model to self-surface likely biases on a topic → candidate rules.

        Returns candidates for HUMAN REVIEW. Curate them; the curated set is your moat.
        """
        raw = self.complete(_DISCOVER_PROMPT.format(topic=topic, n=n))
        return RuleSet.from_list(_parse_json_list(raw))


def _parse_json_list(raw: str) -> list:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("text")]
