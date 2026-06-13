"""Multi-perspective lens framework for `precorrect.discover`.

A **Lens** is a named analytical perspective: a role/identity prompt that, given a source
document, asks the model to surface *trap* yes/no questions from that angle — questions
where the source's specific claim diverges from the popular/conventional account.

The framework + a GENERIC base prompt ship with the OSS package; the OSS registry ships
EMPTY (plus one generic example). Domain-specific lens identities are the user's (or your
private) plugin and are never shipped — that is the moat.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List

# Domain-AGNOSTIC base, appended to every lens role-prompt.
# Placeholders filled by discover: {topic} {kb_filename} {kb_content} {lens_name} {forced_domain}
GENERIC_LENS_BASE = """
Analyze the SOURCE below through your assigned perspective ("{lens_name}"). Produce YES/NO
probe questions that test whether a model relying only on general knowledge (WITHOUT this
source) would DIVERGE from what this source actually says — i.e. traps where the source's
specific claim is the OPPOSITE of the popular/conventional account.

RULES:
- Each question answerable yes/no; `expected_answer` = the answer ACCORDING TO THIS SOURCE.
- Each question cites a `source_ref` grounding it in the text. Do NOT invent references.
- `domain` MUST be "{forced_domain}"; `section_id` MUST be "{kb_filename}_{lens_name}".
- Prefer questions where conventional knowledge answers the OPPOSITE of the source.

TOPIC: {topic} | SOURCE: {kb_filename} | LENS: {lens_name}

SOURCE:
---
{kb_content}
---

Reply ONLY JSON: [{{"id":"Q01","question":"...","expected_answer":true,"source_ref":"...","category":"...","difficulty":"expert","section_id":"{kb_filename}_{lens_name}","question_type":"source_attribution","domain":"{forced_domain}"}}]
If you find no in-domain claims: reply []"""


@dataclass
class Lens:
    name: str            # e.g. "statutory", "structural"
    forced_domain: str   # arbitrary label the caller picks, e.g. "legal_statutory"
    role_prompt: str     # the perspective's identity/instructions (domain-specific = moat)

    def template(self) -> str:
        """Full prompt template = role prompt + the generic base (unformatted)."""
        return self.role_prompt.rstrip() + "\n" + GENERIC_LENS_BASE

    def render(self, topic: str, kb_content: str, kb_filename: str = "kb") -> str:
        """Filled prompt. Formats ONLY the generic base (which holds the placeholders); the
        user-authored role_prompt is concatenated raw, so literal braces in it never crash."""
        base = GENERIC_LENS_BASE.format(
            topic=topic, kb_filename=kb_filename, kb_content=kb_content,
            lens_name=self.name, forced_domain=self.forced_domain,
        )
        return self.role_prompt.rstrip() + "\n" + base


class LensRegistry:
    """A pluggable set of perspectives. Ships empty in OSS; the caller supplies lenses."""

    def __init__(self, lenses: Iterable[Lens] = ()):
        self._lenses = list(lenses)

    def __len__(self) -> int:
        return len(self._lenses)

    def __iter__(self):
        return iter(self._lenses)

    @classmethod
    def from_dict(cls, d: dict) -> "LensRegistry":
        """d = {name: (forced_domain, role_prompt)}."""
        return cls(Lens(name=n, forced_domain=fd, role_prompt=rp) for n, (fd, rp) in d.items())

    @classmethod
    def from_plugin(cls, factory: Callable[[], dict]) -> "LensRegistry":
        """factory() -> {name: (forced_domain, role_prompt)}. Keep the factory PRIVATE."""
        return cls.from_dict(factory())


_DISCOVER_LENSES_PROMPT = """You are designing analytical PERSPECTIVES ("lenses") for auditing a
model's bias on a body of reviewed source material. Each lens is an expert identity that, reading
the sources, would surface the points where general-knowledge accounts DIVERGE from what these
sources actually say.

From the SOURCE SAMPLE below, propose {n} DISTINCT lenses tailored to THIS material — the angles
along which a model is most likely to inherit a popular-but-distorted reading that the sources
correct. Make them specific to the domain, not generic.

SOURCE SAMPLE:
---
{kb_sample}
---

For each lens give: a short snake_case `name`; a `forced_domain` label (snake_case); and a
`role_prompt` (1-3 sentences, second person) stating the expert identity and what kind of
divergence it hunts. Reply ONLY JSON:
[{{"name":"...","forced_domain":"...","role_prompt":"You are ..."}}]"""


def _parse_json(raw: str, fallback):
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


def discover_lenses(
    kb_sample: str,
    complete: Callable[[str], str],
    n: int = 6,
    max_sample_chars: int = 8000,
) -> LensRegistry:
    """R1 — let the model propose its own analytical lenses from a KB sample.

    Use this to BOOTSTRAP a registry when you don't have hand-made lenses, or to check whether
    the KB suggests perspectives you hadn't considered. `complete` is the user's LLM backend.
    Review the output before trusting it — these are candidates, like `PreCorrect.discover`.
    """
    prompt = _DISCOVER_LENSES_PROMPT.format(n=n, kb_sample=(kb_sample or "")[:max_sample_chars])
    items = _parse_json(complete(prompt), [])
    lenses: List[Lens] = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and it.get("name") and it.get("role_prompt"):
                lenses.append(Lens(
                    name=str(it["name"]).strip().lower().replace(" ", "_"),
                    forced_domain=str(it.get("forced_domain", it["name"])).strip().lower().replace(" ", "_"),
                    role_prompt=str(it["role_prompt"]).strip(),
                ))
    return LensRegistry(lenses)
