"""Build a *vertical evaluator* from the user's own KB — the `evaluate` side of
discover → correct → **evaluate**.

The idea: the user's curated KB is their field's authority. Probing it through lenses yields
trap questions whose `expected_answer` is *the position the KB takes* on a contested point —
that set of (question, correct-answer, source_ref) IS a field-specific **answer-key**. An
`Evaluator` holds that key and scores any generated text against it: for each key position the
text actually addresses, does the text AFFIRM the correct position or CONTRADICT it?

100% domain-agnostic: the mechanism ships; the key is derived from the *user's* KB at build
time and never hard-coded here. This is the generic counterpart of a private, hand-tuned auditor
— the vertical evaluator the user gets from their own corpus, without us supplying the answers.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

from .lens import LensRegistry, discover_lenses

Complete = Callable[[str], str]


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


@dataclass
class KeyEntry:
    """One normative position the KB takes on a contested point (an answer-key item)."""
    question: str
    correct_answer: bool          # the answer ACCORDING TO THE KB
    source_ref: str = ""
    topic: str = ""

    def triggers(self) -> List[str]:
        """5-char-prefix content tokens, for cheap morphology-tolerant 'is this addressed?' matching
        (so 'Venus'/'Venusian', singular/plural collapse together)."""
        words = re.findall(r"[A-Za-zÀ-ú]{4,}", f"{self.question} {self.source_ref}".lower())
        return list(dict.fromkeys(w[:5] for w in words))


@dataclass
class EvalResult:
    score: Optional[float]                       # affirmed / (affirmed + contradicted); None if nothing addressed
    addressed: int = 0
    affirmed: int = 0
    contradicted_n: int = 0
    contradicted: List[dict] = field(default_factory=list)   # the key positions the output got WRONG
    n_key: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


_ADDR_OVERLAP = 1   # min shared prefix-tokens for the large-key pre-filter (judge makes the real call)


class Evaluator:
    """A field-specific evaluator: an answer-key (derived from the KB) + an `evaluate` method."""

    def __init__(self, answer_key: List[KeyEntry], complete: Complete, topic: str = ""):
        self.answer_key = answer_key
        self.complete = complete
        self.topic = topic

    def __len__(self) -> int:
        return len(self.answer_key)

    # --- which key positions might this output touch? (cheap prefix pre-filter for LARGE keys) ---
    def _addressed(self, output: str) -> List[int]:
        ow = {w[:5] for w in re.findall(r"[A-Za-zÀ-ú]{4,}", output.lower())}
        return [i for i, k in enumerate(self.answer_key)
                if len(ow & set(k.triggers())) >= _ADDR_OVERLAP]

    def evaluate(self, output: str, max_check: int = 40) -> EvalResult:
        """Score `output` against the KB-derived answer-key. One batched judge call.

        For a small key (<= max_check) every position is sent to the judge, which itself decides
        AFFIRM / CONTRADICT / NOT_ADDRESSED — the reliable arbiter. The keyword pre-filter only
        kicks in to cap the batch when the key is large.
        """
        idx = (list(range(len(self.answer_key))) if len(self.answer_key) <= max_check
               else self._addressed(output))[:max_check]
        if not idx:
            return EvalResult(score=None, n_key=len(self.answer_key))
        items = [self.answer_key[i] for i in idx]
        spec = "\n".join(
            f'K{j}: "{k.question}" — correct answer per the field: {"YES" if k.correct_answer else "NO"}'
            f'{" (" + k.source_ref + ")" if k.source_ref else ""}'
            for j, k in enumerate(items)
        )
        prompt = (
            "Below is an OUTPUT and a list of the field's CORRECT POSITIONS (an answer-key). "
            "For EACH position, judge ONLY against the OUTPUT:\n"
            "  AFFIRM  = the output is consistent with the correct position\n"
            "  CONTRADICT = the output asserts the opposite of the correct position\n"
            "  NOT_ADDRESSED = the output does not actually take a stance on this point\n"
            "Be strict: choose NOT_ADDRESSED unless the output clearly takes a side.\n\n"
            f"--- OUTPUT ---\n{output[:6000]}\n\n--- CORRECT POSITIONS ---\n{spec}\n\n"
            'Reply ONLY JSON: [{"i":0,"verdict":"AFFIRM|CONTRADICT|NOT_ADDRESSED"}]'
        )
        verdicts = _parse(self.complete(prompt), [])
        vmap = {}
        if isinstance(verdicts, list):
            for v in verdicts:
                if isinstance(v, dict) and "i" in v:
                    try:
                        vmap[int(v["i"])] = str(v.get("verdict", "")).strip().upper()
                    except (ValueError, TypeError):
                        pass
        affirmed = contradicted = 0
        contra: List[dict] = []
        for j, k in enumerate(items):
            verd = vmap.get(j, "NOT_ADDRESSED")
            if verd == "AFFIRM":
                affirmed += 1
            elif verd == "CONTRADICT":
                contradicted += 1
                contra.append({"question": k.question, "correct_answer": k.correct_answer,
                               "source_ref": k.source_ref})
        addressed = affirmed + contradicted
        score = round(affirmed / addressed, 3) if addressed else None
        return EvalResult(score=score, addressed=addressed, affirmed=affirmed,
                          contradicted_n=contradicted, contradicted=contra, n_key=len(self.answer_key))

    # --- persistence: extract the key once, reuse it ---
    def save(self, path: str) -> None:
        import pathlib
        data = {"topic": self.topic, "answer_key": [asdict(k) for k in self.answer_key]}
        pathlib.Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str, complete: Complete) -> "Evaluator":
        import pathlib
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        key = [KeyEntry(**k) for k in data.get("answer_key", [])]
        return cls(key, complete, topic=data.get("topic", ""))


def extract_answer_key(
    complete: Complete,
    kb_docs: Dict[str, str],
    topic: str,
    lenses: LensRegistry,
    max_kb_chars: int = 6000,
) -> List[KeyEntry]:
    """Probe the KB through lenses → the field's normative positions on contested points.

    Reuses the lens trap-question machinery: each question's `expected_answer` is the KB's own
    stance, and `source_ref` grounds it. No model self-probing here — we are extracting the KEY,
    not measuring bias. Review the result: these are candidate positions, curate before trusting.
    """
    key: List[KeyEntry] = []
    seen = set()
    for name, text in kb_docs.items():
        kb = (text or "")[:max_kb_chars]
        for lens in lenses:
            prompt = lens.render(topic, kb, kb_filename=name)
            for q in _parse(complete(prompt), []) or []:
                if not (isinstance(q, dict) and q.get("question")):
                    continue
                qtext = str(q["question"]).strip()
                sig = qtext.lower()
                if sig in seen:
                    continue
                seen.add(sig)
                key.append(KeyEntry(
                    question=qtext,
                    correct_answer=bool(q.get("expected_answer", True)),
                    source_ref=str(q.get("source_ref", "")).strip(),
                    topic=topic,
                ))
    return key


def build_evaluator_from_kb(
    complete: Complete,
    kb_docs: Dict[str, str],
    topic: str,
    lenses: Optional[LensRegistry] = None,
    n_lenses: int = 6,
    max_kb_chars: int = 6000,
) -> Evaluator:
    """Build a vertical evaluator from the user's KB.

    kb_docs = {doc_name: text}. If `lenses` is None, bootstrap them from a KB sample via
    `discover_lenses` (review them — they shape what the key covers). Returns an `Evaluator`
    whose answer-key encodes the field's correct positions; `.evaluate(output)` scores any text
    against it and flags the positions it contradicts.
    """
    if lenses is None or len(lenses) == 0:
        sample = "\n\n".join(list(kb_docs.values()))[:8000]
        lenses = discover_lenses(sample, complete, n=n_lenses)
    key = extract_answer_key(complete, kb_docs, topic, lenses, max_kb_chars=max_kb_chars)
    return Evaluator(key, complete, topic=topic)
