"""Unit tests for precorrect.evaluate — no API key needed (the model is mocked)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from precorrect import (
    build_evaluator_from_kb, extract_answer_key, Evaluator, KeyEntry, EvalResult,
)
from precorrect.lens import LensRegistry, Lens


def _stub(verdict="AFFIRM"):
    """A complete() that answers each pipeline stage by recognizing the prompt."""
    def complete(prompt: str) -> str:
        if "DISTINCT lenses" in prompt:                       # discover_lenses
            return '[{"name":"factual","forced_domain":"facts","role_prompt":"You are a fact checker."}]'
        if "probe questions" in prompt:                       # lens trap-question generation
            return ('[{"question":"Is a Venusian day longer than its year?",'
                    '"expected_answer":true,"source_ref":"facts.txt"}]')
        if "CORRECT POSITIONS" in prompt:                     # evaluate judge
            return f'[{{"i":0,"verdict":"{verdict}"}}]'
        return "[]"
    return complete


# ─── KeyEntry ────────────────────────────────────────────────────────────────

def test_keyentry_triggers_prefix_collapse():
    k = KeyEntry(question="Venusian rotation period", correct_answer=True)
    trig = k.triggers()
    assert "venus" in trig          # 'venusian' -> 5-char prefix 'venus'
    assert "rotat" in trig


# ─── build + extract ─────────────────────────────────────────────────────────

def test_build_evaluator_from_kb_extracts_key():
    ev = build_evaluator_from_kb(_stub(), {"facts.txt": "Venus day > year."}, topic="facts")
    assert isinstance(ev, Evaluator)
    assert len(ev) == 1
    assert ev.answer_key[0].correct_answer is True
    assert ev.answer_key[0].source_ref == "facts.txt"


def test_extract_answer_key_dedups():
    lenses = LensRegistry([Lens("a", "facts", "You are A."), Lens("b", "facts", "You are B.")])
    # both lenses yield the SAME question via the stub -> deduped to 1
    key = extract_answer_key(_stub(), {"d.txt": "text"}, "facts", lenses)
    assert len(key) == 1


# ─── evaluate ────────────────────────────────────────────────────────────────

def test_evaluate_affirm_scores_one():
    key = [KeyEntry("Is a Venusian day longer than its year?", True, "facts.txt")]
    ev = Evaluator(key, _stub("AFFIRM"))
    res = ev.evaluate("On Venus a day is longer than a year.")
    assert isinstance(res, EvalResult)
    assert res.affirmed == 1 and res.contradicted_n == 0
    assert res.score == 1.0


def test_evaluate_contradict_flags_position():
    key = [KeyEntry("Is a Venusian day longer than its year?", True, "facts.txt")]
    ev = Evaluator(key, _stub("CONTRADICT"))
    res = ev.evaluate("On Venus a day is shorter than its year.")
    assert res.contradicted_n == 1 and res.score == 0.0
    assert res.contradicted[0]["source_ref"] == "facts.txt"


def test_evaluate_no_addressed_returns_none():
    key = [KeyEntry("Is a Venusian day longer than its year?", True)]
    ev = Evaluator(key, _stub("NOT_ADDRESSED"))
    res = ev.evaluate("Completely unrelated text about cooking.")
    assert res.score is None and res.addressed == 0


# ─── persistence ─────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    key = [KeyEntry("Q?", False, "ref", "topic")]
    ev = Evaluator(key, _stub(), topic="topic")
    p = tmp_path / "key.json"
    ev.save(str(p))
    ev2 = Evaluator.load(str(p), _stub())
    assert len(ev2) == 1
    assert ev2.answer_key[0].correct_answer is False
    assert ev2.topic == "topic"
