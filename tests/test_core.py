"""Unit tests for precorrect — no API key needed (the model is mocked)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from precorrect import PreCorrect, Rule, RuleSet


# ─── RuleSet parsing ──────────────────────────────────────────────────────────

def test_ruleset_from_list_strings():
    rs = RuleSet.from_list(["rule one", "rule two"])
    assert len(rs.rules) == 2
    assert rs.rules[0].text == "rule one"
    assert rs.rules[0].triggers == []
    assert rs.rules[0].severity == "info"


def test_ruleset_from_list_dicts():
    rs = RuleSet.from_list([{"text": "r", "triggers": ["kw"], "severity": "critical"}])
    assert rs.rules[0].triggers == ["kw"]
    assert rs.rules[0].severity == "critical"


def test_ruleset_ignores_invalid():
    rs = RuleSet.from_list([{"no_text": "x"}, "", "valid"])
    # empty string still becomes a Rule(text=""); dict without text is skipped
    texts = [r.text for r in rs.rules]
    assert "valid" in texts
    assert all("no_text" not in (r.text or "") for r in rs.rules)


# ─── Rule trigger matching ────────────────────────────────────────────────────

def test_rule_applies_no_triggers_always():
    assert Rule("x").applies_to("anything at all")


def test_rule_applies_trigger_match():
    r = Rule("x", triggers=["yoke", "torah"])
    assert r.applies_to("explain the YOKE of the kingdom")  # case-insensitive
    assert not r.applies_to("explain the meaning of grace")


# ─── Preamble construction ────────────────────────────────────────────────────

def test_build_preamble_empty():
    pc = PreCorrect(complete=lambda p: p)
    assert pc.build_preamble([]) == ""


def test_build_preamble_orders_critical_first():
    pc = PreCorrect(complete=lambda p: p)
    rules = [Rule("info rule", severity="info"), Rule("crit rule", severity="critical")]
    out = pc.build_preamble(rules)
    assert out.index("crit rule") < out.index("info rule")
    assert "correction rules" in out.lower()


# ─── generate() injects before the prompt ─────────────────────────────────────

def test_generate_injects_preamble_before_prompt():
    captured = {}

    def fake_complete(prompt: str) -> str:
        captured["prompt"] = prompt
        return "OK"

    pc = PreCorrect(complete=fake_complete)
    rules = RuleSet.from_list(["never overgeneralize"])
    pc.generate("What is X?", rules=rules)

    full = captured["prompt"]
    assert "never overgeneralize" in full
    assert full.index("never overgeneralize") < full.index("What is X?")


def test_generate_no_rules_passes_prompt_unchanged():
    captured = {}
    pc = PreCorrect(complete=lambda p: captured.setdefault("p", p) or "OK")
    pc.generate("bare prompt")
    assert captured["p"] == "bare prompt"


def test_generate_only_applicable_rules_injected():
    captured = {}
    pc = PreCorrect(complete=lambda p: captured.setdefault("p", p) or "OK")
    rules = RuleSet.from_list([
        {"text": "torah rule", "triggers": ["torah"]},
        {"text": "grace rule", "triggers": ["grace"]},
    ])
    pc.generate("explain the torah", rules=rules)
    assert "torah rule" in captured["p"]
    assert "grace rule" not in captured["p"]


# ─── discover() parses model JSON ─────────────────────────────────────────────

def test_discover_parses_json():
    fake = '[{"text": "rule a", "triggers": ["x"]}, {"text": "rule b"}]'
    pc = PreCorrect(complete=lambda p: fake)
    rs = pc.discover("any topic", n=2)
    assert len(rs.rules) == 2
    assert rs.rules[0].text == "rule a"


def test_discover_handles_garbage():
    pc = PreCorrect(complete=lambda p: "sorry, no JSON here")
    rs = pc.discover("topic")
    assert rs.rules == []


# ─── generate_gated() — inject only when self-consistency is low ───────────────

def test_generate_gated_high_consistency_skips_injection():
    seen = []
    def fake(p):
        seen.append(p)
        return "10" if "CONSISTENT" in p else "bare answer"
    pc = PreCorrect(complete=fake)
    out = pc.generate_gated("Q?", rules=RuleSet.from_list(["RULETEXT"]), probe_n=2)
    assert out == "bare answer"
    assert not any("RULETEXT" in p for p in seen)   # no injection happened


def test_generate_gated_low_consistency_injects():
    seen = []
    def fake(p):
        seen.append(p)
        return "0" if "CONSISTENT" in p else "ans"
    pc = PreCorrect(complete=fake)
    pc.generate_gated("Q?", rules=RuleSet.from_list(["RULETEXT"]), probe_n=2)
    assert any("RULETEXT" in p for p in seen)        # rules were injected


# ─── RuleSet.from_file (.json — no optional dep) ──────────────────────────────

def test_ruleset_from_file_json(tmp_path):
    p = tmp_path / "r.json"
    p.write_text('{"rules": [{"text": "a", "triggers": ["x"]}, "b"]}', encoding="utf-8")
    rs = RuleSet.from_file(str(p))
    assert len(rs.rules) == 2
    assert rs.rules[0].triggers == ["x"]
    assert rs.rules[1].text == "b"


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
