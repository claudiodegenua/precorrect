"""Unit tests for precorrect.lens — no API key needed (the model is mocked)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from precorrect.lens import Lens, discover_lenses


# ─── Lens.render — must NOT crash on literal braces in the user role_prompt ────

def test_render_tolerates_braces_in_role_prompt():
    lens = Lens(name="x", forced_domain="d", role_prompt="You hunt {weird} braces } and {.")
    out = lens.render(topic="my topic", kb_content="some source text", kb_filename="f.txt")
    assert "You hunt {weird} braces } and {." in out   # raw role_prompt survives untouched
    assert "my topic" in out                            # base placeholders are filled
    assert "f.txt" in out


def test_render_substitutes_base_placeholders():
    lens = Lens(name="lensname", forced_domain="dom", role_prompt="You are an expert.")
    out = lens.render(topic="T", kb_content="KBCONTENT")
    assert "lensname" in out and "dom" in out and "KBCONTENT" in out


# ─── discover_lenses — parse model JSON → normalized LensRegistry ──────────────

def test_discover_lenses_parses_and_normalizes():
    fake = '[{"name":"My Lens","forced_domain":"Some Domain","role_prompt":"You are X."}]'
    reg = discover_lenses("sample kb", lambda p: fake, n=1)
    assert len(reg) == 1
    lens = list(reg)[0]
    assert lens.name == "my_lens"            # snake_case normalized
    assert lens.forced_domain == "some_domain"


def test_discover_lenses_garbage_returns_empty():
    reg = discover_lenses("sample kb", lambda p: "no json here", n=3)
    assert len(reg) == 0


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
