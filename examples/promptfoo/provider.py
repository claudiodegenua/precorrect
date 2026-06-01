"""promptfoo custom provider for precorrect.

Registers a Python provider that wraps precorrect around an Anthropic completion.
Set use_precorrect='true'/'false' via test vars to compare outputs side-by-side.
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from precorrect import PreCorrect, RuleSet

import anthropic

_client = anthropic.Anthropic()

def _complete(prompt: str) -> str:
    r = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text

_rules_path = Path(__file__).parent.parent.parent / "rules" / "example_generic.yaml"
_rules = RuleSet.from_file(str(_rules_path))
_pc = PreCorrect(complete=_complete)


def call_api(prompt: str, options: dict, context: dict) -> dict:
    """Called by promptfoo for each test × provider combination."""
    use_pc = str(options.get("vars", {}).get("use_precorrect", "true")).lower() == "true"
    try:
        if use_pc:
            output = _pc.generate(prompt, rules=_rules)
        else:
            output = _complete(prompt)
        return {"output": output}
    except Exception as e:
        return {"error": str(e)}
