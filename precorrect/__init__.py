"""precorrect — debias an LLM *before* generation, not after."""

from .core import Complete, PreCorrect, Rule, RuleSet
from .lens import Lens, LensRegistry, discover_lenses
from .discover import discover_from_kb, discover_from_corpus
from .evaluate import (
    Evaluator, KeyEntry, EvalResult, extract_answer_key, build_evaluator_from_kb,
)

__all__ = [
    "PreCorrect", "RuleSet", "Rule", "Complete",
    "Lens", "LensRegistry", "discover_lenses",
    "discover_from_kb", "discover_from_corpus",
    "Evaluator", "KeyEntry", "EvalResult", "extract_answer_key", "build_evaluator_from_kb",
]
__version__ = "0.1.0"
