# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/); this project follows
[Semantic Versioning](https://semver.org/).

## [0.1.0] — Unreleased

Initial release.

### Added
- `PreCorrect.generate` / `generate_gated` — inject correction rules into a prompt *before*
  generation (gated = inject only where the model is internally inconsistent, no answer-key needed).
- `RuleSet` / `Rule` — bring-your-own rules from a list, a `.yaml`, or a `.json` file.
- `discover_lenses`, `discover_from_corpus`, `discover_from_kb` — bootstrap candidate correction
  rules from your own reviewed corpus.
- `build_evaluator_from_kb` / `Evaluator` — derive a vertical answer-key (benchmark) from your KB
  and score any model output against it (test a model on your vertical, or gate your own generation).
- `rules/critical_reasoning.yaml` — a 12-rule, domain-agnostic base ruleset.
- promptfoo integration example and an offline benchmark harness.
