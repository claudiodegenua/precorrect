# Contributing

Thanks for your interest in **precorrect**.

## Dev setup

```bash
git clone https://github.com/claudiodegenua/precorrect
cd precorrect
pip install -e ".[yaml]"
python -m pytest -q
```

## Guidelines

- Keep the shipped package **dependency-free** — the optional `yaml` extra is the only exception.
  The user supplies the LLM backend (`complete: Callable[[str], str]`); precorrect never imports a vendor SDK.
- Domain-specific rule sets and lens identities are intentionally **out of scope** — see
  `rules/template.yaml` for the expected format. Ship the *mechanism*, not tuned content.
- Add a test under `tests/` for any new public behavior; the model is mocked, so tests need no API key.
- For new features, open an issue first to discuss scope.

MIT licensed — see `LICENSE`.
