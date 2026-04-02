# Contributing

Thanks for contributing to SALOMI.

This repository is best understood as a **research codebase** with a mix of:

- core package code,
- validation tests,
- exploratory experiment scripts,
- and historical documents preserved for transparency.

## Contribution priorities

The most valuable contributions are:

- clearer evaluation and reproducibility,
- stricter bpp accounting,
- better end-to-end perplexity validation,
- runtime/backend correctness and performance improvements,
- documentation cleanup that reduces narrative ambiguity,
- and test coverage for actively used code paths.

## Before opening a PR

Please try to:

1. explain whether your change affects research claims, engineering quality, or both,
2. say which files contain the authoritative interpretation of the change,
3. note whether the change impacts strict bpp accounting or perplexity results,
4. run the most relevant tests you can,
5. avoid presenting exploratory metrics as final conclusions without caveats.

## Repository hygiene

Please keep these distinctions clear:

- `docs/` and `RESEARCH.md` should reflect the most defensible current narrative
- `tests/` should hold repeatable validation where possible
- `onebit/research/` may contain exploratory work, but new files should be named clearly and avoid overstating conclusions
- historical paper-style docs should not silently become the only source of truth

## Style guidance

- Prefer small, focused changes.
- Add or update tests when they materially reduce regression risk.
- Keep comments concise and useful.
- If a result depends on a narrow setup, say so explicitly.

## Claim discipline

For research-facing changes, please distinguish between:

- per-layer reconstruction metrics,
- end-to-end hidden-state metrics,
- language modeling quality such as perplexity,
- and strict storage accounting.

These are not interchangeable in this repository.

## If you are unsure

If a change improves an optimistic local metric but has not been validated end-to-end, document it as a promising experiment rather than a settled result.
