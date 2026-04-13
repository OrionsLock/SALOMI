# Reproducibility Notes

This repository is a research codebase with both validated tests and many historical experiment scripts. Reproducibility is therefore best approached in layers.

## 1. Recommended starting point

For the most defensible current interpretation of the repository:

1. Read `README.md`
2. Read `RESEARCH.md` (scope index and links to canonical docs)
3. Read `docs/HONEST_ASSESSMENT.md`
4. Read `docs/PROJECT_ANALYSIS_SUMMARY.md`

These documents explain which claims remained credible after stricter evaluation.

## 2. Environment

Recommended baseline environment:

- Python 3.10+
- `pip install -r requirements.txt`

Core dependencies currently documented:

- `numpy`
- `torch`
- `tqdm`
- `transformers`
- `datasets`
- `matplotlib`
- `pytest`
- `pyopencl` for optional backend work

## 3. What is easiest to rerun

The easiest entry points are the test-oriented files under `tests/`.

Examples:

- `tests/test_bpp_strict.py`
- `tests/test_correlation_e2e.py`
- `tests/test_perplexity_real.py`
- `tests/test_speed_benchmark.py`

There are also phased runners:

- `tests/run_phase1_tests.py`
- `tests/run_phase2_tests.py`
- `tests/experiments/run_phase3_experiments.py`

## 4. What requires more care

Some scripts in `onebit/research/` and at repo root are exploratory artifacts rather than polished entry points. They may assume:

- locally cached Hugging Face models or datasets,
- long runtimes,
- GPU availability,
- optional OpenCL support,
- or a particular researcher workflow.

Historical paper-draft materials under `onebit/research/paper/` should not be treated as the final authoritative interpretation when they conflict with later rigorous docs.

## 5. Reporting standards for reruns

When reproducing or extending results, report at minimum:

- exact commit hash,
- script or test used,
- model and dataset split,
- whether the metric is per-layer correlation, end-to-end correlation, or perplexity,
- full bpp accounting including overhead,
- hardware/backend used,
- and any deviations from the documented defaults.

## 6. Known limits

- This repository does not yet present a fully locked package/environment story.
- Some “production” naming in the codebase still reflects research-stage interfaces.
- Several older files preserve superseded or more optimistic interpretations for transparency.

## 7. Best practice

If you are trying to validate the repo for external readers, prefer the combination of:

- `README.md`
- `RESEARCH.md` (index of questions and evidence locations)
- `docs/`
- `tests/`

over isolated historical scripts or result logs.
