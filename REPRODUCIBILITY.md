# Reproducibility Notes

This repository is a research codebase with both validated tests and many historical experiment scripts. Reproducibility is therefore best approached in layers.

## 1. Recommended starting point

For the most defensible current interpretation of the repository:

1. Read `README.md`
2. Read `CURRENT_STATE.md` (canonical results table from the evaluation harness)
3. Read `docs/PHASE0_AUDIT.md` (which earlier numbers were retracted, and why)
4. Read `RESEARCH.md` (scope index and links to canonical docs)
5. Read `docs/HONEST_ASSESSMENT.md`

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

The canonical entry points (all write JSON records to `results/`):

```bash
# single configuration; see --help for methods (fp16, fp32, rtn, gptq, ternary, binary, binary_svid, legacy_*)
python -m onebit.eval.harness --method rtn --bits 4 --group 128

# the full Phase 0 baseline suite (resumable) and its markdown table
python scripts/run_phase0_baselines.py
python scripts/run_phase0_baselines.py --table

# KD-QAT proof-of-concept (CPU works but is slow; use a GPU for real runs)
python scripts/onebit_kd_poc.py --help

# unit tests
pytest tests/test_phase0_correctness.py tests/test_bpp_guard.py tests/test_bpp_strict.py -q
```

Runtime on a 2-core CPU: ~8 minutes per configuration on the full WikiText-2 test set (~280 windows of 1024 tokens);
GPTQ adds a few minutes for calibration. The full suite takes ~2–3 hours.

Older test-oriented files under `tests/` (e.g. `tests/test_improvements.py`, `tests/comprehensive_real_test.py`) now use the
harness for perplexity, but their numbers are not canonical.

## 4. What requires more care

Some scripts in `onebit/research/` and at repo root are exploratory artifacts rather than polished entry points. They may assume:

- locally cached Hugging Face models or datasets,
- long runtimes,
- GPU availability,
- optional OpenCL support,
- or a particular researcher workflow.

Historical paper-draft materials under `onebit/research/paper/` should not be treated as the final authoritative interpretation when they conflict with later rigorous docs.

## 5. Reporting standards for reruns

Rules (see also [`docs/PHASE0_AUDIT.md`](docs/PHASE0_AUDIT.md)):

- **Only harness numbers go in docs.** Perplexity is computed by `onebit/eval/harness.py` on the full WikiText-2-raw test set
  (`--max-windows` is for debugging only and is recorded in the run name as `_winN`).
- **Cite the JSON record** (`results/.../<run_name>.json`); it stores the commit, dirty flag, config, library versions and both bpp figures.
- **State bpp both ways**: `bpp_linear` and `bpp_whole`. Every scale, zero-point, codebook entry and low-rank factor is counted.
- **Compare against correct baselines** at matched bpp (RTN g128, GPTQ) — not against strawmen.
- **Never hardcode** result tables in scripts; generate them from JSON.
- Correlation / MSE may be logged as diagnostics but are not evidence of LM quality.
- Training runs must report training data, token count, seeds and hardware, and must not train on the test split.

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
