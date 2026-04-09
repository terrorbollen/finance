---
name: review-project
description: Comprehensive quality review of the finance trading-signal project. Checks ML/trading best practices, invariant compliance, code structure, test coverage, and known anti-patterns. Use when you want an honest audit of project health.
allowed-tools: Read, Grep, Glob, Bash
---

# Project Quality Review

You are performing a thorough, honest quality audit of this finance/ML/trading project.
Be direct. Identify real problems — do not pad with praise. Organise findings by severity:
**CRITICAL** (silent bugs, data leakage, broken invariants) → **HIGH** (best-practice violations, missing tests) → **MEDIUM** (maintainability, structure) → **LOW** (style, minor).

Work through every section below. Read the actual code, do not guess.

---

## 1. ML / Trading correctness (most important)

Look for violations of these known anti-patterns in ML for trading:

### Lookahead / data leakage
- Are features computed using future data? (e.g. normalisation using the full window, rolling stats that bleed future values)
- Are labels derived from prices that overlap with the feature window?
- Is there shuffle=True anywhere on time-series data? (violates INVARIANTS.md)
- Walk-forward: are train/val/test folds strictly non-overlapping with a purge gap?

### Normalisation
- Is inference normalisation using training-set statistics (from `signal_model_config.json`), NOT live-window statistics?
- Check `signals/generator.py` and `backtesting/backtester.py` for the fallback path described in INVARIANTS.md.

### Label encoding
- Is the BUY=0, HOLD=1, SELL=2 encoding consistent across `SignalModel`, `SignalGenerator`, and `Backtester`?

### Class imbalance handling
- Is focal loss and sample weights never used together? (INVARIANTS.md)
- Read `models/training.py` to verify.

### Feature/model config integrity
- Do feature columns at inference always match `feature_columns` in the saved config?
- Does adding/changing features trigger a config regeneration + retrain?

### Calibration
- Is the calibrator refitted after every training run?
- Does calibration horizon match one of the model's `prediction_horizons`?

### Holdout discipline
- Is `strict_holdout` always True in non-debug paths?
- Does the backtester actually raise if start_date < holdout_start_date?

### Module coupling
- Does `signals/` import from `backtesting/` or vice versa? (violates INVARIANTS.md)
  ```bash
  grep -r "from backtesting" signals/
  grep -r "from signals" backtesting/
  ```

---

## 2. Code quality and structure

### Architecture compliance
- Read `ARCHITECTURE.md` and verify each module only talks to its allowed dependencies.
- Are there any circular imports?
- Are there any large functions (>60 lines) that should be decomposed?

### Error handling
- Are there bare `except:` or `except Exception:` clauses that swallow real errors?
- Is validation only at system boundaries (CLI args, external API calls), not internal?

### Type safety
- Run a scan for untyped function signatures or `Any` escapes that hide real type errors:
  ```
  Grep pattern: def \w+\([^)]*\)\s*[^-]  (missing return type)
  ```
- Check for `# type: ignore` comments — each should be justified.

### Dead code / tech debt
- Are there TODO/FIXME comments that represent real unresolved issues?
  ```
  Grep pattern: TODO|FIXME|HACK|XXX
  ```
- Are there unused imports or unreferenced functions?

### Constants and magic numbers
- Are raw numeric thresholds (e.g. `0.1`, `252`, `0.05`) defined as named constants or documented inline?

---

## 3. Testing

- List all test files under `tests/` and the modules they cover.
- Identify modules with **no test coverage**.
- Check whether tests use realistic fixtures or mock internals in ways that decouple from production behaviour (violates CONTRIBUTIONS.md).
- Are there tests for the invariants listed in INVARIANTS.md? (data leakage, label encoding, normalisation source)
- Are there tests for edge cases: empty data, all-HOLD model output, missing config file?
- Run a quick file count: how many test files vs source files?

---

## 4. Guidelines compliance

Check the project follows its own documented standards:

- **CONTRIBUTIONS.md**: no new abstractions unless required; no unnecessary error handling; minimal changes.
- **CLAUDE.md**: task board, scope discipline, CHANGELOG entries.
- **TASKS.md**: are all tasks properly tracked? Are any stale `in_progress` tasks left unclaimed?
  Read TASKS.md and report any orphaned locks or tasks.
- **CHANGELOG.md**: does it have recent entries matching recent commits?

---

## 5. Dependency and environment hygiene

- Read `pyproject.toml` or `uv.lock` and flag any pinned-to-exact or very old versions of key libraries (tensorflow, scikit-learn, mlflow).
- Are dev dependencies (pytest, ruff, mypy) separated from runtime deps?
- Is there a `.python-version` that matches what `uv` resolves?

---

## 6. ML literature / best-practice checklist

Evaluate against accepted standards for ML-driven trading systems:

| Practice | Compliant? | Notes |
|---|---|---|
| No future leakage in features | ? | Check above |
| Walk-forward cross-validation | ? | Check `models/walk_forward.py` |
| Out-of-sample holdout (not used in any tuning) | ? | Check INVARIANTS.md compliance |
| Statistical significance of win rate (binomial test) | ? | Check `backtesting/metrics.py` |
| Transaction cost modelling (commission + slippage) | ? | Check `backtesting/backtester.py` |
| Multiple-ticker validation (not cherry-picked) | ? | Check STRATEGY.md compliance |
| Confidence calibration (isotonic regression) | ? | Check `signals/calibration.py` |
| Position sizing (Kelly or fraction thereof) | ? | Check `signals/generator.py` |
| Regime awareness (trend filter / ADX) | ? | Check signal pipeline |
| Known limitations documented | ? | Check STRATEGY.md |
| Ensemble or uncertainty quantification | ? | Single model is a known limitation |

---

## Output format

Produce a structured report with these sections:

### Summary
One paragraph overall assessment: what is the project's biggest strength, and its biggest risk right now.

### Critical findings
Bullet list. Each item: module:line (if known), what the problem is, why it matters, suggested fix.

### High findings
Same format.

### Medium findings
Same format.

### Low / style findings
Brief list only.

### ML/trading best-practice scorecard
Fill in the table from section 6 with Compliant / Partial / No for each row, plus a one-line note.

### Recommended next actions
Top 3–5 things to do, ordered by impact on signal reliability.
