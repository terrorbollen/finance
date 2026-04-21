# Discussion

Open questions, architectural ideas, and deferred decisions. Not ready to be TASKS.md tickets yet.

Add entries with `/add-to-discussion`.

---

## GPU upgrade for walk-forward training speed
_2026-04-21 · status: decided: skip_

**Context:** Walk-forward training with 287 windows was running slowly. Explored whether a better GPU would meaningfully speed it up.

**Key points:**
- M4 Pro already uses Metal GPU via TensorFlow's built-in support
- The model is tiny (0.5 MB weights) — GPU is not the bottleneck
- LSTMs are sequentially dependent; they defeat GPU parallelism almost entirely
- The 287 windows are also sequential (expanding window) and can't be parallelised
- eGPU on Mac is a dead end — Apple dropped support in macOS Sonoma
- A cloud A100 would give ~2–3× at best for this model size
- Fastest lever: increase `step_days` to halve window count, or switch architecture

**Conclusion:** Not worth pursuing. Hardware is not the constraint here.

---

## TCN as a replacement backbone for LSTM
_2026-04-21 · status: deferred_

**Context:** Raised as a way to speed up walk-forward training and potentially improve accuracy. LSTM is slow due to sequential dependency; TCN uses dilated causal convolutions which parallelise well on GPU.

**Key points:**
- Migration scope is small: one `elif` branch in `SignalModel._build_model()`, ~15 lines
- The `backbone` parameter already exists — adding `"tcn"` is a natural extension
- Full residual TCN (more accurate) is ~50 lines, still entirely within `signal_model.py`
- No changes needed outside `signal_model.py` — training, backtesting, inference all unaffected
- Literature is mixed on TCN vs LSTM for financial time series — neither dominates consistently
- With only 8 features and seq_len=30, architecture matters less than feature/label quality
- No new dependencies needed (`Conv1D` is built into Keras)

**Conclusion:** Worth doing eventually, but only after establishing a clean baseline from the current walk-forward run. If the model is generating too many HOLDs or losing trades, the problem is likely features or label thresholds — not architecture. Revisit once backtest results are in hand.

---
