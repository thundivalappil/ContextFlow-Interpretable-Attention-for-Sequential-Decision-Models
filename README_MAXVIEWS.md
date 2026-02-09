# ContextFlow 🚦 — Interpretable Attention for Sequential Decisions (PyTorch)

**ContextFlow** is a compact, portfolio-ready project that learns **which time steps mattered most** in a sequence — and returns an **influence map** (attention-style weights) alongside predictions.

> Think: *“Attention, but for decisions on time-series / event streams.”*

---

## Why this repo gets views
- ✅ **Modern attention mechanics (Q·Kᵀ / softmax / weights) with interpretability-first outputs**
- ✅ **One-command run** (`train` + `infer`)
- ✅ **Clear, hackable codebase** (easy to replace synthetic data with real CSV / logs)
- ✅ **Business-ready use cases** (finance risk signals, ops anomaly scoring, audit trails)

---

## What you can do
- Detect a hidden “signal event” in a noisy sequence
- Score sequences (binary classification) and **explain** the score by time-step influence
- Replace the synthetic dataset with:
  - stock/order events
  - payment/fraud event sequences
  - machine sensor time-series
  - server logs / clickstream events

---

## Demo (after training)

```bash
python run.py train
python run.py infer --ckpt runs/contextflow_ckpt.pt --topk 5
```

Example output:
```text
P(class=1) = 0.93
Most influential time steps:
  t=07  weight=0.22
  t=08  weight=0.19
  t=06  weight=0.17
  ...
```

---

## Install

```bash
pip install -r requirements.txt
```

---

## Project structure

```
contextflow/
  config.py      # all knobs in one place
  data.py        # synthetic sequence dataset (replace with your own)
  model.py       # ContextFlow (attention-style influence)
  utils.py       # helpers
train.py         # training loop (AdamW, clipping, checkpoint)
infer.py         # single inference + top-k influential steps
run.py           # one-command launcher
```

---

## How it works (high level)

1. **Input**: a sequence `x` shaped `[B, T, D]` and a boolean `mask` for valid time steps  
2. **Context scoring**: the model learns a distribution `w` over time steps (sums to 1)  
3. **Pooling**: context vector = weighted sum of time steps  
4. **Prediction**: classifier head returns logits for class 0/1  
5. **Explainability**: you get `w` back as the *influence map*

---

## Training tips to boost accuracy / reduce loss (quick checklist)
- Increase dataset size (synthetic: raise `n_samples`)  
- Train longer (more `epochs`)  
- Tune `lr` (most common reason for bad loss curves)  
- Add LR scheduler (Cosine / OneCycle)  
- Add label smoothing (stability)  
- Add early stopping (prevent overfit)  
- Use mixed precision on GPU (faster)  

---

## “Full Transformer features” roadmap (optional upgrades)
If you want this repo to read like a *Transformer-complete* mini project, add:
- **Positional encoding** (sin/cos or learned)
- **Multi-head attention** (several heads + concat)
- **FFN + Add&Norm blocks**
- Optional **causal mask** (for autoregressive mode)

These upgrades keep it toy-sized but “modern”.

---

## License
MIT (recommended for maximum adoption/views).
