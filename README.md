# ContextFlow — Learning Importance in Sequential Data (From Scratch)

ContextFlow is a lightweight **sequence context modeling** project built for clarity and interpretability.

Instead of generating text or predicting the next token, ContextFlow answers a different question:

> **Which moments in an ordered input most influence a decision?**

It is useful for tasks like:
- time-series signal detection (finance / operations)
- event-sequence risk scoring
- anomaly / pattern discovery in ordered logs

## What you get
- A compact, well-commented model (`contextflow/`) that produces:
  - a prediction (class probabilities)
  - an **influence map** over time steps (interpretable weights)
- A synthetic dataset that mimics “hidden signal event” detection
- Simple training + inference scripts

## Quickstart

```bash
pip install -r requirements.txt
python run.py train
python run.py infer --ckpt runs/contextflow_ckpt.pt
```

## Project structure

```
contextflow/
  config.py      # all knobs in one place
  data.py        # synthetic sequence dataset
  model.py       # ContextFlow model
  utils.py       # helpers
train.py         # training loop
infer.py         # show which steps mattered
run.py           # one-command launcher
```

## Notes
- This repo is designed as a **learning + portfolio** project.
- The model is intentionally small and readable.
- Swap the synthetic dataset with your own CSV/Parquet sequence data to adapt it to real use-cases.
