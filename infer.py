"""Run a single inference and show which time steps mattered most."""
import argparse
import torch

from contextflow import ContextFlow, ContextFlowConfig
from contextflow.data import SyntheticSequenceSpec
from contextflow.utils import pick_device, load_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="runs/contextflow_ckpt.pt")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    cfg = ContextFlowConfig()
    device = pick_device(cfg.device)

    model = ContextFlow(cfg).to(device)
    load_checkpoint(args.ckpt, model)
    model.eval()

    # Build one sample (optionally with signal)
    spec = SyntheticSequenceSpec(max_len=cfg.max_len, input_dim=cfg.input_dim)
    x = torch.randn(1, spec.max_len, spec.input_dim)
    mask = torch.ones(1, spec.max_len, dtype=torch.bool)

    with torch.no_grad():
        logits, weights = model(x.to(device), mask=mask.to(device))
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        w = weights[0].cpu()

    top = torch.topk(w, k=min(args.topk, w.numel()))
    idxs = top.indices.tolist()
    vals = top.values.tolist()

    print(f"P(class=1) = {prob:.4f}")
    print("Most influential time steps:")
    for i, v in zip(idxs, vals):
        print(f"  t={i:02d}  weight={v:.4f}")

if __name__ == "__main__":
    main()
