import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from contextflow import ContextFlow, ContextFlowConfig
from contextflow.data import SyntheticSequenceDataset, SyntheticSequenceSpec
from contextflow.utils import set_seed, pick_device, accuracy_from_logits, save_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_path", type=str, default="runs/contextflow_ckpt.pt")
    args = ap.parse_args()

    cfg = ContextFlowConfig()
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.device is not None: cfg.device = args.device

    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    spec = SyntheticSequenceSpec(max_len=cfg.max_len, input_dim=cfg.input_dim)
    train_ds = SyntheticSequenceDataset(n_samples=12000, spec=spec, seed=cfg.seed)
    val_ds   = SyntheticSequenceDataset(n_samples=2000, spec=spec, seed=cfg.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = ContextFlow(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_acc = 0.0
        total_loss = 0.0

        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits, _ = model(x, mask=mask)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_acc  += accuracy_from_logits(logits, y)

        # Validation
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for x, mask, y in val_loader:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                logits, _ = model(x, mask=mask)
                val_acc += accuracy_from_logits(logits, y)

        val_acc /= len(val_loader)
        train_acc = total_acc / len(train_loader)
        train_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint(args.save_path, model, opt, meta={"best_val_acc": best_val, "epoch": epoch})
            print(f"  ✓ saved checkpoint -> {args.save_path}")

    print(f"Done. Best val_acc={best_val:.4f}")

if __name__ == "__main__":
    main()
