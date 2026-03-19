"""Train strategy nets from a saved checkpoint's strategy buffers."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from network import (
    make_betting_net, make_discard_net,
    StrategyBuffer, train_strategy_network,
)

CKPT_PATH = sys.argv[1] if len(sys.argv) > 1 else "submission/checkpoints/checkpoint_latest.pt"
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
BATCH_SIZE = 1024
N_STEPS    = 2000
LR         = 1e-3
device     = 'cpu'

print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=device)

for p in [0, 1]:
    for tag, net_fn, buf_key in [
        ("betting", make_betting_net, f"sb_buf_{p}"),
        ("discard", make_discard_net, f"sd_buf_{p}"),
    ]:
        buf = StrategyBuffer(len(ckpt[buf_key]) + 1)
        buf.buffer = ckpt[buf_key]
        buf.n_seen = len(buf.buffer)

        print(f"\nTraining strategy_{tag}_p{p} on {len(buf)} samples...")
        if len(buf) < BATCH_SIZE:
            print(f"  Skipping: only {len(buf)} samples (need {BATCH_SIZE})")
            continue

        net = net_fn().to(device)
        opt = optim.Adam(net.parameters(), lr=LR)
        loss = train_strategy_network(net, buf, opt, BATCH_SIZE, N_STEPS, device)
        print(f"  Final loss: {loss:.4f}")

        path = os.path.join(OUT_DIR, f"strategy_{tag}_p{p}_final.pt")
        torch.save(net.state_dict(), path)
        print(f"  Saved: {path}")

print("\nDone.")
