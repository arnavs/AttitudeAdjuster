"""
Deep CFR training loop.

Networks:
  value_betting_net   - single shared net, trained every TRAIN_EVERY iters on
                        regret samples from both players (position encoded in input)
  strategy_betting_net - one per player, trained once at end on strategy samples

Discards are handled by a heuristic (best-equity keep-pair), not learned.

Usage:
  cd ~/MachineYearning
  python submission/train.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from multiprocessing import Pool
from torch.utils.tensorboard import SummaryWriter

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network import (
    make_betting_net,
    ReservoirBuffer, StrategyBuffer,
    train_value_network, train_strategy_network,
)
from traversal import run_traversal

# ── hyperparameters ───────────────────────────────────────────────────────────
N_ITERATIONS    = 50_000
K_TRAVERSALS    = 100         # per iteration per player (one per core)
TRAIN_EVERY     = 10        # retrain value nets every N iterations
VALUE_BET_BUF   = 1_500_000
STRAT_BET_BUF   = 1_500_000
BATCH_SIZE      = 1024
N_TRAIN_STEPS   = 400
N_CORES         = 8
LR              = 1e-3
SAVE_EVERY      = 50
SAVE_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
WARM_CKPT       = None # os.path.join(SAVE_DIR, "old_start.pt")  # set to None to train from scratch

os.makedirs(SAVE_DIR, exist_ok=True)


# ── worker (subprocess) ───────────────────────────────────────────────────────

def _worker(args):
    """Run one traversal in a subprocess. Returns buffer contents."""
    (traverser, iteration, vb_state) = args

    import sys, os
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(submission_dir)
    sys.path.insert(0, submission_dir)
    sys.path.insert(0, repo_root)

    import torch
    from network import make_betting_net, ReservoirBuffer, StrategyBuffer
    from traversal import run_traversal

    vb_net = make_betting_net()
    vb_net.load_state_dict(vb_state)
    vb_net.eval()
    vb_nets = [vb_net, vb_net]  # same net for both positions

    vb_buf = ReservoirBuffer(10_000)
    sb_bufs = [StrategyBuffer(10_000), StrategyBuffer(10_000)]

    run_traversal(
        traverser,
        vb_nets,
        vb_buf, sb_bufs,
        iteration,
    )

    return (
        traverser,
        vb_buf.buffer,
        sb_bufs[0].buffer,
        sb_bufs[1].buffer,
    )


def _merge(main_buf, new_items):
    """Add new_items into main_buf respecting reservoir sampling."""
    for item in new_items:
        main_buf.n_seen += 1
        if len(main_buf.buffer) < main_buf.capacity:
            main_buf.buffer.append(item)
        else:
            idx = np.random.randint(0, main_buf.n_seen)
            if idx < main_buf.capacity:
                main_buf.buffer[idx] = item


# ── main ──────────────────────────────────────────────────────────────────────

def train():
    device = 'cpu'

    vb_net = make_betting_net().to(device)
    vb_net.eval()

    # warm-start value net from a prior checkpoint if available
    if WARM_CKPT and os.path.exists(WARM_CKPT):
        ckpt = torch.load(WARM_CKPT, map_location=device, weights_only=True)
        vb_net.load_state_dict(ckpt["vb_net"])
        print(f"Warm-started value net from {WARM_CKPT}")

    vb_opt = optim.Adam(vb_net.parameters(), lr=LR)

    vb_buf = ReservoirBuffer(VALUE_BET_BUF)
    sb_bufs = [StrategyBuffer(STRAT_BET_BUF),  StrategyBuffer(STRAT_BET_BUF)]

    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, "runs"))

    print(f"Deep CFR: {N_ITERATIONS} iters | "
          f"{K_TRAVERSALS} traversals/iter | "
          f"{N_CORES} cores | "
          f"train every {TRAIN_EVERY} iters")

    t0 = time.time()

    with Pool(processes=N_CORES) as pool:
        for iteration in range(1, N_ITERATIONS + 1):

            for traverser in [0, 1]:
                jobs = [
                    (traverser, iteration, vb_net.state_dict())
                    for _ in range(K_TRAVERSALS)
                ]
                results = pool.map(_worker, jobs)

                for (_, vb_items, sb0_items, sb1_items) in results:
                    _merge(vb_buf, vb_items)
                    _merge(sb_bufs[0], sb0_items)
                    _merge(sb_bufs[1], sb1_items)

            # retrain value network
            if iteration % TRAIN_EVERY == 0:
                if len(vb_buf) >= BATCH_SIZE:
                    vb_net = make_betting_net().to(device)
                    vb_opt = optim.Adam(vb_net.parameters(), lr=LR)
                    vb_loss = train_value_network(vb_net, vb_buf, vb_opt,
                                        BATCH_SIZE, N_TRAIN_STEPS, device)
                    vb_net.eval()
                    if vb_loss is not None:
                        writer.add_scalar("loss/value_betting", vb_loss, iteration)

                writer.add_scalar("buffer/vb", len(vb_buf), iteration)
                for p in [0, 1]:
                    writer.add_scalar(f"buffer/sb_p{p}", len(sb_bufs[p]), iteration)

                elapsed = time.time() - t0
                rate    = iteration / elapsed
                eta     = (N_ITERATIONS - iteration) / rate
                writer.add_scalar("speed/it_per_s", rate, iteration)
                print(f"iter {iteration:6d} | "
                      f"vb={len(vb_buf)} | "
                      f"{rate:.1f} it/s | ETA {eta/3600:.1f}h")

            if iteration % SAVE_EVERY == 0:
                torch.save({
                    "iteration": iteration,
                    "vb_net": vb_net.state_dict(),
                    "vb_buf": vb_buf.buffer,
                    "sb_buf_0": sb_bufs[0].buffer, "sb_buf_1": sb_bufs[1].buffer,
                }, os.path.join(SAVE_DIR, f"checkpoint_{iteration}.pt"))
                print(f"  saved checkpoint_{iteration}.pt")

    # train final strategy networks
    print("\nTraining final strategy networks...")
    for p in [0, 1]:
        net = make_betting_net().to(device)
        opt = optim.Adam(net.parameters(), lr=LR)
        loss = train_strategy_network(net, sb_bufs[p], opt, BATCH_SIZE, 2000, device)
        path = os.path.join(SAVE_DIR, f"strategy_betting_p{p}_final.pt")
        torch.save(net.state_dict(), path)
        loss_str = f"{loss:.4f}" if loss is not None else "skipped (buffer too small)"
        print(f"  p{p} betting strategy loss={loss_str} -> {path}")

    print("Done.")


if __name__ == "__main__":
    train()
