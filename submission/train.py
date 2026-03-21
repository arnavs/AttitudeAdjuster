"""
Deep CFR training loop.

Networks (per player):
  value_betting_net   - trained every TRAIN_EVERY iters on betting regret samples
  strategy_betting_net - trained once at end on betting strategy samples

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
from encoder import FOLD

# position feature is the first scalar after card encodings

# ── hyperparameters ───────────────────────────────────────────────────────────
N_ITERATIONS    = 50_000
K_TRAVERSALS    = 500      # per iteration per player (one per core)
TRAIN_EVERY     = 10        # retrain value nets every N iterations
VALUE_BET_BUF   = 3_000_000
STRAT_BET_BUF   = 3_000_000
BATCH_SIZE      = 1024
N_TRAIN_STEPS   = 10000
N_CORES         = 8
LR              = 1e-3
SAVE_EVERY      = 100
STRAT_EVERY     = 100       # train + save strategy nets every N iterations
SAVE_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
WARM_CKPT       = None # os.path.join(SAVE_DIR, "old_start.pt")  # set to None to train from scratch

os.makedirs(SAVE_DIR, exist_ok=True)


# ── worker (subprocess) ───────────────────────────────────────────────────────

def _worker(args):
    """Run one traversal in a subprocess. Returns buffer contents."""
    (traverser, iteration, vb_states) = args

    import sys, os
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(submission_dir)
    sys.path.insert(0, submission_dir)
    sys.path.insert(0, repo_root)

    import torch
    from network import make_betting_net, ReservoirBuffer, StrategyBuffer
    from traversal import run_traversal

    vb_nets = []
    for p in [0, 1]:
        vb_net = make_betting_net(); vb_net.load_state_dict(vb_states[p]); vb_net.eval(); vb_nets.append(vb_net)

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

    vb_nets = [make_betting_net().to(device), make_betting_net().to(device)]
    for net in vb_nets:
        net.eval()

    # warm-start value nets from a prior checkpoint if available
    if WARM_CKPT and os.path.exists(WARM_CKPT):
        ckpt = torch.load(WARM_CKPT, map_location=device, weights_only=True)
        for p in [0, 1]:
            vb_nets[p].load_state_dict(ckpt[f"vb_net_{p}"])
        print(f"Warm-started value nets from {WARM_CKPT}")

    vb_opts = [optim.Adam(n.parameters(), lr=LR) for n in vb_nets]

    vb_bufs = [ReservoirBuffer(VALUE_BET_BUF), ReservoirBuffer(VALUE_BET_BUF)]
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
                    (traverser, iteration,
                     [vb_nets[0].state_dict(), vb_nets[1].state_dict()])
                    for _ in range(K_TRAVERSALS)
                ]
                results = pool.map(_worker, jobs)

                for (_, vb_items, sb0_items, sb1_items) in results:
                    _merge(vb_bufs[traverser], vb_items)
                    _merge(sb_bufs[0], sb0_items)
                    _merge(sb_bufs[1], sb1_items)

            print(f"\r  iter {iteration} / {N_ITERATIONS} ({iteration % TRAIN_EVERY}/{TRAIN_EVERY} to next train)", end="", flush=True)

            # retrain value networks
            if iteration % TRAIN_EVERY == 0:
                for p in [0, 1]:
                    if len(vb_bufs[p]) >= BATCH_SIZE:
                        vb_nets[p] = make_betting_net().to(device)
                        vb_opts[p] = optim.Adam(vb_nets[p].parameters(), lr=LR)
                        vb_loss = train_value_network(vb_nets[p], vb_bufs[p], vb_opts[p],
                                            BATCH_SIZE, N_TRAIN_STEPS, device)
                        vb_nets[p].eval()
                        if vb_loss is not None:
                            writer.add_scalar(f"loss/value_betting_p{p}", vb_loss, iteration)

                # ── diagnostics ──
                for p, label in [(0, "SB"), (1, "BB")]:
                    if len(vb_bufs[p]) >= BATCH_SIZE:
                        vecs, advs, masks = vb_bufs[p].sample(min(len(vb_bufs[p]), 10_000))

                        # 1. loss per position
                        with torch.no_grad():
                            x = torch.tensor(vecs, dtype=torch.float32, device=device)
                            pred = vb_nets[p](x).cpu().numpy()
                        sq_err = ((pred - advs) ** 2 * masks).sum(axis=1) / masks.sum(axis=1).clip(1)
                        writer.add_scalar(f"diag/loss_{label}", sq_err.mean(), iteration)

                        # 2. average regret magnitude
                        regret_mag = np.abs(advs * masks).sum(axis=1).mean()
                        writer.add_scalar(f"diag/regret_mag_{label}", regret_mag, iteration)

                    # 3. fold frequency from strategy buffers
                    if len(sb_bufs[p]) >= 100:
                        sv, ss, sw = sb_bufs[p].sample(min(len(sb_bufs[p]), 5_000))
                        fold_freq = ss[:, FOLD].mean()
                        writer.add_scalar(f"diag/fold_freq_{label}", fold_freq, iteration)

                for p in [0, 1]:
                    writer.add_scalar(f"buffer/vb_p{p}", len(vb_bufs[p]), iteration)
                    writer.add_scalar(f"buffer/sb_p{p}", len(sb_bufs[p]), iteration)

                elapsed = time.time() - t0
                rate    = iteration / elapsed
                eta     = (N_ITERATIONS - iteration) / rate
                writer.add_scalar("speed/it_per_s", rate, iteration)
                print(f"iter {iteration:6d} | "
                      f"vb=({len(vb_bufs[0])},{len(vb_bufs[1])}) sb=({len(sb_bufs[0])},{len(sb_bufs[1])}) | "
                      f"{rate:.1f} it/s | ETA {eta/3600:.1f}h")

            # train + save strategy nets periodically
            if iteration % STRAT_EVERY == 0:
                for p in [0, 1]:
                    if len(sb_bufs[p]) >= BATCH_SIZE:
                        snet = make_betting_net().to(device)
                        sopt = optim.Adam(snet.parameters(), lr=LR)
                        sloss = train_strategy_network(snet, sb_bufs[p], sopt, BATCH_SIZE, N_TRAIN_STEPS, device)
                        path = os.path.join(SAVE_DIR, f"strategy_betting_p{p}_iter{iteration}.pt")
                        torch.save(snet.state_dict(), path)
                        sloss_str = f"{sloss:.4f}" if sloss is not None else "n/a"
                        print(f"  strategy p{p} loss={sloss_str} -> {path}")

            # save just the net weights (no buffers — fast)
            if iteration % SAVE_EVERY == 0:
                ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_iter{iteration}.pt")
                torch.save({
                    "iteration": iteration,
                    "vb_net_0": vb_nets[0].state_dict(), "vb_net_1": vb_nets[1].state_dict(),
                }, ckpt_path)
                print(f"  saved {ckpt_path}")

    # train final strategy networks
    print("\nTraining final strategy networks...")
    for p in [0, 1]:
        net = make_betting_net().to(device)
        opt = optim.Adam(net.parameters(), lr=LR)
        loss = train_strategy_network(net, sb_bufs[p], opt, BATCH_SIZE, N_TRAIN_STEPS, device)
        path = os.path.join(SAVE_DIR, f"strategy_betting_p{p}_final.pt")
        torch.save(net.state_dict(), path)
        loss_str = f"{loss:.4f}" if loss is not None else "skipped (buffer too small)"
        print(f"  p{p} betting strategy loss={loss_str} -> {path}")

    print("Done.")


if __name__ == "__main__":
    train()
