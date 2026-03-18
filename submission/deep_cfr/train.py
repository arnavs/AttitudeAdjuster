"""
Deep CFR training loop.

Each iteration:
  1. Run K traversals in parallel (one per core) for each player
  2. Add samples to value buffer and strategy buffer
  3. Every TRAIN_EVERY iterations: retrain value network from scratch
  4. After all iterations: train strategy network Pi on strategy buffer

Usage:
  python train.py
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
from multiprocessing import Pool
import pickle

from network import CFRNet, ReservoirBuffer, StrategyBuffer
from network import train_value_network, train_strategy_network
from traversal import run_traversal, GameState
from encoder import INPUT_DIM, N_ACTIONS

# ── hyperparameters ──────────────────────────────────────────────────────────
N_ITERATIONS   = 100_000   # total CFR iterations
K_TRAVERSALS   = 8         # traversals per iteration (one per core)
TRAIN_EVERY    = 10        # retrain value net every N iterations
VALUE_BUF_SIZE = 500_000   # reservoir buffer capacity
STRAT_BUF_SIZE = 2_000_000
BATCH_SIZE     = 1024
N_TRAIN_STEPS  = 200       # gradient steps per value net training
N_CORES        = 8
LR             = 1e-3
SAVE_EVERY     = 1000      # save checkpoint every N iterations
SAVE_DIR       = "checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)


# ── worker function (runs in subprocess) ─────────────────────────────────────
def _worker(args):
    """
    Run one traversal and return collected samples.
    Runs in a subprocess — no shared state.
    """
    traverser, iteration, value_net_state, strategy_net_state = args

    import torch
    from network import CFRNet, ReservoirBuffer, StrategyBuffer
    from traversal import run_traversal

    # reconstruct networks in subprocess
    value_net = CFRNet()
    value_net.load_state_dict(value_net_state)
    value_net.eval()

    strategy_net = CFRNet()
    strategy_net.load_state_dict(strategy_net_state)
    strategy_net.eval()

    # small local buffers to collect this traversal's samples
    v_buf = ReservoirBuffer(capacity=10_000)
    s_buf = StrategyBuffer(capacity=10_000)

    run_traversal(traverser, value_net, strategy_net,
                  v_buf, s_buf, iteration)

    return v_buf.buffer, s_buf.buffer


# ── main training loop ────────────────────────────────────────────────────────
def train():
    device = 'cpu'

    # two value networks (one per player) + one strategy network per player
    value_nets    = [CFRNet().to(device), CFRNet().to(device)]
    strategy_nets = [CFRNet().to(device), CFRNet().to(device)]

    for net in value_nets + strategy_nets:
        net.eval()

    value_optimizers    = [optim.Adam(n.parameters(), lr=LR) for n in value_nets]
    strategy_optimizers = [optim.Adam(n.parameters(), lr=LR) for n in strategy_nets]

    value_bufs    = [ReservoirBuffer(VALUE_BUF_SIZE), ReservoirBuffer(VALUE_BUF_SIZE)]
    strategy_bufs = [StrategyBuffer(STRAT_BUF_SIZE), StrategyBuffer(STRAT_BUF_SIZE)]

    print(f"Starting Deep CFR training: {N_ITERATIONS} iterations, "
          f"{K_TRAVERSALS} traversals/iter, {N_CORES} cores")
    print(f"Training value net every {TRAIN_EVERY} iterations")

    t_start = time.time()

    with Pool(processes=N_CORES) as pool:
        for iteration in range(1, N_ITERATIONS + 1):

            # ── collect traversals for both players ──────────────────────────
            for traverser in [0, 1]:
                vnet_state = value_nets[traverser].state_dict()
                snet_state = strategy_nets[traverser].state_dict()

                # prepare K jobs
                jobs = [
                    (traverser, iteration, vnet_state, snet_state)
                    for _ in range(K_TRAVERSALS)
                ]

                results = pool.map(_worker, jobs)

                # merge samples into main buffers
                for v_samples, s_samples in results:
                    for item in v_samples:
                        value_bufs[traverser].buffer.append(item)
                        value_bufs[traverser].n_seen += 1
                        # enforce capacity via reservoir
                        if len(value_bufs[traverser].buffer) > VALUE_BUF_SIZE:
                            idx = np.random.randint(0, value_bufs[traverser].n_seen)
                            if idx < VALUE_BUF_SIZE:
                                value_bufs[traverser].buffer[idx] = \
                                    value_bufs[traverser].buffer[-1]
                            value_bufs[traverser].buffer.pop()

                    for item in s_samples:
                        strategy_bufs[traverser].buffer.append(item)
                        strategy_bufs[traverser].n_seen += 1
                        if len(strategy_bufs[traverser].buffer) > STRAT_BUF_SIZE:
                            idx = np.random.randint(0, strategy_bufs[traverser].n_seen)
                            if idx < STRAT_BUF_SIZE:
                                strategy_bufs[traverser].buffer[idx] = \
                                    strategy_bufs[traverser].buffer[-1]
                            strategy_bufs[traverser].buffer.pop()

            # ── retrain value networks ───────────────────────────────────────
            if iteration % TRAIN_EVERY == 0:
                for p in [0, 1]:
                    if len(value_bufs[p]) >= BATCH_SIZE:
                        # retrain from scratch (reinit weights)
                        value_nets[p] = CFRNet().to(device)
                        value_optimizers[p] = optim.Adam(
                            value_nets[p].parameters(), lr=LR)
                        loss = train_value_network(
                            value_nets[p], value_bufs[p],
                            value_optimizers[p],
                            batch_size=BATCH_SIZE,
                            n_steps=N_TRAIN_STEPS,
                            device=device
                        )
                        value_nets[p].eval()

                elapsed = time.time() - t_start
                v0 = len(value_bufs[0])
                v1 = len(value_bufs[1])
                s0 = len(strategy_bufs[0])
                s1 = len(strategy_bufs[1])
                its_per_sec = iteration / elapsed
                remaining   = (N_ITERATIONS - iteration) / its_per_sec
                print(f"iter {iteration:6d} | "
                      f"v_buf=({v0},{v1}) s_buf=({s0},{s1}) | "
                      f"{its_per_sec:.1f} it/s | "
                      f"ETA {remaining/3600:.1f}h")

            # ── save checkpoint ──────────────────────────────────────────────
            if iteration % SAVE_EVERY == 0:
                _save_checkpoint(iteration, value_nets, strategy_nets,
                                 value_bufs, strategy_bufs)

    # ── final: train strategy networks ───────────────────────────────────────
    print("\nTraining final strategy networks...")
    for p in [0, 1]:
        strategy_nets[p] = CFRNet().to(device)
        strategy_optimizers[p] = optim.Adam(
            strategy_nets[p].parameters(), lr=LR)
        loss = train_strategy_network(
            strategy_nets[p], strategy_bufs[p],
            strategy_optimizers[p],
            batch_size=BATCH_SIZE,
            n_steps=2000,
            device=device
        )
        print(f"Player {p} strategy net final loss: {loss:.4f}")

    # save final strategy networks
    for p in [0, 1]:
        path = os.path.join(SAVE_DIR, f"strategy_net_p{p}_final.pt")
        torch.save(strategy_nets[p].state_dict(), path)
        print(f"Saved strategy network for player {p} to {path}")

    print("Training complete.")


def _save_checkpoint(iteration, value_nets, strategy_nets,
                     value_bufs, strategy_bufs):
    path = os.path.join(SAVE_DIR, f"checkpoint_{iteration}.pt")
    torch.save({
        "iteration"    : iteration,
        "value_net_0"  : value_nets[0].state_dict(),
        "value_net_1"  : value_nets[1].state_dict(),
        "strategy_net_0": strategy_nets[0].state_dict(),
        "strategy_net_1": strategy_nets[1].state_dict(),
    }, path)
    print(f"  Checkpoint saved: {path}")


if __name__ == "__main__":
    train()
