"""
Deep CFR training loop.

Networks (per player):
  value_betting_net   - trained every TRAIN_EVERY iters on betting regret samples
  value_discard_net   - trained every TRAIN_EVERY iters on discard regret samples
  strategy_betting_net - trained once at end on betting strategy samples
  strategy_discard_net - trained once at end on discard strategy samples

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

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network import (
    make_betting_net, make_discard_net,
    ReservoirBuffer, StrategyBuffer,
    train_value_network, train_strategy_network,
)
from traversal import run_traversal

# ── hyperparameters ───────────────────────────────────────────────────────────
N_ITERATIONS    = 100_000
K_TRAVERSALS    = 8         # per iteration per player (one per core)
TRAIN_EVERY     = 10        # retrain value nets every N iterations
VALUE_BUF_SIZE  = 500_000
STRAT_BUF_SIZE  = 2_000_000
BATCH_SIZE      = 1024
N_TRAIN_STEPS   = 200
N_CORES         = 8
LR              = 1e-3
SAVE_EVERY      = 1000
SAVE_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

os.makedirs(SAVE_DIR, exist_ok=True)


# ── worker (subprocess) ───────────────────────────────────────────────────────

def _worker(args):
    """Run one traversal in a subprocess. Returns buffer contents."""
    (traverser, iteration,
     vb_state, vd_state,
     sb_state, sd_state) = args

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import torch
    from network import (make_betting_net, make_discard_net,
                         ReservoirBuffer, StrategyBuffer)
    from traversal import run_traversal

    vb_net = make_betting_net(); vb_net.load_state_dict(vb_state); vb_net.eval()
    vd_net = make_discard_net(); vd_net.load_state_dict(vd_state); vd_net.eval()
    sb_net = make_betting_net(); sb_net.load_state_dict(sb_state); sb_net.eval()
    sd_net = make_discard_net(); sd_net.load_state_dict(sd_state); sd_net.eval()

    vb_buf = ReservoirBuffer(10_000)
    vd_buf = ReservoirBuffer(10_000)
    sb_buf = StrategyBuffer(10_000)
    sd_buf = StrategyBuffer(10_000)

    run_traversal(
        traverser,
        vb_net, vd_net, sb_net, sd_net,
        vb_buf, vd_buf, sb_buf, sd_buf,
        iteration,
    )

    return vb_buf.buffer, vd_buf.buffer, sb_buf.buffer, sd_buf.buffer


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

    # two players × four networks each
    vb_nets = [make_betting_net().to(device), make_betting_net().to(device)]
    vd_nets = [make_discard_net().to(device), make_discard_net().to(device)]
    sb_nets = [make_betting_net().to(device), make_betting_net().to(device)]
    sd_nets = [make_discard_net().to(device), make_discard_net().to(device)]

    for net in vb_nets + vd_nets + sb_nets + sd_nets:
        net.eval()

    vb_opts = [optim.Adam(n.parameters(), lr=LR) for n in vb_nets]
    vd_opts = [optim.Adam(n.parameters(), lr=LR) for n in vd_nets]

    vb_bufs = [ReservoirBuffer(VALUE_BUF_SIZE), ReservoirBuffer(VALUE_BUF_SIZE)]
    vd_bufs = [ReservoirBuffer(VALUE_BUF_SIZE), ReservoirBuffer(VALUE_BUF_SIZE)]
    sb_bufs = [StrategyBuffer(STRAT_BUF_SIZE),  StrategyBuffer(STRAT_BUF_SIZE)]
    sd_bufs = [StrategyBuffer(STRAT_BUF_SIZE),  StrategyBuffer(STRAT_BUF_SIZE)]

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
                     vb_nets[traverser].state_dict(),
                     vd_nets[traverser].state_dict(),
                     sb_nets[traverser].state_dict(),
                     sd_nets[traverser].state_dict())
                    for _ in range(K_TRAVERSALS)
                ]
                results = pool.map(_worker, jobs)

                for vb_items, vd_items, sb_items, sd_items in results:
                    _merge(vb_bufs[traverser], vb_items)
                    _merge(vd_bufs[traverser], vd_items)
                    _merge(sb_bufs[traverser], sb_items)
                    _merge(sd_bufs[traverser], sd_items)

            # retrain value networks
            if iteration % TRAIN_EVERY == 0:
                for p in [0, 1]:
                    if len(vb_bufs[p]) >= BATCH_SIZE:
                        vb_nets[p] = make_betting_net().to(device)
                        vb_opts[p] = optim.Adam(vb_nets[p].parameters(), lr=LR)
                        train_value_network(vb_nets[p], vb_bufs[p], vb_opts[p],
                                            BATCH_SIZE, N_TRAIN_STEPS, device)
                        vb_nets[p].eval()

                    if len(vd_bufs[p]) >= BATCH_SIZE:
                        vd_nets[p] = make_discard_net().to(device)
                        vd_opts[p] = optim.Adam(vd_nets[p].parameters(), lr=LR)
                        train_value_network(vd_nets[p], vd_bufs[p], vd_opts[p],
                                            BATCH_SIZE, N_TRAIN_STEPS, device)
                        vd_nets[p].eval()

                elapsed = time.time() - t0
                rate    = iteration / elapsed
                eta     = (N_ITERATIONS - iteration) / rate
                print(f"iter {iteration:6d} | "
                      f"vb=({len(vb_bufs[0])},{len(vb_bufs[1])}) "
                      f"vd=({len(vd_bufs[0])},{len(vd_bufs[1])}) | "
                      f"{rate:.1f} it/s | ETA {eta/3600:.1f}h")

            if iteration % SAVE_EVERY == 0:
                torch.save({
                    "iteration": iteration,
                    "vb_0": vb_nets[0].state_dict(), "vb_1": vb_nets[1].state_dict(),
                    "vd_0": vd_nets[0].state_dict(), "vd_1": vd_nets[1].state_dict(),
                }, os.path.join(SAVE_DIR, f"checkpoint_{iteration}.pt"))
                print(f"  saved checkpoint_{iteration}.pt")

    # train final strategy networks
    print("\nTraining final strategy networks...")
    for p in [0, 1]:
        for tag, net_fn, buf, name in [
            ("betting", make_betting_net, sb_bufs[p], f"strategy_betting_p{p}"),
            ("discard", make_discard_net, sd_bufs[p], f"strategy_discard_p{p}"),
        ]:
            net = net_fn().to(device)
            opt = optim.Adam(net.parameters(), lr=LR)
            loss = train_strategy_network(net, buf, opt, BATCH_SIZE, 2000, device)
            path = os.path.join(SAVE_DIR, f"{name}_final.pt")
            torch.save(net.state_dict(), path)
            print(f"  p{p} {tag} strategy loss={loss:.4f} -> {path}")

    print("Done.")


if __name__ == "__main__":
    train()
