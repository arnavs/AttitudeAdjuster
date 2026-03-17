"""
Precompute preflop equity table (v2).

Output: submission/preflop_equity.npy
Shape:  (80730, 2), float32
  Axis 0: 5-card hand index via combinatorial number system, C(27,5) = 80730
  Axis 1: kept for compatibility, column 1 (BB) is the primary column.
Value:  avg(max) equity — for each scenario, pick the best keep pair, then average.

Strategy: 
    - Compute BB equities
    - Assume SB discards strategically 
    - Assume that SB thinks I have a uniform hand when SB discards
        - Nash would require SB to know that I have a slightly better hand

Key fix from v1: uses avg(max) instead of max(avg), correctly modeling
that we see the flop before choosing our keep pair.
"""

import itertools
import time
import numpy as np
from math import comb
from multiprocessing import Pool
from gym_env import PokerEnv, WrappedEval

MC_SAMPLES = 50
DISCARD_TEMP = 40.0
N_HANDS = 80730
N_WORKERS = 8

evaluator = None  # initialized per worker


def _init_worker():
    global evaluator
    evaluator = WrappedEval()


def hand_idx(cards):
    """Combinatorial number system index for a sorted 5-card tuple."""
    c = sorted(cards)
    return comb(c[0], 1) + comb(c[1], 2) + comb(c[2], 3) + comb(c[3], 4) + comb(c[4], 5)


def _softmax_keep(hand5, flop, pool):
    """Sample a keep pair (i, j) from hand5 via softmax over equity given flop."""
    equities = []
    keep_pairs = list(itertools.combinations(range(5), 2))
    p = np.array(pool)
    for i, j in keep_pairs:
        h1, h2 = hand5[i], hand5[j]
        wins = 0.0
        for _ in range(20):  # cheap estimate for discard simulation
            sample = np.random.choice(p, size=4, replace=False)
            o1, o2, turn, river = sample
            board = [PokerEnv.int_to_card(int(c)) for c in list(flop) + [turn, river]]
            our_rank = evaluator.evaluate([PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)], board)
            opp_rank = evaluator.evaluate([PokerEnv.int_to_card(int(o1)), PokerEnv.int_to_card(int(o2))], board)
            if our_rank < opp_rank:
                wins += 1.0
            elif our_rank == opp_rank:
                wins += 0.5
        equities.append(wins / 20)
    equities = np.array(equities)
    log_w = DISCARD_TEMP * equities
    log_w -= log_w.max()
    probs = np.exp(log_w)
    probs /= probs.sum()
    chosen = np.random.choice(len(keep_pairs), p=probs)
    i, j = keep_pairs[chosen]
    return hand5[i], hand5[j]


def _compute_hand(hand):
    """Compute avg(max) equity for a 5-card hand.

    For each MC scenario (opponent cards, flop, turn, river):
      1. Simulate opponent's softmax discard
      2. Evaluate all 10 of our keep-pairs on this board
      3. Take the best (modeling that we see flop before discarding)
    Average over all scenarios.
    """
    hi = hand_idx(hand)
    pool = [c for c in range(27) if c not in hand]
    pool_arr = np.array(pool)
    keep_pairs = list(itertools.combinations(range(5), 2))

    wins = 0.0
    for _ in range(MC_SAMPLES):
        sample = np.random.choice(pool_arr, size=10, replace=False)
        opp5 = sample[:5]
        flop = sample[5:8]
        turn, river = sample[8], sample[9]

        # Opponent keeps best pair via softmax (only knows flop, not turn/river)
        opp_pool = [c for c in pool if c not in opp5 and c not in flop]
        ok1, ok2 = _softmax_keep(opp5, flop, opp_pool)

        board = [PokerEnv.int_to_card(int(c)) for c in [*flop, turn, river]]
        opp_treys = [PokerEnv.int_to_card(ok1), PokerEnv.int_to_card(ok2)]
        opp_rank = evaluator.evaluate(opp_treys, board)

        # Evaluate all 10 keep-pairs, take the best
        best = 0.0
        for i, j in keep_pairs:
            h1, h2 = hand[i], hand[j]
            our_treys = [PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)]
            our_rank = evaluator.evaluate(our_treys, board)
            if our_rank < opp_rank:
                eq = 1.0
            elif our_rank == opp_rank:
                eq = 0.5
            else:
                eq = 0.0
            if eq > best:
                best = eq
        wins += best

    equity = wins / MC_SAMPLES
    # Store in both columns for compatibility (column 1 = BB is primary)
    return hi, equity


def compute_table():
    table = np.zeros(N_HANDS, dtype=np.float32)
    all_hands = list(itertools.combinations(range(27), 5))
    print(f"Computing {len(all_hands)} hands with {N_WORKERS} workers, {MC_SAMPLES} samples each...")
    t0 = time.time()

    with Pool(N_WORKERS, initializer=_init_worker) as p:
        for idx, (hi, equity) in enumerate(p.imap(_compute_hand, all_hands, chunksize=1)):
            table[hi] = equity
            if idx > 0 and idx % 1000 == 0:
                elapsed = time.time() - t0
                eta = elapsed / idx * (len(all_hands) - idx)
                print(f"  {idx}/{len(all_hands)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    return table


if __name__ == "__main__":
    table = compute_table()
    np.save("submission/preflop_equity.npy", table)
    print(f"Saved to submission/preflop_equity.npy, shape={table.shape}")
