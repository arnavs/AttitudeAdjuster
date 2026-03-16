"""
Precompute preflop equity table.

Output: submission/preflop_equity.npy
Shape:  (80730, 2), float32
  Axis 0: 5-card hand index via combinatorial number system, C(27,5) = 80730
  Axis 1: position (0=SB, 1=BB)
Value:  max equity over all 10 keep pairs via MC simulation.

SB equity: accounts for seeing BB's softmax discard before choosing keep pair.
BB equity: simpler pool, no info about opp discards.
"""

import itertools
import time
import numpy as np
from math import comb
from multiprocessing import Pool
from gym_env import PokerEnv, WrappedEval

MC_SAMPLES = 3
DISCARD_TEMP = 10.0
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
    for i, j in keep_pairs:
        h1, h2 = hand5[i], hand5[j]
        excluded = set([h1, h2] + list(flop))
        p = np.array([c for c in pool if c not in excluded])
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


def equity_as_bb(h1, h2, pool, n_samples=MC_SAMPLES):
    """BB equity: no info about SB discards. Sample opp 5 cards, flop, turn, river."""
    pool = np.array(pool)
    wins = 0.0
    for _ in range(n_samples):
        sample = np.random.choice(pool, size=10, replace=False)
        opp5 = sample[:5]
        flop = sample[5:8]
        turn, river = sample[8], sample[9]
        # Opp (SB) keeps best pair from their 5 given flop
        opp_pool = [c for c in range(27) if c not in [h1, h2] and c not in sample]
        ok1, ok2 = _softmax_keep(opp5, flop, opp_pool)
        board = [PokerEnv.int_to_card(int(c)) for c in [*flop, turn, river]]
        our_rank = evaluator.evaluate([PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)], board)
        opp_rank = evaluator.evaluate([PokerEnv.int_to_card(ok1), PokerEnv.int_to_card(ok2)], board)
        if our_rank < opp_rank:
            wins += 1.0
        elif our_rank == opp_rank:
            wins += 0.5
    return wins / n_samples


def equity_as_sb(h1, h2, pool, n_samples=MC_SAMPLES):
    """SB equity: sees BB's softmax discard, pool shrinks by 3 discarded cards."""
    pool_arr = np.array(pool)
    wins = 0.0
    for _ in range(n_samples):
        sample = np.random.choice(pool_arr, size=8, replace=False)  # opp5 + flop3
        opp5 = sample[:5]
        flop = sample[5:8]
        # Simulate BB's softmax discard
        bb_pool = [c for c in pool if c not in opp5 and c not in flop]
        ok1, ok2 = _softmax_keep(opp5, flop, bb_pool)
        opp_discards = [c for c in opp5 if c != ok1 and c != ok2]
        # SB now knows opp_discards — exclude from turn/river pool
        remaining = np.array([c for c in pool if c not in opp5 and c not in flop and c not in opp_discards])
        turn, river = np.random.choice(remaining, size=2, replace=False)
        board = [PokerEnv.int_to_card(int(c)) for c in [*flop, turn, river]]
        our_rank = evaluator.evaluate([PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)], board)
        opp_rank = evaluator.evaluate([PokerEnv.int_to_card(ok1), PokerEnv.int_to_card(ok2)], board)
        if our_rank < opp_rank:
            wins += 0.9
        elif our_rank == opp_rank:
            wins += 0.5
    return wins / n_samples


def best_equity(hand, pool, equity_fn):
    """Max equity over all 10 keep pairs."""
    best = 0.0
    for i, j in itertools.combinations(range(5), 2):
        eq = equity_fn(hand[i], hand[j], pool)
        if eq > best:
            best = eq
    return best


def _compute_hand(hand):
    hi = hand_idx(hand)
    pool = [c for c in range(27) if c not in hand]
    sb = best_equity(hand, pool, equity_as_sb)
    bb = best_equity(hand, pool, equity_as_bb)
    return hi, sb, bb


def compute_table():
    table = np.zeros((N_HANDS, 2), dtype=np.float32)
    all_hands = list(itertools.combinations(range(27), 5))
    print(f"Computing {len(all_hands)} hands x 2 positions with {N_WORKERS} workers...")
    t0 = time.time()

    with Pool(N_WORKERS, initializer=_init_worker) as p:
        for idx, (hi, sb, bb) in enumerate(p.imap(_compute_hand, all_hands, chunksize=100)):
            table[hi, 0] = sb
            table[hi, 1] = bb
            if idx > 0:
                elapsed = time.time() - t0
                eta = elapsed / idx * (len(all_hands) - idx)
                print(f"  {idx}/{len(all_hands)}  elapsed={elapsed:.0f}s", flush=True)

    return table


if __name__ == "__main__":
    table = compute_table()
    np.save("submission/preflop_equity.npy", table)
    print(f"Saved to submission/preflop_equity.npy, shape={table.shape}")
