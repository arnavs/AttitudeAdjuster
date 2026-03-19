"""
Iterative discard solver.

Computes BB's discard strategy table, bucketed by suit-isomorphic (hand, flop).
Each run is one iteration toward Nash:
  - If no table exists: SB uses equity against random opponents (level 0)
  - If table exists: SB uses BB's table to weight opponent range (level 2)

Usage:
  cd ~/MachineYearning
  python submission/solve_discard.py              # one iteration
  python submission/solve_discard.py --iters 3    # three iterations
  python submission/solve_discard.py --samples 10 # more MC samples per entry
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
from itertools import combinations, permutations
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv, WrappedEval

KEEP_PAIRS = list(combinations(range(5), 2))
TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "bb_discard_table.pkl")
N_CARDS = 27
TEMP = 6.0
N_CORES = 8


def canonical_hand_flop(hand, flop):
    """Canonical form for (hand, flop) under suit permutation."""
    ranks_h = [c % 9 for c in hand]
    suits_h = [c // 9 for c in hand]
    ranks_f = [c % 9 for c in flop]
    suits_f = [c // 9 for c in flop]
    best = None
    for perm in permutations(range(3)):
        mapped_hand = tuple(sorted(ranks_h[i] + perm[suits_h[i]] * 9 for i in range(5)))
        mapped_flop = tuple(sorted(ranks_f[i] + perm[suits_f[i]] * 9 for i in range(3)))
        key = (mapped_hand, mapped_flop)
        if best is None or key < best:
            best = key
    return best


def canonical_hand_flop_with_index(hand, flop):
    """Canonical key + which keep-pair index positions 0,1 of hand map to."""
    ranks_h = [c % 9 for c in hand]
    suits_h = [c // 9 for c in hand]
    ranks_f = [c % 9 for c in flop]
    suits_f = [c // 9 for c in flop]
    best = None
    best_unsorted = None
    for perm in permutations(range(3)):
        unsorted = [ranks_h[i] + perm[suits_h[i]] * 9 for i in range(5)]
        mapped_hand = tuple(sorted(unsorted))
        mapped_flop = tuple(sorted(ranks_f[i] + perm[suits_f[i]] * 9 for i in range(3)))
        key = (mapped_hand, mapped_flop)
        if best is None or key < best:
            best = key
            best_unsorted = unsorted

    sorted_hand = sorted(best_unsorted)
    card0 = best_unsorted[0]
    card1 = best_unsorted[1]
    pos0 = sorted_hand.index(card0)
    # handle duplicate cards: if card0 == card1, pos1 must differ
    pos1 = sorted_hand.index(card1) if card1 != card0 else pos0 + 1
    if card1 != card0:
        pos1 = sorted_hand.index(card1)
    if pos0 > pos1:
        pos0, pos1 = pos1, pos0
    kp_idx = KEEP_PAIRS.index((pos0, pos1))
    return best, kp_idx


def canonical_hand_flop_with_keep_pair(hand, flop, keep_i, keep_j):
    """Canonical key + canonical keep-pair slot for hand indices keep_i, keep_j."""
    ordered_hand = [hand[keep_i], hand[keep_j]]
    ordered_hand.extend(hand[idx] for idx in range(5) if idx != keep_i and idx != keep_j)
    return canonical_hand_flop_with_index(tuple(ordered_hand), flop)


def sb_best_keep(evaluator, sb_hand, flop, sb_known_dead, bb_table=None, bb_discs=None, n=4):
    """SB picks best keep-pair by equity. If bb_table exists, weight by BB's keep probs."""
    dead = sb_known_dead | set(sb_hand)
    live = [c for c in range(N_CARDS) if c not in dead]
    n_remaining = 5 - len(flop)

    best_eq, best_i, best_j = -1.0, 0, 1
    for i, j in KEEP_PAIRS:
        k1, k2 = sb_hand[i], sb_hand[j]
        wins, total_w = 0.0, 0.0
        for _ in range(n):
            if len(live) < 2 + n_remaining:
                break
            sampled = np.random.choice(live, size=2 + n_remaining, replace=False)
            r1, r2 = int(sampled[0]), int(sampled[1])

            w = 1.0
            if bb_table is not None and bb_discs is not None:
                bb_full = [r1, r2] + list(bb_discs)
                bb_key, kp_idx = canonical_hand_flop_with_index(tuple(bb_full), tuple(flop))
                if bb_key in bb_table:
                    w = bb_table[bb_key][kp_idx]
                if w < 1e-9:
                    continue

            remaining_board = [int(c) for c in sampled[2:]]
            board5 = list(flop) + remaining_board
            full_board = [PokerEnv.int_to_card(c) for c in board5]
            my_rank = evaluator.evaluate(
                [PokerEnv.int_to_card(k1), PokerEnv.int_to_card(k2)], full_board)
            opp_rank = evaluator.evaluate(
                [PokerEnv.int_to_card(r1), PokerEnv.int_to_card(r2)], full_board)
            outcome = 1.0 if my_rank < opp_rank else 0.5 if my_rank == opp_rank else 0.0
            wins += w * outcome
            total_w += w

        eq = wins / max(total_w, 1e-9)
        if eq > best_eq:
            best_eq, best_i, best_j = eq, i, j
    return best_i, best_j


def compute_bb_values(evaluator, hand, flop, bb_table=None, n_samples=5):
    """Compute equity for each BB keep-pair against SB's best response."""
    n_remaining = 5 - len(flop)

    values = np.zeros(len(KEEP_PAIRS), dtype=np.float64)
    seen_slots = np.zeros(len(KEEP_PAIRS), dtype=bool)
    for ki, kj in KEEP_PAIRS:
        k1, k2 = hand[ki], hand[kj]
        bb_discards = [hand[x] for x in range(5) if x != ki and x != kj]

        # physical pool: SB can't have BB's cards (all 5) or flop cards
        sample_pool = [c for c in range(N_CARDS) if c not in set(hand) | set(flop)]

        # SB's knowledge: only BB's discards + flop are known dead
        sb_known_dead = set(flop) | set(bb_discards)

        wins, counted = 0.0, 0
        for _ in range(n_samples):
            if len(sample_pool) < 5 + n_remaining:
                break
            sampled = np.random.choice(sample_pool, size=5 + n_remaining, replace=False)
            sb_hand = [int(c) for c in sampled[:5]]

            si, sj = sb_best_keep(evaluator, sb_hand, flop, sb_known_dead,
                                  bb_table=bb_table, bb_discs=bb_discards, n=4)
            r1, r2 = sb_hand[si], sb_hand[sj]

            remaining_board = [int(c) for c in sampled[5:]]
            board5 = list(flop) + remaining_board
            full_board = [PokerEnv.int_to_card(c) for c in board5]
            my_rank = evaluator.evaluate(
                [PokerEnv.int_to_card(k1), PokerEnv.int_to_card(k2)], full_board)
            opp_rank = evaluator.evaluate(
                [PokerEnv.int_to_card(r1), PokerEnv.int_to_card(r2)], full_board)
            if my_rank < opp_rank:
                wins += 1.0
            elif my_rank == opp_rank:
                wins += 0.5
            counted += 1
        _, kp_idx = canonical_hand_flop_with_keep_pair(hand, flop, ki, kj)
        values[kp_idx] = wins / max(counted, 1)
        seen_slots[kp_idx] = True

    assert seen_slots.all(), f"Expected keep-pair remapping to cover all slots, got {seen_slots}"
    return values


def _worker(args):
    """Process a batch of infosets in a subprocess."""
    keys_and_reps, bb_table_path, n_samples = args

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gym_env import WrappedEval

    evaluator = WrappedEval()
    bb_table = None
    if bb_table_path and os.path.exists(bb_table_path):
        with open(bb_table_path, "rb") as f:
            bb_table = pickle.load(f)

    results = {}
    for key, (hand, flop) in keys_and_reps:
        values = compute_bb_values(evaluator, hand, flop, bb_table=bb_table, n_samples=n_samples)
        logits = TEMP * values
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        results[key] = probs
    return results


def run_iteration(bb_table=None, n_samples=5):
    """One iteration: compute BB's best response for all infosets."""
    evaluator = WrappedEval()
    seen_keys = {}

    print("Enumerating infosets...")
    t0 = time.time()
    count = 0
    for hand in combinations(range(N_CARDS), 5):
        remaining = [c for c in range(N_CARDS) if c not in hand]
        for flop in combinations(remaining, 3):
            key = canonical_hand_flop(hand, flop)
            if key not in seen_keys:
                seen_keys[key] = (hand, flop)
            count += 1
            if count % 5000000 == 0:
                print(f"  {count/1e6:.0f}M enumerated, {len(seen_keys):,} distinct ({time.time()-t0:.0f}s)")

    print(f"  {len(seen_keys):,} distinct infosets ({time.time()-t0:.0f}s)")

    # save current table for workers to load
    bb_table_path = None
    if bb_table is not None:
        bb_table_path = TABLE_PATH + ".tmp"
        with open(bb_table_path, "wb") as f:
            pickle.dump(bb_table, f)

    # split work across cores
    items = list(seen_keys.items())
    chunk_size = max(1, len(items) // (N_CORES * 10))
    batches = []
    for i in range(0, len(items), chunk_size):
        batch = items[i:i + chunk_size]
        batches.append((batch, bb_table_path, n_samples))

    print(f"Computing BB values across {N_CORES} cores, {len(batches)} batches...")
    t1 = time.time()
    new_table = {}

    with Pool(processes=N_CORES) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, batches)):
            new_table.update(result)
            if (i + 1) % 10 == 0:
                print(f"  {len(new_table):,}/{len(seen_keys):,} ({time.time()-t1:.0f}s)")

    # cleanup temp file
    if bb_table_path and os.path.exists(bb_table_path):
        os.remove(bb_table_path)

    print(f"  Done: {len(new_table):,} entries ({time.time()-t1:.0f}s)")
    return new_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    bb_table = None
    if os.path.exists(TABLE_PATH):
        with open(TABLE_PATH, "rb") as f:
            bb_table = pickle.load(f)
        print(f"Loaded existing table: {len(bb_table):,} entries")
    else:
        print("No existing table, starting from level-0")

    for i in range(args.iters):
        print(f"\n=== Iteration {i+1}/{args.iters} ===")
        bb_table = run_iteration(bb_table=bb_table, n_samples=args.samples)

        os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
        with open(TABLE_PATH, "wb") as f:
            pickle.dump(bb_table, f)
        print(f"Saved: {len(bb_table):,} entries to {TABLE_PATH}")

    print("Done.")


if __name__ == "__main__":
    main()
