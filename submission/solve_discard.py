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

def canonical_hand_flop_with_index(hand, flop):
    """Canonical key + which keep-pair index positions 0,1 of hand map to."""
    ranks_h = [c % 9 for c in hand]
    suits_h = [c // 9 for c in hand]
    ranks_f = [c % 9 for c in flop]
    suits_f = [c // 9 for c in flop]
    best = None
    best_unsorted = None
    for perm in permutations(range(3)):
        # apply the suit permutation
        unsorted = [ranks_h[i] + perm[suits_h[i]] * 9 for i in range(5)]
        # sort it 
        mapped_hand = tuple(sorted(unsorted))
        mapped_flop = tuple(sorted(ranks_f[i] + perm[suits_f[i]] * 9 for i in range(3)))
        # store the key
        key = (mapped_hand, mapped_flop)
        if best is None or key < best:
            best = key
            best_unsorted = unsorted

    # sorted version of best suit permutation 
    sorted_hand = sorted(best_unsorted)
    card0 = best_unsorted[0] # suit-permuted version of card 0
    card1 = best_unsorted[1] # suit-permuted version of card 1
    pos0 = sorted_hand.index(card0) # CANONICAL POSITION
    pos1 = sorted_hand.index(card1) 
    if pos0 > pos1:
        pos0, pos1 = pos1, pos0 # KEEP_PAIRS requires monotonicity
    kp_idx = KEEP_PAIRS.index((pos0, pos1))
    return best, kp_idx # KEEP_PAIRS is a scalar

def sb_best_keep(evaluator, sb_hand, flop, sb_known_dead, bb_table=None, bb_discs=None, n=4):
    """SB picks best keep-pair by equity. If bb_table exists, weight by BB's keep probs."""
    dead = sb_known_dead | set(sb_hand)
    live = [c for c in range(N_CARDS) if c not in dead]
    best_eq, best_i, best_j = -1.0, 0, 1
    for i, j in KEEP_PAIRS:
        k1, k2 = sb_hand[i], sb_hand[j]
        wins, total_w = 0.0, 0.0
        for _ in range(n):
            # draw turn, river, and BB holes
            sampled = np.random.choice(live, size=4, replace=False)
            r1, r2 = int(sampled[0]), int(sampled[1])

            w = 1.0
            if bb_table is not None and bb_discs is not None:
                # initialize BB hand
                bb_full = [r1, r2] + list(bb_discs)
                # BB's problem key and keep index
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
    return best_i, best_j # raw indices


def compute_bb_values(evaluator, hand, flop, bb_table=None, n_samples=5):
    """Compute equity for each BB keep-pair against SB's best response."""

    values = np.zeros(len(KEEP_PAIRS), dtype=np.float64)
    seen_slots = np.zeros(len(KEEP_PAIRS), dtype=bool)
    for idx, (ki, kj) in enumerate(KEEP_PAIRS):
        k1, k2 = hand[ki], hand[kj]
        bb_discards = [hand[x] for x in range(5) if x != ki and x != kj] # raw cards

        # physical pool: SB can't have BB's cards (all 5) or flop cards
        sample_pool = [c for c in range(N_CARDS) if c not in set(hand) | set(flop)] # raw cards

        # SB's knowledge: only BB's discards + flop are known dead
        sb_known_dead = set(flop) | set(bb_discards) # raw cards

        wins, counted = 0.0, 0
        for _ in range(n_samples):
            # full SB hand + (river, turn)
            sampled = np.random.choice(sample_pool, size=7, replace=False) # raw cards
            sb_hand = [int(c) for c in sampled[:5]] # raw cards

            si, sj = sb_best_keep(evaluator, sb_hand, flop, sb_known_dead,
                                  bb_table=bb_table, bb_discs=bb_discards, n=4) # raw indices
            r1, r2 = sb_hand[si], sb_hand[sj] # raw cards

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
        values[idx] = wins / max(counted, 1) # raw indices
    return values # raw value vector


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
        # remap from representative-hand order to canonical-sorted order
        canonical_values = np.zeros(10, dtype=np.float64)
        for orig_idx, (ki, kj) in enumerate(KEEP_PAIRS):
            kept = [hand[ki], hand[kj]]
            discs = [hand[x] for x in range(5) if x != ki and x != kj]
            full = list(kept) + list(discs)
            _, canon_idx = canonical_hand_flop_with_index(tuple(full), tuple(flop))
            canonical_values[canon_idx] = values[orig_idx]
        logits = TEMP * canonical_values
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
            key = canonical_hand_flop_with_index(hand, flop)[0]
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
