"""
Validate preflop equity table after precompute finishes.
Usage: python validate_table.py [path_to_table]
"""
import sys
import numpy as np
from math import comb
import itertools
from gym_env import PokerEnv

TABLE_PATH = sys.argv[1] if len(sys.argv) > 1 else "submission/preflop_equity.npy"

def hand_idx(cards):
    c = sorted(cards)
    return comb(c[0], 1) + comb(c[1], 2) + comb(c[2], 3) + comb(c[3], 4) + comb(c[4], 5)

def card_name(c):
    return PokerEnv.RANKS[c % len(PokerEnv.RANKS)] + PokerEnv.SUITS[c // len(PokerEnv.RANKS)]

def hand_name(cards):
    return " ".join(card_name(c) for c in cards)

table = np.load(TABLE_PATH)
print(f"Shape: {table.shape}, dtype: {table.dtype}")
if table.ndim == 2:
    col = table[:, 1]  # BB column for v1
else:
    col = table  # 1D for v2
print()

# 1. Basic stats
nonzero = col[col > 0]
print(f"min={col.min():.4f}  mean={col.mean():.4f}  median={np.median(col):.4f}  max={col.max():.4f}  std={col.std():.4f}  zeros={np.sum(col==0)}")
pcts = np.percentile(nonzero, [5, 10, 25, 50, 75, 90, 95])
print(f"Percentiles (nonzero): p5={pcts[0]:.3f} p10={pcts[1]:.3f} p25={pcts[2]:.3f} p50={pcts[3]:.3f} p75={pcts[4]:.3f} p90={pcts[5]:.3f} p95={pcts[6]:.3f}")
n_unique = len(np.unique(np.round(col, 4)))
print(f"Unique values (rounded to 4dp): {n_unique}")
print()

# 3. Spot-check known hands
test_hands = []
# Strong: pair of aces + high cards
for hand in itertools.combinations(range(27), 5):
    cards = [card_name(c) for c in hand]
    ranks = [c[0] for c in cards]
    if ranks.count('A') >= 2:
        test_hands.append(hand)
        if len(test_hands) >= 5:
            break

# Weak: low cards, no pairs
weak_found = 0
for hand in itertools.combinations(range(27), 5):
    cards = [card_name(c) for c in hand]
    ranks = [c[0] for c in cards]
    if len(set(ranks)) == 5 and 'A' not in ranks and '9' not in ranks:
        test_hands.append(hand)
        weak_found += 1
        if weak_found >= 5:
            break

print("Spot checks:")
for hand in test_hands:
    hi = hand_idx(hand)
    eq = col[hi]
    print(f"  {hand_name(hand):25s}  equity={eq:.3f}")
print()

# 4. Preflop decision impact
n_fold = np.sum(col < 0.45)
n_raise = np.sum(col > 0.8)
total = len(col)
print(f"Preflop decisions (out of {total} hands):")
print(f"  fold(<0.45)={n_fold} ({100*n_fold/total:.1f}%)  raise(>0.8)={n_raise} ({100*n_raise/total:.1f}%)  call={total-n_fold-n_raise} ({100*(total-n_fold-n_raise)/total:.1f}%)")
