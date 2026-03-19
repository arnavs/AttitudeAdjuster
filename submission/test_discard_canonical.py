import sys
sys.path.insert(0, ".")

import random
from itertools import combinations

import numpy as np

from submission.solve_discard import (
    KEEP_PAIRS,
    canonical_hand_flop,
    canonical_hand_flop_with_keep_pair,
    compute_bb_values,
)
from gym_env import WrappedEval


def test_keep_pair_remap_is_permutation(samples=200):
    rng = random.Random(0)
    all_hands = list(combinations(range(27), 5))

    for _ in range(samples):
        hand = rng.choice(all_hands)
        remaining = [c for c in range(27) if c not in hand]
        flop = tuple(sorted(rng.sample(remaining, 3)))

        key = canonical_hand_flop(hand, flop)
        mapped_slots = []
        for keep_i, keep_j in KEEP_PAIRS:
            mapped_key, kp_idx = canonical_hand_flop_with_keep_pair(hand, flop, keep_i, keep_j)
            assert mapped_key == key
            mapped_slots.append(kp_idx)

        assert sorted(mapped_slots) == list(range(len(KEEP_PAIRS))), (
            hand,
            flop,
            mapped_slots,
        )


def test_compute_bb_values_uses_canonical_slots():
    hand = (0, 15, 20, 22, 25)
    flop = (1, 8, 9)
    values = compute_bb_values(WrappedEval(), hand, flop, bb_table=None, n_samples=1)

    assert values.shape == (len(KEEP_PAIRS),)
    assert np.isfinite(values).all()

    mapped_slots = []
    for keep_i, keep_j in KEEP_PAIRS:
        _, kp_idx = canonical_hand_flop_with_keep_pair(hand, flop, keep_i, keep_j)
        mapped_slots.append(kp_idx)

    assert sorted(mapped_slots) == list(range(len(KEEP_PAIRS)))


def test_runtime_slot_meaning_depends_on_hand_order():
    hand_sorted = (1, 2, 5, 7, 9)
    hand_runtime = (9, 1, 2, 5, 7)
    slot = 0

    sorted_cards = tuple(hand_sorted[idx] for idx in KEEP_PAIRS[slot])
    runtime_cards = tuple(hand_runtime[idx] for idx in KEEP_PAIRS[slot])

    assert sorted_cards != runtime_cards


if __name__ == "__main__":
    test_keep_pair_remap_is_permutation()
    test_compute_bb_values_uses_canonical_slots()
    test_runtime_slot_meaning_depends_on_hand_order()
    print("discard canonicalization tests passed")
