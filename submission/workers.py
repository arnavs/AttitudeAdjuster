"""Standalone worker functions for multiprocessing. Kept in a separate module
so that the competition server's module renaming doesn't break pickling."""
import itertools
import numpy as np
from gym_env import PokerEnv, WrappedEval

_ALL_KEEP_PAIRS = list(itertools.combinations(range(5), 2))


def _discard_likelihood(args):
    """Compute softmax likelihood of keeping (h1, h2) given the full 5-card hand."""
    h1, h2, opp_discards, my_discards, community, flop_cards, blind_position, discard_samples, temp = args
    evaluator = WrappedEval()
    opp_hand = [h1, h2] + opp_discards
    if blind_position == 1:  # I am BB and discard first
        excluded = set(opp_hand + my_discards + community)
    else:  # I am SB and discard later
        excluded = set(opp_hand + community)
    pool = np.array([c for c in range(27) if c not in excluded])

    keep_equities = np.zeros(len(_ALL_KEEP_PAIRS))
    for idx, (i, j) in enumerate(_ALL_KEEP_PAIRS):
        k1, k2 = opp_hand[i], opp_hand[j]
        wins = 0.0
        for _ in range(discard_samples):
            sample = np.random.choice(pool, size=4, replace=False)
            opp_h1, opp_h2, turn, river = sample
            board = [PokerEnv.int_to_card(c) for c in flop_cards + [turn, river]]
            our_rank = evaluator.evaluate([PokerEnv.int_to_card(k1), PokerEnv.int_to_card(k2)], board)
            opp_rank = evaluator.evaluate([PokerEnv.int_to_card(opp_h1), PokerEnv.int_to_card(opp_h2)], board)
            if our_rank < opp_rank:
                wins += 1.0
            elif our_rank == opp_rank:
                wins += 0.5
        keep_equities[idx] = wins / discard_samples

    log_probs = temp * keep_equities
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    probs /= probs.sum()
    return probs[0]


def _opp_strength_worker(args):
    """Compute opponent hand strength for a single pair."""
    h1, h2, community, opp_discards, my_discards, hand_samples, board_samples = args
    evaluator = WrappedEval()
    opp_treys = [PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)]
    dead = set(opp_discards) | set(my_discards) | set(community) | {h1, h2}
    pool = [c for c in range(27) if c not in dead]
    all_hands = list(itertools.combinations(pool, 2))
    if not all_hands:
        return 0.5

    hand_count = min(len(all_hands), hand_samples)
    if hand_count == len(all_hands):
        sampled_hands = all_hands
    else:
        sampled_idx = np.random.choice(len(all_hands), size=hand_count, replace=False)
        sampled_hands = [all_hands[i] for i in sampled_idx]

    if len(community) == 5:
        boards = [community]
    elif len(community) == 4:
        rivers = pool if len(pool) <= board_samples else list(
            np.random.choice(pool, size=board_samples, replace=False))
        boards = [community + [int(r)] for r in rivers]
    else:
        bs = min(max(4, board_samples), len(pool) * max(len(pool) - 1, 1))
        boards = []
        seen = set()
        while len(boards) < bs and len(seen) < len(pool) * max(len(pool) - 1, 1):
            t, r = np.random.choice(pool, size=2, replace=False)
            key = tuple(sorted((int(t), int(r))))
            if key in seen:
                continue
            seen.add(key)
            boards.append(community + [int(t), int(r)])

    total = 0.0
    count = 0
    for board_cards in boards:
        board = [PokerEnv.int_to_card(c) for c in board_cards]
        opp_rank = evaluator.evaluate(opp_treys, board)
        board_set = set(board_cards)
        for r1, r2 in sampled_hands:
            if r1 in board_set or r2 in board_set:
                continue
            rand_treys = [PokerEnv.int_to_card(r1), PokerEnv.int_to_card(r2)]
            rand_rank = evaluator.evaluate(rand_treys, board)
            total += 1.0 if opp_rank < rand_rank else (0.5 if opp_rank == rand_rank else 0.0)
            count += 1
    return total / count if count > 0 else 0.5
