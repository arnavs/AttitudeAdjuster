"""
Hand evaluator for the 27-card deck.

Wraps WrappedEval from gym_env.py, which already handles:
  - Ace as high (above 9) or low (below 2) in straights
  - The 27-card deck (no 10/J/Q/K, no clubs)

Card encoding:
  int in [0, 26]
  rank = card % 9   (0=2, 1=3, 2=4, 3=5, 4=6, 5=7, 6=8, 7=9, 8=A)
  suit = card // 9  (0=d, 1=h, 2=s)
"""

# LEVEL 1: ASSUMES THAT OPP IS LEVEL 0 (NOT DISCARDING OPTIMALLY)
# TBD: FIX THIS 

import sys
import os
import numpy as np
from itertools import combinations

# allow importing from the submission directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import WrappedEval, PokerEnv

_evaluator = WrappedEval()


def evaluate(hole_cards, board):
    """
    hole_cards: list of 2 card ints
    board:      list of 3, 4, or 5 card ints
    Returns:    int rank (lower = better, as per treys convention)
    """
    h = [PokerEnv.int_to_card(c) for c in hole_cards]
    b = [PokerEnv.int_to_card(c) for c in board]
    return _evaluator.evaluate(h, b)


def winner(hole0, hole1, board5):
    """
    hole0, hole1: lists of 2 card ints
    board5:       list of 5 card ints
    Returns: 0 if player 0 wins, 1 if player 1 wins, -1 if tie
    """
    r0 = evaluate(hole0, board5)
    r1 = evaluate(hole1, board5)
    if r0 < r1:
        return 0
    elif r1 < r0:
        return 1
    else:
        return -1


def mc_equity(hole_cards, board, dead, n_samples=100):
    """
    Monte Carlo equity for hole_cards against a random opponent,
    with remaining board cards sampled from the live deck.

    hole_cards: list of 2 card ints (our kept hand)
    board:      list of 3-5 card ints (community cards so far)
    dead:       set of card ints that are out of play
    n_samples:  number of rollouts

    Returns float in [0, 1] (win probability, 0.5 for ties)
    """
    live = [c for c in range(27) if c not in dead and c not in hole_cards]
    n_board_remaining = 5 - len(board)

    wins = 0.0
    for _ in range(n_samples):
        sample = np.random.choice(live,
                                  size=2 + n_board_remaining,
                                  replace=False)
        opp_hole = list(sample[:2])
        full_board = board + list(sample[2:])
        r = winner(hole_cards, opp_hole, full_board)
        if r == 0:
            wins += 1.0
        elif r == -1:
            wins += 0.5

    return wins / n_samples


def best_keep_pair(hand5, board3, dead, n_mc=50):
    """
    Find the best pair of cards to keep from a 5-card hand on a given flop.

    hand5:  list of 5 card ints
    board3: list of 3 card ints (flop)
    dead:   set of card ints out of play (our discards, opp discards, etc.)
    n_mc:   MC samples per pair evaluation

    Returns: (keep_idx_1, keep_idx_2), equities_list
      keep_idx_1, keep_idx_2: indices into hand5 of the best pair to keep
      equities_list: equity for each of the 10 keep pairs
    """
    dead_full = set(dead) | set(hand5) | set(board3)
    equities = []

    for i, j in combinations(range(5), 2):
        h1, h2 = hand5[i], hand5[j]
        # remove kept cards from dead for equity calc
        eq = mc_equity([h1, h2], board3, dead_full, n_samples=n_mc)
        equities.append(eq)

    best_idx = int(np.argmax(equities))
    keep_pair = list(combinations(range(5), 2))[best_idx]
    return keep_pair, equities


def equity_all_pairs(hand5, board3, dead, n_mc=50):
    """
    Return equity for all 10 keep pairs. Used for discard node in CFR.
    Returns numpy array of shape (10,).
    """
    dead_full = set(dead) | set(hand5) | set(board3)
    equities = []
    for i, j in combinations(range(5), 2):
        h1, h2 = hand5[i], hand5[j]
        dead_for_pair = dead_full - {h1, h2}
        eq = mc_equity([h1, h2], board3, dead_for_pair, n_samples=n_mc)
        equities.append(eq)
    return np.array(equities, dtype=np.float32)
