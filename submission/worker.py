import numpy as np
from gym_env import PokerEnv, WrappedEval

_evaluator = None


def init_worker():
    global _evaluator
    _evaluator = WrappedEval()


def mc_equity_worker(args):
    h1, h2, flop, pool, n_samples = args
    pool = np.array(pool)
    wins = 0
    for _ in range(n_samples):
        sample = np.random.choice(pool, size=4, replace=False)
        opp_h1, opp_h2, turn, river = sample
        board = [PokerEnv.int_to_card(c) for c in flop + [turn, river]]
        our_rank = _evaluator.evaluate([PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)], board)
        opp_rank = _evaluator.evaluate([PokerEnv.int_to_card(opp_h1), PokerEnv.int_to_card(opp_h2)], board)
        if our_rank < opp_rank:
            wins += 1
        elif our_rank == opp_rank:
            wins += 0.5
    return wins / n_samples
