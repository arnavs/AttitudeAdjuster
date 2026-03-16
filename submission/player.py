import os
import time
import itertools
import numpy as np
from math import comb
from multiprocessing import Pool
from agents.agent import Agent
from gym_env import PokerEnv, WrappedEval

_PREFLOP_TABLE_PATH = os.path.join(os.path.dirname(__file__), "preflop_equity.npy")
N_WORKERS = 4

_worker_evaluator = None


def _init_worker():
    global _worker_evaluator
    _worker_evaluator = WrappedEval()


def _mc_equity_worker(args):
    h1, h2, flop, pool, n_samples = args
    pool = np.array(pool)
    wins = 0
    for _ in range(n_samples):
        sample = np.random.choice(pool, size=4, replace=False)
        opp_h1, opp_h2, turn, river = sample
        board = [PokerEnv.int_to_card(c) for c in flop + [turn, river]]
        our_rank = _worker_evaluator.evaluate([PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)], board)
        opp_rank = _worker_evaluator.evaluate([PokerEnv.int_to_card(opp_h1), PokerEnv.int_to_card(opp_h2)], board)
        if our_rank < opp_rank:
            wins += 1
        elif our_rank == opp_rank:
            wins += 0.5
    return wins / n_samples


def _hand_idx(cards):
    c = sorted(cards)
    return comb(c[0], 1) + comb(c[1], 2) + comb(c[2], 3) + comb(c[3], 4) + comb(c[4], 5)


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self.opp_pairs = None    # list of (card1, card2) tuples
        self.opp_weights = None  # np.ndarray, uniform prior
        self.current_hand = None
        self.flop_cards = None   # stored once at flop, reused on turn/river
        self.zeroed_streets = set()
        self.hand_start_time = None
        self.cumulative_chips = 0
        self.evaluator = WrappedEval()
        self.MC_SAMPLES = 400
        self.pool = Pool(N_WORKERS, initializer=_init_worker)
        if os.path.exists(_PREFLOP_TABLE_PATH):
            self.preflop_table = np.load(_PREFLOP_TABLE_PATH)
        else:
            self.preflop_table = None
            import warnings
            warnings.warn(f"Preflop equity table not found at {_PREFLOP_TABLE_PATH}. Falling back to equity=0.5.")

    def __name__(self):
        return "PlayerAgent"

    def _mc_equity(self, h1, h2, flop, pool, n_samples=100):
        """Estimate win probability of (h1, h2) vs a random opponent pair, with random turn+river from pool."""
        pool = np.array(pool)
        wins = 0
        for _ in range(n_samples):
            sample = np.random.choice(pool, size=4, replace=False)
            opp_h1, opp_h2, turn, river = sample
            board = [PokerEnv.int_to_card(c) for c in flop + [turn, river]]
            our_rank = self.evaluator.evaluate([PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)], board)
            opp_rank = self.evaluator.evaluate([PokerEnv.int_to_card(opp_h1), PokerEnv.int_to_card(opp_h2)], board)
            if our_rank < opp_rank:  # lower is better in treys
                wins += 1
            elif our_rank == opp_rank: 
                wins += 0.5
        return wins / n_samples

    def _update_prior_discard(self, observation):
        """Update weights by likelihood of opp keeping their hole cards given observed discards."""
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discards = [c for c in observation["my_discarded_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]

        my_cards = [c for c in observation["my_cards"] if c != -1]

        args = []
        for h1, h2 in self.opp_pairs:
            excluded = set([h1, h2] + opp_discards + my_discards + my_cards + community)
            pool = [c for c in range(27) if c not in excluded]
            args.append((h1, h2, self.flop_cards, pool, self.MC_SAMPLES))
        equities = np.array(self.pool.map(_mc_equity_worker, args))

        TEMP = 10.0
        log_weights = TEMP * equities
        log_weights -= log_weights.max()
        self.opp_weights *= np.exp(log_weights)
        self.opp_weights /= self.opp_weights.sum()

    def _best_discard(self, observation):
        """Try all 10 keep pairs, return indices of the best by MC equity."""
        my_cards = [c for c in observation["my_cards"] if c != -1]
        flop = [c for c in observation["community_cards"] if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]

        DISCARD_TEMP = 50.0
        keep_pairs = list(itertools.combinations(range(5), 2))
        equities = []
        for i, j in keep_pairs:
            h1, h2 = my_cards[i], my_cards[j]
            excluded = set([h1, h2] + my_cards + flop + opp_discards)
            pool = [c for c in range(27) if c not in excluded]
            equities.append(self._mc_equity(h1, h2, flop, pool, n_samples=self.MC_SAMPLES))
        equities = np.array(equities)
        log_w = DISCARD_TEMP * equities
        log_w -= log_w.max()
        probs = np.exp(log_w)
        probs /= probs.sum()
        chosen = np.random.choice(len(keep_pairs), p=probs)
        return keep_pairs[chosen]

    def _equity_vs_pair(self, h1, h2, my_cards_treys, community, observation):
        """Equity of our hand vs a single opp pair. Exact: river direct, turn enumerate."""
        opp_treys = [PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)]
        if len(community) == 5:
            board = [PokerEnv.int_to_card(c) for c in community]
            our_rank = self.evaluator.evaluate(my_cards_treys, board)
            opp_rank = self.evaluator.evaluate(opp_treys, board)
            return 1.0 if our_rank < opp_rank else (0.5 if our_rank == opp_rank else 0.0)
        else:
            dead = set(observation["my_cards"]) | set(observation["my_discarded_cards"]) | set(observation["opp_discarded_cards"]) | set(community) | {h1, h2}
            dead.discard(-1)
            pool = np.array([c for c in range(27) if c not in dead])
            if len(community) == 4:  # turn: enumerate river
                wins = 0.0
                for river in pool:
                    board = [PokerEnv.int_to_card(c) for c in community + [int(river)]]
                    our_rank = self.evaluator.evaluate(my_cards_treys, board)
                    opp_rank = self.evaluator.evaluate(opp_treys, board)
                    wins += 1.0 if our_rank < opp_rank else (0.5 if our_rank == opp_rank else 0.0)
                return wins / len(pool)
            else:  # flop: MC over (turn, river)
                wins = 0.0
                for _ in range(self.MC_SAMPLES):
                    turn, river = np.random.choice(pool, size=2, replace=False)
                    board = [PokerEnv.int_to_card(c) for c in community + [int(turn), int(river)]]
                    our_rank = self.evaluator.evaluate(my_cards_treys, board)
                    opp_rank = self.evaluator.evaluate(opp_treys, board)
                    wins += 1.0 if our_rank < opp_rank else (0.5 if our_rank == opp_rank else 0.0)
                return wins / self.MC_SAMPLES

    def _thompson_action(self, observation):
        """Sample N pairs from posterior, compute equity for each, raise if majority are wins."""
        N = 20
        valid_actions = observation["valid_actions"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        my_cards_treys = [PokerEnv.int_to_card(c) for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]

        indices = np.random.choice(len(self.opp_pairs), size=N, p=self.opp_weights / self.opp_weights.sum())
        win_rate = np.mean([self._equity_vs_pair(*self.opp_pairs[i], my_cards_treys, community, observation) for i in indices])

        if win_rate > 0.5 and valid_actions[self.action_types.RAISE.value]:
            raise_amount = np.random.randint(min_raise, max_raise + 1)
            return self.action_types.RAISE.value, raise_amount, 0, 0

        call_amount = observation["opp_bet"] - observation["my_bet"]
        pot_odds = call_amount / (observation["pot_size"] + call_amount) if call_amount > 0 else 0.0
        if win_rate >= pot_odds:
            if valid_actions[self.action_types.CALL.value]:
                return self.action_types.CALL.value, 0, 0, 0
            return self.action_types.CHECK.value, 0, 0, 0

        if valid_actions[self.action_types.CHECK.value]:
            return self.action_types.CHECK.value, 0, 0, 0
        return self.action_types.FOLD.value, 0, 0, 0

    def _init_prior(self, observation):
        """Initialize uniform prior over opponent hole card pairs after both players have discarded."""
        my_cards = set(c for c in observation["my_cards"] if c != -1)
        my_discards = set(c for c in observation["my_discarded_cards"] if c != -1)
        opp_discards = set(c for c in observation["opp_discarded_cards"] if c != -1)
        community = set(c for c in observation["community_cards"] if c != -1)

        known = my_cards | my_discards | opp_discards | community
        remaining = [c for c in range(27) if c not in known]

        self.opp_pairs = list(itertools.combinations(remaining, 2))
        self.opp_weights = np.ones(len(self.opp_pairs), dtype=np.float64)

    def act(self, observation, reward, terminated, truncated, info):
        self.cumulative_chips += reward

        hand_number = info.get("hand_number")
        hands_remaining = 1000 - (hand_number or 0)
        if self.cumulative_chips > 1.51 * hands_remaining:
            if observation["valid_actions"][self.action_types.DISCARD.value]:
                return self.action_types.DISCARD.value, 0, 0, 1
            return self.action_types.FOLD.value, 0, 0, 0

        if hand_number != self.current_hand:
            self.current_hand = hand_number
            self.opp_pairs = None
            self.opp_weights = None
            self.flop_cards = None
            self.zeroed_streets = set()
            self.hand_start_time = time.time()

        self.logger.info(f"Hand {hand_number} street {observation["street"]}")

        # discard phase
        if observation["valid_actions"][self.action_types.DISCARD.value]:
            k1, k2 = self._best_discard(observation)
            return self.action_types.DISCARD.value, 0, k1, k2

        # Initialize prior once both discards are known (start of flop betting onward)
        opp_discards_known = all(c != -1 for c in observation["opp_discarded_cards"])
        if opp_discards_known and self.flop_cards is None:
            self.flop_cards = [c for c in observation["community_cards"] if c != -1][:3]

        if opp_discards_known and self.opp_pairs is None:
            self._init_prior(observation)
            self._update_prior_discard(observation)
            self.logger.info(f"Prior initialized: {len(self.opp_pairs)} pairs, weights sum={self.opp_weights.sum():.3f}")

        # hand_elapsed = time.time() - (self.hand_start_time or time.time())
        # if hand_elapsed > 3.5:
        #     self.logger.info(f"Hand {hand_number} time budget exceeded ({hand_elapsed:.1f}s), checking/folding")
        #     if observation["valid_actions"][self.action_types.CHECK.value]:
        #         return self.action_types.CHECK.value, 0, 0, 0
        #     return self.action_types.FOLD.value, 0, 0, 0

        street = observation["street"]
        if street == 0:
            return self._act_preflop(observation)
        elif street == 1:
            return self._act_flop(observation)
        elif street == 2:
            return self._act_turn(observation)
        elif street == 3:
            return self._act_river(observation)

    def _act_preflop(self, observation):
        valid_actions = observation["valid_actions"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]

        if self.preflop_table is not None:
            my_cards = [c for c in observation["my_cards"] if c != -1]
            pos = observation["blind_position"]  # 0=SB, 1=BB
            hi = _hand_idx(my_cards)
            equity = float(self.preflop_table[hi, pos])
        else:
            equity = 0.4

        call_amount = observation["opp_bet"] - observation["my_bet"]
        pot_odds = call_amount / (observation["pot_size"] + call_amount) if call_amount > 0 else 0.0

        if equity > 0.55 and valid_actions[self.action_types.RAISE.value]:
            raise_amount = np.random.randint(min_raise, max_raise + 1)
            return self.action_types.RAISE.value, raise_amount, 0, 0
        if equity >= pot_odds:
            if valid_actions[self.action_types.CALL.value]:
                return self.action_types.CALL.value, 0, 0, 0
            return self.action_types.CHECK.value, 0, 0, 0
        if valid_actions[self.action_types.CHECK.value]:
            return self.action_types.CHECK.value, 0, 0, 0
        return self.action_types.FOLD.value, 0, 0, 0


    def _act_flop(self, observation):
        return self._thompson_action(observation)

    def _act_turn(self, observation):
        if 2 not in self.zeroed_streets:
            community = set(c for c in observation["community_cards"] if c != -1)
            for i, (h1, h2) in enumerate(self.opp_pairs):
                if h1 in community or h2 in community:
                    self.opp_weights[i] = 0.0
            self.opp_weights /= self.opp_weights.sum()
            self.zeroed_streets.add(2)
        return self._thompson_action(observation)

    def _act_river(self, observation):
        if 3 not in self.zeroed_streets:
            community = set(c for c in observation["community_cards"] if c != -1)
            for i, (h1, h2) in enumerate(self.opp_pairs):
                if h1 in community or h2 in community:
                    self.opp_weights[i] = 0.0
            self.opp_weights /= self.opp_weights.sum()
            self.zeroed_streets.add(3)
        return self._thompson_action(observation)
