import os
import time
import itertools
import numpy as np
from math import comb
from agents.agent import Agent
from gym_env import PokerEnv, WrappedEval

_PREFLOP_TABLE_PATH = os.path.join(os.path.dirname(__file__), "preflop_equity.npy")


def _hand_idx(cards):
    c = sorted(cards)
    return comb(c[0], 1) + comb(c[1], 2) + comb(c[2], 3) + comb(c[3], 4) + comb(c[4], 5)


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType

        # --- Strategy constants ---
        self.PREFLOP_FOLD_THRESH = 0.44
        self.PREFLOP_RAISE_THRESH = 0.66
        self.PREFLOP_CALL_MARGIN = 0.05
        self.BET_THRESH = 0.56
        self.RAISE_THRESH = 0.77
        self.CALL_MARGIN = 0.12
        self.ADJ_SCALE = 0.14
        self.ADJ_FLOOR = -0.25
        self.FOLDOUT_RATIO = 1.55
        self.SB_BOOST = 1.05
        self.TIME_BUDGET = 2.85
        self.DISCARD_TEMP = 42.0
        self.PRIOR_DISCARD_TEMP = 9.0
        self.THOMPSON_N = 32
        self.MC_SAMPLES = 64
        self.UPDATE_RAISE_TEMP = 3.0
        self.OPP_STRENGTH_HAND_SAMPLES = 30
        self.OPP_STRENGTH_BOARD_SAMPLES = 20
        self.CLOSE_DECISION_BAND = 0.05
        self.PROBE_BET_FREQ = 0.12
        self.PROBE_RAISE_FREQ = 0.08

        self.opp_pairs = None
        self.opp_weights = None
        self.current_hand = None
        self.flop_cards = None
        self.zeroed_streets = set()
        self.hand_start_time = None
        self.equity_cache = {}
        self.opp_strength_cache = {}
        self.cumulative_chips = 0
        self.opp_showdown_wins = 1
        self.opp_showdowns = 2
        self.hands_won = 0
        self.opp_pressure_events = 0
        self.opp_postflop_observations = 0
        self.evaluator = WrappedEval()

        if os.path.exists(_PREFLOP_TABLE_PATH):
            self.preflop_table = np.load(_PREFLOP_TABLE_PATH)
        else:
            self.preflop_table = None
            import warnings
            warnings.warn(
                f"Preflop equity table not found at {_PREFLOP_TABLE_PATH}. Falling back to equity=0.5."
            )

    def __name__(self):
        return "PlayerAgent"

    def _elapsed_hand_time(self):
        return time.time() - (self.hand_start_time or time.time())

    def _normalize_weights(self):
        if self.opp_weights is None:
            return
        total = float(self.opp_weights.sum())
        if total > 0 and np.isfinite(total):
            self.opp_weights /= total
            return
        # self.opp_weights = np.where(np.isfinite(self.opp_weights), self.opp_weights, 0.0)
        # mask = self.opp_weights >= 0
        # self.opp_weights = mask.astype(np.float64)
        # total = float(self.opp_weights.sum())
        # if total == 0:
        #     self.opp_weights = np.ones_like(self.opp_weights, dtype=np.float64)
        #     total = float(self.opp_weights.sum())
        # self.opp_weights /= total

    def _dynamic_mc_samples(self, street, close_decision=False):
        base = {0: 0, 1: self.MC_SAMPLES, 2: max(20, self.MC_SAMPLES // 2), 3: 1}.get(street, self.MC_SAMPLES)
        if close_decision:
            base = int(base * 1.5)
        return max(8, base)

    def _mc_equity(self, h1, h2, flop, pool, n_samples=100):
        """Estimate win probability of (h1, h2) vs a random opponent pair, with random turn+river from pool."""
        pool = np.array(pool)
        wins = 0.0
        for _ in range(n_samples):
            sample = np.random.choice(pool, size=4, replace=False)
            opp_h1, opp_h2, turn, river = sample
            board = [PokerEnv.int_to_card(c) for c in flop + [turn, river]]
            our_rank = self.evaluator.evaluate([PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)], board)
            opp_rank = self.evaluator.evaluate([PokerEnv.int_to_card(opp_h1), PokerEnv.int_to_card(opp_h2)], board)
            if our_rank < opp_rank:
                wins += 1.0
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
        discard_samples = max(18, self.MC_SAMPLES // 2)
        for h1, h2 in self.opp_pairs:
            excluded = set([h1, h2] + opp_discards + my_discards + community)
            pool = [c for c in range(27) if c not in excluded]
            args.append((h1, h2, self.flop_cards, pool, discard_samples))
        equities = np.array([self._mc_equity(*a) for a in args])

        log_weights = self.PRIOR_DISCARD_TEMP * equities
        log_weights -= log_weights.max()
        self.opp_weights *= np.exp(log_weights)
        self._normalize_weights()

    def _best_discard(self, observation):
        """Try all 10 keep pairs, return indices of the best by MC equity."""
        my_cards = [c for c in observation["my_cards"] if c != -1]
        flop = [c for c in observation["community_cards"] if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]

        keep_pairs = list(itertools.combinations(range(5), 2))
        discard_samples = max(20, self.MC_SAMPLES // 2)
        equities = []
        for i, j in keep_pairs:
            h1, h2 = my_cards[i], my_cards[j]
            excluded = set([h1, h2] + my_cards + flop + opp_discards)
            pool = [c for c in range(27) if c not in excluded]
            equities.append(self._mc_equity(h1, h2, flop, pool, n_samples=discard_samples))
        equities = np.array(equities)
        log_w = self.DISCARD_TEMP * equities
        log_w -= log_w.max()
        probs = np.exp(log_w)
        probs /= probs.sum()
        chosen = np.random.choice(len(keep_pairs), p=probs)
        return keep_pairs[chosen]

    def _equity_vs_pair(self, h1, h2, my_cards_treys, community, observation):
        """Equity of our hand vs a single opp pair. Exact on river/turn, MC on flop."""
        street = observation["street"]
        cache_key = ("eq", h1, h2, street)
        if cache_key in self.equity_cache:
            return self.equity_cache[cache_key]

        opp_treys = [PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)]
        if len(community) == 5:
            board = [PokerEnv.int_to_card(c) for c in community]
            our_rank = self.evaluator.evaluate(my_cards_treys, board)
            opp_rank = self.evaluator.evaluate(opp_treys, board)
            result = 1.0 if our_rank < opp_rank else (0.5 if our_rank == opp_rank else 0.0)
            self.equity_cache[cache_key] = result
            return result

        dead = (
            set(observation["my_cards"])
            | set(observation["my_discarded_cards"])
            | set(observation["opp_discarded_cards"])
            | set(community)
            | {h1, h2}
        )
        dead.discard(-1)
        pool = np.array([c for c in range(27) if c not in dead])
        if len(pool) == 0:
            return 0.5

        if len(community) == 4:
            wins = 0.0
            for river in pool:
                board = [PokerEnv.int_to_card(c) for c in community + [int(river)]]
                our_rank = self.evaluator.evaluate(my_cards_treys, board)
                opp_rank = self.evaluator.evaluate(opp_treys, board)
                wins += 1.0 if our_rank < opp_rank else (0.5 if our_rank == opp_rank else 0.0)
            result = wins / len(pool)
        else:
            wins = 0.0
            n_samples = self._dynamic_mc_samples(street)
            for _ in range(n_samples):
                turn, river = np.random.choice(pool, size=2, replace=False)
                board = [PokerEnv.int_to_card(c) for c in community + [int(turn), int(river)]]
                our_rank = self.evaluator.evaluate(my_cards_treys, board)
                opp_rank = self.evaluator.evaluate(opp_treys, board)
                wins += 1.0 if our_rank < opp_rank else (0.5 if our_rank == opp_rank else 0.0)
            result = wins / n_samples

        self.equity_cache[cache_key] = result
        return result

    def _noisy_raise(self, frac, min_raise, max_raise, aggressive=False):
        """Noisy raise sizing around a target fraction of the legal interval."""
        frac = max(0.0, min(1.0, frac))
        target = min_raise + frac * (max_raise - min_raise)
        spread = max_raise - min_raise
        if aggressive:
            lo = max(min_raise, target - 0.25 * spread)
            hi = min(max_raise, target + 0.2 * spread)
        else:
            lo = max(min_raise, target - 0.4 * spread)
            hi = min(max_raise, target + 0.05 * spread)
        raise_amount = int(np.random.uniform(lo, hi))
        return max(min_raise, min(raise_amount, max_raise))

    def _aggression_factor(self):
        if self.opp_postflop_observations <= 0:
            return 0.0
        return min(1.0, self.opp_pressure_events / self.opp_postflop_observations)

    def _risk_adjustment(self, hands_remaining):
        ratio = self.cumulative_chips / max(hands_remaining, 1)
        adjustment = self.ADJ_SCALE * max(ratio, self.ADJ_FLOOR)
        hand_number = 1000 - hands_remaining
        if hand_number > 50:
            win_rate_overall = self.hands_won / hand_number
            if win_rate_overall < 0.4:
                adjustment = max(adjustment, 0.08)
        return adjustment

    def _estimate_win_rate(self, observation, my_cards_treys, community, close_decision=False):
        if self.opp_pairs is None or self.opp_weights is None:
            return 0.5
        street = observation["street"]
        base_n = {1: self.THOMPSON_N, 2: self.THOMPSON_N + 8, 3: self.THOMPSON_N + 16}.get(street, self.THOMPSON_N)
        if close_decision:
            base_n = int(base_n * 1.5)
        if self._elapsed_hand_time() > self.TIME_BUDGET * 0.7:
            base_n = max(10, base_n // 2)

        probs = self.opp_weights / self.opp_weights.sum()
        indices = np.random.choice(len(self.opp_pairs), size=base_n, p=probs)
        return float(np.mean([
            self._equity_vs_pair(*self.opp_pairs[i], my_cards_treys, community, observation)
            for i in indices
        ]))

    def _thompson_action(self, observation, hands_remaining):
        """Posterior-sampled action with adaptive defend/probe logic."""
        valid_actions = observation["valid_actions"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        my_cards_treys = [PokerEnv.int_to_card(c) for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        street = observation["street"]

        win_rate = self._estimate_win_rate(observation, my_cards_treys, community)
        call_amount = observation["opp_bet"] - observation["my_bet"]
        pot_odds = call_amount / (observation["pot_size"] + call_amount) if call_amount > 0 else 0.0
        adjustment = self._risk_adjustment(hands_remaining)
        aggression = self._aggression_factor()

        if valid_actions[self.action_types.CHECK.value]:
            bet_threshold = self.BET_THRESH + adjustment
            if abs(win_rate - bet_threshold) < self.CLOSE_DECISION_BAND and self._elapsed_hand_time() < self.TIME_BUDGET * 0.65:
                win_rate = self._estimate_win_rate(observation, my_cards_treys, community, close_decision=True)

            if win_rate > bet_threshold and valid_actions[self.action_types.RAISE.value]:
                frac = (win_rate - bet_threshold) / max(1e-6, 1.0 - bet_threshold)
                raise_amount = self._noisy_raise(frac, min_raise, max_raise, aggressive=street >= 2)
                return self.action_types.RAISE.value, raise_amount, 0, 0

            probe_low = max(0.38, bet_threshold - 0.14)
            probe_freq = self.PROBE_BET_FREQ + 0.08 * max(0.0, 0.5 - aggression)
            if (
                street < 3
                and valid_actions[self.action_types.RAISE.value]
                and probe_low <= win_rate < bet_threshold
                and np.random.random() < probe_freq
            ):
                raise_amount = self._noisy_raise(0.12, min_raise, max_raise, aggressive=False)
                return self.action_types.RAISE.value, raise_amount, 0, 0
            return self.action_types.CHECK.value, 0, 0, 0

        raise_threshold = self.RAISE_THRESH + adjustment - 0.04 * aggression
        call_margin = max(0.03, self.CALL_MARGIN + adjustment - 0.08 * aggression - 0.08 * pot_odds)
        call_threshold = pot_odds + call_margin

        if min(abs(win_rate - raise_threshold), abs(win_rate - call_threshold)) < self.CLOSE_DECISION_BAND and self._elapsed_hand_time() < self.TIME_BUDGET * 0.65:
            win_rate = self._estimate_win_rate(observation, my_cards_treys, community, close_decision=True)

        if win_rate > raise_threshold and valid_actions[self.action_types.RAISE.value]:
            frac = (win_rate - raise_threshold) / max(1e-6, 1.0 - raise_threshold)
            raise_amount = self._noisy_raise(frac, min_raise, max_raise, aggressive=True)
            return self.action_types.RAISE.value, raise_amount, 0, 0

        if (
            street < 3
            and aggression < 0.45
            and valid_actions[self.action_types.RAISE.value]
            and pot_odds < 0.28
            and 0.43 <= win_rate < call_threshold
            and np.random.random() < self.PROBE_RAISE_FREQ
        ):
            raise_amount = self._noisy_raise(0.18, min_raise, max_raise, aggressive=False)
            return self.action_types.RAISE.value, raise_amount, 0, 0

        if win_rate >= call_threshold and valid_actions[self.action_types.CALL.value]:
            return self.action_types.CALL.value, 0, 0, 0
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

    def _opp_hand_strength(self, h1, h2, community, observation):
        """Approximate opponent hand strength against random hands from their perspective."""
        street = observation["street"]
        cache_key = ("opp", h1, h2, street)
        if cache_key in self.opp_strength_cache:
            return self.opp_strength_cache[cache_key]

        opp_treys = [PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)]
        dead = set(observation["opp_discarded_cards"]) | set(observation["my_discarded_cards"]) | set(community) | {h1, h2}
        dead.discard(-1)
        pool = [c for c in range(27) if c not in dead]
        all_hands = list(itertools.combinations(pool, 2))
        if not all_hands:
            return 0.5

        hand_count = min(len(all_hands), self.OPP_STRENGTH_HAND_SAMPLES)
        if hand_count == len(all_hands):
            sampled_hands = all_hands
        else:
            sampled_idx = np.random.choice(len(all_hands), size=hand_count, replace=False)
            sampled_hands = [all_hands[i] for i in sampled_idx]

        if len(community) == 5:
            boards = [community]
        elif len(community) == 4:
            rivers = pool if len(pool) <= self.OPP_STRENGTH_BOARD_SAMPLES else np.random.choice(
                pool, size=self.OPP_STRENGTH_BOARD_SAMPLES, replace=False
            )
            boards = [community + [int(river)] for river in rivers]
        else:
            board_samples = min(max(4, self.OPP_STRENGTH_BOARD_SAMPLES), len(pool) * max(len(pool) - 1, 1))
            boards = []
            seen = set()
            while len(boards) < board_samples and len(seen) < len(pool) * max(len(pool) - 1, 1):
                turn, river = np.random.choice(pool, size=2, replace=False)
                key = tuple(sorted((int(turn), int(river))))
                if key in seen:
                    continue
                seen.add(key)
                boards.append(community + [int(turn), int(river)])

        total = 0.0
        count = 0
        for board_cards in boards:
            board = [PokerEnv.int_to_card(c) for c in board_cards]
            opp_rank = self.evaluator.evaluate(opp_treys, board)
            board_set = set(board_cards)
            for r1, r2 in sampled_hands:
                if r1 in board_set or r2 in board_set:
                    continue
                rand_treys = [PokerEnv.int_to_card(r1), PokerEnv.int_to_card(r2)]
                rand_rank = self.evaluator.evaluate(rand_treys, board)
                total += 1.0 if opp_rank < rand_rank else (0.5 if opp_rank == rand_rank else 0.0)
                count += 1

        result = total / count if count > 0 else 0.5
        self.opp_strength_cache[cache_key] = result
        return result

    def _update_prior_raise(self, observation):
        """Shift posterior toward hands the opponent is more likely to raise."""
        if self.opp_pairs is None or self.opp_weights is None:
            return

        community = [c for c in observation["community_cards"] if c != -1]
        raise_fraction = (observation["opp_bet"] - observation["my_bet"]) / max(observation["pot_size"], 1)
        if raise_fraction <= 0:
            return

        opp_win_rate = self.opp_showdown_wins / max(self.opp_showdowns, 1)
        aggression = self._aggression_factor()
        temp = self.UPDATE_RAISE_TEMP * raise_fraction * (0.55 + 0.45 * opp_win_rate) * (0.85 + 0.3 * aggression)
        temp = min(temp, 2.2)

        active_pairs = [(i, h1, h2) for i, (h1, h2) in enumerate(self.opp_pairs) if self.opp_weights[i] > 0]
        strengths = np.array([
            self._opp_hand_strength(h1, h2, community, observation)
            for _, h1, h2 in active_pairs
        ])
        log_weights = temp * strengths
        log_weights -= log_weights.max()
        for k, (i, _, _) in enumerate(active_pairs):
            self.opp_weights[i] *= np.exp(log_weights[k])
        self._normalize_weights()

    def observe(self, observation, reward, terminated, truncated, info):
        self.cumulative_chips += reward
        if terminated:
            if reward > 0:
                self.hands_won += 1
            if observation["opp_bet"] == observation["my_bet"]:
                self.opp_showdowns += 1
                if reward < 0:
                    self.opp_showdown_wins += 1
            return

        if observation["street"] >= 1 and all(c != -1 for c in observation["opp_discarded_cards"]):
            self.opp_postflop_observations += 1
            if observation["opp_bet"] > observation["my_bet"]:
                self.opp_pressure_events += 1
                self._update_prior_raise(observation)

    def act(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number")
        hands_remaining = 1000 - (hand_number or 0)
        if self.cumulative_chips > self.FOLDOUT_RATIO * hands_remaining:
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
            self.equity_cache = {}
            self.opp_strength_cache = {}

        self.logger.info(f"Hand {hand_number} street {observation['street']}")

        if observation["valid_actions"][self.action_types.DISCARD.value]:
            k1, k2 = self._best_discard(observation)
            return self.action_types.DISCARD.value, 0, k1, k2

        opp_discards_known = all(c != -1 for c in observation["opp_discarded_cards"])
        if opp_discards_known and self.flop_cards is None:
            self.flop_cards = [c for c in observation["community_cards"] if c != -1][:3]

        if opp_discards_known and self.opp_pairs is None:
            self._init_prior(observation)
            self._update_prior_discard(observation)
            self.logger.info(
                f"Prior initialized: {len(self.opp_pairs)} pairs, weights sum={self.opp_weights.sum():.3f}"
            )

        hand_elapsed = self._elapsed_hand_time()
        if hand_elapsed > self.TIME_BUDGET:
            self.logger.info(f"Hand {hand_number} time budget exceeded ({hand_elapsed:.1f}s), checking/folding")
            if observation["valid_actions"][self.action_types.CHECK.value]:
                return self.action_types.CHECK.value, 0, 0, 0
            return self.action_types.FOLD.value, 0, 0, 0

        street = observation["street"]
        if street == 0:
            return self._act_preflop(observation, hands_remaining)
        if street == 1:
            return self._act_flop(observation, hands_remaining)
        if street == 2:
            return self._act_turn(observation, hands_remaining)
        return self._act_river(observation, hands_remaining)

    def _act_preflop(self, observation, hands_remaining):
        valid_actions = observation["valid_actions"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]

        if self.preflop_table is not None:
            my_cards = [c for c in observation["my_cards"] if c != -1]
            pos = observation["blind_position"]
            hi = _hand_idx(my_cards)
            equity = float(self.preflop_table[hi])
            if pos == 0:
                equity = min(equity * self.SB_BOOST, 1.0)
        else:
            equity = 0.5

        call_amount = observation["opp_bet"] - observation["my_bet"]
        pot_odds = call_amount / (observation["pot_size"] + call_amount) if call_amount > 0 else 0.0
        adjustment = self._risk_adjustment(hands_remaining)
        raise_thresh = self.PREFLOP_RAISE_THRESH + 0.5 * adjustment
        fold_thresh = self.PREFLOP_FOLD_THRESH + 0.5 * adjustment
        call_thresh = pot_odds + max(0.02, self.PREFLOP_CALL_MARGIN + 0.4 * adjustment)

        if equity < fold_thresh and not valid_actions[self.action_types.CHECK.value]:
            return self.action_types.FOLD.value, 0, 0, 0

        if equity > raise_thresh and valid_actions[self.action_types.RAISE.value]:
            frac = (equity - raise_thresh) / max(1e-6, 1.0 - raise_thresh)
            raise_amount = self._noisy_raise(frac, min_raise, max_raise, aggressive=False)
            return self.action_types.RAISE.value, raise_amount, 0, 0
        if valid_actions[self.action_types.CHECK.value]:
            return self.action_types.CHECK.value, 0, 0, 0
        if equity >= call_thresh and valid_actions[self.action_types.CALL.value]:
            return self.action_types.CALL.value, 0, 0, 0
        return self.action_types.FOLD.value, 0, 0, 0

    def _act_flop(self, observation, hands_remaining):
        return self._thompson_action(observation, hands_remaining)

    def _act_turn(self, observation, hands_remaining):
        if 2 not in self.zeroed_streets:
            community = set(c for c in observation["community_cards"] if c != -1)
            for i, (h1, h2) in enumerate(self.opp_pairs):
                if h1 in community or h2 in community:
                    self.opp_weights[i] = 0.0
            self._normalize_weights()
            self.zeroed_streets.add(2)
        return self._thompson_action(observation, hands_remaining)

    def _act_river(self, observation, hands_remaining):
        if 3 not in self.zeroed_streets:
            community = set(c for c in observation["community_cards"] if c != -1)
            for i, (h1, h2) in enumerate(self.opp_pairs):
                if h1 in community or h2 in community:
                    self.opp_weights[i] = 0.0
            self._normalize_weights()
            self.zeroed_streets.add(3)
        return self._thompson_action(observation, hands_remaining)
