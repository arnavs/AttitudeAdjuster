"""
Runtime player for the CMU poker tournament.

Uses trained Deep CFR strategy networks (betting + discard, one per player position)
layered with:
  - Hand-level posterior (Bayesian range inference from discards + betting)
  - Match-level opponent model (fold frequency, aggression)
  - Match-score heuristics (foldout, variance adjustment)
"""

import os
import sys
import time
import itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import Agent
from gym_env import PokerEnv, WrappedEval
from encoder import (
    encode_infoset, betting_mask, discard_mask,
    FOLD, CHECK, CALL, BET_SMALL, BET_MED, BET_LARGE,
    discard_action_to_keep_pair,
)
from network import make_betting_net, make_discard_net, get_policy_distribution

_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

FOLDOUT_RATIO       = 1.55
POSTERIOR_THRESHOLD = 15.0
TIME_BUDGET         = 2


class PlayerAgent(Agent):
    def __init__(self, stream=True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self.evaluator    = WrappedEval()

        # load strategy networks
        self.bet_nets = {}
        self.disc_nets = {}
        for p in [0, 1]:
            bn = make_betting_net()
            dn = make_discard_net()
            bp = os.path.join(_CKPT_DIR, f"strategy_betting_p{p}_final.pt")
            dp = os.path.join(_CKPT_DIR, f"strategy_discard_p{p}_final.pt")
            if not os.path.exists(bp):
                raise FileNotFoundError(f"Betting net not found: {bp}")
            if not os.path.exists(dp):
                raise FileNotFoundError(f"Discard net not found: {dp}")
            bn.load_state_dict(torch.load(bp, map_location='cpu'))
            dn.load_state_dict(torch.load(dp, map_location='cpu'))
            bn.eval(); dn.eval()
            self.bet_nets[p]  = bn
            self.disc_nets[p] = dn

        # match-level state
        self.cumulative_chips     = 0
        self.hands_played         = 0

        # hand-level state
        self.current_hand   = None
        self.hand_start     = None
        self.opp_pairs      = None
        self.opp_weights    = None
        self.zeroed_streets = set()
        self.posterior_events = set()

    def __name__(self):
        return "PlayerAgent"

    # ── observe ───────────────────────────────────────────────────────────────

    def observe(self, observation, reward, terminated, truncated, info):
        self.cumulative_chips += reward
        if terminated:
            self.hands_played += 1
            return

        street = observation["street"]
        acting_agent = observation.get("acting_agent")
        if street >= 1 and acting_agent == observation["blind_position"]:
            self._update_posterior(observation)

    # ── act ───────────────────────────────────────────────────────────────────

    def act(self, observation, reward, terminated, truncated, info):
        hand_number     = info.get("hand_number", 0)
        hands_remaining = max(1, 1000 - hand_number)

        # reset hand state
        if hand_number != self.current_hand:
            self.current_hand   = hand_number
            self.hand_start     = time.time()
            self.opp_pairs      = None
            self.opp_weights    = None
            self.zeroed_streets = set()
            self.posterior_events = set()

        # layer 1: foldout heuristic
        if self.cumulative_chips > FOLDOUT_RATIO * hands_remaining:
            return self._safe_action(observation)

        # time budget
        if time.time() - self.hand_start > TIME_BUDGET:
            return self._safe_action(observation)

        # discard
        if observation["valid_actions"][self.action_types.DISCARD.value]:
            return self._act_discard(observation)

        # init posterior once discards are known
        if all(c != -1 for c in observation["opp_discarded_cards"]) and self.opp_pairs is None:
            self._init_posterior(observation)

        # get network strategy
        position = observation["blind_position"]
        vec      = encode_infoset(observation, is_discard_node=False)
        mask     = betting_mask(observation["valid_actions"])

        if mask.sum() == 0:
            return self._safe_action(observation)

        probs = get_policy_distribution(self.bet_nets[position], vec, mask)

        # layer 2: posterior blending on turn/river
        if observation["street"] >= 2 and self.opp_pairs is not None:
            probs = self._blend_posterior(observation, probs, mask)

        # normalize and sample
        probs = probs * mask
        if probs.sum() == 0:
            return self._safe_action(observation)
        probs /= probs.sum()

        action = int(np.random.choice(len(probs), p=probs))
        return self._to_gym(action, observation)

    # ── discard ───────────────────────────────────────────────────────────────

    def _act_discard(self, observation):
        position = observation["blind_position"]
        vec      = encode_infoset(observation, is_discard_node=True)
        mask     = discard_mask()
        probs    = get_policy_distribution(self.disc_nets[position], vec, mask)
        action   = int(np.argmax(probs))
        ki, kj   = discard_action_to_keep_pair(action)
        # sanity check: fall back if net's pick has terrible equity
        my_cards  = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        if self._fast_equity(my_cards[ki], my_cards[kj], community, n=20) < 0.25:
            return self._heuristic_discard(observation)
        return self.action_types.DISCARD.value, 0, ki, kj

    def _heuristic_discard(self, observation):
        """Fallback: keep the pair with highest flop equity."""
        my_cards  = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        keep_pairs = list(itertools.combinations(range(len(my_cards)), 2))
        best_eq, best_ki, best_kj = -1, 0, 1
        for ki, kj in keep_pairs:
            eq = self._fast_equity(my_cards[ki], my_cards[kj], community, n=30)
            if eq > best_eq:
                best_eq, best_ki, best_kj = eq, ki, kj
        return self.action_types.DISCARD.value, 0, best_ki, best_kj

    # ── posterior ─────────────────────────────────────────────────────────────

    def _init_posterior(self, observation):
        my_cards  = set(c for c in observation["my_cards"] if c != -1)
        my_discs  = set(c for c in observation["my_discarded_cards"] if c != -1)
        opp_discs = set(c for c in observation["opp_discarded_cards"] if c != -1)
        community = set(c for c in observation["community_cards"] if c != -1)

        known     = my_cards | my_discs | opp_discs | community
        remaining = [c for c in range(27) if c not in known]

        self.opp_pairs   = list(itertools.combinations(remaining, 2))
        self.opp_weights = np.ones(len(self.opp_pairs), dtype=np.float64)
        self._normalize()
        self._update_posterior_discard(observation)

    def _update_posterior_discard(self, observation):
        """Weight by relative discard likelihood: Pr[keep (h1,h2) | full hand, flop]."""
        community = [c for c in observation["community_cards"] if c != -1]
        opp_discs = [c for c in observation["opp_discarded_cards"] if c != -1]
        DISCARD_TEMP = 6.0
        KEEP_PAIRS = list(itertools.combinations(range(5), 2))

        my_discs  = set(c for c in observation["my_discarded_cards"] if c != -1)
        we_are_bb = observation["blind_position"] == 1
        # SB discards after seeing BB's discards; BB discards first (blind)
        opp_is_sb = we_are_bb

        for i, (h1, h2) in enumerate(self.opp_pairs):
            if self.opp_weights[i] < 1e-9:
                continue
            full_hand = [h1, h2] + opp_discs
            dead = set(full_hand)
            if opp_is_sb:
                dead |= my_discs
            equities = []
            for ki, kj in KEEP_PAIRS:
                k1, k2 = full_hand[ki], full_hand[kj]
                equities.append(self._fast_equity(k1, k2, community, n=20, extra_dead=dead))
            equities = np.array(equities)
            log_probs = DISCARD_TEMP * equities
            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            probs /= probs.sum()
            # action 0 = keep (0,1) = keep (h1, h2)
            self.opp_weights[i] *= probs[0]
        self._normalize()

    def _update_posterior(self, observation):
        if self.opp_pairs is None:
            return

        event = (
            observation["street"],
            observation["my_bet"],
            observation["opp_bet"],
            tuple(observation["community_cards"]),
            tuple(observation["opp_discarded_cards"]),
        )
        if event in self.posterior_events:
            return

        street = observation["street"]
        if street not in self.zeroed_streets:
            community = set(c for c in observation["community_cards"] if c != -1)
            for i, (h1, h2) in enumerate(self.opp_pairs):
                if h1 in community or h2 in community:
                    self.opp_weights[i] = 0.0
            self._normalize()
            self.zeroed_streets.add(street)

        opp_bet = observation["opp_bet"]
        my_bet  = observation["my_bet"]
        if opp_bet > my_bet:
            self._update_raise(observation)
        elif opp_bet == my_bet:
            self._update_check(observation)

        self.posterior_events.add(event)

    def _opp_known_dead(self, observation):
        """Cards dead from opponent's perspective: their discards + our discards."""
        my_discs  = set(c for c in observation["my_discarded_cards"] if c != -1)
        opp_discs = set(c for c in observation["opp_discarded_cards"] if c != -1)
        return my_discs | opp_discs

    def _update_raise(self, observation):
        community = [c for c in observation["community_cards"] if c != -1]
        dead = self._opp_known_dead(observation)
        for i, (h1, h2) in enumerate(self.opp_pairs):
            if self.opp_weights[i] < 1e-9:
                continue
            eq = self._fast_equity(h1, h2, community, extra_dead=dead)
            self.opp_weights[i] *= np.exp(2.0 * eq)
        self._normalize()

    def _update_check(self, observation):
        community = [c for c in observation["community_cards"] if c != -1]
        dead = self._opp_known_dead(observation)
        for i, (h1, h2) in enumerate(self.opp_pairs):
            if self.opp_weights[i] < 1e-9:
                continue
            eq = self._fast_equity(h1, h2, community, extra_dead=dead)
            self.opp_weights[i] *= np.exp(-1.0 * eq)
        self._normalize()

    def _fast_equity(self, h1, h2, community, n=10, extra_dead=None):
        """Quick MC equity for posterior updates."""
        dead = set(community) | {h1, h2}
        if extra_dead:
            dead |= extra_dead
        live = [c for c in range(27) if c not in dead]
        n_remaining = 5 - len(community)
        wins = 0.0
        for _ in range(n):
            sample = np.random.choice(live, size=2 + n_remaining, replace=False)
            r1, r2 = int(sample[0]), int(sample[1])
            board5 = community + [int(c) for c in sample[2:]]
            our = self.evaluator.evaluate(
                [PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)],
                [PokerEnv.int_to_card(c) for c in board5])
            opp = self.evaluator.evaluate(
                [PokerEnv.int_to_card(r1), PokerEnv.int_to_card(r2)],
                [PokerEnv.int_to_card(c) for c in board5])
            if our < opp:
                wins += 1.0
            elif our == opp:
                wins += 0.5
        return wins / n

    def _hand_vs_hand_equity(self, hero_cards, vill_cards, community, n=10, extra_dead=None):
        """MC equity for a fixed hero hand against a fixed villain hand."""
        dead = set(community) | set(hero_cards) | set(vill_cards)
        if extra_dead:
            dead |= extra_dead
        live = [c for c in range(27) if c not in dead]
        n_remaining = 5 - len(community)
        wins = 0.0
        for _ in range(n):
            sample = np.random.choice(live, size=n_remaining, replace=False)
            board5 = community + [int(c) for c in sample]
            hero = self.evaluator.evaluate(
                [PokerEnv.int_to_card(c) for c in hero_cards],
                [PokerEnv.int_to_card(c) for c in board5])
            vill = self.evaluator.evaluate(
                [PokerEnv.int_to_card(c) for c in vill_cards],
                [PokerEnv.int_to_card(c) for c in board5])
            if hero < vill:
                wins += 1.0
            elif hero == vill:
                wins += 0.5
        return wins / n

    def _normalize(self):
        if self.opp_weights is None:
            return
        total = self.opp_weights.sum()
        if total > 1e-12:
            self.opp_weights /= total
        else:
            self.opp_weights = np.ones_like(self.opp_weights) / len(self.opp_weights)

    def _effective_range(self):
        if self.opp_weights is None:
            return 120.0
        p = self.opp_weights / (self.opp_weights.sum() + 1e-12)
        p = p[p > 1e-12]
        return float(2 ** (-np.sum(p * np.log2(p))))

    # ── posterior blending ────────────────────────────────────────────────────

    def _blend_posterior(self, observation, probs, mask):
        eff = self._effective_range()
        if eff >= POSTERIOR_THRESHOLD:
            return probs

        my_cards  = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        my_discs  = set(c for c in observation["my_discarded_cards"] if c != -1)
        opp_discs = set(c for c in observation["opp_discarded_cards"] if c != -1)
        all_discs = my_discs | opp_discs

        if len(my_cards) != 2:
            return probs

        # weighted equity against posterior
        equity = 0.0
        total_w = 0.0
        for i, (h1, h2) in enumerate(self.opp_pairs):
            w = self.opp_weights[i]
            if w < 1e-4:
                continue
            eq = self._hand_vs_hand_equity(my_cards, [h1, h2], community, n=8, extra_dead=all_discs)
            equity  += w * eq
            total_w += w
        if total_w > 0:
            equity /= total_w

        pot = observation["pot_size"]
        call_amt = observation["opp_bet"] - observation["my_bet"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        small_raise = int(np.clip(pot // 3, min_raise, max_raise)) if max_raise >= min_raise else 0
        med_raise = int(np.clip(2 * pot // 3, min_raise, max_raise)) if max_raise >= min_raise else 0
        large_raise = max_raise
        range_confidence = max(0.0, (POSTERIOR_THRESHOLD - eff) / POSTERIOR_THRESHOLD)
        pressure = max(0.0, equity - 0.5)

        def showdown_ev(commitment):
            return equity * (pot + commitment) - (1.0 - equity) * commitment

        def bet_ev(raise_amount, aggression_bonus):
            if raise_amount <= 0:
                return -1e9
            fold_equity = np.clip(
                0.06 + 0.22 * range_confidence + aggression_bonus + 0.12 * pressure,
                0.02,
                0.42,
            )
            risk_penalty = raise_amount * max(0.0, 0.56 - equity)
            return fold_equity * pot + (1.0 - fold_equity) * showdown_ev(raise_amount) - risk_penalty

        scores = np.full_like(probs, -1e9, dtype=np.float64)

        if mask[FOLD] > 0:
            scores[FOLD] = -0.25 * call_amt
        if mask[CHECK] > 0:
            scores[CHECK] = showdown_ev(0)
        if mask[CALL] > 0:
            scores[CALL] = showdown_ev(call_amt)
        if mask[BET_SMALL] > 0:
            scores[BET_SMALL] = bet_ev(small_raise, aggression_bonus=0.05)
        if mask[BET_MED] > 0:
            scores[BET_MED] = bet_ev(med_raise, aggression_bonus=0.02)
        if mask[BET_LARGE] > 0:
            scores[BET_LARGE] = bet_ev(large_raise, aggression_bonus=-0.04)

        ev_probs = np.zeros_like(probs, dtype=np.float64)
        valid_idx = np.flatnonzero(mask > 0)
        if len(valid_idx) == 0:
            return probs
        logits = scores[valid_idx]
        logits -= logits.max()
        temperature = 0.75 - 0.25 * range_confidence
        ev_probs[valid_idx] = np.exp(logits / temperature)
        if ev_probs.sum() > 0:
            ev_probs /= ev_probs.sum()

        weight = min(0.4, (POSTERIOR_THRESHOLD - eff) / POSTERIOR_THRESHOLD)
        return (1 - weight) * probs + weight * ev_probs

    # ── helpers ───────────────────────────────────────────────────────────────

    def _safe_action(self, observation):
        valid = observation["valid_actions"]
        if valid[self.action_types.DISCARD.value]:
            return self.action_types.DISCARD.value, 0, 0, 1
        if valid[self.action_types.CHECK.value]:
            return self.action_types.CHECK.value, 0, 0, 0
        return self.action_types.FOLD.value, 0, 0, 0

    def _to_gym(self, action, observation):
        at    = self.action_types
        valid = observation["valid_actions"]
        min_r = observation["min_raise"]
        max_r = observation["max_raise"]
        pot   = observation["pot_size"]

        if action == FOLD:
            return at.FOLD.value, 0, 0, 0
        elif action == CHECK:
            if valid[at.CHECK.value]:
                return at.CHECK.value, 0, 0, 0
            return at.CALL.value, 0, 0, 0
        elif action == CALL:
            if valid[at.CALL.value]:
                return at.CALL.value, 0, 0, 0
            return at.CHECK.value, 0, 0, 0
        elif action == BET_SMALL:
            if valid[at.RAISE.value]:
                amt = int(np.clip(pot // 3, min_r, max_r))
                return at.RAISE.value, amt, 0, 0
            if valid[at.CALL.value]:
                return at.CALL.value, 0, 0, 0
            return at.CHECK.value, 0, 0, 0
        elif action == BET_MED:
            if valid[at.RAISE.value]:
                amt = int(np.clip(2 * pot // 3, min_r, max_r))
                return at.RAISE.value, amt, 0, 0
            if valid[at.CALL.value]:
                return at.CALL.value, 0, 0, 0
            return at.CHECK.value, 0, 0, 0
        elif action == BET_LARGE:
            if valid[at.RAISE.value]:
                amt = int(np.clip(max_r, min_r, max_r))
                return at.RAISE.value, amt, 0, 0
            if valid[at.CALL.value]:
                return at.CALL.value, 0, 0, 0
            return at.CHECK.value, 0, 0, 0
        return self._safe_action(observation)
