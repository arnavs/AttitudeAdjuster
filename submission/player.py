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
    FOLD, CHECK, CALL, BET_SMALL, BET_LARGE,
    discard_action_to_keep_pair, N_DISCARD_ACTIONS,
)
from network import make_betting_net, make_discard_net, get_strategy

_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

FOLDOUT_RATIO       = 1.55
POSTERIOR_THRESHOLD = 15.0
TIME_BUDGET         = 1.2
OPP_FOLD_THRESHOLD  = 0.65


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
            if os.path.exists(bp):
                bn.load_state_dict(torch.load(bp, map_location='cpu'))
            else:
                self.logger.warning(f"Betting net p{p} not found, using random")
            if os.path.exists(dp):
                dn.load_state_dict(torch.load(dp, map_location='cpu'))
            else:
                self.logger.warning(f"Discard net p{p} not found, using random")
            bn.eval(); dn.eval()
            self.bet_nets[p]  = bn
            self.disc_nets[p] = dn

        # match-level state
        self.cumulative_chips     = 0
        self.hands_played         = 0
        self.opp_postflop_actions = 0
        self.opp_postflop_folds   = 0
        self.last_action_was_bet  = False

        # hand-level state
        self.current_hand   = None
        self.hand_start     = None
        self.opp_pairs      = None
        self.opp_weights    = None
        self.zeroed_streets = set()

    def __name__(self):
        return "PlayerAgent"

    # ── observe ───────────────────────────────────────────────────────────────

    def observe(self, observation, reward, terminated, truncated, info):
        self.cumulative_chips += reward
        if terminated:
            self.hands_played += 1
            # track if opp folded to our bet
            if reward > 0 and self.last_action_was_bet:
                self.opp_postflop_folds += 1
            return

        street = observation["street"]
        if street >= 1:
            self.opp_postflop_actions += 1
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

        # layer 1: foldout heuristic
        if self.cumulative_chips > FOLDOUT_RATIO * hands_remaining:
            return self._safe_action(observation)

        # time budget
        if time.time() - self.hand_start > TIME_BUDGET:
            return self._safe_action(observation)

        # discard
        if observation["valid_actions"][self.action_types.DISCARD.value]:
            self.last_action_was_bet = False
            return self._act_discard(observation)

        # init posterior once discards are known
        self._maybe_init_posterior(observation)

        # get network strategy
        position = observation["blind_position"]
        vec      = encode_infoset(observation, is_discard_node=False)
        mask     = betting_mask(observation["valid_actions"])

        if mask.sum() == 0:
            return self._safe_action(observation)

        probs = get_strategy(self.bet_nets[position], vec, mask)

        # layer 2: posterior blending on turn/river
        if observation["street"] >= 2 and self.opp_pairs is not None:
            probs = self._blend_posterior(observation, probs, mask)

        # layer 3: opponent model exploit
        probs = self._apply_opp_model(probs, mask)

        # layer 4: variance adjustment
        probs = self._variance_adjust(probs, mask, hands_remaining)

        # normalize and sample
        probs = probs * mask
        if probs.sum() == 0:
            return self._safe_action(observation)
        probs /= probs.sum()

        action = int(np.random.choice(len(probs), p=probs))
        self.last_action_was_bet = action in (BET_SMALL, BET_LARGE)
        return self._to_gym(action, observation)

    # ── discard ───────────────────────────────────────────────────────────────

    def _act_discard(self, observation):
        position = observation["blind_position"]
        vec      = encode_infoset(observation, is_discard_node=True)
        mask     = discard_mask()
        probs    = get_strategy(self.disc_nets[position], vec, mask)
        action   = int(np.random.choice(N_DISCARD_ACTIONS, p=probs))
        ki, kj   = discard_action_to_keep_pair(action)
        return self.action_types.DISCARD.value, 0, ki, kj

    # ── posterior ─────────────────────────────────────────────────────────────

    def _maybe_init_posterior(self, observation):
        opp_known = all(c != -1 for c in observation["opp_discarded_cards"])
        if not opp_known or self.opp_pairs is not None:
            return

        my_cards  = set(c for c in observation["my_cards"] if c != -1)
        my_discs  = set(c for c in observation["my_discarded_cards"] if c != -1)
        opp_discs = set(c for c in observation["opp_discarded_cards"] if c != -1)
        community = set(c for c in observation["community_cards"] if c != -1)

        known     = my_cards | my_discs | opp_discs | community
        remaining = [c for c in range(27) if c not in known]

        self.opp_pairs   = list(itertools.combinations(remaining, 2))
        self.opp_weights = np.ones(len(self.opp_pairs), dtype=np.float64)
        self._normalize()

    def _update_posterior(self, observation):
        if self.opp_pairs is None:
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

    def _update_raise(self, observation):
        community = [c for c in observation["community_cards"] if c != -1]
        for i, (h1, h2) in enumerate(self.opp_pairs):
            if self.opp_weights[i] < 1e-9:
                continue
            eq = self._fast_equity(h1, h2, community)
            self.opp_weights[i] *= np.exp(2.0 * eq)
        self._normalize()

    def _update_check(self, observation):
        community = [c for c in observation["community_cards"] if c != -1]
        for i, (h1, h2) in enumerate(self.opp_pairs):
            if self.opp_weights[i] < 1e-9:
                continue
            eq = self._fast_equity(h1, h2, community)
            self.opp_weights[i] *= np.exp(-1.0 * eq)
        self._normalize()

    def _fast_equity(self, h1, h2, community, n=10):
        """Quick MC equity for posterior updates."""
        dead = set(community) | {h1, h2}
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

        if len(my_cards) != 2:
            return probs

        # weighted equity against posterior
        equity = 0.0
        total_w = 0.0
        for i, (h1, h2) in enumerate(self.opp_pairs):
            w = self.opp_weights[i]
            if w < 1e-4:
                continue
            eq = self._fast_equity(h1, h2, community, n=8)
            equity  += w * eq
            total_w += w
        if total_w > 0:
            equity /= total_w

        # simple EV-based probs
        ev_probs = np.zeros_like(probs)
        pot = observation["pot_size"]
        call_amt = observation["opp_bet"] - observation["my_bet"]

        if mask[FOLD] > 0:
            ev_probs[FOLD]  = 0.0
        if mask[CHECK] > 0:
            ev_probs[CHECK] = equity
        if mask[CALL] > 0:
            ev_probs[CALL]  = equity * (pot + call_amt) - (1 - equity) * call_amt
        if mask[BET_SMALL] > 0:
            ev_probs[BET_SMALL] = equity * pot * 1.33
        if mask[BET_LARGE] > 0:
            ev_probs[BET_LARGE] = equity * pot * 2.0

        # softmax
        ev_probs = ev_probs * mask
        ev_probs -= ev_probs[mask > 0].max()
        ev_probs  = np.where(mask > 0, np.exp(ev_probs / 0.5), 0.0)
        if ev_probs.sum() > 0:
            ev_probs /= ev_probs.sum()

        weight = min(0.8, (POSTERIOR_THRESHOLD - eff) / POSTERIOR_THRESHOLD)
        return (1 - weight) * probs + weight * ev_probs

    # ── opponent model ────────────────────────────────────────────────────────

    def _apply_opp_model(self, probs, mask):
        if self.opp_postflop_actions < 20:
            return probs
        fold_rate = self.opp_postflop_folds / max(self.opp_postflop_actions, 1)
        if fold_rate > OPP_FOLD_THRESHOLD and mask[BET_SMALL] > 0 and mask[CHECK] > 0:
            shift = min(0.25, (fold_rate - OPP_FOLD_THRESHOLD) * 1.5)
            transfer = shift * probs[CHECK]
            probs[CHECK]     -= transfer
            probs[BET_SMALL] += transfer * 0.6
            probs[BET_LARGE] += transfer * 0.4
        return probs

    # ── variance adjustment ───────────────────────────────────────────────────

    def _variance_adjust(self, probs, mask, hands_remaining):
        if self.hands_played == 0:
            return probs
        ev_per_hand = self.cumulative_chips / self.hands_played
        if ev_per_hand >= 0:
            return probs
        urgency = min(1.0, abs(ev_per_hand) * hands_remaining / 100.0)
        shift   = urgency * 0.12
        if mask[FOLD] > 0 and mask[CALL] > 0:
            t = shift * probs[FOLD]
            probs[FOLD] -= t
            probs[CALL] += t
        if mask[BET_SMALL] > 0 and mask[BET_LARGE] > 0:
            t = shift * probs[BET_SMALL] * 0.5
            probs[BET_SMALL] -= t
            probs[BET_LARGE] += t
        return probs

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
            return at.CHECK.value, 0, 0, 0
        elif action == BET_LARGE:
            if valid[at.RAISE.value]:
                amt = int(np.clip(max_r, min_r, max_r))
                return at.RAISE.value, amt, 0, 0
            return at.CHECK.value, 0, 0, 0
        return self._safe_action(observation)
