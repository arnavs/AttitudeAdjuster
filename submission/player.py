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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import Agent
from gym_env import PokerEnv, WrappedEval
from encoder import (
    encode_infoset, betting_mask,
    FOLD, CHECK, CALL, BET_SMALL, BET_MED, BET_LARGE,
    KEEP_PAIRS,
)
from network import make_betting_net, get_policy_distribution

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
        for p in [0, 1]:
            bn = make_betting_net()
            bp = os.path.join(_CKPT_DIR, f"strategy_betting_p{p}_final.pt")
            if not os.path.exists(bp):
                raise FileNotFoundError(f"Betting net not found: {bp}")
            bn.load_state_dict(torch.load(bp, map_location='cpu'))
            bn.eval()
            self.bet_nets[p] = bn

        # match-level state
        self.cumulative_chips     = 0
        self.hands_played         = 0

        # hand-level state
        self.current_hand   = None
        self.hand_start     = None
        self.opp_pairs      = None
        self.opp_weights    = None
        self.zeroed_streets   = set()
        self._eq_cache       = {}     # (street,) -> np.array of equities per opp pair
        self._eq_cache_street = None

    def __name__(self):
        return "PlayerAgent"

    # ── observe ───────────────────────────────────────────────────────────────

    def observe(self, observation, reward, terminated, truncated, info):
        self.cumulative_chips += reward
        if terminated:
            self.hands_played += 1
            return

        street = observation["street"]
        if street >= 1:
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
            self.zeroed_streets   = set()
            self._eq_cache       = {}
            self._eq_cache_street = None

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
        """MC equity against random opponent (Level 1). Pick best keep-pair."""
        my_cards  = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        opp_discs = [c for c in observation["opp_discarded_cards"] if c != -1]

        dead = set(my_cards) | set(community) | set(opp_discs)
        pool = [c for c in range(27) if c not in dead]
        n_remaining = 5 - len(community)
        n_samples = 200
        temp = 6.0
        flop_cards = [PokerEnv.int_to_card(c) for c in community[:3]]

        wins = np.zeros(len(KEEP_PAIRS), dtype=np.float64)
        for _ in range(n_samples):
            sampled = np.random.choice(pool, size=5 + n_remaining, replace=False)
            opp_hand = [int(c) for c in sampled[:5]]
            runout = [int(c) for c in sampled[5:]]

            # opponent keeps pair via MC equity softmax (Level 1.5)
            opp_dead = set(opp_hand) | set(community) | set(opp_discs)
            opp_pool = [c for c in range(27) if c not in opp_dead]
            opp_eq = np.zeros(len(KEEP_PAIRS), dtype=np.float64)
            n_inner = 5
            for oi_idx, (oi, oj) in enumerate(KEEP_PAIRS):
                ok1, ok2 = opp_hand[oi], opp_hand[oj]
                ow = 0.0
                for _ in range(n_inner):
                    inner_s = np.random.choice(opp_pool, size=2 + n_remaining, replace=False)
                    ib = community + [int(c) for c in inner_s[2:]]
                    ib_cards = [PokerEnv.int_to_card(c) for c in ib]
                    mr = self.evaluator.evaluate(
                        [PokerEnv.int_to_card(ok1), PokerEnv.int_to_card(ok2)], ib_cards)
                    vr = self.evaluator.evaluate(
                        [PokerEnv.int_to_card(int(inner_s[0])), PokerEnv.int_to_card(int(inner_s[1]))], ib_cards)
                    ow += 1.0 if mr < vr else 0.5 if mr == vr else 0.0
                opp_eq[oi_idx] = ow / n_inner
            opp_logits = temp * opp_eq
            opp_logits -= opp_logits.max()
            opp_probs = np.exp(opp_logits)
            opp_probs /= opp_probs.sum()
            opp_action = int(np.random.choice(len(KEEP_PAIRS), p=opp_probs))
            opp_k1, opp_k2 = opp_hand[KEEP_PAIRS[opp_action][0]], opp_hand[KEEP_PAIRS[opp_action][1]]

            board5 = community + runout
            board5_treys = [PokerEnv.int_to_card(c) for c in board5]
            opp_rank = self.evaluator.evaluate(
                [PokerEnv.int_to_card(opp_k1), PokerEnv.int_to_card(opp_k2)], board5_treys)
            for idx, (ki, kj) in enumerate(KEEP_PAIRS):
                my_rank = self.evaluator.evaluate(
                    [PokerEnv.int_to_card(my_cards[ki]), PokerEnv.int_to_card(my_cards[kj])], board5_treys)
                if my_rank < opp_rank:
                    wins[idx] += 1.0
                elif my_rank == opp_rank:
                    wins[idx] += 0.5
        equities = wins / n_samples

        logits = temp * equities
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(np.random.choice(len(KEEP_PAIRS), p=probs))
        best_ki, best_kj = KEEP_PAIRS[action]

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
        """Weight by how likely opponent kept each candidate pair (Level 1: best equity)."""
        opp_discs = [c for c in observation["opp_discarded_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        flop_cards = [PokerEnv.int_to_card(c) for c in community[:3]]

        for i, (h1, h2) in enumerate(self.opp_pairs):
            if self.opp_weights[i] < 1e-9:
                continue
            # reconstruct opponent's full hand
            opp_full = [h1, h2] + opp_discs
            # rank each keep-pair; lower rank = better hand
            keep_rank = self.evaluator.evaluate(
                [PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)], flop_cards)
            is_best = True
            for ki, kj in KEEP_PAIRS:
                if (ki, kj) == (0, 1):
                    continue
                alt_rank = self.evaluator.evaluate(
                    [PokerEnv.int_to_card(opp_full[ki]), PokerEnv.int_to_card(opp_full[kj])], flop_cards)
                if alt_rank < keep_rank:
                    is_best = False
                    break
            # if keeping (h1,h2) isn't the best choice, downweight
            self.opp_weights[i] *= 1.0 if is_best else 0.75

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

    def _opp_known_dead(self, observation):
        """Cards dead from opponent's perspective: their discards + our discards."""
        my_discs  = set(c for c in observation["my_discarded_cards"] if c != -1)
        opp_discs = set(c for c in observation["opp_discarded_cards"] if c != -1)
        return my_discs | opp_discs

    def _get_cached_equities(self, observation):
        """Compute & cache equities for all opp pairs once per street."""
        street = observation["street"]
        if self._eq_cache_street == street:
            return self._eq_cache
        community = [c for c in observation["community_cards"] if c != -1]
        dead = self._opp_known_dead(observation)
        eqs = np.zeros(len(self.opp_pairs))
        for i, (h1, h2) in enumerate(self.opp_pairs):
            if self.opp_weights[i] < 1e-9:
                continue
            eqs[i] = self._fast_equity(h1, h2, community, extra_dead=dead)
        self._eq_cache = eqs
        self._eq_cache_street = street
        return eqs

    def _update_raise(self, observation):
        eqs = self._get_cached_equities(observation)
        for i in range(len(self.opp_pairs)):
            if self.opp_weights[i] < 1e-9:
                continue
            self.opp_weights[i] *= np.exp(2.0 * eqs[i])
        self._normalize()

    def _update_check(self, observation):
        eqs = self._get_cached_equities(observation)
        for i in range(len(self.opp_pairs)):
            if self.opp_weights[i] < 1e-9:
                continue
            self.opp_weights[i] *= np.exp(-1.0 * eqs[i])
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
        if mask[BET_MED] > 0:
            ev_probs[BET_MED]   = equity * pot * 1.67
        if mask[BET_LARGE] > 0:
            ev_probs[BET_LARGE] = equity * pot * 2.0

        # softmax
        ev_probs = ev_probs * mask
        ev_probs -= ev_probs[mask > 0].max()
        ev_probs  = np.where(mask > 0, np.exp(ev_probs / 0.5), 0.0)
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
