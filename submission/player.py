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

        # load strategy networks (betting only; discards use heuristic)
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
        return self._heuristic_discard(observation)

    def _best_keep_by_rank(self, five_cards, board_treys):
        """Return (best_i, best_j) for strongest hand rank on this board. No MC."""
        best_rank, best_i, best_j = float('inf'), 0, 1
        for i, j in itertools.combinations(range(5), 2):
            rank = self.evaluator.evaluate(
                [PokerEnv.int_to_card(five_cards[i]), PokerEnv.int_to_card(five_cards[j])],
                board_treys)
            if rank < best_rank:
                best_rank, best_i, best_j = rank, i, j
        return best_i, best_j

    def _keep_pair_equities(self, five_cards, community, extra_dead=None, n=10):
        """Compute MC equity for all 10 keep-pairs from a 5-card hand."""
        dead = set(five_cards)
        if extra_dead:
            dead |= extra_dead
        equities = []
        for i, j in itertools.combinations(range(5), 2):
            eq = self._fast_equity(five_cards[i], five_cards[j], community, n=n, extra_dead=dead)
            equities.append(eq)
        return np.array(equities)

    # FROM BB's PERSPECTIVE
    def _sb_best_keep(self, sb_hand, community, dead_cards, n=8):
        """SB picks best keep-pair by MC equity with dead cards excluded."""
        best_eq, best_i, best_j = -1.0, 0, 1
        for i, j in itertools.combinations(range(5), 2):
            eq = self._fast_equity(sb_hand[i], sb_hand[j], community, n=n,
                                   extra_dead=dead_cards | set(sb_hand))
            if eq > best_eq:
                best_eq, best_i, best_j = eq, i, j
        return best_i, best_j

    def _precompute_sb_keeps(self, pool, community, sb_dead, n=1):
        """Precompute SB's best keep-pair (by equity) for all possible 5-card hands from pool."""
        cache = {}
        for hand in itertools.combinations(pool, 5):
            hand_list = list(hand)
            bi, bj = self._sb_best_keep(hand_list, community, sb_dead, n=n)
            cache[hand] = (hand_list[bi], hand_list[bj])
        return cache

    def _bb_keep_pair_values(self, full_hand, community, n_samples=20):
        """BB level-1: equity for each keep-pair vs SB choosing best pair."""
        n_remaining = 5 - len(community)

        values = []
        for ki, kj in itertools.combinations(range(len(full_hand)), 2):
            k1, k2 = full_hand[ki], full_hand[kj]
            bb_discards = set(full_hand) - {k1, k2}
            sb_dead = set(community) | bb_discards
            # pool excludes BB's full hand + community (BB holds k1,k2; discards are visible)
            pool = [c for c in range(27) if c not in (set(community) | set(full_hand))]

            # precompute SB's best keep (equity-based) for all possible SB hands
            sb_cache = self._precompute_sb_keeps(pool, community, sb_dead, n=4)

            wins, counted = 0.0, 0
            for _ in range(n_samples):
                if len(pool) < 5 + n_remaining:
                    break
                sampled = np.random.choice(pool, size=5 + n_remaining, replace=False)
                sb_hand_key = tuple(sorted(int(c) for c in sampled[:5]))
                r1, r2 = sb_cache[sb_hand_key]
                remaining_board = [int(c) for c in sampled[5:]]

                board5 = community + remaining_board
                board5_treys = [PokerEnv.int_to_card(c) for c in board5]
                my_rank = self.evaluator.evaluate(
                    [PokerEnv.int_to_card(k1), PokerEnv.int_to_card(k2)], board5_treys)
                opp_rank = self.evaluator.evaluate(
                    [PokerEnv.int_to_card(r1), PokerEnv.int_to_card(r2)], board5_treys)
                if my_rank < opp_rank:
                    wins += 1.0
                elif my_rank == opp_rank:
                    wins += 0.5
                counted += 1
            values.append(wins / max(counted, 1))
        return np.array(values, dtype=np.float64)

    def _discard_policy_probs(self, full_hand, community, actor_is_sb, visible_opp_discs=None,
                              n_samples=20, bb_infer_samples=8, temp=6.0):
        """Shared discard policy used both for acting and posterior updates."""
        visible_opp_discs = [c for c in (visible_opp_discs or []) if c != -1]
        keep_pairs = list(itertools.combinations(range(len(full_hand)), 2))

        if actor_is_sb and len(visible_opp_discs) == 3:
            dead = set(full_hand) | set(visible_opp_discs) | set(community)
            pool = [c for c in range(27) if c not in dead]
            n_remaining = 5 - len(community)
            bb_policy_cache = {}
            values = []

            for ki, kj in keep_pairs:
                k1, k2 = full_hand[ki], full_hand[kj]
                weighted_wins, total_weight = 0.0, 0.0
                for _ in range(n_samples):
                    if len(pool) < 2 + n_remaining:
                        break
                    sampled = np.random.choice(pool, size=2 + n_remaining, replace=False)
                    r1, r2 = int(sampled[0]), int(sampled[1])
                    bb_full = [r1, r2] + visible_opp_discs
                    bb_key = tuple(bb_full)
                    if bb_key not in bb_policy_cache:
                        bb_values = self._bb_keep_pair_values(
                            bb_full, community, n_samples=bb_infer_samples)
                        bb_logits = temp * bb_values
                        bb_logits -= bb_logits.max()
                        bb_probs = np.exp(bb_logits)
                        bb_probs /= bb_probs.sum()
                        bb_policy_cache[bb_key] = bb_probs[0]
                    weight = bb_policy_cache[bb_key]
                    if weight < 1e-9:
                        continue

                    remaining_board = [int(c) for c in sampled[2:]]
                    board5 = community + remaining_board
                    board5_treys = [PokerEnv.int_to_card(c) for c in board5]
                    my_rank = self.evaluator.evaluate(
                        [PokerEnv.int_to_card(k1), PokerEnv.int_to_card(k2)], board5_treys)
                    opp_rank = self.evaluator.evaluate(
                        [PokerEnv.int_to_card(r1), PokerEnv.int_to_card(r2)], board5_treys)
                    outcome = 1.0 if my_rank < opp_rank else 0.5 if my_rank == opp_rank else 0.0
                    weighted_wins += weight * outcome
                    total_weight += weight
                values.append(weighted_wins / max(total_weight, 1e-9))
            values = np.array(values, dtype=np.float64)
        else:
            values = self._bb_keep_pair_values(
                full_hand, community, n_samples=n_samples)

        logits = temp * values
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return values, probs

    def _heuristic_discard(self, observation):
        """BB level-1, SB level-2 using the same discard model for acting and inference."""
        my_cards  = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        opp_discs = [c for c in observation["opp_discarded_cards"] if c != -1]
        we_are_sb = observation["blind_position"] == 0

        values, _ = self._discard_policy_probs(
            my_cards, community,
            actor_is_sb=we_are_sb,
            visible_opp_discs=opp_discs,
            n_samples=24, bb_infer_samples=8, temp=6.0,
        )
        best_idx = int(np.argmax(values))
        keep_pairs = list(itertools.combinations(range(len(my_cards)), 2))
        best_ki, best_kj = keep_pairs[best_idx]
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
        """Weight by relative discard likelihood under the same discard model used for acting."""
        community = [c for c in observation["community_cards"] if c != -1]
        opp_discs = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discs  = [c for c in observation["my_discarded_cards"] if c != -1]

        we_are_bb = observation["blind_position"] == 1
        opp_is_sb = we_are_bb

        for i, (h1, h2) in enumerate(self.opp_pairs):
            if self.opp_weights[i] < 1e-9:
                continue
            full_hand = [h1, h2] + opp_discs
            _, probs = self._discard_policy_probs(
                full_hand, community,
                actor_is_sb=opp_is_sb,
                visible_opp_discs=my_discs if opp_is_sb else [],
                n_samples=12, bb_infer_samples=6, temp=6.0,
            )
            # action 0 = keep (0,1) = keep (h1, h2)
            self.opp_weights[i] *= probs[0]
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
