"""
Runtime player for the CMU poker tournament.

Layers (in order of priority):
  1. Match-score heuristics (foldout, variance adjustment)
  2. Match-level opponent model (exploit persistent tendencies)
  3. Hand-level posterior (exploit tight range on turn/river)
  4. Deep CFR strategy network Pi (Nash baseline)
"""

import os
import time
import itertools
import numpy as np
import torch

from agents.agent import Agent
from gym_env import PokerEnv, WrappedEval
from deep_cfr.network import CFRNet, get_strategy as net_get_strategy
from deep_cfr.encoder import (
    encode_infoset, action_mask, N_ACTIONS,
    FOLD, CHECK, CALL, BET_SMALL, BET_LARGE,
    keep_pair_to_action, action_to_keep_pair, KEEP_PAIRS,
)

_STRATEGY_NET_PATH = os.path.join(
    os.path.dirname(__file__), "checkpoints", "strategy_net_p{p}_final.pt"
)

# ── constants ────────────────────────────────────────────────────────────────
FOLDOUT_RATIO        = 1.55   # fold everything if ahead by this * hands_remaining
POSTERIOR_THRESHOLD  = 15.0   # effective pairs below this -> use posterior blending
ENTROPY_BLEND_SCALE  = 0.06   # how aggressively to blend posterior into network output
DISCARD_TEMP         = 8.0    # softmax temperature for discard mixing
OPP_FOLD_THRESHOLD   = 0.65   # if opp folds >65% to our bets, increase aggression
TIME_BUDGET          = 1.2    # seconds per hand before falling back to safe action
MC_SAMPLES_DISCARD   = 40
MC_SAMPLES_EQUITY    = 60


class PlayerAgent(Agent):
    def __init__(self, stream=True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self.evaluator    = WrappedEval()

        # load strategy networks (one per position: SB=0, BB=1)
        self.strategy_nets = {}
        for p in [0, 1]:
            net = CFRNet()
            path = _STRATEGY_NET_PATH.format(p=p)
            if os.path.exists(path):
                net.load_state_dict(torch.load(path, map_location='cpu'))
                net.eval()
                self.strategy_nets[p] = net
            else:
                self.logger.warning(f"Strategy net for player {p} not found at {path}")
                self.strategy_nets[p] = net  # untrained fallback

        # ── match-level state (persists across all 1000 hands) ───────────────
        self.cumulative_chips    = 0
        self.hands_played        = 0
        self.hands_won           = 0

        # match-level opponent model
        self.opp_postflop_actions   = 0   # total postflop actions observed
        self.opp_postflop_folds     = 0   # times opp folded to our bet
        self.opp_postflop_bets      = 0   # times opp bet/raised postflop
        self.opp_vpip               = 0   # times opp voluntarily put chips in preflop
        self.opp_preflop_hands      = 0

        # ── hand-level state (reset each hand) ───────────────────────────────
        self.current_hand      = None
        self.hand_start_time   = None
        self.opp_pairs         = None    # list of (h1, h2) tuples
        self.opp_weights       = None    # np.array of posterior weights
        self.flop_cards        = None
        self.zeroed_streets    = set()
        self.equity_cache      = {}
        self.last_our_bet_action = None  # track if we bet last action

    def __name__(self):
        return "PlayerAgent"

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry points
    # ─────────────────────────────────────────────────────────────────────────

    def observe(self, observation, reward, terminated, truncated, info):
        self.cumulative_chips += reward
        if terminated:
            if reward > 0:
                self.hands_won += 1
            self.hands_played += 1
            return

        # update match-level opponent model
        street = observation["street"]
        if street >= 1:
            opp_bet = observation["opp_bet"]
            my_bet  = observation["my_bet"]
            if opp_bet > my_bet:
                self.opp_postflop_bets += 1

            # did opp just fold after our bet?
            # (detected via reward signal in terminated branch above,
            #  so we track our last action separately)

        # update posterior from new information
        if street >= 1:
            self._maybe_init_posterior(observation)
            self._maybe_zero_posterior(observation, street)
            if opp_bet > my_bet:
                self._update_posterior_bet(observation)
            elif opp_bet == my_bet:
                self._update_posterior_check(observation)

    def act(self, observation, reward, terminated, truncated, info):
        hand_number    = info.get("hand_number", 0)
        hands_remaining = max(1, 1000 - hand_number)

        # ── reset hand state ─────────────────────────────────────────────────
        if hand_number != self.current_hand:
            self.current_hand    = hand_number
            self.hand_start_time = time.time()
            self.opp_pairs       = None
            self.opp_weights     = None
            self.flop_cards      = None
            self.zeroed_streets  = set()
            self.equity_cache    = {}
            self.last_our_bet_action = None

        # ── layer 1: match-score heuristics ──────────────────────────────────
        if self.cumulative_chips > FOLDOUT_RATIO * hands_remaining:
            return self._safe_action(observation)

        # ── time budget fallback ──────────────────────────────────────────────
        if time.time() - self.hand_start_time > TIME_BUDGET:
            return self._safe_action(observation)

        # ── handle discard ───────────────────────────────────────────────────
        if observation["valid_actions"][self.action_types.DISCARD.value]:
            return self._act_discard(observation)

        # ── init posterior once discards are known ───────────────────────────
        self._maybe_init_posterior(observation)

        # ── get network strategy ─────────────────────────────────────────────
        position = observation["blind_position"]
        net      = self.strategy_nets[position]
        vec      = encode_infoset(observation, is_discard_node=False)
        valid    = observation["valid_actions"]
        mask     = np.zeros(N_ACTIONS, dtype=np.float32)
        # map gym valid_actions to our action indices
        mask[FOLD]      = float(valid[self.action_types.FOLD.value])
        mask[CHECK]     = float(valid[self.action_types.CHECK.value])
        mask[CALL]      = float(valid[self.action_types.CALL.value])
        mask[BET_SMALL] = float(valid[self.action_types.RAISE.value])
        mask[BET_LARGE] = float(valid[self.action_types.RAISE.value])

        probs = net_get_strategy(net, vec, mask)

        # ── layer 3: posterior blending ───────────────────────────────────────
        street = observation["street"]
        if street >= 2 and self.opp_pairs is not None:
            probs = self._blend_posterior(observation, probs, mask)

        # ── layer 2: match-level exploit ──────────────────────────────────────
        probs = self._apply_opponent_model(observation, probs, mask)

        # ── layer 4: variance adjustment ─────────────────────────────────────
        probs = self._variance_adjustment(
            probs, mask, hands_remaining)

        # ── sample action ────────────────────────────────────────────────────
        probs = probs * mask
        if probs.sum() == 0:
            return self._safe_action(observation)
        probs /= probs.sum()

        action_idx = int(np.random.choice(N_ACTIONS, p=probs))
        return self._idx_to_gym_action(action_idx, observation)

    # ─────────────────────────────────────────────────────────────────────────
    # Discard
    # ─────────────────────────────────────────────────────────────────────────

    def _act_discard(self, observation):
        """
        Use network strategy for discard, blended with equity-based softmax.
        Network handles deception (mixing); equity ensures we don't keep garbage.
        """
        position = observation["blind_position"]
        net      = self.strategy_nets[position]

        vec  = encode_infoset(observation, is_discard_node=True)
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for i in range(10):
            mask[5 + i] = 1.0

        # network strategy (handles mixing for deception)
        net_probs = net_get_strategy(net, vec, mask)

        # equity-based discard probs (ensure we don't keep garbage)
        my_cards  = [c for c in observation["my_cards"] if c != -1]
        flop      = [c for c in observation["community_cards"] if c != -1]
        opp_disc  = [c for c in observation["opp_discarded_cards"] if c != -1]
        dead      = set(my_cards) | set(flop) | set(opp_disc)

        from eval import equity_all_pairs
        equities  = equity_all_pairs(my_cards, flop, dead,
                                     n_mc=MC_SAMPLES_DISCARD)
        eq_logits = DISCARD_TEMP * equities
        eq_logits -= eq_logits.max()
        eq_probs  = np.exp(eq_logits)
        eq_probs /= eq_probs.sum()

        # full action vector
        eq_full = np.zeros(N_ACTIONS, dtype=np.float32)
        eq_full[5:15] = eq_probs

        # blend: 50% network (deception), 50% equity (strength)
        probs = 0.5 * net_probs + 0.5 * eq_full
        probs = probs * mask
        probs /= probs.sum()

        action_idx = int(np.random.choice(N_ACTIONS, p=probs))
        ki, kj     = action_to_keep_pair(action_idx)
        return self.action_types.DISCARD.value, 0, ki, kj

    # ─────────────────────────────────────────────────────────────────────────
    # Posterior management
    # ─────────────────────────────────────────────────────────────────────────

    def _maybe_init_posterior(self, observation):
        """Initialize posterior once both players have discarded."""
        opp_discards_known = all(
            c != -1 for c in observation["opp_discarded_cards"])
        if not opp_discards_known or self.opp_pairs is not None:
            return

        my_cards   = set(c for c in observation["my_cards"] if c != -1)
        my_discs   = set(c for c in observation["my_discarded_cards"] if c != -1)
        opp_discs  = set(c for c in observation["opp_discarded_cards"] if c != -1)
        community  = set(c for c in observation["community_cards"] if c != -1)
        self.flop_cards = [c for c in observation["community_cards"] if c != -1][:3]

        known = my_cards | my_discs | opp_discs | community
        remaining = [c for c in range(27) if c not in known]

        self.opp_pairs   = list(itertools.combinations(remaining, 2))
        self.opp_weights = np.ones(len(self.opp_pairs), dtype=np.float64)

        # apply discard likelihood
        self._update_posterior_discard(observation)
        self._normalize_weights()

    def _update_posterior_discard(self, observation):
        """
        L(h1,h2 | opp_discards, flop, our_discards)
        = softmax probability of keeping (h1,h2) from {h1,h2,d1,d2,d3}
          given the flop and dead cards.
        """
        from eval import equity_all_pairs
        opp_discs = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discs  = [c for c in observation["my_discarded_cards"] if c != -1]
        flop      = [c for c in observation["community_cards"] if c != -1][:3]
        TEMP      = 8.0

        likelihoods = np.zeros(len(self.opp_pairs), dtype=np.float64)
        for idx, (h1, h2) in enumerate(self.opp_pairs):
            orig_hand = [h1, h2] + opp_discs  # reconstruct 5-card hand
            dead = set(my_discs) | set(flop) | set(orig_hand)
            equities = equity_all_pairs(orig_hand, flop, dead, n_mc=20)
            # rank of (h1,h2) = pair index 0 in orig_hand
            sorted_eq = np.sort(equities)[::-1]
            rank = np.where(sorted_eq == equities[0])[0][0]
            likelihoods[idx] = np.exp(-TEMP * rank / 10.0)

        self.opp_weights *= likelihoods

    def _update_posterior_bet(self, observation):
        """Shift posterior toward stronger hands when opponent bets."""
        if self.opp_pairs is None:
            return
        # simple: weight by equity of each pair
        community = [c for c in observation["community_cards"] if c != -1]
        for i, (h1, h2) in enumerate(self.opp_pairs):
            if self.opp_weights[i] < 1e-9:
                continue
            from eval import mc_equity
            dead = set(community) | {h1, h2}
            eq = mc_equity([h1, h2], community, dead, n_samples=10)
            self.opp_weights[i] *= np.exp(2.0 * eq)
        self._normalize_weights()

    def _update_posterior_check(self, observation):
        """Shift posterior toward weaker hands when opponent checks."""
        if self.opp_pairs is None:
            return
        community = [c for c in observation["community_cards"] if c != -1]
        for i, (h1, h2) in enumerate(self.opp_pairs):
            if self.opp_weights[i] < 1e-9:
                continue
            from eval import mc_equity
            dead = set(community) | {h1, h2}
            eq = mc_equity([h1, h2], community, dead, n_samples=10)
            self.opp_weights[i] *= np.exp(-1.0 * eq)
        self._normalize_weights()

    def _maybe_zero_posterior(self, observation, street):
        """Zero out impossible hands (cards that appeared on board)."""
        if self.opp_pairs is None or street in self.zeroed_streets:
            return
        community = set(c for c in observation["community_cards"] if c != -1)
        for i, (h1, h2) in enumerate(self.opp_pairs):
            if h1 in community or h2 in community:
                self.opp_weights[i] = 0.0
        self._normalize_weights()
        self.zeroed_streets.add(street)

    def _normalize_weights(self):
        if self.opp_weights is None:
            return
        total = float(self.opp_weights.sum())
        if total > 1e-12:
            self.opp_weights /= total
        else:
            self.opp_weights = np.ones_like(self.opp_weights) / len(self.opp_weights)

    def _effective_range_size(self):
        """2^entropy of posterior — effective number of opponent pairs."""
        if self.opp_weights is None:
            return 120.0
        p = self.opp_weights / (self.opp_weights.sum() + 1e-12)
        p = p[p > 1e-12]
        entropy = -np.sum(p * np.log2(p))
        return float(2 ** entropy)

    # ─────────────────────────────────────────────────────────────────────────
    # Posterior blending
    # ─────────────────────────────────────────────────────────────────────────

    def _blend_posterior(self, observation, net_probs, mask):
        """
        On turn/river, blend network output with direct EV calculation
        against the posterior. More weight to posterior as range tightens.
        """
        eff = self._effective_range_size()
        if eff >= POSTERIOR_THRESHOLD:
            return net_probs

        # compute EV of each action against posterior
        ev_probs = self._direct_ev_probs(observation, mask)
        if ev_probs is None:
            return net_probs

        # blend weight: 0 at threshold, 1 at effective_range=1
        weight = max(0.0, (POSTERIOR_THRESHOLD - eff) / POSTERIOR_THRESHOLD)
        weight = min(weight, 0.85)  # cap at 85% posterior weight

        blended = (1 - weight) * net_probs + weight * ev_probs
        return blended

    def _direct_ev_probs(self, observation, mask):
        """
        Compute action EVs directly against posterior distribution.
        Returns softmax over EVs as probability vector.
        """
        my_cards  = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        pot       = observation["pot_size"]
        call_amt  = observation["opp_bet"] - observation["my_bet"]

        if len(my_cards) != 2:
            return None

        # equity against posterior
        equity = self._posterior_equity(my_cards, community)

        # simple EV estimates
        ev = np.full(N_ACTIONS, -np.inf, dtype=np.float32)
        if mask[FOLD] > 0:
            ev[FOLD]  = 0.0
        if mask[CHECK] > 0:
            ev[CHECK] = equity * pot - (1 - equity) * pot
        if mask[CALL] > 0:
            ev[CALL]  = equity * (pot + call_amt) - (1 - equity) * call_amt
        if mask[BET_SMALL] > 0:
            bet_s = max(2, pot // 3)
            fold_eq = self._fold_equity_estimate()
            ev[BET_SMALL] = (fold_eq * pot +
                             (1-fold_eq) * (equity*(pot+bet_s) - (1-equity)*bet_s))
        if mask[BET_LARGE] > 0:
            bet_l = observation.get("max_raise", pot)
            fold_eq = self._fold_equity_estimate()
            ev[BET_LARGE] = (fold_eq * pot +
                             (1-fold_eq) * (equity*(pot+bet_l) - (1-equity)*bet_l))

        # softmax over legal actions
        legal_ev = np.where(mask > 0, ev, -np.inf)
        legal_ev -= legal_ev[mask > 0].max()
        probs = np.where(mask > 0, np.exp(legal_ev / 0.5), 0.0)
        if probs.sum() > 0:
            probs /= probs.sum()
        return probs

    def _posterior_equity(self, my_cards, community):
        """Equity against weighted posterior."""
        if self.opp_pairs is None:
            return 0.5
        from eval import mc_equity
        total_w = 0.0
        total_eq = 0.0
        dead_base = set(my_cards) | set(community)
        for i, (h1, h2) in enumerate(self.opp_pairs):
            w = self.opp_weights[i]
            if w < 1e-4:
                continue
            dead = dead_base | {h1, h2}
            eq = mc_equity(my_cards, community, dead, n_samples=MC_SAMPLES_EQUITY)
            total_eq += w * eq
            total_w  += w
        return total_eq / total_w if total_w > 0 else 0.5

    def _fold_equity_estimate(self):
        """Estimate fold equity from match-level opponent model."""
        if self.opp_postflop_actions < 10:
            return 0.5  # not enough data
        fold_rate = self.opp_postflop_folds / max(self.opp_postflop_actions, 1)
        return float(np.clip(fold_rate, 0.1, 0.9))

    # ─────────────────────────────────────────────────────────────────────────
    # Match-level opponent model
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_opponent_model(self, observation, probs, mask):
        """
        If opponent is over-folding, increase our bet frequency.
        If opponent is over-aggressive, tighten calling range.
        """
        if self.opp_postflop_actions < 20:
            return probs  # not enough data

        fold_rate = self.opp_postflop_folds / max(self.opp_postflop_actions, 1)

        if fold_rate > OPP_FOLD_THRESHOLD:
            # shift probability from check toward bet
            excess_fold = fold_rate - OPP_FOLD_THRESHOLD
            shift = min(0.3, excess_fold * 1.5)
            if mask[BET_SMALL] > 0 and mask[CHECK] > 0:
                transfer = shift * probs[CHECK]
                probs[CHECK]    -= transfer
                probs[BET_SMALL]+= transfer * 0.7
                probs[BET_LARGE]+= transfer * 0.3

        return probs

    # ─────────────────────────────────────────────────────────────────────────
    # Variance adjustment
    # ─────────────────────────────────────────────────────────────────────────

    def _variance_adjustment(self, probs, mask, hands_remaining):
        """
        When behind: increase variance (call more, fold less, bet bigger).
        When ahead by a lot: handled by foldout heuristic above.
        When marginally ahead: slight tightening.
        """
        ev_per_hand = self.cumulative_chips / max(self.hands_played, 1)
        if ev_per_hand >= 0 or hands_remaining < 50:
            return probs  # not behind or too late to matter

        # we're losing: shift toward higher variance actions
        deficit = abs(ev_per_hand) * hands_remaining
        urgency = min(1.0, deficit / 100.0)  # scale 0-1

        shift = urgency * 0.15
        # reduce fold, increase call/bet_large
        if mask[FOLD] > 0 and mask[CALL] > 0:
            transfer = shift * probs[FOLD]
            probs[FOLD] -= transfer
            probs[CALL] += transfer
        if mask[BET_SMALL] > 0 and mask[BET_LARGE] > 0:
            transfer = shift * probs[BET_SMALL] * 0.5
            probs[BET_SMALL] -= transfer
            probs[BET_LARGE] += transfer

        return probs

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _safe_action(self, observation):
        """Minimal safe action: check if possible, else fold."""
        valid = observation["valid_actions"]
        if valid[self.action_types.DISCARD.value]:
            return self.action_types.DISCARD.value, 0, 0, 1
        if valid[self.action_types.CHECK.value]:
            return self.action_types.CHECK.value, 0, 0, 0
        return self.action_types.FOLD.value, 0, 0, 0

    def _idx_to_gym_action(self, action_idx, observation):
        """Convert our action index to gym (action_type, amount, k1, k2)."""
        valid   = observation["valid_actions"]
        min_r   = observation["min_raise"]
        max_r   = observation["max_raise"]
        pot     = observation["pot_size"]
        at      = self.action_types

        if action_idx == FOLD:
            return at.FOLD.value, 0, 0, 0
        elif action_idx == CHECK:
            if valid[at.CHECK.value]:
                return at.CHECK.value, 0, 0, 0
            return at.CALL.value, 0, 0, 0
        elif action_idx == CALL:
            if valid[at.CALL.value]:
                return at.CALL.value, 0, 0, 0
            return at.CHECK.value, 0, 0, 0
        elif action_idx == BET_SMALL:
            if valid[at.RAISE.value]:
                amount = int(np.clip(pot // 3, min_r, max_r))
                return at.RAISE.value, amount, 0, 0
            return at.CHECK.value, 0, 0, 0
        elif action_idx == BET_LARGE:
            if valid[at.RAISE.value]:
                amount = int(np.clip(max_r, min_r, max_r))
                return at.RAISE.value, amount, 0, 0
            return at.CHECK.value, 0, 0, 0
        else:
            # discard action — shouldn't reach here in betting context
            return at.FOLD.value, 0, 0, 0
