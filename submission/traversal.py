"""
External sampling MCCFR traversal for Deep CFR.

Two separate networks:
  betting_net:  handles fold/check/call/bet_small/bet_large
  discard_net:  handles which pair of cards to keep (10 options)

Discard order:
  BB discards first (sees own 5 cards + flop only)
  SB discards second (sees own 5 cards + flop + BB's discards)
  Then flop betting begins with BB acting first.
"""

import copy
import numpy as np
from encoder import (
    encode_infoset,
    betting_mask,
    KEEP_PAIRS, discard_action_to_keep_pair,
    FOLD, CHECK, CALL, BET_SMALL, BET_MED, BET_LARGE,
    N_BETTING_ACTIONS, N_DISCARD_ACTIONS,
)
from network import get_strategy
from solve_discard import TEMP, lookup_bb_prob, lookup_bb_probs
from gym_env import PokerEnv, WrappedEval


# ── bet sizing ────────────────────────────────────────────────────────────────

def compute_bet_sizes(pot, max_raise, min_raise):
    """
    Return abstract raise sizes in the same units as gym_env's `raise_amount`.

    small = 1/3 pot, med = 2/3 pot, large = all-in.
    All clamped to [min_raise, max_raise].
    """
    if max_raise < min_raise:
        return 0, 0, 0
    small = int(np.clip(pot // 3, min_raise, max_raise))
    med   = int(np.clip(2 * pot // 3, min_raise, max_raise))
    large = int(max_raise)
    return small, med, large


# ── game state ────────────────────────────────────────────────────────────────

class GameState:
    STARTING_STACK = 100
    SMALL_BLIND    = 1
    BIG_BLIND      = 2

    def __init__(self, deck=None):
        if deck is None:
            deck = np.arange(27)
            np.random.shuffle(deck)
        self.deck = list(deck)
        ptr = 0

        # deal 5 hole cards each, 5 community cards
        self.hole      = [self.deck[ptr:ptr+5], self.deck[ptr+5:ptr+10]]
        ptr += 10
        self.community = self.deck[ptr:ptr+5]
        ptr += 5

        # discards: -1 = not yet discarded
        self.discarded     = [[-1, -1, -1], [-1, -1, -1]]
        self.discard_done  = [False, False]

        # blinds
        self.stacks = [self.STARTING_STACK - self.SMALL_BLIND,
                       self.STARTING_STACK - self.BIG_BLIND]
        self.bets   = [self.SMALL_BLIND, self.BIG_BLIND]
        self.pot    = self.SMALL_BLIND + self.BIG_BLIND

        self.street          = 0
        self.acting_player   = 0   # SB acts first preflop
        self.min_raise       = self.BIG_BLIND
        self.last_street_bet = 0

        self.terminal = False
        self.winner   = None   # 0, 1, or -1 (tie)

    def board(self):
        """Community cards revealed so far."""
        if self.street == 0:
            return []
        return list(self.community[:self.street + 2])

    def obs(self, player):
        """Build observation dict for player."""
        board = self.board()
        padded_board = board + [-1] * (5 - len(board))

        # post-discard: show only kept cards
        if self.discard_done[player]:
            kept = [c for c in self.hole[player] if c != -1]
            padded_hole = kept + [-1] * (5 - len(kept))
        else:
            padded_hole = list(self.hole[player]) + [-1] * (5 - len(self.hole[player]))

        opp = 1 - player
        opp_disc = self.discarded[opp] if self.discard_done[opp] else [-1, -1, -1]

        return {
            "my_cards"           : padded_hole,
            "community_cards"    : padded_board,
            "my_discarded_cards" : list(self.discarded[player]),
            "opp_discarded_cards": list(opp_disc),
            "blind_position"     : player,
            "street"             : self.street,
            "pot_size"           : self.pot,
            "my_bet"             : self.bets[player],
            "opp_bet"            : self.bets[opp],
            "my_stack"           : self.stacks[player],
            "opp_stack"          : self.stacks[opp],
            "min_raise"          : self.min_raise,
            "max_raise"          : self.max_raise_amount(),
        }

    def max_raise_amount(self):
        return self.STARTING_STACK - max(self.bets)

    def legal_betting_mask(self, player):
        """6-dim binary mask of legal betting actions."""
        opp = 1 - player
        call_amount = self.bets[opp] - self.bets[player]
        mask = np.zeros(N_BETTING_ACTIONS, dtype=np.float32)

        # fold is always legal in the environment on betting nodes
        if call_amount > 0:
            mask[FOLD] = 1.0

        if call_amount == 0:
            mask[CHECK] = 1.0
        else:
            mask[CALL] = 1.0

        max_raise = self.max_raise_amount()
        if max_raise >= self.min_raise:
            small, med, large = compute_bet_sizes(self.pot, max_raise, self.min_raise)
            if small >= self.min_raise:
                mask[BET_SMALL] = 1.0
            if med >= self.min_raise:
                mask[BET_MED] = 1.0
            if large >= self.min_raise:
                mask[BET_LARGE] = 1.0

        return mask

    def apply_discard(self, player, keep_idx_1, keep_idx_2):
        """Player keeps two cards by index from their 5-card hand."""
        hand = list(self.hole[player])
        kept      = [hand[keep_idx_1], hand[keep_idx_2]]
        discarded = [hand[k] for k in range(5) if k not in (keep_idx_1, keep_idx_2)]
        self.hole[player]         = kept
        self.discarded[player]    = discarded
        self.discard_done[player] = True
        self.acting_player        = 1 - player

    def apply_bet(self, player, action):
        """
        Apply a betting action. Returns True if the street should advance.
        """
        opp = 1 - player
        call_amount = self.bets[opp] - self.bets[player]
        max_raise = self.max_raise_amount()
        small, med, large = compute_bet_sizes(self.pot, max_raise, self.min_raise)

        if action == FOLD:
            self.terminal = True
            self.winner   = opp
            return True

        elif action == CHECK:
            # street ends: BB checks preflop, or SB checks postflop
            if self.street == 0 and player == 1:
                return True
            elif self.street >= 1 and player == 0:
                return True
            else:
                self.acting_player = opp
                return False

        elif action == CALL:
            actual = min(call_amount, self.stacks[player])
            self.stacks[player] -= actual
            self.bets[player]   += actual
            self.pot            += actual

            # In heads-up preflop, the SB completing the blind does not end the street.
            if self.street == 0 and player == 0 and self.bets[player] == self.BIG_BLIND:
                self.acting_player = opp
                return False
            return True

        elif action in (BET_SMALL, BET_MED, BET_LARGE):
            raise_amount = small if action == BET_SMALL else (med if action == BET_MED else large)
            raise_amount = int(np.clip(raise_amount, self.min_raise, max_raise))
            new_bet = self.bets[opp] + raise_amount
            contribution = new_bet - self.bets[player]

            self.stacks[player] -= contribution
            self.bets[player]    = new_bet
            self.pot            += contribution

            raise_so_far = self.bets[opp] - self.last_street_bet
            max_raise = self.STARTING_STACK - max(self.bets)
            min_raise_no_limit = raise_so_far + raise_amount
            self.min_raise = max(self.BIG_BLIND, min(min_raise_no_limit, max_raise))
            self.acting_player = opp
            return False

        return False

    def advance_street(self):
        """Move to next street. Returns True if hand is over (showdown)."""
        self.street += 1
        if self.street > 3:
            self._resolve_showdown()
            return True
        self.last_street_bet = self.bets[0]
        self.min_raise       = self.BIG_BLIND
        self.acting_player   = 1  # BB leads postflop
        return False

    def _resolve_showdown(self):
        evaluator = WrappedEval()
        board5    = [PokerEnv.int_to_card(c) for c in self.community[:5]]
        h0 = [PokerEnv.int_to_card(c) for c in self.hole[0] if c != -1]
        h1 = [PokerEnv.int_to_card(c) for c in self.hole[1] if c != -1]
        r0 = evaluator.evaluate(h0, board5)
        r1 = evaluator.evaluate(h1, board5)
        if r0 < r1:
            self.winner = 0
        elif r1 < r0:
            self.winner = 1
        else:
            self.winner = -1
        self.terminal = True

    def payoff(self, player):
        """Terminal payoff to player in chips."""
        assert self.terminal
        if self.winner == -1:
            return 0
        contested = min(self.bets[0], self.bets[1])
        return contested if self.winner == player else -contested

    def clone(self):
        s = copy.copy(self)
        s.hole         = [list(h) for h in self.hole]
        s.community    = list(self.community)
        s.discarded    = [list(d) for d in self.discarded]
        s.discard_done = list(self.discard_done)
        s.stacks       = list(self.stacks)
        s.bets         = list(self.bets)
        return s


# ── traversal ─────────────────────────────────────────────────────────────────

def traverse(state, traverser,
             value_betting_nets,
             value_betting_buf,
             strategy_betting_bufs,
             iteration,
             reach_traverser=1.0, reach_opponent=1.0,
             device='cpu', bb_table=None):
    """
    External sampling MCCFR traversal.
    Returns counterfactual value for traverser at this node.
    """
    if state.terminal:
        return state.payoff(traverser)

    # both all-in post-discard: run out remaining streets to showdown
    if state.stacks[0] == 0 and state.stacks[1] == 0 and all(state.discard_done):
        while state.street <= 3 and not state.terminal:
            state.advance_street()
        return state.payoff(traverser)

    # ── discard nodes (BB first, then SB) ────────────────────────────────────
    if state.street == 1:
        if not state.discard_done[1]:  # BB discards first
            return _traverse_discard(
                state, player=1, traverser=traverser,
                value_betting_nets=value_betting_nets,
                value_betting_buf=value_betting_buf,
                strategy_betting_bufs=strategy_betting_bufs,
                iteration=iteration,
                reach_traverser=reach_traverser,
                reach_opponent=reach_opponent,
                device=device, bb_table=bb_table,
            )
        if not state.discard_done[0]:  # SB discards second
            return _traverse_discard(
                state, player=0, traverser=traverser,
                value_betting_nets=value_betting_nets,
                value_betting_buf=value_betting_buf,
                strategy_betting_bufs=strategy_betting_bufs,
                iteration=iteration,
                reach_traverser=reach_traverser,
                reach_opponent=reach_opponent,
                device=device, bb_table=bb_table,
            )

    if state.terminal:
        return state.payoff(traverser)

    # ── betting node ──────────────────────────────────────────────────────────
    player = state.acting_player
    obs    = state.obs(player)
    mask   = state.legal_betting_mask(player)

    if mask.sum() == 0:
        return state.payoff(traverser)

    vec = encode_infoset(obs, is_discard_node=False)
    strategy = get_strategy(value_betting_nets[player], vec, mask, device)

    is_traverser = (player == traverser)
    strategy_betting_bufs[player].add(vec, strategy, iteration)

    if is_traverser:
        action_values = {}
        for action in range(N_BETTING_ACTIONS):
            if mask[action] == 0:
                continue
            s2           = state.clone()
            street_ended = s2.apply_bet(player, action)
            if street_ended and not s2.terminal:
                s2.advance_street()
            action_values[action] = traverse(
                s2, traverser,
                value_betting_nets,
                value_betting_buf,
                strategy_betting_bufs,
                iteration,
                reach_traverser * strategy[action], reach_opponent,
                device, bb_table=bb_table,
            )

        node_value = sum(strategy[a] * action_values[a]
                         for a in action_values)

        regrets = np.zeros(N_BETTING_ACTIONS, dtype=np.float32)
        for action in action_values:
            regrets[action] = reach_opponent * (action_values[action] - node_value)

        value_betting_buf.add(vec, regrets, mask)
        return node_value

    action = int(np.random.choice(N_BETTING_ACTIONS, p=strategy))

    s2           = state.clone()
    street_ended = s2.apply_bet(player, action)
    if street_ended and not s2.terminal:
        s2.advance_street()

    return traverse(
        s2, traverser,
        value_betting_nets,
        value_betting_buf,
        strategy_betting_bufs,
        iteration,
        reach_traverser, reach_opponent * strategy[action],
        device, bb_table=bb_table,
    )


# FROM SB's PERSPECTIVE
def _bb_discard_prob(bb_table, evaluator, h1, h2, opp_discs, board):
    """Look up probability that BB kept (h1, h2) from [h1, h2] + opp_discs."""
    bb_full = [h1, h2] + list(opp_discs)
    return lookup_bb_prob(bb_table, evaluator, bb_full, board[:3], 0)


def _best_discard_heuristic(state, player, bb_table, n_samples=20):
    """BB: sample from table strategy. SB: table-weighted equity."""
    evaluator = WrappedEval()
    hand = list(state.hole[player])
    board = state.board()
    opp = 1 - player
    n_remaining = 5 - len(board)

    opp_discs = []
    if state.discard_done[opp]:
        opp_discs = list(state.discarded[opp])

    is_sb = (player == 0)

    if not is_sb:
        # BB: sample from table strategy directly
        probs = lookup_bb_probs(bb_table, evaluator, hand, board[:3])
        probs /= probs.sum()
        return int(np.random.choice(N_DISCARD_ACTIONS, p=probs))

    # SB: weighted by BB's table probabilities, softmax over equities
    dead = set(board) | set(hand) | set(opp_discs)
    pool = [c for c in range(27) if c not in dead]
    equities = np.zeros(N_DISCARD_ACTIONS, dtype=np.float64)

    for action in range(N_DISCARD_ACTIONS):
        ki, kj = discard_action_to_keep_pair(action)
        k1, k2 = hand[ki], hand[kj]
        weighted_wins, total_weight = 0.0, 0.0
        for _ in range(n_samples):
            if len(pool) < 2 + n_remaining:
                break
            sampled = np.random.choice(pool, size=2 + n_remaining, replace=False)
            r1, r2 = int(sampled[0]), int(sampled[1])

            weight = _bb_discard_prob(bb_table, evaluator, r1, r2, opp_discs, board)
            if weight < 1e-9:
                continue

            remaining_board = [int(c) for c in sampled[2:]]
            full_board = [PokerEnv.int_to_card(c) for c in board] + \
                         [PokerEnv.int_to_card(int(c)) for c in remaining_board]
            my_rank = evaluator.evaluate(
                [PokerEnv.int_to_card(k1), PokerEnv.int_to_card(k2)], full_board)
            opp_rank = evaluator.evaluate(
                [PokerEnv.int_to_card(r1), PokerEnv.int_to_card(r2)], full_board)
            outcome = 1.0 if my_rank < opp_rank else 0.5 if my_rank == opp_rank else 0.0
            weighted_wins += weight * outcome
            total_weight += weight

        equities[action] = weighted_wins / max(total_weight, 1e-9)

    logits = TEMP * equities
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(np.random.choice(N_DISCARD_ACTIONS, p=probs))


def _traverse_discard(state, player, traverser,
                      value_betting_nets,
                      value_betting_buf,
                      strategy_betting_bufs,
                      iteration,
                      reach_traverser, reach_opponent,
                      device, bb_table=None):
    """Handle a discard decision node using table-based discard."""
    action = _best_discard_heuristic(state, player, bb_table)
    ki, kj = discard_action_to_keep_pair(action)
    s2 = state.clone()
    s2.apply_discard(player, ki, kj)
    return traverse(
        s2, traverser,
        value_betting_nets,
        value_betting_buf,
        strategy_betting_bufs,
        iteration,
        reach_traverser, reach_opponent,
        device, bb_table=bb_table,
    )


def run_traversal(traverser,
                  value_betting_nets,
                  value_betting_buf,
                  strategy_betting_bufs,
                  iteration, device='cpu', bb_table=None):
    """Run one complete traversal for the given traverser."""
    state = GameState()
    return traverse(
        state, traverser,
        value_betting_nets,
        value_betting_buf,
        strategy_betting_bufs,
        iteration, device=device, bb_table=bb_table,
    )
