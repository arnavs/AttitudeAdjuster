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
    betting_mask, discard_mask,
    KEEP_PAIRS, discard_action_to_keep_pair,
    FOLD, CHECK, CALL, BET_SMALL, BET_LARGE,
    N_BETTING_ACTIONS, N_DISCARD_ACTIONS,
)
from network import get_strategy


# ── bet sizing ────────────────────────────────────────────────────────────────

def compute_bet_sizes(pot, stack, min_raise):
    """
    small = 1/3 pot, large = all-in.
    Both clamped to [min_raise, stack].
    """
    small = int(max(min_raise, min(pot / 3, stack)))
    large = int(stack)
    return small, large


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
            "max_raise"          : max(0, self.stacks[player] - max(0, self.bets[opp] - self.bets[player])),
        }

    def legal_betting_mask(self, player):
        """5-dim binary mask of legal betting actions."""
        opp = 1 - player
        call_amount = self.bets[opp] - self.bets[player]
        mask = np.zeros(N_BETTING_ACTIONS, dtype=np.float32)

        # fold: only when facing a bet
        if call_amount > 0:
            mask[FOLD] = 1.0

        # check: only when no bet to call
        if call_amount == 0:
            mask[CHECK] = 1.0

        # call: when facing a bet and have chips
        if call_amount > 0 and self.stacks[player] > 0:
            mask[CALL] = 1.0

        # raises: need chips beyond the call
        if self.stacks[player] > call_amount:
            small, large = compute_bet_sizes(self.pot, self.stacks[player], self.min_raise)
            if small >= self.min_raise:
                mask[BET_SMALL] = 1.0
            if large >= self.min_raise:
                mask[BET_LARGE] = 1.0

        return mask

    def apply_discard(self, player, keep_idx_1, keep_idx_2):
        """Player keeps two cards by index from their 5-card hand."""
        hand = list(self.hole[player])
        kept      = [hand[keep_idx_1], hand[keep_idx_2]]
        discarded = [hand[k] for k in range(5) if k not in (keep_idx_1, keep_idx_2)]
        self.hole[player]      = kept
        self.discarded[player] = discarded
        self.discard_done[player] = True

    def apply_bet(self, player, action):
        """
        Apply a betting action. Returns True if street ends.
        """
        opp = 1 - player
        call_amount = self.bets[opp] - self.bets[player]
        small, large = compute_bet_sizes(self.pot, self.stacks[player], self.min_raise)

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
            return True

        elif action in (BET_SMALL, BET_LARGE):
            amount = small if action == BET_SMALL else large
            amount = min(amount, self.stacks[player])
            self.stacks[player] -= amount
            self.bets[player]   += amount
            self.pot            += amount
            self.min_raise       = amount
            self.acting_player   = opp
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
        from gym_env import PokerEnv, WrappedEval
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
             value_betting_net, value_discard_net,
             strategy_betting_net, strategy_discard_net,
             value_betting_buf, value_discard_buf,
             strategy_betting_buf, strategy_discard_buf,
             iteration,
             reach_traverser=1.0, reach_opponent=1.0,
             device='cpu'):
    """
    External sampling MCCFR traversal.
    Returns counterfactual value for traverser at this node.
    """
    if state.terminal:
        return state.payoff(traverser)

    # ── discard nodes (BB first, then SB) ────────────────────────────────────
    if state.street == 1:
        if not state.discard_done[1]:  # BB discards first
            return _traverse_discard(
                state, player=1, traverser=traverser,
                value_discard_net=value_discard_net,
                strategy_discard_net=strategy_discard_net,
                value_discard_buf=value_discard_buf,
                strategy_discard_buf=strategy_discard_buf,
                value_betting_net=value_betting_net,
                value_discard_net2=value_discard_net,
                strategy_betting_net=strategy_betting_net,
                strategy_discard_net2=strategy_discard_net,
                value_betting_buf=value_betting_buf,
                strategy_betting_buf=strategy_betting_buf,
                iteration=iteration,
                reach_traverser=reach_traverser,
                reach_opponent=reach_opponent,
                device=device,
            )
        if not state.discard_done[0]:  # SB discards second
            return _traverse_discard(
                state, player=0, traverser=traverser,
                value_discard_net=value_discard_net,
                strategy_discard_net=strategy_discard_net,
                value_discard_buf=value_discard_buf,
                strategy_discard_buf=strategy_discard_buf,
                value_betting_net=value_betting_net,
                value_discard_net2=value_discard_net,
                strategy_betting_net=strategy_betting_net,
                strategy_discard_net2=strategy_discard_net,
                value_betting_buf=value_betting_buf,
                strategy_betting_buf=strategy_betting_buf,
                iteration=iteration,
                reach_traverser=reach_traverser,
                reach_opponent=reach_opponent,
                device=device,
            )

    if state.terminal:
        return state.payoff(traverser)

    # ── betting node ──────────────────────────────────────────────────────────
    player = state.acting_player
    obs    = state.obs(player)
    mask   = state.legal_betting_mask(player)

    if mask.sum() == 0:
        return state.payoff(traverser)

    vec      = encode_infoset(obs, is_discard_node=False)
    strategy = get_strategy(value_betting_net, vec, mask, device)

    is_traverser = (player == traverser)

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
                value_betting_net, value_discard_net,
                strategy_betting_net, strategy_discard_net,
                value_betting_buf, value_discard_buf,
                strategy_betting_buf, strategy_discard_buf,
                iteration,
                reach_traverser * strategy[action], reach_opponent,
                device,
            )

        node_value = sum(strategy[a] * action_values[a]
                         for a in action_values)

        regrets = np.zeros(N_BETTING_ACTIONS, dtype=np.float32)
        for action in action_values:
            regrets[action] = reach_opponent * (action_values[action] - node_value)

        value_betting_buf.add(vec, regrets, mask)
        strategy_betting_buf.add(vec, strategy, iteration)
        return node_value

    else:
        action = int(np.random.choice(N_BETTING_ACTIONS, p=strategy))
        strategy_betting_buf.add(vec, strategy, iteration)

        s2           = state.clone()
        street_ended = s2.apply_bet(player, action)
        if street_ended and not s2.terminal:
            s2.advance_street()

        return traverse(
            s2, traverser,
            value_betting_net, value_discard_net,
            strategy_betting_net, strategy_discard_net,
            value_betting_buf, value_discard_buf,
            strategy_betting_buf, strategy_discard_buf,
            iteration,
            reach_traverser, reach_opponent * strategy[action],
            device,
        )


def _traverse_discard(state, player, traverser,
                      value_discard_net, strategy_discard_net,
                      value_discard_buf, strategy_discard_buf,
                      value_betting_net, value_discard_net2,
                      strategy_betting_net, strategy_discard_net2,
                      value_betting_buf, strategy_betting_buf,
                      iteration,
                      reach_traverser, reach_opponent,
                      device):
    """Handle a discard decision node."""
    obs      = state.obs(player)
    mask     = discard_mask()  # all 10 pairs legal
    vec      = encode_infoset(obs, is_discard_node=True)
    strategy = get_strategy(value_discard_net, vec, mask, device)

    is_traverser = (player == traverser)

    if is_traverser:
        action_values = {}
        for action in range(N_DISCARD_ACTIONS):
            ki, kj = discard_action_to_keep_pair(action)
            s2     = state.clone()
            s2.apply_discard(player, ki, kj)
            action_values[action] = traverse(
                s2, traverser,
                value_betting_net, value_discard_net2,
                strategy_betting_net, strategy_discard_net2,
                value_betting_buf, value_discard_buf,
                strategy_betting_buf, strategy_discard_buf,
                iteration,
                reach_traverser * strategy[action], reach_opponent,
                device,
            )

        node_value = sum(strategy[a] * action_values[a]
                         for a in range(N_DISCARD_ACTIONS))

        regrets = np.zeros(N_DISCARD_ACTIONS, dtype=np.float32)
        for action in range(N_DISCARD_ACTIONS):
            regrets[action] = reach_opponent * (action_values[action] - node_value)

        value_discard_buf.add(vec, regrets, mask)
        strategy_discard_buf.add(vec, strategy, iteration)
        return node_value

    else:
        action = int(np.random.choice(N_DISCARD_ACTIONS, p=strategy))
        ki, kj = discard_action_to_keep_pair(action)
        strategy_discard_buf.add(vec, strategy, iteration)

        s2 = state.clone()
        s2.apply_discard(player, ki, kj)
        return traverse(
            s2, traverser,
            value_betting_net, value_discard_net2,
            strategy_betting_net, strategy_discard_net2,
            value_betting_buf, value_discard_buf,
            strategy_betting_buf, strategy_discard_buf,
            iteration,
            reach_traverser, reach_opponent * strategy[action],
            device,
        )


def run_traversal(traverser,
                  value_betting_net, value_discard_net,
                  strategy_betting_net, strategy_discard_net,
                  value_betting_buf, value_discard_buf,
                  strategy_betting_buf, strategy_discard_buf,
                  iteration, device='cpu'):
    """Run one complete traversal for the given traverser."""
    state = GameState()
    return traverse(
        state, traverser,
        value_betting_net, value_discard_net,
        strategy_betting_net, strategy_discard_net,
        value_betting_buf, value_discard_buf,
        strategy_betting_buf, strategy_discard_buf,
        iteration, device=device,
    )
