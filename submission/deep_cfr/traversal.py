"""
External sampling MCCFR traversal for Deep CFR.

On each traversal:
  - One player is the "traverser"
  - At traverser nodes: explore ALL actions
  - At opponent nodes: sample ONE action from opponent's strategy
  - At chance nodes: sample one outcome
  - Collect (infoset_vec, instantaneous_regrets, mask) into value buffer
  - Collect (infoset_vec, current_strategy, mask) into strategy buffer

Game structure:
  Street 0: preflop betting (SB first)
  Street 1: BB discards, SB discards, flop betting (BB first)
  Street 2: turn betting (BB first)
  Street 3: river betting (BB first)

Node types:
  CHANCE   - deal cards, move to next street
  DISCARD  - player chooses which 2 cards to keep
  BETTING  - player chooses fold/check/call/raise
  TERMINAL - hand is over, compute payoff
"""

import numpy as np
from itertools import combinations
from encoder import (
    encode_infoset, action_mask, get_strategy as _get_strategy_from_net,
    N_ACTIONS, N_BETTING_ACTIONS, KEEP_PAIRS,
    keep_pair_to_action, action_to_keep_pair,
    FOLD, CHECK, CALL, BET_SMALL, BET_LARGE,
)
from eval import mc_equity, equity_all_pairs, winner

# ── bet sizing ──────────────────────────────────────────────────────────────
def bet_sizes(pot, stack, min_raise):
    """
    Return (small_bet, large_bet) as integer chip amounts.
    small = 1/3 pot, large = all-in, both clamped to [min_raise, stack].
    """
    small = max(min_raise, min(int(pot / 3), stack))
    large = stack  # all-in
    return small, large


# ── game state ───────────────────────────────────────────────────────────────
class GameState:
    """
    Represents a single hand in progress.
    All card values are ints in [0,26]. -1 = unknown/not yet dealt.
    """
    STARTING_STACK = 100
    SMALL_BLIND    = 1
    BIG_BLIND      = 2

    def __init__(self, deck=None):
        if deck is None:
            deck = np.arange(27)
            np.random.shuffle(deck)
        self.deck = list(deck)
        self._ptr = 0

        # deal 5 cards each, 5 community
        self.hole      = [self._deal(5), self._deal(5)]   # [p0_cards, p1_cards]
        self.community = self._deal(5)                    # all 5 upfront, reveal per street

        # discards
        self.discarded = [[-1,-1,-1], [-1,-1,-1]]
        self.discard_done = [False, False]

        # stacks / bets
        self.stacks = [self.STARTING_STACK, self.STARTING_STACK]
        self.bets   = [self.SMALL_BLIND, self.BIG_BLIND]
        self.stacks[0] -= self.SMALL_BLIND
        self.stacks[1] -= self.BIG_BLIND

        # SB=0, BB=1
        self.street         = 0
        self.acting_player  = 0        # SB acts first preflop
        self.street_bets    = [0, 0]   # bets placed this street (for min-raise calc)
        self.last_street_bet= 0
        self.min_raise      = self.BIG_BLIND
        self.pot            = self.SMALL_BLIND + self.BIG_BLIND
        self.terminal       = False
        self.winner         = None     # 0, 1, or -1 (tie)

    def _deal(self, n):
        cards = self.deck[self._ptr:self._ptr+n]
        self._ptr += n
        return list(cards)

    def board_so_far(self):
        """Community cards revealed so far (3 on flop, 4 turn, 5 river)."""
        if self.street == 0:
            return []
        return self.community[:self.street + 2]

    def obs(self, player):
        """Build an observation dict for player (mirrors gym_env structure)."""
        board = self.board_so_far()
        # pad to 5
        padded_board = board + [-1] * (5 - len(board))

        # hole cards: post-discard show only kept cards
        if self.discard_done[player]:
            kept = [c for c in self.hole[player] if c != -1]
            padded_hole = kept + [-1] * (5 - len(kept))
        else:
            padded_hole = self.hole[player] + [-1] * (5 - len(self.hole[player]))

        opp = 1 - player
        opp_disc = self.discarded[opp] if self.discard_done[opp] else [-1,-1,-1]

        return {
            "my_cards"           : padded_hole,
            "community_cards"    : padded_board,
            "my_discarded_cards" : self.discarded[player],
            "opp_discarded_cards": opp_disc,
            "blind_position"     : player,   # 0=SB,1=BB
            "street"             : self.street,
            "pot_size"           : self.pot,
            "my_bet"             : self.bets[player],
            "opp_bet"            : self.bets[opp],
            "my_stack"           : self.stacks[player],
            "opp_stack"          : self.stacks[opp],
            "min_raise"          : self.min_raise,
        }

    def legal_betting_actions(self, player):
        """
        Returns list of action indices that are legal for player.
        """
        opp = 1 - player
        legal = []
        call_amount = self.bets[opp] - self.bets[player]

        # fold: always legal if facing a bet
        if call_amount > 0:
            legal.append(FOLD)

        # check: legal if no bet to call
        if call_amount == 0:
            legal.append(CHECK)

        # call: legal if facing a bet and have chips
        if call_amount > 0 and self.stacks[player] > 0:
            legal.append(CALL)

        # raises: legal if have chips beyond the call
        small, large = bet_sizes(self.pot, self.stacks[player], self.min_raise)
        if self.stacks[player] > call_amount:
            legal.append(BET_SMALL)
            if large > small:
                legal.append(BET_LARGE)
            elif BET_LARGE not in legal:
                legal.append(BET_LARGE)  # all-in even if same as small

        return legal

    def apply_bet(self, player, action, pot_before):
        """Apply a betting action. Returns True if street ends."""
        opp = 1 - player
        call_amount = self.bets[opp] - self.bets[player]
        small, large = bet_sizes(pot_before, self.stacks[player], self.min_raise)

        if action == FOLD:
            self.terminal = True
            self.winner   = opp
            return True

        elif action == CHECK:
            # street ends if BB checks preflop, or SB checks postflop
            if self.street == 0 and player == 1:
                return True   # BB checks preflop -> next street
            elif self.street >= 1 and player == 0:
                return True   # SB checks postflop -> next street
            else:
                self.acting_player = opp
                return False

        elif action == CALL:
            actual_call = min(call_amount, self.stacks[player])
            self.stacks[player] -= actual_call
            self.bets[player]   += actual_call
            self.pot            += actual_call
            return True   # call always ends street

        elif action in (BET_SMALL, BET_LARGE):
            amount = small if action == BET_SMALL else large
            amount = min(amount, self.stacks[player])
            # total bet for this player this street
            self.stacks[player] -= amount
            self.bets[player]   += amount
            self.pot            += amount
            # update min raise
            raise_amount = amount
            self.min_raise = min(raise_amount,
                                 self.stacks[opp])  # can't raise more than opp has
            self.acting_player = opp
            return False

        return False

    def advance_street(self):
        """Move to the next street. Returns True if game over (showdown)."""
        self.street += 1
        if self.street > 3:
            # showdown
            board5 = self.community[:5]
            p0_hole = [c for c in self.hole[0] if c != -1]
            p1_hole = [c for c in self.hole[1] if c != -1]
            self.winner   = winner(p0_hole, p1_hole, board5)
            self.terminal = True
            return True

        # reset for new street
        self.last_street_bet = self.bets[0]  # both equal after street ends
        self.min_raise       = self.BIG_BLIND
        self.acting_player   = 1  # BB acts first postflop
        return False

    def apply_discard(self, player, keep_idx_1, keep_idx_2):
        """Player keeps cards at keep_idx_1, keep_idx_2 from their 5-card hand."""
        hand = self.hole[player]
        kept     = [hand[keep_idx_1], hand[keep_idx_2]]
        discarded= [hand[i] for i in range(5) if i not in (keep_idx_1, keep_idx_2)]
        self.hole[player]      = kept
        self.discarded[player] = discarded
        self.discard_done[player] = True

    def payoff(self, player):
        """Terminal payoff to player (in chips, relative to starting stack)."""
        assert self.terminal
        if self.winner == -1:
            return 0  # tie
        won = min(self.bets[0], self.bets[1])  # contested amount
        if self.winner == player:
            return won
        else:
            return -won


# ── get strategy from network ────────────────────────────────────────────────
def get_strategy(net, obs, is_discard, legal_actions, device='cpu'):
    """
    Query network for strategy at this node.
    Returns numpy array of shape (N_ACTIONS,).
    """
    from network import get_strategy as net_get_strategy
    vec  = encode_infoset(obs, is_discard_node=is_discard)
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    for a in legal_actions:
        mask[a] = 1.0
    return net_get_strategy(net, vec, mask, device=device)


# ── external sampling traversal ──────────────────────────────────────────────
def traverse(state, traverser, value_net, strategy_net,
             value_buf, strategy_buf, iteration,
             reach_traverser=1.0, reach_opponent=1.0,
             device='cpu', mc_samples=30):
    """
    External sampling MCCFR traversal.

    Returns: counterfactual value for the traverser at this node.

    At traverser nodes:   explore ALL actions, update regrets
    At opponent nodes:    sample ONE action from opponent strategy
    At discard nodes:     same logic, but action space = keep pairs
    At chance nodes:      advance street
    At terminal nodes:    return payoff
    """
    if state.terminal:
        return state.payoff(traverser)

    player = state.acting_player

    # ── discard nodes ────────────────────────────────────────────────────────
    # Discard happens at start of street 1, before betting.
    # BB discards first (player 1), then SB (player 0).
    if state.street == 1:
        if not state.discard_done[1]:  # BB discards first
            return _traverse_discard(
                state, 1, traverser,
                value_net, strategy_net,
                value_buf, strategy_buf, iteration,
                reach_traverser, reach_opponent,
                device, mc_samples
            )
        elif not state.discard_done[0]:  # then SB
            return _traverse_discard(
                state, 0, traverser,
                value_net, strategy_net,
                value_buf, strategy_buf, iteration,
                reach_traverser, reach_opponent,
                device, mc_samples
            )

    # ── terminal ─────────────────────────────────────────────────────────────
    if state.terminal:
        return state.payoff(traverser)

    # ── betting node ─────────────────────────────────────────────────────────
    obs          = state.obs(player)
    legal        = state.legal_betting_actions(player)
    is_traverser = (player == traverser)

    # build mask and get strategy from value net
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    for a in legal:
        mask[a] = 1.0

    vec      = encode_infoset(obs, is_discard_node=False)
    strategy = get_strategy(value_net, obs, False, legal, device)

    if is_traverser:
        # explore ALL actions
        action_values = {}
        for action in legal:
            # clone state
            s2 = _clone_state(state)
            pot_before = s2.pot
            street_ended = s2.apply_bet(player, action, pot_before)
            if street_ended and not s2.terminal:
                s2.advance_street()
            new_reach_t = reach_traverser * strategy[action]
            val = traverse(s2, traverser,
                           value_net, strategy_net,
                           value_buf, strategy_buf, iteration,
                           new_reach_t, reach_opponent,
                           device, mc_samples)
            action_values[action] = val

        # node value under current strategy
        node_value = sum(strategy[a] * action_values[a] for a in legal)

        # instantaneous regrets weighted by opponent reach
        regrets = np.zeros(N_ACTIONS, dtype=np.float32)
        for action in legal:
            regrets[action] = reach_opponent * (action_values[action] - node_value)

        value_buf.add(vec, regrets, mask)

        # add to strategy buffer
        strategy_buf.add(vec, strategy, iteration)

        return node_value

    else:
        # opponent: sample ONE action
        action = np.random.choice(N_ACTIONS, p=strategy)

        # add opponent strategy to strategy buffer
        strategy_buf.add(vec, strategy, iteration)

        s2 = _clone_state(state)
        pot_before = s2.pot
        street_ended = s2.apply_bet(player, action, pot_before)
        if street_ended and not s2.terminal:
            s2.advance_street()

        new_reach_o = reach_opponent * strategy[action]
        return traverse(s2, traverser,
                        value_net, strategy_net,
                        value_buf, strategy_buf, iteration,
                        reach_traverser, new_reach_o,
                        device, mc_samples)


def _traverse_discard(state, discard_player, traverser,
                      value_net, strategy_net,
                      value_buf, strategy_buf, iteration,
                      reach_traverser, reach_opponent,
                      device, mc_samples):
    """Handle a discard node for discard_player."""
    obs   = state.obs(discard_player)
    hand5 = [c for c in obs["my_cards"] if c != -1]
    board = [c for c in obs["community_cards"] if c != -1]

    # legal actions = all 10 keep pairs (indices 5-14)
    legal = [5 + i for i in range(10)]
    mask  = np.zeros(N_ACTIONS, dtype=np.float32)
    for a in legal:
        mask[a] = 1.0

    vec      = encode_infoset(obs, is_discard_node=True)
    strategy = get_strategy(value_net, obs, True, legal, device)

    is_traverser = (discard_player == traverser)

    if is_traverser:
        action_values = {}
        for action in legal:
            ki, kj = action_to_keep_pair(action)
            s2 = _clone_state(state)
            s2.apply_discard(discard_player, ki, kj)
            val = traverse(s2, traverser,
                           value_net, strategy_net,
                           value_buf, strategy_buf, iteration,
                           reach_traverser * strategy[action], reach_opponent,
                           device, mc_samples)
            action_values[action] = val

        node_value = sum(strategy[a] * action_values[a] for a in legal)

        regrets = np.zeros(N_ACTIONS, dtype=np.float32)
        for action in legal:
            regrets[action] = reach_opponent * (action_values[action] - node_value)

        value_buf.add(vec, regrets, mask)
        strategy_buf.add(vec, strategy, iteration)
        return node_value

    else:
        # opponent discard: sample one action
        action = np.random.choice(N_ACTIONS, p=strategy)
        ki, kj = action_to_keep_pair(action)

        strategy_buf.add(vec, strategy, iteration)

        s2 = _clone_state(state)
        s2.apply_discard(discard_player, ki, kj)

        return traverse(s2, traverser,
                        value_net, strategy_net,
                        value_buf, strategy_buf, iteration,
                        reach_traverser, reach_opponent * strategy[action],
                        device, mc_samples)


def _clone_state(state):
    """Shallow clone of game state for tree exploration."""
    import copy
    s = copy.copy(state)
    s.hole       = [list(h) for h in state.hole]
    s.community  = list(state.community)
    s.discarded  = [list(d) for d in state.discarded]
    s.discard_done = list(state.discard_done)
    s.stacks     = list(state.stacks)
    s.bets       = list(state.bets)
    s.street_bets= list(state.street_bets)
    return s


def run_traversal(traverser, value_net, strategy_net,
                  value_buf, strategy_buf, iteration, device='cpu'):
    """
    Run one complete traversal for the given traverser.
    Returns the root counterfactual value.
    """
    state = GameState()
    # preflop: SB acts first
    state.acting_player = 0
    return traverse(state, traverser,
                    value_net, strategy_net,
                    value_buf, strategy_buf, iteration,
                    device=device)
