"""
Infoset encoder for Deep CFR.

Cards are integers 0-26:
  rank = card % 9   (0=2, 1=3, ..., 7=9, 8=A)
  suit = card // 9  (0=d, 1=h, 2=s)

Each card -> 2-dim vector: (rank/8, suit/2).
Unknown/absent cards -> (0, 0).

Input vector layout (44 dims total):
  [0:10]    my_hole_cards     5 x 2  (all 5 preflop/discard; 2 filled + 3 zeros postflop)
  [10:20]   community_cards   5 x 2  (zeros for unseen)
  [20:26]   my_discards       3 x 2  (zeros until I discard)
  [26:32]   opp_discards      3 x 2  (zeros until revealed)
  [32]      hand_strength     1 dim  (1 - rank/7462, higher = better; 0.5 if unavailable)
  [33]      position          1 dim  (0=SB, 1=BB)
  [34:38]   street            4 dims one-hot
  [38]      pot_size          1 dim normalized
  [39]      my_stack          1 dim normalized
  [40]      opp_stack         1 dim normalized
  [41]      my_bet            1 dim normalized
  [42]      opp_bet           1 dim normalized
  [43]      is_discard_node   1 dim

Betting actions (6):
  0: fold
  1: check
  2: call
  3: bet_small  (~1/3 pot)
  4: bet_med    (~2/3 pot)
  5: bet_large  (all-in)

Discard actions (10):
  index i -> KEEP_PAIRS[i], which is a (col1, col2) tuple of indices into hand5
"""

import numpy as np
from itertools import combinations
from gym_env import PokerEnv, WrappedEval

CARD_DIM  = 2
N_RANKS   = 9
N_SUITS   = 3

N_HOLE      = 5
N_COMMUNITY = 5
N_DISCARDS  = 3

N_SCALARS = 12  # hand_strength(1) + position(1) + street(4) + pot(1) + stacks(2) + bets(2) + is_discard(1)

INPUT_DIM = (N_HOLE + N_COMMUNITY + N_DISCARDS + N_DISCARDS) * CARD_DIM + N_SCALARS
# = 32 + 12 = 44

N_BETTING_ACTIONS = 6
N_DISCARD_ACTIONS = 10

KEEP_PAIRS = list(combinations(range(5), 2))  # C(5,2) = 10

# betting action indices
FOLD      = 0
CHECK     = 1
CALL      = 2
BET_SMALL = 3
BET_MED   = 4
BET_LARGE = 5

_evaluator = WrappedEval()
_MAX_RANK  = 7462


def encode_card(card_int):
    """card_int in [0,26] -> 2-dim vector (rank/8, suit/2). -1 -> zeros."""
    if card_int < 0:
        return np.zeros(CARD_DIM, dtype=np.float32)
    rank = card_int % N_RANKS
    suit = card_int // N_RANKS
    return np.array([rank / 8.0, suit / 2.0], dtype=np.float32)


def encode_cards(card_list, n_slots):
    """
    Encode a list of card ints into n_slots * CARD_DIM dims.
    Sorts cards for permutation invariance.
    Pads with zeros if fewer than n_slots cards.
    """
    vec = np.zeros(n_slots * CARD_DIM, dtype=np.float32)
    cards = sorted([c for c in card_list if c >= 0])
    for i, c in enumerate(cards[:n_slots]):
        vec[i * CARD_DIM:(i + 1) * CARD_DIM] = encode_card(c)
    return vec


def hand_strength(my_cards, community):
    """
    Single evaluator call: normalized hand rank (1 = nuts, 0 = worst).
    Returns 0.5 if we don't have exactly 2 hole cards and 3+ board cards.
    """
    hole = [c for c in my_cards if c >= 0]
    board = [c for c in community if c >= 0]
    if len(hole) != 2 or len(board) < 3:
        return 0.5
    treys_hole = [PokerEnv.int_to_card(c) for c in hole]
    treys_board = [PokerEnv.int_to_card(c) for c in board[:5]]
    rank = _evaluator.evaluate(treys_hole, treys_board)
    return 1.0 - rank / _MAX_RANK


def encode_infoset(observation, is_discard_node=False):
    """
    Convert a gym observation dict into a 44-dim input vector.
    """
    parts = []

    my_cards  = list(observation["my_cards"])
    community = list(observation["community_cards"])
    my_disc   = list(observation["my_discarded_cards"])
    opp_disc  = list(observation["opp_discarded_cards"])

    parts.append(encode_cards(my_cards,  N_HOLE))
    parts.append(encode_cards(community, N_COMMUNITY))
    parts.append(encode_cards(my_disc,   N_DISCARDS))
    parts.append(encode_cards(opp_disc,  N_DISCARDS))

    hs = hand_strength(my_cards, community)

    position = float(observation["blind_position"])
    street   = int(observation["street"])

    street_onehot = np.zeros(4, dtype=np.float32)
    street_onehot[street] = 1.0

    pot       = observation["pot_size"] / 200.0
    my_stack  = observation.get("my_stack", 100) / 100.0
    opp_stack = observation.get("opp_stack", 100) / 100.0
    my_bet    = observation["my_bet"] / 100.0
    opp_bet   = observation["opp_bet"] / 100.0

    scalars = np.array([
        hs,
        position,
        *street_onehot,
        pot,
        my_stack,
        opp_stack,
        my_bet,
        opp_bet,
        float(is_discard_node),
    ], dtype=np.float32)

    parts.append(scalars)

    vec = np.concatenate(parts)
    assert vec.shape == (INPUT_DIM,), f"Expected {INPUT_DIM}, got {vec.shape}"
    return vec


def betting_mask(valid_actions):
    """
    Build a 6-dim binary mask from gym valid_actions array.
    Gym ActionType order: FOLD=0, RAISE=1, CHECK=2, CALL=3, DISCARD=4
    Our order:           FOLD=0, CHECK=1, CALL=2, BET_SMALL=3, BET_MED=4, BET_LARGE=5
    """
    mask = np.zeros(N_BETTING_ACTIONS, dtype=np.float32)
    mask[FOLD]      = float(valid_actions[0])  # FOLD
    mask[CHECK]     = float(valid_actions[2])  # CHECK
    mask[CALL]      = float(valid_actions[3])  # CALL
    mask[BET_SMALL] = float(valid_actions[1])  # RAISE
    mask[BET_MED]   = float(valid_actions[1])  # RAISE
    mask[BET_LARGE] = float(valid_actions[1])  # RAISE
    return mask


def discard_mask():
    """All 10 keep pairs are always legal at a discard node."""
    return np.ones(N_DISCARD_ACTIONS, dtype=np.float32)

def discard_action_to_keep_pair(action_idx):
    """Convert discard action index [0..9] to (keep_idx_1, keep_idx_2)."""
    return KEEP_PAIRS[action_idx]
