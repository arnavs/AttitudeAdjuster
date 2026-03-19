"""
Infoset encoder for Deep CFR.

Cards are integers 0-26:
  rank = card % 9   (0=2, 1=3, ..., 7=9, 8=A)
  suit = card // 9  (0=d, 1=h, 2=s)

Each card -> 12-dim vector: 9-dim rank one-hot + 3-dim suit one-hot.
Unknown/absent cards -> 12 zeros.

Input vector layout (203 dims total):
  [0:60]    my_hole_cards     5 x 12  (all 5 preflop/discard; 2 filled + 3 zeros postflop)
  [60:120]  community_cards   5 x 12  (zeros for unseen)
  [120:156] my_discards       3 x 12  (zeros until I discard)
  [156:192] opp_discards      3 x 12  (zeros until revealed)
  [192]     position          1 dim (0=SB, 1=BB)
  [193:197] street            4 dims one-hot
  [197]     pot_size          1 dim normalized
  [198]     my_stack          1 dim normalized
  [199]     opp_stack         1 dim normalized
  [200]     my_bet            1 dim normalized
  [201]     opp_bet           1 dim normalized
  [202]     is_discard_node   1 dim

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

CARD_DIM  = 12
N_RANKS   = 9
N_SUITS   = 3

N_HOLE      = 5
N_COMMUNITY = 5
N_DISCARDS  = 3

INPUT_DIM = (N_HOLE + N_COMMUNITY + N_DISCARDS + N_DISCARDS) * CARD_DIM + 11
# = 203

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


def encode_card(card_int):
    """card_int in [0,26] -> 12-dim vector. -1 -> zeros."""
    vec = np.zeros(CARD_DIM, dtype=np.float32)
    if card_int < 0:
        return vec
    rank = card_int % N_RANKS
    suit = card_int // N_RANKS
    vec[rank] = 1.0
    vec[N_RANKS + suit] = 1.0
    return vec


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


def encode_infoset(observation, is_discard_node=False):
    """
    Convert a gym observation dict into a 203-dim input vector.
    Same vector used for both betting and discard networks.
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
