"""
Infoset encoder for Deep CFR.

Cards are integers 0-26:
  rank = card % 9   (0=2, 1=3, ..., 7=9, 8=A)
  suit = card // 9  (0=d, 1=h, 2=s)

Each card -> 12-dim vector: 9-dim rank one-hot + 3-dim suit one-hot.
Unknown/absent cards -> 12 zeros.

Input vector layout (203 dims total):
  [0:60]    my_hole_cards     5 cards x 12  (all 5 preflop/discard; 2 filled + 3 zeros postflop)
  [60:120]  community_cards   5 cards x 12  (zeros for unseen)
  [120:156] my_discards       3 cards x 12  (zeros until I discard)
  [156:192] opp_discards      3 cards x 12  (zeros until revealed)
  [192]     position          1 dim (0=SB, 1=BB)
  [193:197] street            4 dims one-hot
  [197]     pot_size          1 dim normalized
  [198]     my_stack          1 dim normalized
  [199]     opp_stack         1 dim normalized
  [200]     my_bet            1 dim normalized
  [201]     opp_bet           1 dim normalized
  [202]     is_discard_node   1 dim

Output vector (15 dims):
  [0]  fold
  [1]  check
  [2]  call
  [3]  bet_small
  [4]  bet_large
  [5]  keep_(0,1)
  [6]  keep_(0,2)
  [7]  keep_(0,3)
  [8]  keep_(0,4)
  [9]  keep_(1,2)
  [10] keep_(1,3)
  [11] keep_(1,4)
  [12] keep_(2,3)
  [13] keep_(2,4)
  [14] keep_(3,4)
"""

import numpy as np
from itertools import combinations

CARD_DIM = 12
N_RANKS = 9
N_SUITS = 3

N_HOLE = 5
N_COMMUNITY = 5
N_DISCARDS = 3

INPUT_DIM = (N_HOLE + N_COMMUNITY + N_DISCARDS + N_DISCARDS) * CARD_DIM + 11
# = 203

N_BETTING_ACTIONS = 5   # fold, check, call, bet_small, bet_large
KEEP_PAIRS = list(combinations(range(5), 2))  # C(5,2) = 10
N_DISCARD_ACTIONS = len(KEEP_PAIRS)           # 10
N_ACTIONS = N_BETTING_ACTIONS + N_DISCARD_ACTIONS  # 15

BETTING_MASK = np.array([1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0], dtype=np.float32)
DISCARD_MASK = np.array([0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1], dtype=np.float32)

# action indices
FOLD       = 0
CHECK      = 1
CALL       = 2
BET_SMALL  = 3
BET_LARGE  = 4


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
    Sorts cards first for permutation invariance.
    Pads with zeros if fewer than n_slots cards.
    """
    vec = np.zeros(n_slots * CARD_DIM, dtype=np.float32)
    # sort for permutation invariance
    cards = sorted([c for c in card_list if c >= 0])
    for i, c in enumerate(cards[:n_slots]):
        vec[i*CARD_DIM:(i+1)*CARD_DIM] = encode_card(c)
    return vec


def encode_infoset(observation, is_discard_node=False):
    """
    Convert a gym observation dict into a 203-dim input vector.

    observation keys (from gym_env.py):
      my_cards            - list of 5 ints (-1 = unknown)
      community_cards     - list of 5 ints (-1 = unknown)
      my_discarded_cards  - list of 3 ints (-1 = unknown)
      opp_discarded_cards - list of 3 ints (-1 = unknown)
      blind_position      - 0=SB, 1=BB
      street              - 0,1,2,3
      pot_size            - int
      my_bet              - int
      opp_bet             - int
    """
    parts = []

    # --- card slots ---
    my_cards  = [c for c in observation["my_cards"]]       # may include -1
    community = [c for c in observation["community_cards"]]
    my_disc   = [c for c in observation["my_discarded_cards"]]
    opp_disc  = [c for c in observation["opp_discarded_cards"]]

    parts.append(encode_cards(my_cards,  N_HOLE))       # 60 dims
    parts.append(encode_cards(community, N_COMMUNITY))  # 60 dims
    parts.append(encode_cards(my_disc,   N_DISCARDS))   # 36 dims
    parts.append(encode_cards(opp_disc,  N_DISCARDS))   # 36 dims

    # --- scalar features ---
    position = float(observation["blind_position"])     # 0 or 1
    street   = int(observation["street"])

    street_onehot = np.zeros(4, dtype=np.float32)
    street_onehot[street] = 1.0

    pot      = observation["pot_size"] / 200.0          # normalize by max pot
    my_stack = observation.get("my_stack", 100) / 100.0
    opp_stack= observation.get("opp_stack", 100) / 100.0
    my_bet   = observation["my_bet"] / 100.0
    opp_bet  = observation["opp_bet"] / 100.0

    scalars = np.array([
        position,
        *street_onehot,
        pot,
        my_stack,
        opp_stack,
        my_bet,
        opp_bet,
        float(is_discard_node),
    ], dtype=np.float32)               # 11 dims

    parts.append(scalars)

    vec = np.concatenate(parts)
    assert vec.shape == (INPUT_DIM,), f"Expected {INPUT_DIM}, got {vec.shape}"
    return vec


def action_mask(is_discard_node, valid_actions=None):
    """
    Return a 15-dim binary mask of legal actions.
    valid_actions: the gym valid_actions array (5 dims, for betting nodes).
    """
    if is_discard_node:
        return DISCARD_MASK.copy()

    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    if valid_actions is not None:
        mask[:N_BETTING_ACTIONS] = valid_actions[:N_BETTING_ACTIONS].astype(np.float32)
    else:
        mask[:N_BETTING_ACTIONS] = 1.0
    return mask


def keep_pair_to_action(keep_idx_1, keep_idx_2):
    """Convert (keep_card_1, keep_card_2) indices into action index [5..14]."""
    pair = tuple(sorted([keep_idx_1, keep_idx_2]))
    return N_BETTING_ACTIONS + KEEP_PAIRS.index(pair)


def action_to_keep_pair(action_idx):
    """Convert action index [5..14] back to (keep_idx_1, keep_idx_2)."""
    assert action_idx >= N_BETTING_ACTIONS
    return KEEP_PAIRS[action_idx - N_BETTING_ACTIONS]
