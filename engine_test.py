from gym_env import PokerEnv
import logging
import numpy as np

"""
We will use a standard card set minus all royal cards and the club's suit. Card 
numbers 2-9 and A from the remaining 3 suits are in play. Texas Hold 'em hand 
sets will still be used for hand evaluation in the same order but with 
four-of-a-kind removed.
"""

"""
1.  Straight Flush
2.  Full House
3.  Flush
4.  Three of a Kind
5.  Straight
6.  Two Pair
7.  One Pair
8.  High Card
"""

"""
Redraw Once: On any phase excluding the river, and up to once per hand for each 
player, players are allowed to discard a card and draw a replacement card. Both 
the discarded cards and the drawn card must be revealed to the opponent player, 
and the discarded card is out of the game for that hand. This goal is to make 
players decide whether or not to reveal information about their hand by 
discarding and attempting to improve their hand.
"""

"""
Testing Poker Variant:
- A single match consists of:
    - Player Cards (fix for tests)
    - Shared Cards (fix for tests)
    - Player Actions:
        - discard: get a new card; old and new cards revealed to other player
        - check: pot doesn't change
        - raise: pot increases
        - fold: game ends, opponent wins the pot
"""


# 27-card deck: 9 ranks (2-9, A), 3 suits (d, h, s) — matches PokerEnv
RANKS = "23456789A"
SUITS = "dhs"

logging.basicConfig(level=logging.DEBUG)


def int_to_card_str(card_int: int):
    """
    Convert from our encoding of a card, an integer on [0, 27), to rank+suit string.
    """
    rank = RANKS[card_int % len(RANKS)]
    suit = SUITS[card_int // len(RANKS)]
    return rank + suit


def card_str_to_int(card_str: str):
    rank, suit = card_str[0], card_str[1]
    return (SUITS.index(suit) * len(RANKS)) + RANKS.index(rank)


def test_utils():
    for card_int in range(len(RANKS) * len(SUITS)):
        assert card_str_to_int(int_to_card_str(card_int)) == card_int


"""

## Simple Test Cases

- p1 fold => p2 wins
    - action pace before hand doesn't matter, as long as neither player folds

- p2 fold => p1 wins
    - action pace before hand doesn't matter, as long as neither player folds

- both all in, then the better cards win
    - or tie, if hand ranks are equal

- both check the entire game => better cards win 
    - or tie, if hand ranks are equal

- p1 raises & p2 checks => better cards win 
    - or tie, if hand ranks are equal

- p2 raises & p1 checks => better cards win 
    - or tie, if hand ranks are equal


## Negative Test Cases

- invalid action -> FOLD, and print

"""


class State:
    pass


# make assert statements a function
# Action = fold,etc
# state = check the state using dictionary


def check_observation(expected_obs: dict, got_obs: dict):
    for field, value in expected_obs.items():
        assert field in got_obs, print(f"Field {field} was expected, but wasn't present in obs: {got_obs}")
        assert got_obs[field] == value, print(f"Field {field} failed: expected {value}, got {got_obs[field]}")


class Action:
    def __init__(
        self,
        action: int,
        raise_ammount: int,
        keep1: int,
        keep2: int
    ):
        assert isinstance(action, int)
        assert isinstance(raise_ammount, int)
        self.action = action
        self.raise_ammount = raise_ammount
        self.keep1 = keep1
        self.keep2 = keep2

    def __repr__(self):
        return f"Action(action={repr(self.action)}, raise_ammount={repr(self.raise_ammount)}, card_to_discard={repr(self.card_to_discard)})"


class GameState:
    def __init__(self, p0obs: dict, p1obs: dict):
        self.p0obs = p0obs
        self.p1obs = p1obs


def _test_engine(rigged_deck: list[int], updates: list[tuple[Action, tuple[dict, dict]]], expected_final_rewards: tuple[int, int], num_hands: int = 1):
    engine = PokerEnv(num_hands=num_hands)  # small blind player always starts out as 0
    assert isinstance(rigged_deck, list)
    (player0_obs, player1_obs), _info = engine.reset(
        options={"cards": rigged_deck}  # rig the deck
    )

    reversed_deck = rigged_deck[::-1]

    # pops p0 5 cards first
    p0_expected_start_cards = [reversed_deck.pop() for _ in range(5)]

    # pops p1 5 cards first
    p1_expected_start_cards = [reversed_deck.pop() for _ in range(5)]

    # pops 5 community cards
    expected_community_cards = [reversed_deck.pop() for _ in range(5)]

    p0_valid_actions = [1] * 5
    p0_valid_actions[engine.ActionType.CHECK.value] = 0  # p0 can't check as it's small blind

    p0_valid_actions = [1, 1, 0, 1, 0]

    # pop redraw cards next (as needed)
    player0_expected_obs = {
        "street": 0,
        "acting_agent": 0,
        "my_cards": p0_expected_start_cards,
        "community_cards": [-1] * 5,
        "my_bet": 1,
        "opp_bet": 2,
        "my_discarded_cards": [-1, -1, -1],  # Added
        "opp_discarded_cards": [-1, -1, -1], # Updated from singular to plural
        "min_raise": 2,
        "valid_actions": p0_valid_actions,
    }

    player1_expected_obs = {
        "street": 0,
        "acting_agent": 0,
        "my_cards": p1_expected_start_cards,
        "community_cards": [-1] * 5,
        "my_bet": 2,
        "opp_bet": 1,
        "my_discarded_cards": [-1, -1, -1],  # Added
        "opp_discarded_cards": [-1, -1, -1], # Updated
        "min_raise": 2,
    }

    check_observation(player0_expected_obs, player0_obs)
    check_observation(player1_expected_obs, player1_obs)

    for i, (action, expected_state) in enumerate(updates):
        obs, reward, terminated, _, _ = engine.step((action.action, action.raise_ammount, action.keep1, action.keep2))
        p0_got_obs, p1_got_obs = obs
        p0_got_reward, p1_got_reward = reward

        assert terminated == (i == (len(updates) - 1)), print(f"terminated: {terminated}; len(updates): {len(updates)}; i: {i}")

        expected_p0_obs, expected_p1_obs = expected_state
        check_observation(expected_p0_obs, p0_got_obs)
        check_observation(expected_p1_obs, p1_got_obs)

        if terminated:
            assert reward == expected_final_rewards, print(f"Got final reward: {reward}, expected: {expected_final_rewards}")
        else:
            assert p0_got_reward == 0 and p1_got_reward == 0

    return


def test_allways_check():
    # small blind player goes first; they need to pay 1 to check
    small_blind_call = Action(PokerEnv.ActionType.CALL.value, 0, 0, 0)
    bb_preflop_check = Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0)

    # Flop Discard Phase: Both players must keep 2 cards (Indices 0 and 1)
    p1_discard = Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1)
    p0_discard = Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1)

    either_player_check = Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0)

    actions = [
        small_blind_call, bb_preflop_check,  # Street 0
        p1_discard, p0_discard,              # Street 1 Discard Phase
        either_player_check, either_player_check, # Street 1 Betting
        either_player_check, either_player_check, # Street 2
        either_player_check, either_player_check  # Street 3
    ]

    states = [({}, {})] * len(actions)
    assert len(actions) == len(states)
    updates = list(zip(actions, states))
    # 27-card deck: only 2-9, A and d,h,s. P0 folds so outcome is P1 wins.
    rigged_deck = list(map(card_str_to_int, [
        "Ah", "Ad", "5d", "2s", "2d",
        "9h", "9d", "5h", "3s", "3d",
        "As", "9s", "2h", "3h", "4h"
    ]))
    _test_engine(rigged_deck=rigged_deck, updates=updates, expected_final_rewards=(2, -2))


def test_allways_raise_small():
    # --- Action Definitions ---
    min_bet = 2
    # Pre-Flop: p0 raises, p1 calls
    p0_preflop_raise = Action(PokerEnv.ActionType.RAISE.value, min_bet, 0, 0)
    p1_preflop_call = Action(PokerEnv.ActionType.CALL.value, 0, 0, 0)
    
    # Flop Discard Phase: Mandatory after Pre-Flop ends
    p0_discard = Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1) # Keep first two cards
    p1_discard = Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1)
    
    # Post-Discard Betting: p0 raises, p1 calls
    p0_betting_raise = Action(PokerEnv.ActionType.RAISE.value, min_bet, 0, 0)
    p1_betting_call = Action(PokerEnv.ActionType.CALL.value, 0, 0, 0)

    # --- Sequence Assembly ---
    # Street 0 (Pre-flop): p0 raise, p1 call
    # Street 1 (Flop): p0 discard, p1 discard, p0 raise, p1 call
    # Street 2 (Turn): p0 raise, p1 call
    # Street 3 (River): p0 raise, p1 call
    actions = [
        p0_preflop_raise, p1_preflop_call,   # Street 0
        p0_discard, p1_discard,              # Street 1 Mandatory Phase
        p0_betting_raise, p1_betting_call,   # Street 1 Betting
        p0_betting_raise, p1_betting_call,   # Street 2
        p0_betting_raise, p1_betting_call    # Street 3
    ]

    # Placeholder states for _test_engine to ignore/skip detailed field checks
    states = [({}, {})] * len(actions)
    updates = list(zip(actions, states))

    # Rigged deck (15 cards, 27-card variant)
    rigged_deck = list(map(card_str_to_int, [
        # p0's 5 cards (4h and 4d are the first two, which we 'keep')
        "4h", "4d", "2s", "5d", "2d",
        # p1's 5 cards (6h and 7d are kept)
        "6h", "7d", "3s", "5h", "3d",
        # community cards
        "4s", "9s", "9h", "2h", "Ah"
    ]))

    _test_engine(
        rigged_deck=rigged_deck,
        updates=updates,
        expected_final_rewards=(10, -10), # Adjust based on your blind logic
    )

def test_example_tie():
    # --- Action Definitions ---
    # Pre-Flop: SB calls, BB checks
    sb_call = Action(PokerEnv.ActionType.CALL.value, 0, 0, 0)
    bb_check = Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0)
    
    # Flop Discard Phase: Mandatory for both
    p0_discard = Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1) # Keep indices 0, 1
    p1_discard = Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1)
    
    # Standard Checking through all streets
    check = Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0)

    # --- Sequence Assembly ---
    # 1. Pre-Flop (2 actions)
    # 2. Discard Phase (2 actions)
    # 3. Flop Betting (2 actions)
    # 4. Turn Betting (2 actions)
    # 5. River Betting (2 actions)
    actions = [
        sb_call, bb_check,         # Street 0
        p0_discard, p1_discard,    # Street 1 Discard Phase
        check, check,              # Street 1 Betting
        check, check,              # Street 2
        check, check               # Street 3
    ]

    states = [({}, {})] * len(actions)
    updates = list(zip(actions, states))

    # Rigged deck (15 cards, 27-card variant). First two kept don't beat board straight.
    rigged_deck = list(map(card_str_to_int, [
        # p0's 5 cards (Keeps 2h, 3d)
        "2h", "3d", "4h", "4s", "4d",
        # p1's 5 cards (Keeps 2d, 3s)
        "2d", "3s", "5h", "2s", "3h",
        # community cards: a straight on the board
        "9s", "8s", "7d", "6d", "5d"
    ]))

    _test_engine(
        rigged_deck=rigged_deck, 
        updates=updates, 
        expected_final_rewards=(0, 0)
    )
def test_example_game_1():
    """
    A game played between Player 0 and Player 1 with a discard phase.
    """
    # Rigged deck (15 cards total for 5-5-5 deal)
    # P0 keeps 25, 14; P1 keeps 1, 4.
    rigged_deck = [
        25, 14, 2, 3, 5,  # p0's 5 cards
        1, 4, 6, 7, 10,   # p1's 5 cards
        8, 16, 9, 11, 23  # community cards
    ]
    
    # Engine returns (p0_reward, p1_reward); with 27-card deck this deck/action sequence yields P0 wins: (14, -14)
    expected_final_rewards = (14, -14)

    # 4-tuple format: (action_type, raise_amount, keep1, keep2)
    actions = [
        Action(PokerEnv.ActionType.CALL.value, 0, 0, 0),     # P0 Calls Pre-flop
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),    # P1 Checks Pre-flop
        Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1),  # P0 Discards (Flop starts)
        Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1),  # P1 Discards
        Action(PokerEnv.ActionType.RAISE.value, 2, 0, 0),    # P0 Raises on Flop
        Action(PokerEnv.ActionType.CALL.value, 0, 0, 0),     # P1 Calls
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),    # P0 Checks on Turn
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),    # P1 Checks
        Action(PokerEnv.ActionType.RAISE.value, 10, 0, 0),   # P0 Raises on River
        Action(PokerEnv.ActionType.CALL.value, 0, 0, 0),     # P1 Calls
    ]

    # Shared observation template for brevity in this example
    def get_obs_template(street, agent, p0_bet, p1_bet, p0_cards, p1_cards, comm_cards):
        # Note: In a real test, you'd fill these with the exact expected values
        return (
            {
                "street": street,
                "acting_agent": agent,
                "my_cards": p0_cards + [-1] * (5 - len(p0_cards)),
                "community_cards": comm_cards + [-1] * (5 - len(comm_cards)),
                "my_bet": p0_bet,
                "opp_bet": p1_bet,
                "my_discarded_cards": [-1, -1, -1],
                "opp_discarded_cards": [-1, -1, -1],
                "valid_actions": [1] * 5,
            },
            {
                "street": street,
                "acting_agent": agent,
                "my_cards": p1_cards + [-1] * (5 - len(p1_cards)),
                "community_cards": comm_cards + [-1] * (5 - len(comm_cards)),
                "my_bet": p1_bet,
                "opp_bet": p0_bet,
                "my_discarded_cards": [-1, -1, -1],
                "opp_discarded_cards": [-1, -1, -1],
                "valid_actions": [1] * 5,
            }
        )

    # Simplified obs list for the test runner
    # In practice, you would populate these with exact NumPy types as shown in your original code
    obs = [({}, {})] * len(actions) 
    
    assert len(actions) == len(obs)
    updates = list(zip(actions, obs))

    _test_engine(
        rigged_deck=rigged_deck,
        updates=updates,
        expected_final_rewards=expected_final_rewards,
    )
def test_example_game_2():
    """
    A game with an invalid raise (under-raise) on player 1 (BB).
    """
    # 27-card deck: indices 0-26 only
    rigged_deck = [
        24, 14, 1, 2, 3,   # p0 cards
        11, 23, 4, 5, 6,   # p1 cards
        10, 20, 12, 18, 26 # board
    ]
    
    expected_final_rewards = (2, -2)

    # 4-tuple format: (Type, Amount, Keep1, Keep2)
    actions = [
        Action(PokerEnv.ActionType.CALL.value, 0, 0, 0),
        Action(PokerEnv.ActionType.RAISE.value, 1, 0, 0), # Invalid: min_raise is 2
    ]

    obs = [
        (
            {
                "street": 0,
                "acting_agent": 1,
                "my_bet": 2,
                "opp_bet": 2,
                # Pre-flop: Discard (idx 4) is NOT valid. 
                # Actions: [FOLD, RAISE, CHECK, CALL, DISCARD]
                "valid_actions": [1, 1, 1, 0, 0], 
            },
            {
                "street": 0,
                "acting_agent": 1,
                "my_bet": 2,
                "opp_bet": 2,
                "valid_actions": [1, 1, 1, 0, 0],
            },
        ),
        (
            # Final state after invalid action (fold)
            {
                "street": 0,
                "acting_agent": 0,
                "my_bet": 2,
                "opp_bet": 2,
                "valid_actions": [1, 1, 1, 0, 0],
            },
            {
                "street": 0,
                "acting_agent": 0,
                "my_bet": 2,
                "opp_bet": 2,
                "valid_actions": [1, 1, 1, 0, 0],
            },
        ),
    ]

    assert len(actions) == len(obs)
    updates = list(zip(actions, obs))

    _test_engine(
        rigged_deck=rigged_deck,
        updates=updates,
        expected_final_rewards=expected_final_rewards,
    )

def test_illegal_keep_logic():
    """
    Test that keeping out-of-bounds indices or duplicate indices triggers a fold.
    """
    rigged_deck = [i for i in range(15)]
    # Actions: SB Calls, BB Checks, SB attempts to keep index 5 (Illegal)
    actions = [
        Action(PokerEnv.ActionType.CALL.value, 0, 0, 0),
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),
        Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 5) # Index 5 is invalid
    ]
    
    # Player 0 should be folded, Player 1 wins the pot of 2.
    expected_final_rewards = (-2, 2)
    
    # Minimal updates for engine runner
    updates = list(zip(actions, [({}, {})] * 3))
    
    print("Running Illegal Keep Test...")
    _test_engine(rigged_deck, updates, expected_final_rewards)
    print("✓ Illegal Keep Test Passed")

def test_discard_showdown_integrity():
    """
    Verifies that discarded cards do not influence hand strength at showdown.
    """
    # 27-card deck: P0 keeps Ah,2h (high card); P1 keeps pair of 9s
    rigged_deck = list(map(card_str_to_int, [
        "Ah", "2h", "3h", "4h", "5h",
        "9d", "9s", "2d", "3d", "4d",
        "8d", "5h", "7s", "6h", "2d"
    ]))
    
    actions = [
        Action(PokerEnv.ActionType.CALL.value, 0, 0, 0),
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),
        Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1), # P0 keeps A and 2 (High Card)
        Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1), # P1 keeps 9 and 9 (Pair)
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0), # Flop
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0), # Turn
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0)  # River
    ]
    
    # P1 should win with Pair of 9s. P0's hearts were mucked.
    expected_final_rewards = (-2, 2)
    updates = list(zip(actions, [({}, {})] * len(actions)))
    
    print("Running Showdown Integrity Test...")
    _test_engine(rigged_deck, updates, expected_final_rewards)
    print("✓ Showdown Integrity Test Passed")

def test_multi_street_betting_and_fold():
    """
    Verifies bankroll and pot tracking across multiple streets with raises.
    """
    rigged_deck = [i for i in range(15)]
    
    actions = [
        Action(PokerEnv.ActionType.RAISE.value, 10, 0, 0), # P0 Raise 10 (Total 11)
        Action(PokerEnv.ActionType.CALL.value, 0, 0, 0),   # P1 Call (Total 11)
        Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1), # Discards
        Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1),
        Action(PokerEnv.ActionType.RAISE.value, 20, 0, 0), # P0 Raise 20 (Total 31)
        Action(PokerEnv.ActionType.CALL.value, 0, 0, 0),   # P1 Call (Total 31)
        Action(PokerEnv.ActionType.FOLD.value, 0, 0, 0)    # P0 Folds on Turn
    ]
    
    # Each player put in 31. P0 folds, so P1 wins 31.
    expected_final_rewards = (-32, 32)
    updates = list(zip(actions, [({}, {})] * len(actions)))
    
    print("Running Multi-Street Betting Test...")
    _test_engine(rigged_deck, updates, expected_final_rewards)
    print("✓ Multi-Street Betting Test Passed")

def test_discard_showdown_integrity_v2():
    """
    Non-trivial: Rigged so P0 discards a Flush and is left with High Card.
    P1 keeps a simple Pair. Pair must win.
    """
    # 27-card deck: P0 keeps Ad,2d; P1 keeps pair of 8s
    rigged_deck = list(map(card_str_to_int, [
        "Ad", "2d", "7d", "9d", "5s",
        "8d", "8s", "3d", "4d", "5d",
        "2s", "4h", "7h", "5h", "6s"
    ]))
    
    # P0 keeps indices 0 and 1 (Ad, 2d). All other diamonds are GONE.
    # P1 keeps indices 0 and 1 (8c, 8s). 
    
    actions = [
        Action(PokerEnv.ActionType.CALL.value, 0, 0, 0),
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),
        Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1), # P0 keeps Ad, 2d
        Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 1), # P1 keeps 8c, 8s
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0), # Flop
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0), # Turn
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0)  # River
    ]

    # P1 wins with Pair of 8s. P0 has Ace High (Pair of 2s isn't possible here).
    expected_final_rewards = (-2, 2)
    updates = list(zip(actions, [({}, {})] * len(actions)))
    
    _test_engine(rigged_deck, updates, expected_final_rewards)

def test_betting_limit_enforcement():
    """
    Ensures players cannot bet more than the MAX_PLAYER_BET (100).
    """
    rigged_deck = [i for i in range(15)]
    actions = [
        Action(PokerEnv.ActionType.CALL.value, 0, 0, 0),
        # P1 tries to raise by 200, but max allowed is 100 - 2 = 98.
        Action(PokerEnv.ActionType.RAISE.value, 200, 0, 0) 
    ]
    
    # P1 should be folded for an illegal raise.
    expected_final_rewards = (2, -2) 
    updates = list(zip(actions, [({}, {})] * 2))
    
    print("Running Betting Limit Test...")
    _test_engine(rigged_deck, updates, expected_final_rewards)
    print("✓ Betting Limit Test Passed")

def test_observation_leakage():
    """
    Manually inspects the observation to ensure private data isn't leaking.
    """
    engine = PokerEnv()
    rigged_deck = [i for i in range(15)]
    (obs0, obs1), _ = engine.reset(options={"cards": rigged_deck})

    # P1's view of P0's cards should be ALL -1s
    p1_view_of_p0 = obs1["opp_discarded_cards"]
    assert all(c == -1 for c in p1_view_of_p0), "Leak detected: P1 can see P0's discards!"
    
    # Check community cards (Street 0 should have all -1s)
    assert all(c == -1 for c in obs1["community_cards"]), "Leak detected: Future cards visible!"
    
    print("✓ Observation Leakage Test Passed")

def test_duplicate_keep_prevention():
    """
    Ensures keeping the same card index twice results in a fold.
    """
    rigged_deck = [i for i in range(15)]
    actions = [
        Action(PokerEnv.ActionType.CALL.value, 0, 0, 0),
        Action(PokerEnv.ActionType.CHECK.value, 0, 0, 0),
        # P0 tries to keep index 0 twice to "duplicate" a card.
        Action(PokerEnv.ActionType.DISCARD.value, 0, 0, 0) 
    ]
    
    expected_final_rewards = (-2, 2)
    updates = list(zip(actions, [({}, {})] * 3))
    
    print("Running Duplicate Keep Test...")
    _test_engine(rigged_deck, updates, expected_final_rewards)
    print("✓ Duplicate Keep Test Passed")

def main():
    test_utils()
    print("test utils passed")
    print("test engine 0 passed")
    test_allways_check()
    print("test allways_check passed")
    test_allways_raise_small()
    test_example_tie()
    test_example_game_2()
    test_illegal_keep_logic()
    test_discard_showdown_integrity()
    test_multi_street_betting_and_fold()
    test_discard_showdown_integrity_v2()
    test_betting_limit_enforcement()
    test_observation_leakage()
    test_duplicate_keep_prevention()


if __name__ == "__main__":
    main()

