import sys
sys.path.insert(0, '.')

import numpy as np

import logging

from gym_env import PokerEnv, WrappedEval
from submission.traversal import GameState, compute_bet_sizes
from submission.encoder import FOLD, CHECK, CALL, BET_SMALL, BET_LARGE


def assert_state_matches_env(state, env, label):
    assert state.street == env.street, f"{label}: street mismatch {state.street} != {env.street}"
    assert state.acting_player == env.acting_agent, f"{label}: acting mismatch {state.acting_player} != {env.acting_agent}"
    assert state.bets == env.bets, f"{label}: bets mismatch {state.bets} != {env.bets}"
    assert state.pot == sum(env.bets), f"{label}: pot mismatch {state.pot} != {sum(env.bets)}"
    assert state.min_raise == env.min_raise, f"{label}: min_raise mismatch {state.min_raise} != {env.min_raise}"
    assert state.discard_done == env.discard_completed, f"{label}: discard flags mismatch {state.discard_done} != {env.discard_completed}"


def step_both(state, env, state_action, env_action, label):
    street_ended = state.apply_bet(state.acting_player, state_action)
    if street_ended and not state.terminal:
        state.advance_street()
    obs, reward, terminated, truncated, info = env.step(env_action)
    assert not truncated
    assert_state_matches_env(state, env, label)
    return obs, reward, terminated, info


def make_test_env(cards):
    env = PokerEnv.__new__(PokerEnv)
    env.logger = logging.getLogger("submission.test")
    env.small_blind_amount = 1
    env.big_blind_amount = 2
    env.min_raise = 2
    env.acting_agent = PokerEnv.SMALL_BLIND_PLAYER
    env.last_street_bet = 0
    env.evaluator = WrappedEval()
    env.reset(options={"cards": cards.copy(), "small_blind_player": 0})
    return env


# deterministic deck so env and GameState see the same deal order
cards = np.arange(27)
state = GameState(deck=cards.copy())
env = make_test_env(cards)

assert_state_matches_env(state, env, "initial")
print(f"PASS: initial state bets={state.bets} pot={state.pot} acting={state.acting_player}")

# cards valid and disjoint
assert len(state.hole[0]) == 5
assert len(state.hole[1]) == 5
assert len(set(state.hole[0]) & set(state.hole[1])) == 0
assert len(set(state.hole[0]) & set(state.community)) == 0
assert len(set(state.hole[1]) & set(state.community)) == 0
print("PASS: cards valid and disjoint")

# preflop fold
s_fold = GameState(deck=cards.copy())
ended = s_fold.apply_bet(0, FOLD)
assert ended and s_fold.terminal and s_fold.winner == 1
assert s_fold.payoff(0) == -1 and s_fold.payoff(1) == 1
print("PASS: SB fold payoff matches blinds")

# preflop call/check must match gym_env: SB call does not end the street, BB can still act
obs, reward, terminated, info = step_both(
    state,
    env,
    CALL,
    (PokerEnv.ActionType.CALL.value, 0, 0, 0),
    "sb_call",
)
assert not terminated
assert state.street == 0 and state.acting_player == 1
print(f"PASS: SB call keeps preflop open acting={state.acting_player} bets={state.bets}")

obs, reward, terminated, info = step_both(
    state,
    env,
    CHECK,
    (PokerEnv.ActionType.CHECK.value, 0, 0, 0),
    "bb_check_to_flop",
)
assert not terminated
assert state.street == 1 and state.acting_player == 1
print(f"PASS: BB check ends preflop and advances to flop street={state.street} acting={state.acting_player}")

# discard order and information flow
obs_bb = state.obs(1)
assert all(c == -1 for c in obs_bb["opp_discarded_cards"])
assert len([c for c in obs_bb["my_cards"] if c != -1]) == 5
print("PASS: BB sees no opponent discards before discarding")

state.apply_discard(1, 0, 1)
obs, reward, terminated, truncated, info = env.step((PokerEnv.ActionType.DISCARD.value, 0, 0, 1))
assert not terminated and not truncated
assert_state_matches_env(state, env, "bb_discard")
obs_sb = state.obs(0)
assert all(c != -1 for c in obs_sb["opp_discarded_cards"])
print(f"PASS: SB sees BB discards {obs_sb['opp_discarded_cards']}")

state.apply_discard(0, 0, 1)
obs, reward, terminated, truncated, info = env.step((PokerEnv.ActionType.DISCARD.value, 0, 0, 1))
assert not terminated and not truncated
assert_state_matches_env(state, env, "sb_discard")
print("PASS: SB discard matches env state")

# raise semantics: BET_SMALL must behave like gym raise_amount, not as raw chip contribution
max_raise = state.max_raise_amount()
small_raise, _, _ = compute_bet_sizes(state.pot, max_raise, state.min_raise)
old_bets = list(state.bets)
obs, reward, terminated, info = step_both(
    state,
    env,
    BET_SMALL,
    (PokerEnv.ActionType.RAISE.value, small_raise, 0, 0),
    "bb_small_raise",
)
expected_new_bet = old_bets[0] + small_raise
assert state.bets[1] == expected_new_bet, f"expected BB bet {expected_new_bet}, got {state.bets[1]}"
print(f"PASS: BET_SMALL maps to raise_amount semantics, bets={state.bets}, min_raise={state.min_raise}")

# facing that raise, SB fold should end the hand consistently
ended = state.apply_bet(0, FOLD)
assert ended and state.terminal and state.winner == 1
print("PASS: fold after raise terminates correctly")

print("\nAll submission GameState regression tests passed.")
