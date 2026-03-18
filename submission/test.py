import sys
sys.path.insert(0, '.')
import numpy as np
from traversal import GameState
from encoder import FOLD, CHECK, CALL, BET_SMALL, BET_LARGE

# ── basic setup ───────────────────────────────────────────────────────────────
s = GameState()
assert s.stacks == [99, 98], f"stacks wrong: {s.stacks}"
assert s.bets   == [1, 2],   f"bets wrong: {s.bets}"
assert s.pot    == 3,        f"pot wrong: {s.pot}"
assert s.street == 0
assert s.acting_player == 0
print(f"PASS: initial state stacks={s.stacks} bets={s.bets} pot={s.pot}")

# cards valid and disjoint
assert len(s.hole[0]) == 5
assert len(s.hole[1]) == 5
assert len(set(s.hole[0]) & set(s.hole[1])) == 0
assert len(set(s.hole[0]) & set(s.community)) == 0
assert len(set(s.hole[1]) & set(s.community)) == 0
assert all(0 <= c <= 26 for c in s.hole[0] + s.hole[1] + s.community)
print("PASS: cards valid and disjoint")

# ── preflop fold ──────────────────────────────────────────────────────────────
s2 = s.clone()
ended = s2.apply_bet(0, FOLD)
assert ended
assert s2.terminal
assert s2.winner == 1
assert s2.payoff(0) == -1   # SB loses small blind
assert s2.payoff(1) == 1
print(f"PASS: SB fold winner={s2.winner} payoffs=({s2.payoff(0)},{s2.payoff(1)})")

# ── preflop call -> flop ──────────────────────────────────────────────────────
s3 = s.clone()
ended = s3.apply_bet(0, CALL)  # SB calls
assert ended
assert not s3.terminal
assert s3.bets[0] == s3.bets[1] == 2
assert s3.pot == 4
gone = s3.advance_street()
assert not gone
assert s3.street == 1
assert s3.acting_player == 1   # BB leads postflop
print(f"PASS: SB call -> flop street={s3.street} acting={s3.acting_player}")

# ── discard order ─────────────────────────────────────────────────────────────
# BB discards first, SB second
assert not s3.discard_done[0]
assert not s3.discard_done[1]

# BB obs: opp_discarded_cards should be all -1
obs_bb = s3.obs(1)
assert all(c == -1 for c in obs_bb["opp_discarded_cards"]), \
    f"BB sees SB discards before SB discards: {obs_bb['opp_discarded_cards']}"
assert len([c for c in obs_bb["my_cards"] if c != -1]) == 5, \
    "BB should see all 5 hole cards at discard time"
print("PASS: BB obs correct before any discard")

# BB discards
s3.apply_discard(1, 0, 1)
assert s3.discard_done[1]
assert not s3.discard_done[0]
assert len(s3.hole[1]) == 2
assert len(s3.discarded[1]) == 3
assert all(c != -1 for c in s3.discarded[1])
print(f"PASS: BB discarded {s3.discarded[1]}, kept {s3.hole[1]}")

# SB obs: should now see BB's discards
obs_sb = s3.obs(0)
assert all(c != -1 for c in obs_sb["opp_discarded_cards"]), \
    f"SB can't see BB's discards: {obs_sb['opp_discarded_cards']}"
assert len([c for c in obs_sb["my_cards"] if c != -1]) == 5, \
    "SB should still see all 5 hole cards before their discard"
print(f"PASS: SB sees BB discards {obs_sb['opp_discarded_cards']}")

# SB discards
s3.apply_discard(0, 0, 1)
assert s3.discard_done[0]
assert len(s3.hole[0]) == 2
print(f"PASS: SB discarded {s3.discarded[0]}, kept {s3.hole[0]}")

# post-discard: both players see 2 hole cards
obs_bb2 = s3.obs(1)
obs_sb2 = s3.obs(0)
assert len([c for c in obs_bb2["my_cards"] if c != -1]) == 2
assert len([c for c in obs_sb2["my_cards"] if c != -1]) == 2
print("PASS: post-discard both players see 2 hole cards")

# ── flop betting -> turn ──────────────────────────────────────────────────────
ended = s3.apply_bet(1, CHECK)   # BB checks
assert not ended
assert s3.acting_player == 0
ended = s3.apply_bet(0, CHECK)   # SB checks
assert ended
assert not s3.terminal
gone = s3.advance_street()
assert not gone
assert s3.street == 2
print(f"PASS: flop check-check -> turn street={s3.street}")

# ── full hand to showdown ─────────────────────────────────────────────────────
s4 = GameState()
s4.apply_bet(0, CALL)
s4.advance_street()
s4.apply_discard(1, 0, 1)
s4.apply_discard(0, 0, 1)

for street in [1, 2, 3]:
    s4.apply_bet(1, CHECK)
    s4.apply_bet(0, CHECK)
    if not s4.terminal:
        s4.advance_street()

assert s4.terminal
assert s4.winner in [-1, 0, 1]
p0, p1 = s4.payoff(0), s4.payoff(1)
assert p0 + p1 == 0, f"not zero sum: {p0} + {p1}"
print(f"PASS: showdown winner={s4.winner} payoffs=({p0},{p1}) zero-sum={p0+p1==0}")

# ── raise sizing ──────────────────────────────────────────────────────────────
s5 = GameState()
s5.apply_bet(0, CALL)
s5.advance_street()
s5.apply_discard(1, 0, 1)
s5.apply_discard(0, 0, 1)
pot_before = s5.pot
ended = s5.apply_bet(1, BET_SMALL)   # BB bets small
assert not ended
assert s5.pot > pot_before
assert s5.acting_player == 0
print(f"PASS: BB bet small pot {pot_before}->{s5.pot} acting={s5.acting_player}")

ended = s5.apply_bet(0, FOLD)
assert ended and s5.terminal and s5.winner == 1
print(f"PASS: SB folds to bet winner={s5.winner}")

print("\nAll GameState tests passed.")