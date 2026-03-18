from traversal import GameState

# basic setup
s = GameState()
print(s.stacks)   # should be [99, 98]
print(s.bets)     # should be [1, 2]
print(s.pot)      # should be 3
print(s.street)   # should be 0
print(s.acting_player)  # should be 0 (SB acts first preflop)

# check hole cards are disjoint and valid
assert len(s.hole[0]) == 5
assert len(s.hole[1]) == 5
assert len(set(s.hole[0]) & set(s.hole[1])) == 0
assert len(set(s.hole[0]) & set(s.community)) == 0
print("cards ok")

# preflop: SB folds
s2 = s.clone()
ended = s2.apply_bet(0, 0)  # FOLD=0
assert s2.terminal
assert s2.winner == 1
print(f"SB fold: winner={s2.winner}, payoff p0={s2.payoff(0)}, p1={s2.payoff(1)}")

# preflop: SB calls, BB checks -> advance to flop
s3 = s.clone()
ended = s3.apply_bet(0, 2)  # CALL
assert ended
assert not s3.terminal
assert s3.bets == [2, 2]
s3.advance_street()
assert s3.street == 1
assert s3.acting_player == 1  # BB leads postflop
print("preflop call -> flop ok")