/Users/arnavsood/MachineYearning/.venv/bin/python -c "
import sys, os, time
sys.path.insert(0, 'submission')
sys.path.insert(0, '.')
from traversal import GameState, CALL
from gym_env import PokerEnv, WrappedEval
from player import PlayerAgent

agent = PlayerAgent.__new__(PlayerAgent)
agent.evaluator = WrappedEval()
agent.action_types = PokerEnv.ActionType

s = GameState()
s.apply_bet(0, CALL)
s.advance_street()

obs_bb = s.obs(1)
obs_bb['blind_position'] = 1
t0 = time.time()
result = agent._heuristic_discard(obs_bb)
print(f'BB discard: {time.time()-t0:.3f}s, keep=({result[2]},{result[3]})')

s.apply_discard(1, result[2], result[3])
obs_sb = s.obs(0)
obs_sb['blind_position'] = 0
t0 = time.time()
result = agent._heuristic_discard(obs_sb)
print(f'SB discard: {time.time()-t0:.3f}s, keep=({result[2]},{result[3]})')
" 2>&1 | grep -v "Gym\|upgrade\|Users of\|migration"
