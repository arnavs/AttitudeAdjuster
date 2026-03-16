import time
import numpy as np
from submission.player import PlayerAgent
from gym_env import PokerEnv

env = PokerEnv()
agent = PlayerAgent(stream=False)

times = []
obs, info = env.reset()
reward, terminated, truncated = 0, False, False
hand = info.get('hand_number')
hand_start = time.time()
hands_timed = 0

for _ in range(5000):
    action = agent.act(obs[0], reward, terminated, truncated, info)
    obs, info = env.reset()
    reward, terminated, truncated = 0, False, False
    new_hand = time.time()
    times.append(new_hand - hand_start)
    hand_start = new_hand
    hands_timed += 1
    if hands_timed >= 100:
        break

times = np.array(times)
print(f'Hands timed: {len(times)}')
print(f'Mean: {times.mean():.3f}s  Median: {np.median(times):.3f}s  Max: {times.max():.3f}s  p95: {np.percentile(times,95):.3f}s')
