# Deep CFR Correctness Review Guide

## File Map

| File | Purpose |
|------|---------|
| `submission/encoder.py` | Infoset → 203-dim vector. Card encoding, action constants. |
| `submission/network.py` | CFRNet architecture, `get_strategy` (regret matching), `get_policy_distribution` (softmax), replay buffers, training functions. |
| `submission/traversal.py` | `GameState` (game simulator) + `traverse` (MCCFR recursion). |
| `submission/train.py` | Training loop: workers, buffer merging, checkpointing. |
| `submission/player.py` | Runtime agent: loads strategy nets, blends with posterior + opponent model. |
| `submission/hand_eval.py` | Standalone eval helpers. Not used by the main pipeline. |

## Critical Invariants to Check

### 1. GameState must match gym_env

The trained strategy is only valid if `GameState` explores the same game tree as the real environment. Key areas:

- **Raise semantics**: `apply_bet` line 193 does `new_bet = self.bets[opp] + raise_amount`. gym_env line 375 does `self.bets[acting] = self.bets[opp] + raise_amount`. Must match.
- **min_raise update**: `apply_bet` lines 201-204. Compare with gym_env lines 376-380. Currently floors at `BIG_BLIND` to prevent `min_raise=0` (which causes infinite recursion).
- **Preflop SB limp**: `apply_bet` line 185-188. SB calling to match BB does NOT end the street. BB still gets to act.
- **Street-ending conditions**: CHECK at line 170-172. BB checking preflop ends street. SB checking postflop ends street.
- **max_raise**: `max_raise_amount()` line 117-118 returns `STARTING_STACK - max(self.bets)`. Compare with gym_env.
- **Fold legality**: line 126-127. Currently `if call_amount > 0`. gym_env only allows fold when facing a bet. Verify against `gym_env._get_valid_actions`.
- **Discard order**: BB discards first (traversal line 274), SB second (line 290). `apply_discard` line 152 switches `acting_player`.
- **Board reveal**: `board()` line 82-84. Street 0→empty, 1→3 cards, 2→4, 3→5.
- **Showdown**: `_resolve_showdown` line 220-234. Lower treys rank = better hand.
- **Payoff**: `payoff` line 236-242. `contested = min(bets)`. Zero-sum.

### 2. Traversal algorithm correctness

- **Traverser's nodes**: explores ALL legal actions, computes regrets (lines 324-351).
- **Opponent's nodes**: samples ONE action from strategy (lines 354-370).
- **Regret formula**: `regrets[a] = reach_opponent * (action_value[a] - node_value)` (line 349). Standard external sampling MCCFR.
- **Strategy recording**: happens for BOTH players at their decision nodes, indexed by `player` not `traverser` (line 322, 388). Correct for average strategy computation.
- **Value net used for strategy**: `get_strategy(value_betting_nets[player], ...)` at line 319. Regret matching on value net output. Correct during training.
- **All-in shortcut**: lines 273-282. When both stacks=0, run out streets to showdown. Prevents infinite recursion from zero-raises. If discards not done, does them with fixed indices (0,1).
- **Reach probabilities**: `reach_traverser` updated at traverser nodes (line 340), `reach_opponent` at opponent nodes (line 368). Correct.

### 3. Network and training

- **CFRNet**: 203→256→256→128→N. Same architecture for value and strategy nets, different output dims (5 betting, 10 discard).
- **Value net training** (`train_value_network`): MSE loss on regret targets, masked to legal actions. Net is recreated from scratch each training round (line 169).
- **Strategy net training** (`train_strategy_network`): Cross-entropy on strategy targets, weighted by iteration number. Trained once at end.
- **ReservoirBuffer**: fixed capacity, reservoir sampling. Mixes old and new samples.
- **StrategyBuffer**: same structure, stores `(infoset, strategy, iteration_weight)`.
- **Per-player nets**: `value_betting_nets[player]`, indexed by acting player. Each player has their own value and strategy nets.

### 4. Runtime player (player.py)

- **Inference**: uses `get_policy_distribution` (masked softmax), NOT `get_strategy` (regret matching). Correct for strategy nets.
- **Equity direction in `_blend_posterior`**: uses `_hand_vs_hand_equity(my_cards, [h1,h2], ...)` — computes OUR equity against candidate opponent hand. Verify this returns our win rate, not theirs.
- **`_fast_equity`**: computes equity for a SINGLE hand `(h1,h2)` against a RANDOM opponent. Used in `_update_raise` and `_update_check` for posterior updates. Different from `_hand_vs_hand_equity`.
- **`_to_gym` fallbacks**: if BET_SMALL chosen but RAISE not valid, falls back to CHECK (line 379). Should this be CALL when facing a bet?
- **Foldout**: `_safe_action` checks for free, folds otherwise. Correct (unlike the old bot which always folded).
- **Posterior zeroing**: `_update_posterior` zeros pairs overlapping with new community cards (lines 191-197). Tracked by `zeroed_streets` to avoid double-zeroing.

### 5. Encoder

- **Card encoding**: `card_int % 9` = rank, `card_int // 9` = suit. 12-dim per card (9 rank + 3 suit one-hot).
- **Sorting**: `encode_cards` sorts cards for permutation invariance within each group.
- **Betting mask mapping**: gym ActionType order (FOLD=0, RAISE=1, CHECK=2, CALL=3) → encoder order (FOLD=0, CHECK=1, CALL=2, BET_SMALL=3, BET_LARGE=4). Verify line 142-146.
- **Input dim**: 203 = (5+5+3+3)×12 + 11 scalars. Verify the count.

## Known Issues / Open Questions

1. **Fold legality**: is fold legal when `call_amount == 0`? Check gym_env `_get_valid_actions`. If not, line 126-127 should guard with `if call_amount > 0`.
2. **All-in discard shortcut**: when both all-in before discards, `apply_discard(p, 0, 1)` keeps indices 0 and 1. This is arbitrary — the discard net never gets to decide. Acceptable since both players are already committed, but the EV depends on which cards are kept for showdown.
3. **`opp_last_action` field**: does the gym_env observation include this? If not, `_consume_opponent_action` in the ChatGPT PR is dead code.
4. **Strategy net vs value net at runtime**: we use strategy nets with softmax. If strategy nets weren't trained long enough (small buffer), the policy could be poor. Fall back to value net + regret matching as backup?
5. **Per-player strategy nets**: `player.py` loads `strategy_betting_p0` and `strategy_betting_p1`. At runtime, it indexes by `observation["blind_position"]`. Verify that position 0 = SB = player 0 in both the env and the trained nets.

## Quick Smoke Tests

```bash
# single traversal
cd ~/MachineYearning
python -c "
import sys; sys.path.insert(0, 'submission')
from network import make_betting_net, make_discard_net, ReservoirBuffer, StrategyBuffer
from traversal import run_traversal
vb=[make_betting_net().eval(),make_betting_net().eval()]
vd=[make_discard_net().eval(),make_discard_net().eval()]
sb=[make_betting_net().eval(),make_betting_net().eval()]
sd=[make_discard_net().eval(),make_discard_net().eval()]
vb_buf=ReservoirBuffer(1000);vd_buf=ReservoirBuffer(1000)
sb_bufs=[StrategyBuffer(1000),StrategyBuffer(1000)]
sd_bufs=[StrategyBuffer(1000),StrategyBuffer(1000)]
val=run_traversal(0,vb,vd,sb,sd,vb_buf,vd_buf,sb_bufs,sd_bufs,1)
print(f'val={val:.1f} vb={len(vb_buf)} vd={len(vd_buf)}')
"

# run test suite
python submission/test.py
```
