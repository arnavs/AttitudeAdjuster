# Deep CFR review notes

Potential bug candidates found while reviewing `submission/`:

1. **Raises are modeled inconsistently with the real environment.**
   - In `submission/traversal.py`, `GameState.apply_bet()` treats `BET_SMALL` / `BET_LARGE` as the total amount added to the acting player's contribution, even when the player is facing a bet.
   - The real environment in `gym_env.py` interprets a raise amount as the increment _after matching the opponent's current bet_ (`self.bets[player] = self.bets[opp] + raise_amount`).
   - Example from the local `GameState`: from the initial state `[1, 2]`, `BET_SMALL` moves the acting player to `[3, 2]`. In the environment, the minimum legal raise from the same spot should produce `[4, 2]` because the small blind must first call 1 and then raise by at least 2.
   - This also means `min_raise` is updated incorrectly in the simulator, so traversal targets will not match the actual game tree.

2. **Posterior blending uses the opponent hand's random equity instead of our hand's equity versus that opponent hand.**
   - In `submission/player.py`, `_blend_posterior()` loops over candidate opponent pairs but calls `_fast_equity(h1, h2, community, n=8)`.
   - `_fast_equity()` evaluates the first two arguments as the hand whose win rate is being measured.
   - So the blend currently weights actions using the estimated strength of the opponent candidate hand against a random hand, not the equity of our actual kept cards against that candidate hand.
   - This will push the exploit layer in the wrong direction whenever the posterior becomes concentrated.

3. **Final strategy training can crash when the strategy buffers are too small.**
   - `submission/network.py::train_strategy_network()` returns `None` when `len(buffer) < batch_size`.
   - `submission/train.py` always formats the result with `{loss:.4f}`.
   - If a shorter debug run or early termination leaves either strategy buffer under `BATCH_SIZE`, the training script will raise a `TypeError` during final checkpoint reporting instead of failing gracefully.

4. **The training pipeline carries unused strategy networks during traversal.**
   - Workers deserialize `sb_state` / `sd_state`, but `submission/traversal.py` derives traversal strategy from `value_betting_net` / `value_discard_net` only.
   - The extra strategy networks are not consulted inside the recursion, so the worker payload and bookkeeping appear inconsistent with the intended Deep CFR split between advantage networks and the final average-strategy network.
   - This might be dead code rather than a functional bug, but it is worth verifying because it is easy to think the traversal is using more information than it actually is.
