# Codebase Review: Strategic Depth and Thompson Sampling Correctness

## Scope
Reviewed the repository with emphasis on `submission/player.py`, while cross-checking engine semantics in `gym_env.py` and baseline behavior in `agents/prob_agent.py`.

## Executive Summary
- **Current submission cannot run as-is** because `submission/player.py` has a syntax error in an f-string logger line.
- The postflop policy is **Bayesian-style range sampling**, but it is **not a mathematically correct Thompson Sampling implementation** in the usual bandit sense.
- Strategic depth is moderate: there is a useful discard search and range weighting, but betting strategy remains shallow (static thresholds, random sizing, limited update signals).

## (A) Strategic Depth Assessment

### Strengths
1. **Multi-stage decision architecture**
   - Distinct preflop/flop/turn/river handlers.
   - Dedicated discard optimization over all 10 keep-combinations.
2. **Opponent range representation**
   - Maintains explicit opponent hole-card pair hypotheses (`opp_pairs`) with weights (`opp_weights`).
3. **Street-aware equity computation**
   - River exact showdown eval.
   - Turn exact enumeration of river cards.
   - Flop Monte Carlo rollout.
4. **Tournament-level heuristic**
   - “Lock win” mode based on cumulative chip lead and remaining hands.

### Weaknesses / Missing Depth
1. **No betting-action Bayesian updates**
   - Range is updated from discard evidence only. Opponent raises/calls/folds during betting streets do not alter posterior.
2. **One-dimensional aggression logic**
   - Raise iff sampled win-rate > 0.5; otherwise use pot-odds call/check/fold. This misses stack leverage, position, or calibrated value/bluff mixes.
3. **Raise sizing is unstructured**
   - Uniform random sizing in `[min_raise, max_raise]` regardless of edge, board texture, or pot geometry.
4. **No exploit persistence**
   - Aside from cumulative chips for lock-win, no per-opponent long-run adaptation.

## (B) Thompson Sampling Correctness Assessment

### What the code currently does
- Samples opponent hole pairs from a categorical posterior (`opp_weights`).
- Computes hand equity versus each sampled pair.
- Uses mean sampled equity (`win_rate`) to decide raise/call/check/fold.

### Why this is not “correct Thompson Sampling”
In standard Thompson Sampling, you sample model parameters from a posterior over reward-generating parameters (often per-arm) and then choose the action maximizing sampled expected reward.

Here:
1. **No posterior over action values**
   - Posterior is over opponent private states only, not over action-value parameters.
2. **Decision rule is threshold-based, not argmax over sampled returns**
   - Uses `win_rate > 0.5` and pot-odds checks, rather than maximizing sampled EV across legal actions.
3. **No per-action uncertainty propagation**
   - Raise/call/fold EV distributions are not sampled separately, so exploration/exploitation balance is only indirectly represented.

Conclusion: this is better described as a **posterior hand-range sampler with heuristic action thresholds**, not canonical Thompson Sampling.

## Critical Correctness Issue
- `submission/player.py` has an invalid f-string (`observation["street"]` inside double-quoted f-string), causing immediate `SyntaxError` and preventing agent execution.

## Recommended Fix Priority
1. **P0**: Fix syntax error so submission imports and runs.
2. **P1**: Convert policy to EV-based action choice under posterior samples (true TS-style approximate Bayes decision).
3. **P2**: Add posterior updates from observed betting actions.
4. **P3**: Replace uniform random raise sizing with EV-calibrated sizing buckets.
