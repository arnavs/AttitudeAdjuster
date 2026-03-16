1. What I have in mind: a simple EV bot that plays a tight, Nash-like game. 
2. Details: 
    - A **hand layer** that decides what to do within a hand
        - One part describes how we learn
        - Another part describes how we act
    - A **heuristics layer** that overrides that behavior sometimes and manages persistence between hands. 
3. Hand Layer (Learning): 
    - Key object is a prior distribution over opponent's hole cards (h1, h2). 
    - At discard time, there are 16 or 19 unseen cards, depending on whether we are big blind or small blind: 
        - 27 card deck, minus 5 cards in our hand, minus 3 flop cards (f1, f2, f3), minus their 3 discards (d1, d2, d3). 
        - So there are C(16, 2) = 120 possible hole pairs. 
    - Begin with a uniform prior over the 120 possible pairs (h1, h2). 
    - For each pair, (h1, h2), we have the following: 
        - The likelihood of discarding (d1, d2, d3) given opp hand (d1, d2, d3, h1, h2) and flop (f1, f2, f3) is a softmax of the equity against a uniform distribution. That is: 
            - Compute the equity of each pair in opp hand and then assume they discard proportionally via logit.
            - If we are SB, opp is not conditioning on our discard. 
            - If we are BB, opp is conditioning on our discard.
        - Posterior is computed via Bayes' rule. 
    - At each street, we compute expected value with respect to the distribution over opponent's hole.
        - Posterior updates based on opponent's bets:
            - If opp makes a big raise (> 50% of pot): shift posterior toward stronger opp hands via `weights *= exp(alpha * opp_equity)`.
            - If opp folds: shift posterior toward weaker opp hands via `weights *= exp(-alpha * opp_equity)`.
            - `alpha` is a per-opponent aggression scalar, updated via EMA from showdowns opp loses.
4. Hand Layer (Acting):
    - **Pre-flop**: compute MC equity for all C(5,2)=10 possible keep pairs vs uniform random opponent. Act based on the **median** equity (we don't yet know which 2 cards we'll keep).
    - **Post-flop (flop/turn/river)**: act based on weighted equity against the posterior distribution over opponent hole pairs.
        - Equity at flop uses a precomputed lookup table keyed by (my 2 hole cards, opp 2 hole cards, flop 3 cards), averaged over all discard scenarios.
        - Equity at turn/river uses exact enumeration over remaining board cards, excluding dead cards (both discards).
    - **Raise probability**: `P(raise) = min(equity, 0.7)`. Sampled stochastically — naturally incorporates bluffing at low equity.
    - **Raise sizing**: uniform random in `[min_raise, max_raise]`.
    - **If not raising**: call when equity ≥ pot odds, else check or fold.
5. Heuristics Layer: We list heuristics and persistence between states here.
    - If we are up enough chips to lock a win (our chip lead > 1.55 * remaining hands), we always fold.