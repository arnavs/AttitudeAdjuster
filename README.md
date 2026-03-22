# LOU Attitude Adjuster

This is my bot for CMU's [AI Poker Contest](https://cmudsc.com/pokerai-2026/). It's named after the [Culture Warship](https://theculture.fandom.com/wiki/Attitude_Adjuster), and also because of what it does: blend a Deep CFR approach with a Bayesian posterior over opponent's holes (i.e., an attitude adjuster).

> [!NOTE]  
> **LLM Usage**: I worked heavily with [Claude Code](https://code.claude.com/docs/en/overview) while building this. 

> [!NOTE]  
> **Submission**: I didn't have time to finish the training run. So what I actually submitted was a Bayesian heuristic bot `submission/frank_exchange.py`, named after the VFP [_Frank Exchange of Views_](https://theculture.fandom.com/wiki/Frank_Exchange_of_Views). It placed around 37th out of 110 or so.

## The Game 

The game we're learning is a modified version of HUNL ("Heads-Up No Limit") poker. In particular: 

   0. The deck is 27 cards (3 suits, no Jacks, Queens, or Kings). Aces can be high or low.
   1. Each player (Small Blind or Big Blind) is dealt 5 preflop. After preflop betting, there is a **discard phase.**
      -    BB discards 3 cards first, and these are revealed to SB. 
      -    Then SB discards 3 cards, and these are revealed to BB. 
   2. Then normal HUNL poker is played on this reduced deck. 

Bots are given stacks of 100 chips at the start of each of 1000 hands. The bot up in chips at the end of 1000 hands win. 

## Constraints

This bot had to be prepared under strict constraints. In particular: 

   * 1GB upload limit. 
   * 4GB RAM, 4 vCPUs, and 1500 seconds total to move on the server. 
   * One week from revelation of the variant to the competition. 
   * I only had my laptop (16GB RAM, 8 virtual M4 cores) available for training. 
      > The RAM point is crucial, since the algorithm requires holding a buffer in memory, and hitting swap memory was a noticeable slowdown.

## Approach 

Hart and Mas-Colell proposed the [regret matching algorithm](https://www.ma.imperial.ac.uk/~dturaev/Hart0.pdf) to find (correlated) Nash equilibria of incomplete-information games. This was operationalized by [Brown and Sandholm](https://arxiv.org/pdf/1809.04040) in an algorithm called Counterfactual Regret Minimization (CFR.) CFR is generally considered the state of the art for incomplete information games (and in particular [Poker](https://www.ijcai.org/proceedings/2017/0772.pdf)), but the tabular lookup is expensive. 

People have experimented with various abstractions to reduce the effective state space size (e.g., applying a suit permutation to hands and the flop.) Here that would reduce the state space for Big Blind from C(27, 5) * C(22, 3) = 124M to about 20M. Further reductions may be possible if you know something about the game (e.g., if you can do lossy compression to the 20M permuted keys.) 

But this requires being a good poker player. The best learning algorithm cannot save you from a bad abstraction. So one approach is to use what is effectively an autoencoder to learn [these abstractions nonparametrically](https://arxiv.org/pdf/1811.00164) via an algorithm called Deep CFR. Under these constraints, however, Deep CFR is noisy (and even in general it throws out domain-specific knowledge.)

Our approach is _posterior blending_. We maintain a Bayesian posterior over opponent's hole cards, beginning with a uniform distribution over the C(16, 2) possible pairs once all discards are known (27 deck cards - 5 hand cards - 3 flop cards - 3 opp discards = 16.) This posterior is updated based on a simple heuristic (though bootstrapping from the strategy network is possible.) The strategy network outputs a distribution over actions which is combined linearly with that induced by the posterior, according to the effective range of the posterior (i.e., its entropy, or the number of pairs above some confidence threshold.) That is: 

```
weight = min(0.2, (POSTERIOR_THRESHOLD - eff) / POSTERIOR_THRESHOLD)
return (1 - weight) * probs + weight * ev_probs
```

is the heart of the algorithm. 

The rest is mechanics. In particular: 

1. We decide our discards by a "Level 2.5" rationality scheme. We assume that our opponent is discarding strategically (Level 1.) However, we also _partially_ internalize the effects of our discard decision (as BB, who discards first) on SB. We assume that SB sees our discards and correctly realizes that they cannot be drawn, but do not model SB's inference about BB's hole cards `(h1, h2)` given that BB discarded `(d1, d2, d3)`.

2. Posteriors are updated using a simple Monte Carlo in response to raises and checks. 

3. Actions are softmaxes over a final action distribution. 

## Code Structure

All new code lives in `submission/`. 

   1. The network architecture for the DCFR approach is in `submission/network.py`. 
   2. The traverser is in `submision/traversal.py`. 
   3. The player agent is in `submission/player.py`. 
   4. The training loop is in `submission/train.py`. 
   4. The observation encoder is in `submission/encoder.py`. 

## How to Train the Bot

> [!CAUTION]
> Training is very compute and memory intensive. 

## How to Run the Engine

1. Create a virtual environment:

   ```bash
   python3.12 -m venv .venv
   ```

2. Activate the virtual environment:
   - On Windows:

     ```bash
     .venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source .venv/bin/activate
     ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

1. Basic coverage test:

```bash
pytest --cov=gym_env --cov-report=term-missing --cov-report=html --cov-branch
```

### Testing

1. To test the Attitude Adjuster against ProbabilityAgent, AllInAgent, FoldAgent, CallingStationAgent, RandomAgent:

```bash
python agent_test.py
```

2. To run a full match (1000 hands) of your agent against a specific agent:

```bash
python run.py
```

You can modify which bots play by modifying the agent config file. Write the file path to the corresponding agent for that bot to play. 
