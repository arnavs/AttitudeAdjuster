"""
Track the posterior distribution over opponent hole cards for each hand in a match CSV.

Usage: python posterior_tracker.py matches/match_28112.csv [hand_number]

If hand_number is given, prints detailed per-action posterior updates for that hand.
Otherwise, prints a per-hand summary showing posterior entropy and whether the
true opponent hand was in the top-k.
"""
import ast
import csv
import itertools
import multiprocessing
import sys

import numpy as np
from gym_env import PokerEnv, WrappedEval
from submission.player import _discard_likelihood

RANKS = PokerEnv.RANKS
SUITS = PokerEnv.SUITS
STREET_MAP = {"Pre-Flop": 0, "Flop": 1, "Turn": 2, "River": 3}


def card_str_to_int(card_str):
    return SUITS.index(card_str[1]) * len(RANKS) + RANKS.index(card_str[0])


def parse_card_list(s):
    cards = ast.literal_eval(s)
    return [card_str_to_int(c) for c in cards]


def int_to_card_str(c):
    return RANKS[c % 9] + SUITS[c // 9]


class PosteriorTracker:
    """Mirrors the posterior logic from PlayerAgent, but only observes — never acts."""

    def __init__(self):
        self.evaluator = WrappedEval()
        self.MC_SAMPLES = 64
        self.PRIOR_DISCARD_TEMP = 9.0
        self.UPDATE_RAISE_TEMP = 3.0
        self.UPDATE_CHECK_TEMP = 1.5
        self.OPP_STRENGTH_HAND_SAMPLES = 30
        self.OPP_STRENGTH_BOARD_SAMPLES = 20

        self.opp_showdown_wins = 1
        self.opp_showdowns = 2
        self.opp_pressure_events = 0
        self.opp_postflop_observations = 0

    def _aggression_factor(self):
        if self.opp_postflop_observations <= 0:
            return 0.0
        return min(1.0, self.opp_pressure_events / self.opp_postflop_observations)

    def init_prior(self, my_cards, my_discards, opp_discards, community):
        known = set(my_cards) | set(my_discards) | set(opp_discards) | set(community)
        remaining = [c for c in range(27) if c not in known]
        self.opp_pairs = list(itertools.combinations(remaining, 2))
        self.opp_weights = np.ones(len(self.opp_pairs), dtype=np.float64)

    def _normalize_weights(self):
        total = float(self.opp_weights.sum())
        if total > 0 and np.isfinite(total):
            self.opp_weights /= total

    def _mc_equity(self, h1, h2, flop, pool, n_samples=100):
        pool = np.array(pool)
        wins = 0.0
        for _ in range(n_samples):
            sample = np.random.choice(pool, size=4, replace=False)
            opp_h1, opp_h2, turn, river = sample
            board = [PokerEnv.int_to_card(c) for c in flop + [turn, river]]
            our_rank = self.evaluator.evaluate(
                [PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)], board
            )
            opp_rank = self.evaluator.evaluate(
                [PokerEnv.int_to_card(opp_h1), PokerEnv.int_to_card(opp_h2)], board
            )
            if our_rank < opp_rank:
                wins += 1.0
            elif our_rank == opp_rank:
                wins += 0.5
        return wins / n_samples

    def update_prior_discard(self, opp_discards, my_discards, community, flop_cards, blind_position):
        discard_samples = max(18, self.MC_SAMPLES // 2)
        obs_mini = {"blind_position": blind_position}
        args = [
            (h1, h2, opp_discards, my_discards, community,
             flop_cards, obs_mini, discard_samples, self.PRIOR_DISCARD_TEMP)
            for h1, h2 in self.opp_pairs
        ]
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(4) as pool:
            likelihoods = np.array(pool.map(_discard_likelihood, args))

        self.opp_weights *= likelihoods
        self._normalize_weights()

    def zero_community_overlaps(self, community):
        comm_set = set(community)
        for i, (h1, h2) in enumerate(self.opp_pairs):
            if h1 in comm_set or h2 in comm_set:
                self.opp_weights[i] = 0.0
        self._normalize_weights()

    def _opp_hand_strength(self, h1, h2, community, opp_discards, my_discards):
        opp_treys = [PokerEnv.int_to_card(h1), PokerEnv.int_to_card(h2)]
        dead = set(opp_discards) | set(my_discards) | set(community) | {h1, h2}
        dead.discard(-1)
        pool = [c for c in range(27) if c not in dead]
        all_hands = list(itertools.combinations(pool, 2))
        if not all_hands:
            return 0.5

        hand_count = min(len(all_hands), self.OPP_STRENGTH_HAND_SAMPLES)
        if hand_count == len(all_hands):
            sampled_hands = all_hands
        else:
            sampled_idx = np.random.choice(len(all_hands), size=hand_count, replace=False)
            sampled_hands = [all_hands[i] for i in sampled_idx]

        if len(community) == 5:
            boards = [community]
        elif len(community) == 4:
            rivers = (
                pool
                if len(pool) <= self.OPP_STRENGTH_BOARD_SAMPLES
                else list(np.random.choice(pool, size=self.OPP_STRENGTH_BOARD_SAMPLES, replace=False))
            )
            boards = [community + [int(r)] for r in rivers]
        else:
            board_samples = min(
                max(4, self.OPP_STRENGTH_BOARD_SAMPLES),
                len(pool) * max(len(pool) - 1, 1),
            )
            boards = []
            seen = set()
            while len(boards) < board_samples and len(seen) < len(pool) * max(len(pool) - 1, 1):
                t, r = np.random.choice(pool, size=2, replace=False)
                key = tuple(sorted((int(t), int(r))))
                if key in seen:
                    continue
                seen.add(key)
                boards.append(community + [int(t), int(r)])

        total = 0.0
        count = 0
        for board_cards in boards:
            board = [PokerEnv.int_to_card(c) for c in board_cards]
            opp_rank = self.evaluator.evaluate(opp_treys, board)
            board_set = set(board_cards)
            for r1, r2 in sampled_hands:
                if r1 in board_set or r2 in board_set:
                    continue
                rand_treys = [PokerEnv.int_to_card(r1), PokerEnv.int_to_card(r2)]
                rand_rank = self.evaluator.evaluate(rand_treys, board)
                total += 1.0 if opp_rank < rand_rank else (0.5 if opp_rank == rand_rank else 0.0)
                count += 1
        return total / count if count > 0 else 0.5

    def update_prior_raise(self, street, raise_fraction, community, opp_discards, my_discards):
        if raise_fraction <= 0:
            return

        street_scale = {1: 0.6, 2: 0.85, 3: 1.2}.get(street, 0.8)
        sizing_signal = raise_fraction**0.6
        opp_win_rate = self.opp_showdown_wins / max(self.opp_showdowns, 1)
        aggression = self._aggression_factor()
        temp = (
            self.UPDATE_RAISE_TEMP
            * sizing_signal
            * street_scale
            * (0.55 + 0.45 * opp_win_rate)
            * (1.0 - 0.35 * aggression)
        )
        temp = max(0.1, min(temp, 2.2))

        active_pairs = [(i, h1, h2) for i, (h1, h2) in enumerate(self.opp_pairs) if self.opp_weights[i] > 0]
        strengths = np.array([
            self._opp_hand_strength(h1, h2, community, opp_discards, my_discards)
            for _, h1, h2 in active_pairs
        ])
        log_weights = temp * strengths
        log_weights -= log_weights.max()
        for k, (i, _, _) in enumerate(active_pairs):
            self.opp_weights[i] *= np.exp(log_weights[k])
        self._normalize_weights()

    def update_prior_check(self, community, opp_discards, my_discards):
        aggression = self._aggression_factor()
        temp = self.UPDATE_CHECK_TEMP * (0.5 + 0.5 * aggression)
        temp = min(temp, 1.5)

        active_pairs = [(i, h1, h2) for i, (h1, h2) in enumerate(self.opp_pairs) if self.opp_weights[i] > 0]
        strengths = np.array([
            self._opp_hand_strength(h1, h2, community, opp_discards, my_discards)
            for _, h1, h2 in active_pairs
        ])
        log_weights = -temp * strengths
        log_weights -= log_weights.max()
        for k, (i, _, _) in enumerate(active_pairs):
            self.opp_weights[i] *= np.exp(log_weights[k])
        self._normalize_weights()

    def update_showdown(self, reward):
        self.opp_showdowns += 1
        if reward < 0:
            self.opp_showdown_wins += 1

    def update_aggression(self, opp_bet_greater):
        self.opp_postflop_observations += 1
        if opp_bet_greater:
            self.opp_pressure_events += 1

    def top_k(self, k=10):
        idx = np.argsort(-self.opp_weights)[:k]
        return [(self.opp_pairs[i], float(self.opp_weights[i])) for i in idx]

    def entropy(self):
        p = self.opp_weights / self.opp_weights.sum()
        p = p[p > 0]
        return -float(np.sum(p * np.log2(p)))

    def rank_of(self, true_pair):
        true_pair_sorted = tuple(sorted(true_pair))
        for rank, i in enumerate(np.argsort(-self.opp_weights)):
            if tuple(sorted(self.opp_pairs[i])) == true_pair_sorted:
                return rank + 1
        return -1

    def weight_of(self, true_pair):
        true_pair_sorted = tuple(sorted(true_pair))
        for i, pair in enumerate(self.opp_pairs):
            if tuple(sorted(pair)) == true_pair_sorted:
                return float(self.opp_weights[i])
        return 0.0


def load_hands(csv_path):
    hands = {}
    with open(csv_path) as f:
        first = f.readline()
        if not first.startswith("hand_number"):
            reader = csv.DictReader(f)
        else:
            f.seek(0)
            f.readline()
            reader = csv.DictReader(f)
        for row in reader:
            hnum = int(row["hand_number"])
            hands.setdefault(hnum, []).append(row)
    return hands


def process_hand(tracker, rows, verbose=False):
    """Process a single hand, updating the tracker's posterior.

    Returns (true_opp_pair, final_entropy, final_rank, reward).
    """
    first_row = rows[0]
    my_cards_full = parse_card_list(first_row["team_0_cards"])
    opp_cards_full = parse_card_list(first_row["team_1_cards"])

    # Determine who is team 0 (us) vs team 1 (opponent)
    t0_bet = int(first_row["team_0_bet"])
    t1_bet = int(first_row["team_1_bet"])

    blind_position = 0 if t0_bet <= t1_bet else 1
    prior_initialized = False
    flop_cards = None
    my_cards = my_cards_full[:]
    my_discards = []
    opp_discards = []
    community = []
    zeroed_streets = set()
    last_street = -1
    reward = 0

    events = []

    for row in rows:
        street = STREET_MAP[row["street"]]
        active_team = int(row["active_team"])
        action_type = row["action_type"]
        action_amount = int(row["action_amount"])
        t0_bet = int(row["team_0_bet"])
        t1_bet = int(row["team_1_bet"])

        # Update community cards
        board = parse_card_list(row["board_cards"]) if row["board_cards"] != "[]" else []
        if len(board) > len(community):
            community = board

        # Track discards
        t0_disc = parse_card_list(row["team_0_discarded"]) if row["team_0_discarded"] != "[]" else []
        t1_disc = parse_card_list(row["team_1_discarded"]) if row["team_1_discarded"] != "[]" else []

        if action_type == "DISCARD":
            if active_team == 0:
                keep1 = int(row["action_keep_1"])
                keep2 = int(row["action_keep_2"])
                kept = [my_cards_full[keep1], my_cards_full[keep2]]
                my_discards = [c for c in my_cards_full if c not in kept]
                my_cards = kept
            else:
                keep1 = int(row["action_keep_1"])
                keep2 = int(row["action_keep_2"])
                kept = [opp_cards_full[keep1], opp_cards_full[keep2]]
                opp_discards = [c for c in opp_cards_full if c not in kept]

        # Initialize prior once both discards are known
        if not prior_initialized and len(opp_discards) == 3 and len(my_discards) == 3:
            flop_cards = community[:3]
            tracker.init_prior(my_cards, my_discards, opp_discards, community)

            if verbose:
                print(f"  Prior initialized: {len(tracker.opp_pairs)} pairs")

            tracker.update_prior_discard(opp_discards, my_discards, community, flop_cards, blind_position)

            if verbose:
                true_pair = tuple(sorted(kept))
                rank = tracker.rank_of(true_pair)
                ent = tracker.entropy()
                print(f"  After discard update: entropy={ent:.2f} bits, true pair rank={rank}/{len(tracker.opp_pairs)}")

            prior_initialized = True

        # Zero out pairs that overlap with new community cards
        if prior_initialized and street >= 2 and street not in zeroed_streets:
            tracker.zero_community_overlaps(community)
            zeroed_streets.add(street)
            if verbose:
                true_pair = tuple(sorted(kept))
                rank = tracker.rank_of(true_pair)
                ent = tracker.entropy()
                print(f"  Street {street} community zeroing: entropy={ent:.2f}, rank={rank}")

        # Update posterior on opponent raises (postflop only)
        if prior_initialized and active_team == 1 and street >= 1:
            opp_bet_greater = t1_bet > t0_bet
            tracker.update_aggression(opp_bet_greater)

            if action_type == "RAISE":
                pot_size = t0_bet + t1_bet
                raise_fraction = (t1_bet - t0_bet) / max(pot_size, 1)
                tracker.update_prior_raise(street, raise_fraction, community, opp_discards, my_discards)

                if verbose:
                    true_pair = tuple(sorted(kept))
                    rank = tracker.rank_of(true_pair)
                    ent = tracker.entropy()
                    print(f"  Opp RAISE (street {street}, frac={raise_fraction:.2f}): entropy={ent:.2f}, rank={rank}")

            elif action_type == "CHECK" and t1_bet == t0_bet:
                tracker.update_prior_check(community, opp_discards, my_discards)

                if verbose:
                    true_pair = tuple(sorted(kept))
                    rank = tracker.rank_of(true_pair)
                    ent = tracker.entropy()
                    print(f"  Opp CHECK (street {street}): entropy={ent:.2f}, rank={rank}")

    # Determine reward from final bets
    last_row = rows[-1]
    last_action = last_row["action_type"]
    t0_bet_final = int(last_row["team_0_bet"])
    t1_bet_final = int(last_row["team_1_bet"])

    if last_action == "FOLD":
        active = int(last_row["active_team"])
        if active == 0:
            reward = -min(t0_bet_final, t1_bet_final)
        else:
            reward = min(t0_bet_final, t1_bet_final)
    else:
        # Showdown — need to evaluate hands
        if len(community) == 5 and len(my_cards) == 2 and len(opp_discards) == 3:
            opp_kept = [c for c in opp_cards_full if c not in opp_discards]
            board = [PokerEnv.int_to_card(c) for c in community]
            evaluator = WrappedEval()
            our_rank = evaluator.evaluate([PokerEnv.int_to_card(c) for c in my_cards], board)
            opp_rank = evaluator.evaluate([PokerEnv.int_to_card(c) for c in opp_kept], board)
            if our_rank < opp_rank:
                reward = min(t0_bet_final, t1_bet_final)
            elif our_rank > opp_rank:
                reward = -min(t0_bet_final, t1_bet_final)
            else:
                reward = 0

            if t0_bet_final == t1_bet_final:
                tracker.update_showdown(reward)

    # True opponent pair (after discard)
    opp_kept = [c for c in opp_cards_full if c not in opp_discards] if opp_discards else opp_cards_full[:2]
    true_pair = tuple(sorted(opp_kept))

    if prior_initialized:
        return true_pair, tracker.entropy(), tracker.rank_of(true_pair), reward
    return true_pair, None, None, reward


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "matches/match_28112.csv"
    target_hand = int(sys.argv[2]) if len(sys.argv) > 2 else None

    hands = load_hands(csv_path)
    print(f"Loaded {len(hands)} hands from {csv_path}\n")

    tracker = PosteriorTracker()

    if target_hand is not None:
        if target_hand not in hands:
            print(f"Hand {target_hand} not found")
            return
        rows = hands[target_hand]
        print(f"=== Hand {target_hand} ===")
        first = rows[0]
        print(f"  Our cards:  {first['team_0_cards']}")
        print(f"  Opp cards:  {first['team_1_cards']}")
        print(f"  Board:      {rows[-1]['board_cards']}")
        print()
        true_pair, ent, rank, reward = process_hand(tracker, rows, verbose=True)
        opp_str = ", ".join(int_to_card_str(c) for c in true_pair)
        print(f"\n  True opp hand: ({opp_str})")
        if ent is not None:
            n_pairs = len(tracker.opp_pairs)
            print(f"  Final entropy: {ent:.2f} bits (uniform would be {np.log2(n_pairs):.2f})")
            print(f"  True pair rank: {rank}/{n_pairs}")
            print(f"  True pair weight: {tracker.weight_of(true_pair):.4f}")
            print(f"\n  Top 10 pairs:")
            for (h1, h2), w in tracker.top_k(10):
                marker = " <-- TRUE" if tuple(sorted((h1, h2))) == true_pair else ""
                print(f"    ({int_to_card_str(h1)}, {int_to_card_str(h2)}): {w:.4f}{marker}")
        print(f"  Reward: {reward:+d}")
        return

    # Summary mode
    print(f"{'Hand':>6} {'Entropy':>8} {'Rank':>6} {'Total':>6} {'Top5':>5} {'Reward':>7}")
    print("-" * 50)

    total_in_top5 = 0
    total_in_top10 = 0
    total_hands = 0

    for hnum in sorted(hands.keys()):
        rows = hands[hnum]
        true_pair, ent, rank, reward = process_hand(tracker, rows, verbose=False)

        if ent is None:
            continue

        total_hands += 1
        n_pairs = len(tracker.opp_pairs)
        in_top5 = "Y" if rank <= 5 else ""
        if rank <= 5:
            total_in_top5 += 1
        if rank <= 10:
            total_in_top10 += 1

        print(f"{hnum:6d} {ent:8.2f} {rank:6d} {n_pairs:6d} {in_top5:>5} {reward:+7d}")

    if total_hands > 0:
        print(f"\n{'='*50}")
        print(f"Hands with posterior: {total_hands}")
        print(f"Top-5  accuracy: {total_in_top5}/{total_hands} ({100*total_in_top5/total_hands:.1f}%)")
        print(f"Top-10 accuracy: {total_in_top10}/{total_hands} ({100*total_in_top10/total_hands:.1f}%)")


if __name__ == "__main__":
    main()
