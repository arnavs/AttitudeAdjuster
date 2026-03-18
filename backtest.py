"""
Replay match CSV: keep opponent (Team 1) actions fixed, let our updated bot play Team 0.
"""
import ast
import csv
import sys
from gym_env import PokerEnv
from submission.player import PlayerAgent

STREET_MAP = {"Pre-Flop": 0, "Flop": 1, "Turn": 2, "River": 3}
ACTION_MAP = {
    "FOLD": PokerEnv.ActionType.FOLD.value,
    "RAISE": PokerEnv.ActionType.RAISE.value,
    "CHECK": PokerEnv.ActionType.CHECK.value,
    "CALL": PokerEnv.ActionType.CALL.value,
    "DISCARD": PokerEnv.ActionType.DISCARD.value,
}


def card_str_to_int(card_str):
    """Convert e.g. '7d' -> int index in [0,27)."""
    RANKS = PokerEnv.RANKS  # "23456789A"
    SUITS = PokerEnv.SUITS  # "dhs"
    rank = card_str[0]
    suit = card_str[1]
    return SUITS.index(suit) * len(RANKS) + RANKS.index(rank)


def parse_card_list(s):
    """Parse string like \"['2s', '7d']\" into list of int card indices."""
    cards = ast.literal_eval(s)
    return [card_str_to_int(c) for c in cards]


def load_hands(csv_path):
    """Parse CSV into dict of hand_number -> list of action rows."""
    hands = {}
    with open(csv_path) as f:
        # Skip comment line
        first = f.readline()
        if not first.startswith("hand_number"):
            reader = csv.DictReader(f)
        else:
            f.seek(0)
            # skip comment
            f.readline()
            reader = csv.DictReader(f)
        for row in reader:
            hnum = int(row["hand_number"])
            hands.setdefault(hnum, []).append(row)
    return hands


def reconstruct_deck(row):
    """Reconstruct the 27-card deck order from logged cards.

    Cards are dealt: 5 to player0, 5 to player1, 5 community, rest unknown.
    We just need the first 15 in the right order; remaining don't matter.
    """
    p0_cards = parse_card_list(row["team_0_cards"])
    p1_cards = parse_card_list(row["team_1_cards"])

    # Find community cards from the last row of the hand (most revealed)
    return p0_cards, p1_cards


def get_full_community(rows):
    """Get full 5-card community from the last row that has board cards."""
    for row in reversed(rows):
        board = parse_card_list(row["board_cards"])
        if len(board) == 5:
            return board
        if len(board) > 0:
            # Might be partial; keep looking
            pass
    # Return whatever we have from last row
    return parse_card_list(rows[-1]["board_cards"])


def build_deck_order(rows):
    """Build the full 27-card deck order used by PokerEnv.reset().

    Deal order: 5 to p0, 5 to p1, 5 community, then remaining.
    """
    first_row = rows[0]
    p0_cards = parse_card_list(first_row["team_0_cards"])
    p1_cards = parse_card_list(first_row["team_1_cards"])
    community = get_full_community(rows)

    dealt = set(p0_cards + p1_cards + community)
    remaining = [c for c in range(27) if c not in dealt]

    # PokerEnv deals: 5 to p0, 5 to p1, 5 community
    deck = p0_cards + p1_cards + community + remaining
    return deck


def get_small_blind_player(rows):
    """Determine who was small blind from the first preflop action."""
    # First action in preflop is by small blind (SB acts first preflop)
    # But actually the SB posts first, then acts first
    # From the CSV, first row's team_0_bet and team_1_bet tell us:
    # SB posts 1, BB posts 2
    first = rows[0]
    t0_bet = int(first["team_0_bet"])
    t1_bet = int(first["team_1_bet"])
    if t0_bet <= t1_bet:
        return 0  # team 0 is SB
    else:
        return 1  # team 1 is SB


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "matches/match_16159.csv"

    hands = load_hands(csv_path)
    print(f"Loaded {len(hands)} hands from {csv_path}")

    bot = PlayerAgent(stream=False)
    env = PokerEnv()

    bankroll = 0  # Team 0 cumulative
    wins = 0
    losses = 0
    ties = 0
    skipped = 0
    original_bankroll = 0

    for hnum in sorted(hands.keys()):
        rows = hands[hnum]

        # Track original result
        last_row = rows[-1]
        if hnum + 1 in hands:
            next_first = hands[hnum + 1][0]
            orig_bank_after = int(next_first["team_0_bankroll"])
            orig_delta = orig_bank_after - original_bankroll
            original_bankroll = orig_bank_after
        else:
            orig_delta = None

        # Reconstruct deck and reset env
        deck = build_deck_order(rows)
        sb_player = get_small_blind_player(rows)
        (obs0, obs1), info = env.reset(options={"cards": deck, "small_blind_player": sb_player})

        # Collect Team 1's actions in order
        opp_actions = []
        for row in rows:
            if int(row["active_team"]) == 1:
                action_type = ACTION_MAP[row["action_type"]]
                amount = int(row["action_amount"])
                keep1 = int(row["action_keep_1"])
                keep2 = int(row["action_keep_2"])
                opp_actions.append((action_type, amount, keep1, keep2))

        opp_idx = 0
        terminated = False
        reward = (0, 0)
        truncated = False
        hand_info = {"hand_number": hnum}
        diverged = False

        while not terminated:
            acting = env.acting_agent
            if acting == 0:
                # Our bot acts
                obs = obs0
                action = bot.act(obs, reward[0], terminated, truncated, hand_info)
            else:
                # Replay opponent's logged action
                if opp_idx < len(opp_actions):
                    action = opp_actions[opp_idx]
                    opp_idx += 1
                else:
                    diverged = True
                    break

            (obs0, obs1), reward, terminated, truncated, info = env.step(action)
            info["hand_number"] = hnum
            bot.observe(obs0, reward[0], terminated, truncated, info)

        if diverged:
            skipped += 1
            continue

        delta = reward[0]
        bankroll += delta

        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1

        orig_str = f" (orig: {orig_delta:+d})" if orig_delta is not None else ""
        if abs(delta) >= 20 or (orig_delta is not None and delta != orig_delta):
            print(f"  Hand {hnum:4d}: {delta:+4d}  bankroll={bankroll:+6d}{orig_str}")

    print(f"\n{'='*50}")
    print(f"Final bankroll: {bankroll:+d}")
    print(f"Wins: {wins}  Losses: {losses}  Ties: {ties}  Skipped: {skipped}")
    print(f"Original match bankroll was: {original_bankroll:+d}")
    print(f"Improvement: {bankroll - original_bankroll:+d}")


if __name__ == "__main__":
    main()
