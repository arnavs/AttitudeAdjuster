"""
Neural Fictitious Self-Play (NFSP) training loop.

Two networks per player position (SB=0, BB=1):
  br_net[p]   - best-response Q-network (trained via DQN-style TD updates)
  avg_net[p]  - average strategy network (trained via supervised learning)

At each step the acting player:
  - with prob ETA  : acts greedily from br_net (best response, epsilon-greedy)
  - with prob 1-ETA: acts from avg_net (average strategy, softmax)

Transitions from BR episodes -> CircularBuffer -> train br_net (DQN / MSE)
(state, probs) from AVG episodes -> AvgStrategyBuffer -> train avg_net (cross-entropy)

At the end, avg_net is the deployed policy — saved as
  checkpoints/strategy_betting_p{p}_final.pt
which player.py picks up unchanged.

Usage:
  cd ~/MachineYearning
  python submission/train_nfsp.py
"""

import os, sys, time, copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network import make_betting_net, get_policy_distribution
from encoder import (
    encode_infoset, FOLD, N_BETTING_ACTIONS, KEEP_PAIRS,
)
from traversal import GameState, compute_bet_sizes

# ── hyperparameters ───────────────────────────────────────────────────────────

N_ITERATIONS      = 50_000
EPISODES_PER_ITER = 200      # self-play episodes collected per iteration
TRAIN_EVERY       = 25       # retrain networks every N iterations
N_TRAIN_STEPS_BR  = 500      # gradient steps on BR net per training phase
N_TRAIN_STEPS_AVG = 500      # gradient steps on avg net per training phase
BATCH_SIZE        = 512

ETA               = 0.1      # prob of acting from BR net (vs avg net)
EPSILON           = 0.06     # epsilon-greedy exploration on top of BR net
GAMMA             = 1.0      # no discounting (all payoffs are terminal)

BR_BUF_CAPACITY   = 200_000  # circular replay buffer
AVG_BUF_CAPACITY  = 1_000_000

LR_BR             = 1e-3
LR_AVG            = 1e-3
TAU               = 0.01     # soft target-network update

N_CORES           = 8
SAVE_EVERY        = 50
SAVE_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
WARM_CKPT         = None     # e.g. "submission/checkpoints/nfsp_checkpoint_500.pt"

os.makedirs(SAVE_DIR, exist_ok=True)


# ── replay buffers ────────────────────────────────────────────────────────────

class CircularBuffer:
    """Fixed-size circular buffer for DQN transitions."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer   = []
        self.ptr      = 0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, n):
        idx = np.random.choice(len(self.buffer), size=min(n, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


class AvgStrategyBuffer:
    """Reservoir buffer for (infoset_vec, action_probs) strategy samples."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer   = []
        self.n_seen   = 0

    def add(self, vec, probs):
        self.n_seen += 1
        item = (vec.copy(), probs.copy())
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            idx = np.random.randint(0, self.n_seen)
            if idx < self.capacity:
                self.buffer[idx] = item

    def sample(self, n):
        idx   = np.random.choice(len(self.buffer), size=min(n, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in idx]
        return np.stack([b[0] for b in batch]), np.stack([b[1] for b in batch])

    def __len__(self):
        return len(self.buffer)


# ── policy helpers ────────────────────────────────────────────────────────────

def br_policy(net, vec, mask, epsilon=0.0, device='cpu'):
    """Epsilon-greedy over Q-values. Returns (probs, sampled_action)."""
    with torch.no_grad():
        x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        q = net(x).squeeze(0).cpu().numpy()
    q_masked = np.where(mask > 0, q, -1e9)
    greedy   = int(np.argmax(q_masked))
    n_legal  = int(mask.sum())
    probs    = np.zeros(N_BETTING_ACTIONS, dtype=np.float32)
    for a in range(N_BETTING_ACTIONS):
        if mask[a] > 0:
            probs[a] = epsilon / n_legal + (1.0 - epsilon) * float(a == greedy)
    return probs, int(np.random.choice(N_BETTING_ACTIONS, p=probs))


def avg_policy(net, vec, mask, device='cpu'):
    """Softmax policy from average strategy net. Returns (probs, sampled_action)."""
    probs  = get_policy_distribution(net, vec, mask, device)
    action = int(np.random.choice(N_BETTING_ACTIONS, p=probs))
    return probs, action


def discard_heuristic(state, player):
    """
    Pick discard by MC equity against a random opponent hand.
    Returns (keep_idx_1, keep_idx_2).
    """
    from gym_env import PokerEnv, WrappedEval
    evaluator = WrappedEval()
    hand      = list(state.hole[player])
    board     = state.board()
    opp       = 1 - player
    opp_discs = state.discarded[opp] if state.discard_done[opp] else []
    dead      = set(hand) | set(board) | set(opp_discs)
    pool      = [c for c in range(27) if c not in dead]
    n_rem     = 5 - len(board)
    wins      = np.zeros(len(KEEP_PAIRS), dtype=np.float64)
    for _ in range(20):
        sampled  = np.random.choice(pool, size=5 + n_rem, replace=False)
        opp_hand = [int(c) for c in sampled[:5]]
        runout   = [int(c) for c in sampled[5:]]
        board5   = [PokerEnv.int_to_card(c) for c in board + runout]
        # opponent greedily keeps best-equity pair
        opp_ranks = [
            evaluator.evaluate(
                [PokerEnv.int_to_card(opp_hand[oi]), PokerEnv.int_to_card(opp_hand[oj])], board5)
            for oi, oj in KEEP_PAIRS
        ]
        best_opp = KEEP_PAIRS[int(np.argmin(opp_ranks))]
        opp_rank = evaluator.evaluate(
            [PokerEnv.int_to_card(opp_hand[best_opp[0]]),
             PokerEnv.int_to_card(opp_hand[best_opp[1]])], board5)
        for idx, (ki, kj) in enumerate(KEEP_PAIRS):
            my_rank = evaluator.evaluate(
                [PokerEnv.int_to_card(hand[ki]), PokerEnv.int_to_card(hand[kj])], board5)
            wins[idx] += 1.0 if my_rank < opp_rank else (0.5 if my_rank == opp_rank else 0.0)
    return KEEP_PAIRS[int(np.argmax(wins))]


# ── episode runner ────────────────────────────────────────────────────────────

def run_episode(br_states, avg_states, eta=ETA, epsilon=EPSILON, device='cpu'):
    """
    Run one self-play episode.
    Returns:
      br_transitions  : {player: [(vec, action, reward, next_vec, next_mask, done)]}
      avg_transitions : {player: [(vec, probs)]}  — only from AVG-mode steps
    """
    br_nets  = [make_betting_net() for _ in range(2)]
    avg_nets = [make_betting_net() for _ in range(2)]
    for p in range(2):
        br_nets[p].load_state_dict(br_states[p]);   br_nets[p].eval()
        avg_nets[p].load_state_dict(avg_states[p]); avg_nets[p].eval()

    state = GameState()
    # decide mode once per episode per player
    mode = ['br' if np.random.random() < eta else 'avg' for _ in range(2)]

    trajectory = {0: [], 1: []}  # player -> [(vec, action, mask, mode, probs)]

    while not state.terminal:
        # discard phase
        if state.street == 1:
            if not state.discard_done[1]:
                ki, kj = discard_heuristic(state, 1)
                state.apply_discard(1, ki, kj)
                continue
            if not state.discard_done[0]:
                ki, kj = discard_heuristic(state, 0)
                state.apply_discard(0, ki, kj)
                continue

        # both all-in: run out remaining streets
        if state.stacks[0] == 0 and state.stacks[1] == 0:
            while state.street <= 3 and not state.terminal:
                state.advance_street()
            break

        player = state.acting_player
        mask   = state.legal_betting_mask(player)
        if mask.sum() == 0:
            break

        obs  = state.obs(player)
        vec  = encode_infoset(obs, is_discard_node=False)

        if mode[player] == 'br':
            probs, action = br_policy(br_nets[player], vec, mask, epsilon, device)
        else:
            probs, action = avg_policy(avg_nets[player], vec, mask, device)

        trajectory[player].append((vec.copy(), action, mask.copy(), mode[player], probs.copy()))

        street_ended = state.apply_bet(player, action)
        if street_ended and not state.terminal:
            state.advance_street()

    payoffs = [state.payoff(p) for p in range(2)]

    br_transitions  = {0: [], 1: []}
    avg_transitions = {0: [], 1: []}

    for p in range(2):
        steps = trajectory[p]
        for i, (vec, action, mask, step_mode, probs) in enumerate(steps):
            is_last = (i == len(steps) - 1)
            reward  = float(payoffs[p]) if is_last else 0.0
            done    = is_last

            if not is_last:
                nv, _, nm, _, _ = steps[i + 1]
                next_vec, next_mask = nv.copy(), nm.copy()
            else:
                next_vec  = np.zeros_like(vec)
                next_mask = np.zeros(N_BETTING_ACTIONS, dtype=np.float32)

            br_transitions[p].append((vec, action, reward, next_vec, next_mask, done))
            if step_mode == 'avg':
                avg_transitions[p].append((vec, probs))

    return br_transitions, avg_transitions


# ── multiprocessing worker ────────────────────────────────────────────────────

def _worker(args):
    br_states, avg_states, eta, epsilon = args
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return run_episode(br_states, avg_states, eta, epsilon)


# ── training steps ────────────────────────────────────────────────────────────

def train_br_net(net, target_net, buf, optimizer,
                 batch_size=512, n_steps=500, gamma=1.0, device='cpu'):
    """DQN-style MSE update on best-response Q-network."""
    if len(buf) < batch_size:
        return None
    net.train()
    losses = []
    for _ in range(n_steps):
        batch     = buf.sample(batch_size)
        vecs      = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=device)
        actions   = torch.tensor([b[1] for b in batch],           dtype=torch.long,    device=device)
        rewards   = torch.tensor([b[2] for b in batch],           dtype=torch.float32, device=device)
        next_vecs = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=device)
        next_masks= torch.tensor(np.stack([b[4] for b in batch]), dtype=torch.float32, device=device)
        dones     = torch.tensor([b[5] for b in batch],           dtype=torch.float32, device=device)

        with torch.no_grad():
            next_q = target_net(next_vecs).masked_fill(next_masks == 0, -1e9)
            next_v = next_q.max(dim=1).values
            target = rewards + gamma * next_v * (1 - dones)

        q    = net(vecs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    net.eval()
    return float(np.mean(losses))


def train_avg_net(net, buf, optimizer,
                  batch_size=512, n_steps=500, device='cpu'):
    """Cross-entropy update on average strategy network."""
    if len(buf) < batch_size:
        return None
    net.train()
    losses = []
    for _ in range(n_steps):
        vecs, probs = buf.sample(batch_size)
        x         = torch.tensor(vecs,  dtype=torch.float32, device=device)
        tgt       = torch.tensor(probs, dtype=torch.float32, device=device)
        log_probs = F.log_softmax(net(x), dim=-1)
        loss      = -(tgt * log_probs).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    net.eval()
    return float(np.mean(losses))


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


# ── main ──────────────────────────────────────────────────────────────────────

def train():
    device = 'cpu'

    br_nets     = [make_betting_net().to(device) for _ in range(2)]
    target_nets = [copy.deepcopy(n)              for n in br_nets]
    avg_nets    = [make_betting_net().to(device) for _ in range(2)]
    for net in br_nets + target_nets + avg_nets:
        net.eval()

    if WARM_CKPT and os.path.exists(WARM_CKPT):
        ckpt = torch.load(WARM_CKPT, map_location=device, weights_only=False)
        for p in range(2):
            br_nets[p].load_state_dict(ckpt[f"br_net_{p}"])
            target_nets[p].load_state_dict(ckpt[f"target_net_{p}"])
            avg_nets[p].load_state_dict(ckpt[f"avg_net_{p}"])
        print(f"Warm-started from {WARM_CKPT}")

    br_opts  = [optim.Adam(n.parameters(), lr=LR_BR)  for n in br_nets]
    avg_opts = [optim.Adam(n.parameters(), lr=LR_AVG) for n in avg_nets]

    br_bufs  = [CircularBuffer(BR_BUF_CAPACITY)      for _ in range(2)]
    avg_bufs = [AvgStrategyBuffer(AVG_BUF_CAPACITY)  for _ in range(2)]

    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, "runs_nfsp"))

    print(f"NFSP: {N_ITERATIONS} iters | {EPISODES_PER_ITER} eps/iter | "
          f"eta={ETA} epsilon={EPSILON} | {N_CORES} cores | train every {TRAIN_EVERY}")

    t0 = time.time()

    with Pool(processes=N_CORES) as pool:
        for iteration in range(1, N_ITERATIONS + 1):

            br_states  = [br_nets[p].state_dict()  for p in range(2)]
            avg_states = [avg_nets[p].state_dict() for p in range(2)]

            jobs    = [(br_states, avg_states, ETA, EPSILON)] * EPISODES_PER_ITER
            results = pool.map(_worker, jobs)

            for br_trans, avg_trans in results:
                for p in range(2):
                    for t in br_trans[p]:
                        br_bufs[p].add(t)
                    for t in avg_trans[p]:
                        avg_bufs[p].add(t[0], t[1])

            if iteration % TRAIN_EVERY == 0:
                for p in range(2):
                    br_loss = train_br_net(
                        br_nets[p], target_nets[p], br_bufs[p], br_opts[p],
                        BATCH_SIZE, N_TRAIN_STEPS_BR, GAMMA, device)
                    soft_update(target_nets[p], br_nets[p], TAU)

                    avg_loss = train_avg_net(
                        avg_nets[p], avg_bufs[p], avg_opts[p],
                        BATCH_SIZE, N_TRAIN_STEPS_AVG, device)

                    if br_loss is not None:
                        writer.add_scalar(f"loss/br_p{p}",  br_loss,  iteration)
                    if avg_loss is not None:
                        writer.add_scalar(f"loss/avg_p{p}", avg_loss, iteration)

                    if len(avg_bufs[p]) >= 100:
                        _, ss = avg_bufs[p].sample(min(len(avg_bufs[p]), 5_000))
                        writer.add_scalar(f"diag/fold_freq_p{p}", ss[:, FOLD].mean(), iteration)

                for p in range(2):
                    writer.add_scalar(f"buffer/br_p{p}",  len(br_bufs[p]),  iteration)
                    writer.add_scalar(f"buffer/avg_p{p}", len(avg_bufs[p]), iteration)

                elapsed = time.time() - t0
                rate    = iteration / elapsed
                print(f"iter {iteration:6d} | "
                      f"br=({len(br_bufs[0])},{len(br_bufs[1])}) "
                      f"avg=({len(avg_bufs[0])},{len(avg_bufs[1])}) | "
                      f"{rate:.1f} it/s | ETA {(N_ITERATIONS-iteration)/rate/3600:.1f}h")

            if iteration % SAVE_EVERY == 0:
                torch.save({
                    "iteration":    iteration,
                    "br_net_0":     br_nets[0].state_dict(),
                    "br_net_1":     br_nets[1].state_dict(),
                    "target_net_0": target_nets[0].state_dict(),
                    "target_net_1": target_nets[1].state_dict(),
                    "avg_net_0":    avg_nets[0].state_dict(),
                    "avg_net_1":    avg_nets[1].state_dict(),
                }, os.path.join(SAVE_DIR, f"nfsp_checkpoint_{iteration}.pt"))
                print(f"  saved nfsp_checkpoint_{iteration}.pt")

    # save final avg nets — player.py loads these directly, no changes needed
    print("\nSaving final average strategy networks...")
    for p in range(2):
        path = os.path.join(SAVE_DIR, f"strategy_betting_p{p}_final.pt")
        torch.save(avg_nets[p].state_dict(), path)
        print(f"  p{p} -> {path}")

    print("Done.")


if __name__ == "__main__":
    train()
