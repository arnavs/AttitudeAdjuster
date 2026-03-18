"""
Deep CFR network and replay buffers.

Value network V:    infoset -> advantage (regret) per action
Strategy network Pi: infoset -> average strategy (prob per action)

Both are the same architecture; different training targets.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import INPUT_DIM, N_ACTIONS


# ──────────────────────────────────────────
# Network
# ──────────────────────────────────────────

class CFRNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=256, output_dim=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def get_strategy(net, infoset_vec, mask, device='cpu'):
    """
    Given a value network and an infoset vector, return a
    strategy (prob distribution) via regret matching.

    mask: 15-dim binary array of legal actions.
    Returns: numpy array of probabilities over N_ACTIONS.
    """
    with torch.no_grad():
        x = torch.tensor(infoset_vec, dtype=torch.float32,
                         device=device).unsqueeze(0)
        advantages = net(x).squeeze(0).cpu().numpy()

    # mask illegal actions
    advantages = advantages * mask

    # regret matching: strategy proportional to positive advantages
    pos = np.maximum(advantages, 0.0)
    total = pos.sum()
    if total > 0:
        return pos / total
    else:
        # uniform over legal actions
        legal = mask.astype(np.float32)
        return legal / legal.sum()


# ──────────────────────────────────────────
# Replay buffers with reservoir sampling
# ──────────────────────────────────────────

class ReservoirBuffer:
    """
    Fixed-capacity buffer with reservoir sampling.
    When full, each new sample replaces a random existing sample
    with probability capacity / (capacity + n_seen).
    This ensures uniform sampling over all samples ever seen.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []        # list of (infoset_vec, advantages, mask)
        self.n_seen = 0

    def add(self, infoset_vec, advantages, mask):
        self.n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((infoset_vec.copy(),
                                advantages.copy(),
                                mask.copy()))
        else:
            # reservoir sampling: replace with prob capacity/n_seen
            idx = np.random.randint(0, self.n_seen)
            if idx < self.capacity:
                self.buffer[idx] = (infoset_vec.copy(),
                                    advantages.copy(),
                                    mask.copy())

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer),
                                   size=min(batch_size, len(self.buffer)),
                                   replace=False)
        batch = [self.buffer[i] for i in indices]
        vecs  = np.stack([b[0] for b in batch])
        advs  = np.stack([b[1] for b in batch])
        masks = np.stack([b[2] for b in batch])
        return vecs, advs, masks

    def __len__(self):
        return len(self.buffer)


class StrategyBuffer:
    """
    Buffer for strategy samples (infoset, strategy, iteration_weight).
    Used to train the final policy network Pi.
    Reservoir sampling with iteration weighting as per Deep CFR paper.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []        # list of (infoset_vec, strategy, weight)
        self.n_seen = 0

    def add(self, infoset_vec, strategy, iteration):
        """iteration: CFR iteration number, used as sample weight."""
        self.n_seen += 1
        item = (infoset_vec.copy(), strategy.copy(), float(iteration))
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            idx = np.random.randint(0, self.n_seen)
            if idx < self.capacity:
                self.buffer[idx] = item

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer),
                                   size=min(batch_size, len(self.buffer)),
                                   replace=False)
        batch  = [self.buffer[i] for i in indices]
        vecs   = np.stack([b[0] for b in batch])
        strats = np.stack([b[1] for b in batch])
        weights= np.array([b[2] for b in batch], dtype=np.float32)
        return vecs, strats, weights

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────
# Training steps
# ──────────────────────────────────────────

def train_value_network(net, buffer, optimizer,
                        batch_size=512, n_steps=200, device='cpu'):
    """
    Train value network V on regret samples from buffer.
    Loss: MSE between predicted advantages and sampled regrets,
    weighted by mask (only compute loss on legal actions).
    """
    if len(buffer) < batch_size:
        return None

    net.train()
    losses = []
    for _ in range(n_steps):
        vecs, advs, masks = buffer.sample(batch_size)
        x    = torch.tensor(vecs,  dtype=torch.float32, device=device)
        tgt  = torch.tensor(advs,  dtype=torch.float32, device=device)
        mask = torch.tensor(masks, dtype=torch.float32, device=device)

        pred = net(x)

        # MSE loss only over legal actions
        loss = ((pred - tgt) ** 2 * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    net.eval()
    return float(np.mean(losses))


def train_strategy_network(net, buffer, optimizer,
                           batch_size=512, n_steps=400, device='cpu'):
    """
    Train strategy network Pi on strategy samples.
    Loss: weighted cross-entropy between predicted strategy and
    average strategy samples (weighted by iteration number).
    """
    if len(buffer) < batch_size:
        return None

    net.train()
    losses = []
    for _ in range(n_steps):
        vecs, strats, weights = buffer.sample(batch_size)
        x   = torch.tensor(vecs,    dtype=torch.float32, device=device)
        tgt = torch.tensor(strats,  dtype=torch.float32, device=device)
        w   = torch.tensor(weights, dtype=torch.float32, device=device)

        # normalize weights in batch
        w = w / w.sum()

        logits = net(x)
        # softmax cross-entropy weighted by iteration
        log_probs = F.log_softmax(logits, dim=-1)
        # only where tgt > 0 (legal actions that were played)
        loss = -(tgt * log_probs).sum(dim=-1)
        loss = (loss * w).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    net.eval()
    return float(np.mean(losses))
