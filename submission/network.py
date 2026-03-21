"""
Deep CFR networks and replay buffers.

BettingNet: infoset -> advantage over 5 betting actions

Each has a corresponding value network (V) trained per CFR iteration
and a strategy network (Pi) trained at the end.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import INPUT_DIM, N_BETTING_ACTIONS, N_DISCARD_ACTIONS


# ── network ──────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc   = nn.Linear(dim, dim)
        self.act  = nn.PReLU()

    def forward(self, x):
        return self.act(self.fc(self.norm(x)) + x)

class CFRNet(nn.Module):
    def __init__(self, output_dim, hidden=256):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.PReLU(),
        )
        self.res1 = ResBlock(hidden)
        self.res2 = ResBlock(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.PReLU(),
            nn.Linear(hidden // 2, output_dim),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x)


def make_betting_net(hidden=128):
    return CFRNet(output_dim=N_BETTING_ACTIONS, hidden=hidden)

# ── strategy from network ─────────────────────────────────────────────────────

def get_strategy(net, infoset_vec, legal_mask, device='cpu'):
    """
    Run regret matching on network output to get a strategy.
    Used during training (value/advantage nets).
    """
    with torch.no_grad():
        x = torch.tensor(infoset_vec, dtype=torch.float32,
                         device=device).unsqueeze(0)
        advantages = net(x).squeeze(0).cpu().numpy()

    advantages = advantages * legal_mask
    pos = np.maximum(advantages, 0.0)
    total = pos.sum()
    if total > 0:
        return pos / total
    # uniform over legal actions
    return legal_mask / legal_mask.sum()


def get_policy_distribution(net, infoset_vec, legal_mask, device='cpu'):
    """
    Masked softmax on strategy network output.
    Used at runtime (trained strategy nets).
    """
    with torch.no_grad():
        x = torch.tensor(infoset_vec, dtype=torch.float32,
                         device=device).unsqueeze(0)
        logits = net(x).squeeze(0).cpu().numpy()

    masked_logits = np.where(legal_mask > 0, logits, -1e9)
    max_logit = masked_logits[legal_mask > 0].max()
    probs = np.where(legal_mask > 0, np.exp(masked_logits - max_logit), 0.0)
    total = probs.sum()
    if total > 0:
        return probs / total
    return legal_mask / legal_mask.sum()


# ── replay buffers ────────────────────────────────────────────────────────────

class ReservoirBuffer:
    """
    Fixed-capacity buffer with reservoir sampling.
    Stores (infoset_vec, advantages, mask) tuples.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer   = []
        self.n_seen   = 0

    def add(self, infoset_vec, advantages, mask):
        self.n_seen += 1
        item = (infoset_vec.copy(), advantages.copy(), mask.copy())
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            idx = np.random.randint(0, self.n_seen)
            if idx < self.capacity:
                self.buffer[idx] = item

    def sample(self, batch_size):
        n = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=n, replace=False)
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
    Trains the final policy network Pi.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer   = []
        self.n_seen   = 0

    def add(self, infoset_vec, strategy, iteration):
        self.n_seen += 1
        item = (infoset_vec.copy(), strategy.copy(), float(iteration))
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            idx = np.random.randint(0, self.n_seen)
            if idx < self.capacity:
                self.buffer[idx] = item

    def sample(self, batch_size):
        n = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=n, replace=False)
        batch   = [self.buffer[i] for i in indices]
        vecs    = np.stack([b[0] for b in batch])
        strats  = np.stack([b[1] for b in batch])
        weights = np.array([b[2] for b in batch], dtype=np.float32)
        return vecs, strats, weights

    def __len__(self):
        return len(self.buffer)


# ── training steps ────────────────────────────────────────────────────────────

def train_value_network(net, buffer, optimizer,
                        batch_size=512, n_steps=200, device='cpu'):
    """MSE loss on regret samples, masked to legal actions."""
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
        loss = ((pred - tgt) ** 2 * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    net.eval()
    return float(np.mean(losses))


def train_strategy_network(net, buffer, optimizer,
                           batch_size=512, n_steps=400, device='cpu'):
    """Weighted cross-entropy loss on strategy samples."""
    if len(buffer) < batch_size:
        return None

    net.train()
    losses = []
    for _ in range(n_steps):
        vecs, strats, weights = buffer.sample(batch_size)
        x   = torch.tensor(vecs,    dtype=torch.float32, device=device)
        tgt = torch.tensor(strats,  dtype=torch.float32, device=device)
        w   = torch.tensor(weights, dtype=torch.float32, device=device)
        w   = w / w.sum()

        logits    = net(x)
        log_probs = F.log_softmax(logits, dim=-1)
        loss      = -(tgt * log_probs).sum(dim=-1)
        loss      = (loss * w).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    net.eval()
    return float(np.mean(losses))
