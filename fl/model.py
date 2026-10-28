"""Small CNN classifier plus shared train/eval and weight (de)serialization helpers."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """~47k-parameter CNN for 28x28 grayscale images (MNIST / Fashion-MNIST)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)   # 28 -> 24 -> pool -> 12
        self.conv2 = nn.Conv2d(16, 32, 5)  # 12 -> 8  -> pool -> 4
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_weights(model: nn.Module, wire_dtype: str = "float32") -> dict:
    """state_dict -> {name: numpy array} in the dtype used on the wire."""
    return {k: v.detach().cpu().numpy().astype(wire_dtype) for k, v in model.state_dict().items()}


def set_weights(model: nn.Module, weights: dict):
    model.load_state_dict({k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in weights.items()})


def to_wire(weights: dict, wire_dtype: str) -> dict:
    return {k: v.astype(wire_dtype) for k, v in weights.items()}


def fedavg(updates: list) -> dict:
    """Sample-weighted FedAvg. `updates` is a list of (weights, num_samples)."""
    total = float(sum(n for _, n in updates))
    keys = updates[0][0].keys()
    return {
        k: sum(w[k].astype(np.float64) * (n / total) for w, n in updates).astype(np.float32)
        for k in keys
    }


def train_epochs(model, x, y, epochs, lr, batch_size, momentum=0.9, generator=None) -> float:
    """Plain minibatch SGD over in-memory tensors. Returns mean training loss."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    model.train()
    total_loss, seen = 0.0, 0
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=generator)
        for i in range(0, len(x), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            seen += len(idx)
    return total_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, x, y, batch_size=2000):
    """Returns (accuracy, mean loss) on the given tensors."""
    model.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, len(x), batch_size):
        out = model(x[i:i + batch_size])
        loss_sum += F.cross_entropy(out, y[i:i + batch_size], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + batch_size]).sum().item()
    return correct / len(x), loss_sum / len(x)
