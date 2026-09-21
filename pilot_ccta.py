"""Pilot for commutator-canceling test-time adaptation on sklearn digits."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)


def shift(x: torch.Tensor, kind: str, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if kind == "A":
        noise = 0.10 * torch.randn(x.shape, generator=g)
        return torch.clamp(x + noise, 0, 1)
    noise = 0.20 * torch.randn(x.shape, generator=g)
    return torch.clamp(1.0 - x + noise, 0, 1)


def entropy_loss(model: Net, x: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(model(x), dim=1)
    return -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()


def accuracy(model: Net, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((model(x).argmax(1) == y).float().mean())


def apply(model: Net, delta, lr: float) -> None:
    with torch.no_grad():
        for p, d in zip(model.parameters(), delta):
            p.add_(d, alpha=lr)


def adapt(model: Net, batches: list[tuple[torch.Tensor, torch.Tensor]], method: str, eta: float) -> tuple[list[float], float]:
    params = tuple(model.parameters())
    prev_x = None
    k_norms: list[float] = []
    for x, _ in batches:
        t0 = time.perf_counter()
        cur = entropy_loss(model, x)
        g_cur = torch.autograd.grad(cur, params, create_graph=(method == "ccta_h"), retain_graph=(method == "ccta_h"))
        if method == "ccta_h" and prev_x is not None:
            prev = entropy_loss(model, prev_x)
            g_prev = torch.autograd.grad(prev, params, create_graph=True, retain_graph=True)
            hv_cur_prev = torch.autograd.grad(g_cur, params, grad_outputs=g_prev, retain_graph=True)
            hv_prev_cur = torch.autograd.grad(g_prev, params, grad_outputs=g_cur, retain_graph=True)
            k = tuple(a - b for a, b in zip(hv_cur_prev, hv_prev_cur))
            k_norms.append(float(math.sqrt(sum(float(v.detach().pow(2).sum()) for v in k))))
            delta = tuple(-eta * g - 0.5 * eta * eta * kk for g, kk in zip(g_cur, k))
        else:
            delta = tuple(-eta * g for g in g_cur)
            if method == "ccta_h":
                k_norms.append(0.0)
        apply(model, delta, 1.0)
        prev_x = x
        del cur, g_cur
        if method == "ccta_h":
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return k_norms, time.perf_counter() - t0


def main() -> None:
    data = load_digits()
    x = torch.tensor(data.data / 16.0, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.long)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.30, random_state=123, stratify=y)
    results = []
    for seed in [0, 1, 2, 3, 4]:
        torch.manual_seed(seed)
        model = Net()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(40):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(xtr), ytr)
            loss.backward()
            opt.step()
        a0 = accuracy(model, xte, yte)
        # Fixed batches make the cycle deterministic for each seed.
        A = shift(xte, "A", 1000 + seed)
        B = shift(xte, "B", 2000 + seed)
        order = [(A[i:i+64], yte[i:i+64]) for i in range(0, len(A), 64)] + [(B[i:i+64], yte[i:i+64]) for i in range(0, len(B), 64)] + [(A[i:i+64], yte[i:i+64]) for i in range(0, len(A), 64)]
        for method in ["tent", "ccta_h"]:
            m = Net(); m.load_state_dict(model.state_dict())
            norms, elapsed = adapt(m, order, method, eta=0.03)
            results.append({"seed": seed, "method": method, "source_accuracy": a0,
                            "A_accuracy_before": accuracy(model, A, yte),
                            "A_accuracy_after_cycle": accuracy(m, A, yte),
                            "B_accuracy_after_cycle": accuracy(m, B, yte),
                            "return_gap": accuracy(model, A, yte) - accuracy(m, A, yte),
                            "mean_commutator_norm": float(np.mean(norms)) if norms else 0.0,
                            "max_commutator_norm": float(np.max(norms)) if norms else 0.0,
                            "elapsed_seconds": elapsed})
    (OUT / "pilot_results.json").write_text(json.dumps({"protocol": {"dataset": "sklearn digits", "seeds": [0,1,2,3,4], "eta": 0.03, "cycle": "A-B-A", "methods": ["tent", "ccta_h"]}, "results": results}, indent=2), encoding="utf-8")
    for method in ["tent", "ccta_h"]:
        r = [x for x in results if x["method"] == method]
        print(method, "gap", np.mean([x["return_gap"] for x in r]), "A_after", np.mean([x["A_accuracy_after_cycle"] for x in r]), "K", np.mean([x["mean_commutator_norm"] for x in r]))


if __name__ == "__main__":
    main()
