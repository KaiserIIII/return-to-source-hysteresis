from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formal_experiment import Net, adapt


def test_adapt_honors_stage_passes_for_recurrent_trajectory() -> None:
    model = Net()
    source_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    batch = torch.zeros((64, 64), dtype=torch.float32)
    stages = [("A_1", batch), ("B", batch), ("A_2", batch)]

    result = adapt(
        model,
        source_state,
        stages,
        "source",
        {"eta": 0.01, "anchor_lambda": 0.05, "ema_decay": 0.99, "restore_probability": 0.1, "stage_passes": 2},
    )

    assert result["stage_passes"] == 2
    assert result["adaptation_steps"] == 6
    assert len(result["stages"]) == 6
