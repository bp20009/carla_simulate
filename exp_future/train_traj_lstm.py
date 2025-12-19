#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TrajLSTM(nn.Module):
    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 1,
        horizon_steps: int = 50,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, feature_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.lstm(history)
        step_input = history[:, -1:, :]
        preds = []
        for _ in range(self.horizon_steps):
            step_out, hidden = self.lstm(step_input, hidden)
            dxy = self.decoder(step_out[:, -1, :])
            preds.append(dxy)
            step_input = dxy.unsqueeze(1)
        return torch.stack(preds, dim=1)


def load_xy_by_id(csv_path: Path) -> Dict[int, List[Tuple[int, float, float]]]:
    by_id: Dict[int, List[Tuple[int, float, float]]] = {}
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            vid = int(row["id"])
            frame = int(row["frame"])
            x = float(row["location_x"])
            y = float(row["location_y"])
            by_id.setdefault(vid, []).append((frame, x, y))
    for vid in list(by_id.keys()):
        by_id[vid].sort(key=lambda item: item[0])
    return by_id


def make_samples_delta(
    by_id: Dict[int, List[Tuple[int, float, float]]],
    history_steps: int,
    horizon_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []

    need = history_steps + horizon_steps + 1

    for seq in by_id.values():
        if len(seq) < need:
            continue

        xs = [point[1] for point in seq]
        ys = [point[2] for point in seq]

        dx = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        dy = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]

        for t in range(history_steps, len(dx) - horizon_steps + 1):
            hist = list(zip(dx[t - history_steps : t], dy[t - history_steps : t]))
            fut = list(zip(dx[t : t + horizon_steps], dy[t : t + horizon_steps]))
            x_list.append(torch.tensor(hist, dtype=torch.float32))
            y_list.append(torch.tensor(fut, dtype=torch.float32))

    x_tensor = torch.stack(x_list)
    y_tensor = torch.stack(y_list)
    return x_tensor, y_tensor


def train(
    csv_path: str,
    out_path: str = "traj_lstm.pt",
    history_steps: int = 10,
    horizon_steps: int = 50,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
) -> None:
    by_id = load_xy_by_id(Path(csv_path))
    x_tensor, y_tensor = make_samples_delta(by_id, history_steps, horizon_steps)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TrajLSTM(horizon_steps=horizon_steps)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item()
            count += 1
        avg = total / max(count, 1)
        print(f"epoch {epoch:02d}: loss={avg:.6f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "history_steps": history_steps,
            "horizon_steps": horizon_steps,
            "mode": "delta_xy",
        },
        out_path,
    )
    print(f"saved: {out_path}  samples={len(dataset)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="traj_lstm.pt")
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.csv, args.out, args.history, args.horizon, args.batch, args.epochs)


if __name__ == "__main__":
    main()
