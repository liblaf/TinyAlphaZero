import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend(newbackend="agg")


def plot_loss(
    begin_time: datetime,
    loss: list[float],
    time: list[datetime],
    updated: list[bool],
    output: Path = Path.cwd() / "output" / "loss.png",
) -> None:
    x: list[float] = []
    y: list[float] = []
    for l, t in zip(loss, time):
        dt: float = (t - begin_time).total_seconds() / 3600
        x.append(dt)
        y.append(l)
    x_numpy: np.ndarray = np.array(x)
    y_numpy: np.ndarray = np.array(y)
    updated_numpy: np.ndarray = np.array(updated)
    plt.figure(dpi=600)
    plt.plot(x_numpy, y_numpy)
    plt.scatter(x=x_numpy[updated_numpy], y=y_numpy[updated_numpy], label="Update")
    plt.title("Loss")
    plt.xlabel("Time (hour)")
    plt.ylabel("Loss")
    plt.tight_layout()
    os.makedirs(output.parent, exist_ok=True)
    plt.savefig(output)
    plt.close()


def plot_update_frequency(
    begin_time: datetime,
    time: list[datetime],
    updated: list[bool],
    output: Path = Path.cwd() / "output" / "update_frequency.png",
) -> None:
    x: list[float] = []
    y: list[float] = []
    last_update_time: datetime = begin_time
    for t, u in zip(time, updated):
        if not u:
            continue
        x.append((t - begin_time).total_seconds() / 3600)
        y.append(3600 / (t - last_update_time).total_seconds())
        last_update_time = t
    plt.figure(dpi=600)
    plt.plot(x, y)
    plt.title("Update Frequency")
    plt.xlabel("Time (hour)")
    plt.ylabel("Update Frequency (per hour)")
    plt.tight_layout()
    os.makedirs(output.parent, exist_ok=True)
    plt.savefig(output)
    plt.close()


def plot_win_rate(
    begin_time: datetime,
    match_results: list[tuple[int, int, int]],
    time: list[datetime],
    updated: list[bool],
    output: Path = Path.cwd() / "output" / "win-rate.png",
) -> None:
    x: list[float] = []
    win_rate: list[float] = []
    undefeated_rate: list[float] = []
    for (win, draw, lose), t, u in zip(match_results, time, updated):
        if not u:
            continue
        dt: float = (t - begin_time).total_seconds() / 3600
        x.append(dt)
        win_rate.append(win / (win + draw + lose))
        undefeated_rate.append((win + draw) / (win + draw + lose))
    plt.figure(dpi=600)
    plt.plot(x, undefeated_rate, label="Undefeated", marker="x")
    plt.plot(x, win_rate, label="Win", marker="x")
    plt.legend(loc="best")
    plt.title("Win Rate Against Random")
    plt.xlabel("Time (hour)")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()
    os.makedirs(output.parent, exist_ok=True)
    plt.savefig(output)
    plt.close()
