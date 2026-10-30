"""Графики: кривая обучения, карта выученной политики, траектория склада."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

# backend Agg позволяет сохранять графики без открытия окна
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import DAY_NAMES, InventoryConfig  # noqa: E402
from .training import History  # noqa: E402
from .utils import ensure_dir, moving_average  # noqa: E402

# DejaVu Sans поставляется вместе с matplotlib и содержит кириллицу
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def _save(fig: plt.Figure, path: str | Path) -> Path:
    """Единая точка сохранения: каталог, плотность, закрытие фигуры."""
    target = Path(path)
    ensure_dir(target.parent)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_reward_curve(history: History, path: str | Path, window: int = 100) -> Path:
    """Награда за эпизод, её скользящее среднее и кривая epsilon."""
    rewards = np.asarray(history.rewards, dtype=float)
    episodes = np.arange(1, rewards.size + 1)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(episodes, rewards, color="tab:blue", alpha=0.25, linewidth=0.8,
            label="награда за эпизод")

    smooth = moving_average(rewards, window)
    if smooth.size:
        left = rewards.size - smooth.size + 1
        ax.plot(np.arange(left, rewards.size + 1), smooth, color="tab:blue",
                linewidth=2.2, label=f"среднее за {window} эпизодов")

    ax.set_xlabel("эпизод")
    ax.set_ylabel("суммарная награда за эпизод")
    ax.set_title("Кривая обучения табличного Q-обучения")
    ax.grid(alpha=0.3)

    ax_eps = ax.twinx()
    ax_eps.plot(episodes, history.epsilons, color="tab:orange", linestyle="--",
                linewidth=1.2, label="epsilon (доля случайных действий)")
    ax_eps.set_ylabel("epsilon")
    ax_eps.set_ylim(0.0, 1.05)

    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax_eps.get_legend_handles_labels()
    ax.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    return _save(fig, path)


def plot_policy_heatmap(policy: Sequence[int], cfg: InventoryConfig, path: str | Path) -> Path:
    """Карта жадной политики: остаток × день недели, цвет — фактический заказ.

    Заказ не может превысить свободное место на складе, поэтому две разные
    «номинальные» величины заказа в одном состоянии часто эквивалентны.
    На графике показан именно фактически принимаемый заказ:
    ``min(order_options[a], max_stock - stock)``.
    """
    actions = np.asarray(policy, dtype=int).reshape(cfg.max_stock + 1, cfg.n_days)
    requested = np.asarray(cfg.order_options, dtype=float)[actions]
    free_space = (cfg.max_stock - np.arange(cfg.max_stock + 1)).reshape(-1, 1)
    grid = np.minimum(requested, free_space)

    fig, ax = plt.subplots(figsize=(7.5, 8.0))
    image = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower",
                      vmin=0.0, vmax=float(max(cfg.order_options)))
    colorbar = fig.colorbar(image, ax=ax, ticks=cfg.order_options)
    colorbar.ax.set_yticklabels([f"{quantity} шт." for quantity in cfg.order_options])
    colorbar.set_label("фактически принятый заказ")

    ax.set_xticks(range(cfg.n_days), DAY_NAMES)
    ax.set_yticks(range(0, cfg.max_stock + 1, 2))
    ax.set_xlabel("день недели")
    ax.set_ylabel("остаток на складе, шт.")
    ax.set_title("Выученная политика пополнения склада")
    return _save(fig, path)


def plot_episode_trajectory(
    log: Sequence[dict],
    cfg: InventoryConfig,
    path: str | Path,
    title: str = "Один эпизод обученного агента",
) -> Path:
    """Динамика остатка, заказов, спроса и недостачи за один эпизод."""
    days = np.arange(1, len(log) + 1)
    orders = np.array([row["ordered"] for row in log], dtype=float)
    demand = np.array([row["demand"] for row in log], dtype=float)
    unmet = np.array([row["unmet"] for row in log], dtype=float)
    # остаток на начало дня: стартовый запас, затем остатки предыдущих дней
    stocks_before = np.array([cfg.initial_stock] + [row["stock"] for row in log[:-1]], dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.step(days, stocks_before, where="post", color="tab:blue", linewidth=2.0,
            label="остаток на начало дня")
    ax.bar(days, orders, color="tab:green", alpha=0.45, label="принятый заказ")
    ax.set_xlabel("день эпизода")
    ax.set_ylabel("остаток / заказ, шт.")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    ax_demand = ax.twinx()
    ax_demand.plot(days, demand, color="tab:red", marker="o", markersize=3,
                   linewidth=1.2, label="спрос за день")
    ax_demand.bar(days, unmet, color="tab:red", alpha=0.25, label="недостача")
    ax_demand.set_ylabel("спрос / недостача, шт.")

    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax_demand.get_legend_handles_labels()
    ax.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        fontsize=9,
        frameon=False,
    )
    return _save(fig, path)
