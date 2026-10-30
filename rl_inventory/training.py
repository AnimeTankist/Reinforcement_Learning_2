"""Циклы обучения и оценки агента, накопление статистики по эпизодам."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .agents import Agent
from .environment import InventoryEnv
from .utils import moving_average


@dataclass
class EpisodeStats:
    """Итоги одного эпизода — используются для анализа и графиков."""

    total_reward: float
    days: int
    stockout_days: int      # дни, когда спрос не был удовлетворён полностью
    unmet_units: int        # суммарная недостача за эпизод
    demand_units: int       # суммарный спрос за эпизод
    order_units: int        # суммарно принято на склад за эпизод
    stock_sum: int          # сумма остатков после каждого дня (для среднего)


@dataclass
class History:
    """История обучения: награда по эпизодам, epsilon, средняя |TD|."""

    rewards: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    td_errors: list[float] = field(default_factory=list)
    episodes: list[EpisodeStats] = field(default_factory=list)

    def moving_average(self, window: int) -> np.ndarray:
        """Скользящее среднее награды по окну ``window`` эпизодов."""
        return moving_average(self.rewards, window)

    def stockout_rate(self) -> float:
        """Доля дней с дефицитом по всем эпизодам обучения."""
        days = sum(ep.days for ep in self.episodes)
        if not days:
            return 0.0
        return sum(ep.stockout_days for ep in self.episodes) / days


def train(
    env: InventoryEnv,
    agent: Agent,
    episodes: int,
    log_every: int = 250,
    verbose: bool = True,
) -> History:
    """Обучить агента в среде ``env`` за ``episodes`` эпизодов.

    Эпизод заканчивается, когда среда возвращает ``terminated=True``
    (горизонт ``InventoryConfig.horizon``). После каждого эпизода
    уменьшается epsilon.
    """
    if episodes < 1:
        raise ValueError("episodes должен быть >= 1")
    if log_every < 1:
        raise ValueError("log_every должен быть >= 1")

    history = History()
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        stockout_days = unmet_units = demand_units = order_units = stock_sum = 0
        td_magnitudes: list[float] = []

        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            td_magnitudes.append(abs(agent.update(state, action, reward, next_state, done)))
            state = next_state

            total_reward += reward
            stockout_days += int(info["unmet"] > 0)
            unmet_units += info["unmet"]
            demand_units += info["demand"]
            order_units += info["ordered"]
            stock_sum += info["stock"]

        agent.decay_epsilon()
        history.rewards.append(total_reward)
        history.epsilons.append(float(getattr(agent, "epsilon", 0.0)))
        history.td_errors.append(float(np.mean(td_magnitudes)))
        history.episodes.append(
            EpisodeStats(
                total_reward=total_reward,
                days=len(td_magnitudes),
                stockout_days=stockout_days,
                unmet_units=unmet_units,
                demand_units=demand_units,
                order_units=order_units,
                stock_sum=stock_sum,
            )
        )

        if verbose and episode % log_every == 0:
            recent = np.mean(history.rewards[-log_every:])
            print(
                f"эпизод {episode:5d}/{episodes} | "
                f"средняя награда за {log_every} эп.: {recent:9.2f} | "
                f"epsilon: {history.epsilons[-1]:.3f}"
            )

    return history


def run_episode(env: InventoryEnv, policy: Callable[[int], int]) -> tuple[EpisodeStats, list[dict]]:
    """Пройти один эпизод заданной политикой и вернуть статистику и журнал дней."""
    state = env.reset()
    log: list[dict] = []
    total_reward = 0.0
    stockout_days = unmet_units = demand_units = order_units = stock_sum = 0

    done = False
    while not done:
        action = policy(state)
        state, reward, done, info = env.step(action)
        log.append(dict(info, action=action, reward=reward))

        total_reward += reward
        stockout_days += int(info["unmet"] > 0)
        unmet_units += info["unmet"]
        demand_units += info["demand"]
        order_units += info["ordered"]
        stock_sum += info["stock"]

    stats = EpisodeStats(
        total_reward=total_reward,
        days=len(log),
        stockout_days=stockout_days,
        unmet_units=unmet_units,
        demand_units=demand_units,
        order_units=order_units,
        stock_sum=stock_sum,
    )
    return stats, log


def evaluate(env: InventoryEnv, policy: Callable[[int], int], episodes: int) -> dict[str, float]:
    """Оценить политику на ``episodes`` эпизодах (обычно жадную, epsilon=0)."""
    if episodes < 1:
        raise ValueError("episodes должен быть >= 1")

    rewards: list[float] = []
    stockout_days = total_days = unmet_units = demand_units = 0
    stock_values: list[int] = []

    for _ in range(episodes):
        stats, log = run_episode(env, policy)
        rewards.append(stats.total_reward)
        stockout_days += stats.stockout_days
        unmet_units += stats.unmet_units
        demand_units += stats.demand_units
        total_days += stats.days
        stock_values.extend(row["stock"] for row in log)

    reward_array = np.asarray(rewards, dtype=float)
    return {
        "mean_reward": float(reward_array.mean()),
        "std_reward": float(reward_array.std()),
        "fill_rate": float(1.0 - unmet_units / demand_units) if demand_units else 1.0,
        "stockout_rate": float(stockout_days / total_days) if total_days else 0.0,
        "avg_stock": float(np.mean(stock_values)) if stock_values else 0.0,
        "mean_unmet": float(unmet_units / episodes),
    }
