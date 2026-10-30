"""Точка входа: обучение и оценка агента управления запасами на складе.

Примеры запуска (из каталога проекта):
    python main.py all --episodes 3000 --seed 0
    python main.py train --episodes 6000 --alpha 0.15
    python main.py evaluate
    python main.py demo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from rl_inventory.agents import RandomAgent, TabularQLearning
from rl_inventory.config import InventoryConfig, QLearningConfig, TrainConfig
from rl_inventory.environment import InventoryEnv
from rl_inventory.plots import (
    plot_episode_trajectory,
    plot_policy_heatmap,
    plot_reward_curve,
)
from rl_inventory.training import EpisodeStats, evaluate, run_episode, train
from rl_inventory.utils import ensure_dir, format_metrics, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Q-обучение для прикладной задачи «управление запасами на складе»"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "train", "evaluate", "demo"),
        help="all — обучение + оценка + демонстрация (по умолчанию)",
    )
    parser.add_argument("--episodes", type=int, default=TrainConfig.episodes,
                        help="число эпизодов обучения")
    parser.add_argument("--alpha", type=float, default=QLearningConfig.alpha,
                        help="скорость обучения alpha")
    parser.add_argument("--gamma", type=float, default=QLearningConfig.gamma,
                        help="коэффициент дисконтирования gamma")
    parser.add_argument("--eval-episodes", type=int, default=TrainConfig.eval_episodes,
                        help="число эпизодов оценки")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed,
                        help="seed обучения (влияет на исследование и спрос)")
    parser.add_argument("--window", type=int, default=TrainConfig.window,
                        help="окно скользящего среднего награды")
    parser.add_argument("--output-dir", default=TrainConfig.output_dir,
                        help="каталог для графиков и метрик")
    return parser.parse_args(argv)


def load_agent(
    inventory: InventoryConfig,
    q_cfg: QLearningConfig,
    train_cfg: TrainConfig,
) -> TabularQLearning:
    """Загрузить обученную Q-таблицу из каталога результатов."""
    path = Path(train_cfg.output_dir) / "q_table.npz"
    try:
        return TabularQLearning.load(
            path, q_cfg, inventory.n_states, inventory.n_actions, seed=train_cfg.eval_seed
        )
    except FileNotFoundError as error:
        raise SystemExit(f"{error}\nСначала обучите агента: python main.py train") from error


def run_training(
    inventory: InventoryConfig,
    q_cfg: QLearningConfig,
    train_cfg: TrainConfig,
) -> TabularQLearning:
    """Обучить агента и сохранить Q-таблицу, графики и метрики обучения."""
    print("Обучение агента табличным Q-обучением...")
    env = InventoryEnv(inventory, seed=train_cfg.seed)
    agent = TabularQLearning(q_cfg, env.n_states, env.n_actions, seed=train_cfg.seed)

    started = time.perf_counter()
    history = train(env, agent, train_cfg.episodes, log_every=train_cfg.log_every)
    elapsed = time.perf_counter() - started

    out_dir = ensure_dir(train_cfg.output_dir)
    plot_reward_curve(history, out_dir / "reward_curve.png", window=train_cfg.window)
    plot_policy_heatmap(agent.greedy_policy(), inventory, out_dir / "policy_heatmap.png")
    agent.save(out_dir / "q_table.npz")

    window = min(train_cfg.window, len(history.rewards))
    metrics = {
        "episodes": train_cfg.episodes,
        "mean_reward_first_window": float(np.mean(history.rewards[:window])),
        "mean_reward_last_window": float(np.mean(history.rewards[-window:])),
        "final_epsilon": float(history.epsilons[-1]),
        "train_stockout_rate": float(history.stockout_rate()),
        "training_seconds": round(elapsed, 2),
    }
    save_json(metrics, out_dir / "metrics_training.json")

    print(f"\nОбучение завершено за {elapsed:.1f} с")
    print(f"средняя награда, первые {window} эпизодов  : {metrics['mean_reward_first_window']:9.2f}")
    print(f"средняя награда, последние {window} эпизодов: {metrics['mean_reward_last_window']:9.2f}")
    print(f"итоговый epsilon: {metrics['final_epsilon']:.3f}")
    print(f"результаты сохранены в '{out_dir}'")
    return agent


def print_episode_log(log: list[dict]) -> None:
    """Напечатать журнал одного эпизода в виде таблицы."""
    header = "день | заказ | спрос | продано | недостача | остаток"
    print(header)
    print("-" * len(header))
    for day, row in enumerate(log, start=1):
        print(
            f"{day:4d} | {row['ordered']:5d} | {row['demand']:5d} | "
            f"{row['sold']:7d} | {row['unmet']:9d} | {row['stock']:7d}"
        )


def episode_summary(stats: EpisodeStats) -> str:
    """Короткая строка с итогами эпизода."""
    sold = stats.demand_units - stats.unmet_units
    return (
        f"награда {stats.total_reward:.2f} | продано {sold} из {stats.demand_units} шт. | "
        f"дней с дефицитом {stats.stockout_days} из {stats.days}"
    )


def run_evaluation(
    inventory: InventoryConfig,
    q_cfg: QLearningConfig,
    train_cfg: TrainConfig,
    agent: TabularQLearning | None = None,
) -> dict[str, float]:
    """Сравнить обученную жадную политику со случайной на одинаковом потоке спроса."""
    policy_agent: TabularQLearning = agent if agent is not None else load_agent(
        inventory, q_cfg, train_cfg
    )

    learned_env = InventoryEnv(inventory, seed=train_cfg.eval_seed)
    learned = evaluate(
        learned_env,
        lambda state: policy_agent.act(state, greedy=True),
        train_cfg.eval_episodes,
    )

    random_agent = RandomAgent(inventory.n_actions, seed=train_cfg.eval_seed)
    random_env = InventoryEnv(inventory, seed=train_cfg.eval_seed)  # тот же поток спроса
    baseline = evaluate(random_env, lambda state: random_agent.act(state), train_cfg.eval_episodes)

    print()
    print(format_metrics(f"Обученная политика ({train_cfg.eval_episodes} эпизодов)", learned))
    print()
    print(format_metrics("Случайная политика — базовая линия", baseline))

    out_dir = ensure_dir(train_cfg.output_dir)
    save_json(
        {"q_learning": learned, "random": baseline, "eval_episodes": train_cfg.eval_episodes},
        out_dir / "metrics_evaluation.json",
    )
    return learned


def run_demo(
    inventory: InventoryConfig,
    q_cfg: QLearningConfig,
    train_cfg: TrainConfig,
    agent: TabularQLearning | None = None,
) -> None:
    """Показать один жадный эпизод: таблица по дням и график траектории склада."""
    policy_agent: TabularQLearning = agent if agent is not None else load_agent(
        inventory, q_cfg, train_cfg
    )

    env = InventoryEnv(inventory, seed=train_cfg.eval_seed + 1)
    stats, log = run_episode(env, lambda state: policy_agent.act(state, greedy=True))

    print()
    print("Один эпизод обученного агента:")
    print_episode_log(log)
    print(f"\nИтог эпизода: {episode_summary(stats)}")

    out_dir = ensure_dir(train_cfg.output_dir)
    path = plot_episode_trajectory(log, inventory, out_dir / "inventory_trajectory.png")
    print(f"траектория эпизода сохранена: {path}")


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI: train / evaluate / demo / all."""
    args = parse_args(argv)
    inventory = InventoryConfig()
    q_cfg = QLearningConfig(alpha=args.alpha, gamma=args.gamma)
    train_cfg = TrainConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        window=args.window,
        output_dir=args.output_dir,
    )

    print(
        f"Состояний: {inventory.n_states}, действий: {inventory.n_actions} "
        f"({inventory.order_options}), горизонт: {inventory.horizon} дн."
    )
    print(f"alpha={q_cfg.alpha}, gamma={q_cfg.gamma}, эпизодов={train_cfg.episodes}, seed={train_cfg.seed}")

    if args.command == "evaluate":
        run_evaluation(inventory, q_cfg, train_cfg)
    elif args.command == "demo":
        run_demo(inventory, q_cfg, train_cfg)
    elif args.command == "train":
        run_training(inventory, q_cfg, train_cfg)
    else:
        agent = run_training(inventory, q_cfg, train_cfg)
        run_evaluation(inventory, q_cfg, train_cfg, agent=agent)
        run_demo(inventory, q_cfg, train_cfg, agent=agent)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
