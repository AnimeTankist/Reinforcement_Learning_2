"""Агенты: табличный Q-обучение и случайная стратегия (базовая линия)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .config import QLearningConfig
from .utils import ensure_dir


class Agent(Protocol):
    """Минимальный контракт агента, который понимает ``training.train``."""

    def act(self, state: int, greedy: bool = False) -> int:
        """Выбрать действие в состоянии ``state``."""

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Учесть переход; вернуть TD-ошибку (или 0, если агент не обучается)."""

    def decay_epsilon(self) -> None:
        """Уменьшить долю случайных действий после эпизода."""


class RandomAgent:
    """Случайная стратегия — нужна как базовая линия для сравнения."""

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        if n_actions < 1:
            raise ValueError("n_actions должен быть >= 1")
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)

    def act(self, state: int, greedy: bool = False) -> int:
        """Равновероятный выбор действия: ``greedy`` игнорируется."""
        return int(self._rng.integers(self.n_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Случайная стратегия не обучается."""
        return 0.0

    def decay_epsilon(self) -> None:
        """Нечего уменьшать — метод нужен только для единого интерфейса."""


class TabularQLearning:
    """Табличный Q-learning (off-policy TD(0)) с epsilon-greedy исследованием.

    Обновление после перехода ``(s, a, r, s')``:

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

    для завершающего перехода бутстрэп не выполняется: целевое значение равно ``r``.
    """

    def __init__(
        self,
        cfg: QLearningConfig,
        n_states: int,
        n_actions: int,
        seed: int | None = None,
    ) -> None:
        if n_states < 1 or n_actions < 1:
            raise ValueError("n_states и n_actions должны быть >= 1")
        self.cfg = cfg
        self.n_states = n_states
        self.n_actions = n_actions
        # Q-таблица: строки — состояния, столбцы — действия
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        self.epsilon = cfg.epsilon_start
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ политика
    def act(self, state: int, greedy: bool = False) -> int:
        """Выбрать действие: с вероятностью epsilon — случайное, иначе жадное."""
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        row = self.q_table[state]
        best_actions = np.flatnonzero(row == row.max())  # все действия с максимумом Q
        return int(self._rng.choice(best_actions))       # равные максимумы — случайно

    def greedy_policy(self) -> np.ndarray:
        """Жадная политика в виде массива индексов действий длиной ``n_states``."""
        return np.argmax(self.q_table, axis=1)

    def action_values(self) -> np.ndarray:
        """Копия Q-таблицы для анализа и графиков."""
        return self.q_table.copy()

    # ------------------------------------------------------------------ обучение
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Одно обновление Q-обучения; возвращает TD-ошибку до обновления."""
        target = float(reward)
        if not done:
            target += self.cfg.gamma * float(np.max(self.q_table[next_state]))
        td_error = target - float(self.q_table[state, action])
        self.q_table[state, action] += self.cfg.alpha * td_error
        return td_error

    def decay_epsilon(self) -> None:
        """Уменьшить epsilon после эпизода, не опускаясь ниже ``epsilon_min``."""
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    # ------------------------------------------------------------------ сериализация
    def save(self, path: str | Path) -> Path:
        """Сохранить Q-таблицу и epsilon в ``.npz``-файл."""
        target = Path(path)
        ensure_dir(target.parent)
        with target.open("wb") as handle:
            np.savez(handle, q_table=self.q_table, epsilon=np.array(self.epsilon))
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        cfg: QLearningConfig,
        n_states: int,
        n_actions: int,
        seed: int | None = None,
    ) -> "TabularQLearning":
        """Загрузить агента из ``.npz``-файла, созданного методом ``save``."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"файл с Q-таблицей не найден: {target}")
        with target.open("rb") as handle:
            data = np.load(handle)
            q_table = np.asarray(data["q_table"], dtype=np.float64)
            epsilon = float(data["epsilon"]) if "epsilon" in data else cfg.epsilon_min
        if q_table.shape != (n_states, n_actions):
            raise ValueError(
                f"форма Q-таблицы {q_table.shape} не совпадает с ожидаемой {(n_states, n_actions)}"
            )
        agent = cls(cfg, n_states, n_actions, seed=seed)
        agent.q_table = q_table.copy()
        agent.epsilon = epsilon
        return agent
