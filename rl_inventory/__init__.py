"""Учебный пакет: управление запасами на складе с помощью Q-обучения.

Публичный API пакета:

* ``InventoryConfig`` / ``QLearningConfig`` / ``TrainConfig`` — параметры;
* ``InventoryEnv`` — среда склада (``reset``/``step``);
* ``TabularQLearning`` / ``RandomAgent`` — агенты;
* ``train`` / ``evaluate`` / ``run_episode`` — циклы обучения и оценки.
"""

from .agents import Agent, RandomAgent, TabularQLearning
from .config import DAY_NAMES, InventoryConfig, QLearningConfig, TrainConfig
from .environment import InventoryEnv
from .training import EpisodeStats, History, evaluate, run_episode, train

__all__ = [
    "Agent",
    "DAY_NAMES",
    "EpisodeStats",
    "History",
    "InventoryConfig",
    "InventoryEnv",
    "QLearningConfig",
    "RandomAgent",
    "TabularQLearning",
    "TrainConfig",
    "evaluate",
    "run_episode",
    "train",
]
