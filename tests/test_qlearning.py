"""Проверки табличного Q-обучения: формула обновления, политика, сериализация."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_inventory.agents import RandomAgent, TabularQLearning
from rl_inventory.config import InventoryConfig, QLearningConfig
from rl_inventory.environment import InventoryEnv
from rl_inventory.training import train


class TabularQLearningTest(unittest.TestCase):
    """Юнит-тесты агента Q-обучения."""

    def setUp(self) -> None:
        self.cfg = QLearningConfig(alpha=0.5, gamma=0.9, epsilon_min=0.1, epsilon_decay=0.5)
        self.agent = TabularQLearning(self.cfg, n_states=3, n_actions=2, seed=0)

    def test_update_without_bootstrap(self) -> None:
        td_error = self.agent.update(0, 1, 2.0, 2, True)
        self.assertAlmostEqual(td_error, 2.0)
        self.assertAlmostEqual(self.agent.q_table[0, 1], self.cfg.alpha * 2.0)

    def test_update_bootstraps_from_next_state(self) -> None:
        self.agent.q_table[2, 0] = 10.0
        td_error = self.agent.update(0, 0, 1.0, 2, False)
        expected_target = 1.0 + self.cfg.gamma * 10.0
        self.assertAlmostEqual(td_error, expected_target)
        self.assertAlmostEqual(self.agent.q_table[0, 0], self.cfg.alpha * expected_target)

    def test_epsilon_decays_to_min(self) -> None:
        for _ in range(10):
            self.agent.decay_epsilon()
        self.assertAlmostEqual(self.agent.epsilon, self.cfg.epsilon_min)

    def test_greedy_policy_uses_argmax(self) -> None:
        self.agent.q_table[0] = [1.0, 5.0]
        self.agent.q_table[1] = [3.0, 2.0]
        self.assertEqual(self.agent.greedy_policy().tolist(), [1, 0, 0])

    def test_learning_finds_best_action(self) -> None:
        cfg = QLearningConfig(alpha=0.2, gamma=0.0, epsilon_min=0.0, epsilon_decay=0.9)
        agent = TabularQLearning(cfg, n_states=2, n_actions=3, seed=1)
        # эпизод из одного шага: вознаграждение 1.0 получает только действие 2
        for _ in range(300):
            action = agent.act(0)
            reward = 1.0 if action == 2 else 0.0
            agent.update(0, action, reward, 0, True)
            agent.decay_epsilon()
        self.assertEqual(int(np.argmax(agent.q_table[0])), 2)

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "q_table.npz"
            self.agent.q_table[1, 0] = 3.5
            saved = self.agent.save(path)
            self.assertTrue(saved.exists())

            loaded = TabularQLearning.load(path, self.cfg, n_states=3, n_actions=2)
            np.testing.assert_allclose(loaded.q_table, self.agent.q_table)
            self.assertAlmostEqual(loaded.epsilon, self.agent.epsilon)

    def test_load_rejects_wrong_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "q_table.npz"
            self.agent.save(path)
            with self.assertRaises(ValueError):
                TabularQLearning.load(path, self.cfg, n_states=4, n_actions=2)


class TrainingSmokeTest(unittest.TestCase):
    """Короткий прогон обучения: проверяем накопление статистики."""

    def test_train_fills_history(self) -> None:
        env = InventoryEnv(InventoryConfig(horizon=10), seed=0)
        agent = TabularQLearning(QLearningConfig(), env.n_states, env.n_actions, seed=0)
        history = train(env, agent, episodes=20, log_every=10, verbose=False)

        self.assertEqual(len(history.rewards), 20)
        self.assertEqual(len(history.epsilons), 20)
        self.assertTrue(all(stats.days == 10 for stats in history.episodes))
        self.assertLess(history.epsilons[-1], QLearningConfig().epsilon_start)

    def test_random_agent_returns_valid_actions(self) -> None:
        agent = RandomAgent(n_actions=4, seed=0)
        actions = {agent.act(0) for _ in range(200)}
        self.assertTrue(actions.issubset({0, 1, 2, 3}))


if __name__ == "__main__":
    unittest.main()
