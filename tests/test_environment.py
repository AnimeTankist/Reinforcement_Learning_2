"""Проверки среды склада: кодирование состояний, награда, детерминизм."""

from __future__ import annotations

import unittest

import numpy as np

from rl_inventory.config import InventoryConfig
from rl_inventory.environment import InventoryEnv


class InventoryEnvTest(unittest.TestCase):
    """Юнит-тесты среды «склад»."""

    def setUp(self) -> None:
        self.cfg = InventoryConfig()
        self.env = InventoryEnv(self.cfg, seed=42)

    def test_state_space_size(self) -> None:
        self.assertEqual(self.cfg.n_states, (self.cfg.max_stock + 1) * self.cfg.n_days)
        self.assertEqual(self.cfg.n_states, 147)
        self.assertEqual(self.cfg.n_actions, len(self.cfg.order_options))

    def test_state_index_roundtrip(self) -> None:
        for stock in range(self.cfg.max_stock + 1):
            for day in range(self.cfg.n_days):
                state = self.cfg.state_index(stock, day)
                self.assertEqual(self.cfg.split_state(state), (stock, day))
        with self.assertRaises(ValueError):
            self.cfg.state_index(self.cfg.max_stock + 1, 0)
        with self.assertRaises(ValueError):
            self.cfg.state_index(0, self.cfg.n_days)

    def test_reset_returns_initial_state(self) -> None:
        state = self.env.reset()
        self.assertEqual(state, self.cfg.state_index(self.cfg.initial_stock, 0))
        self.assertEqual(self.env.stock, self.cfg.initial_stock)
        self.assertEqual(self.env.day, 0)

    def test_episode_has_horizon_steps(self) -> None:
        self.env.reset()
        steps = 0
        done = False
        while not done:
            _, _, done, _ = self.env.step(0)
            steps += 1
        self.assertEqual(steps, self.cfg.horizon)
        self.assertTrue(self.env.finished)

    def test_step_after_termination_raises(self) -> None:
        self.env.reset()
        for _ in range(self.cfg.horizon):
            self.env.step(0)
        with self.assertRaises(RuntimeError):
            self.env.step(0)

    def test_invalid_action_raises(self) -> None:
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step(self.cfg.n_actions)
        with self.assertRaises(ValueError):
            self.env.step(-1)

    def test_reward_formula_without_demand(self) -> None:
        # при нулевом среднем спросе Poisson(0) = 0, поэтому награда детерминирована
        cfg = InventoryConfig(demand_mean=(0.0,) * 7)
        env = InventoryEnv(cfg, seed=1)
        env.reset()

        quantity = 10
        action = cfg.order_options.index(quantity)
        _, reward, _, info = env.step(action)

        accepted = min(quantity, cfg.max_stock - cfg.initial_stock)
        expected = -(
            cfg.unit_cost * accepted
            + cfg.fixed_order_cost
            + cfg.holding_cost * (cfg.initial_stock + accepted)
        )
        self.assertAlmostEqual(reward, expected, places=6)
        self.assertEqual(info["demand"], 0)
        self.assertEqual(info["sold"], 0)
        self.assertEqual(info["unmet"], 0)
        self.assertEqual(info["ordered"], accepted)

    def test_stock_stays_in_range(self) -> None:
        rng = np.random.default_rng(3)
        env = InventoryEnv(self.cfg, seed=3)
        for _ in range(5):
            env.reset()
            done = False
            while not done:
                action = int(rng.integers(self.cfg.n_actions))  # произвольный заказ
                _, _, done, info = env.step(action)
                self.assertGreaterEqual(info["stock"], 0)
                self.assertLessEqual(info["stock"], self.cfg.max_stock)
                self.assertLessEqual(info["sold"], info["demand"])
                self.assertGreaterEqual(info["ordered"], 0)

    def test_same_seed_same_trajectory(self) -> None:
        actions = [0, 1, 2, 3, 1] * 6  # ровно 30 шагов эпизода

        def run(seed: int) -> list[float]:
            env = InventoryEnv(self.cfg, seed=seed)
            env.reset()
            rewards = []
            for action in actions:
                _, reward, _, _ = env.step(action)
                rewards.append(reward)
            return rewards

        self.assertEqual(run(7), run(7))
        self.assertNotEqual(run(7), run(8))


if __name__ == "__main__":
    unittest.main()
