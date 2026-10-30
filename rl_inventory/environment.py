"""Среда «склад»: агент делает заказ, затем удовлетворяет случайный спрос.

Интерфейс намеренно повторяет классический gym-стиль ``reset``/``step``,
но без зависимости от gymnasium — нужен только numpy.
"""

from __future__ import annotations

import numpy as np

from .config import InventoryConfig


class InventoryEnv:
    """Дискретная среда управления запасами с конечным горизонтом.

    Порядок событий внутри одного шага (одного дня):

    1. склад принимает заказ — не больше свободного места, оплачивается
       фактически принятое количество;
    2. приходит случайный суточный спрос ``d ~ Poisson(demand_mean[день])``;
    3. продаётся ``min(остаток, d)`` единиц, разница ``d - продано`` — это
       упущенные продажи (недостача);
    4. начисляется награда ``R = price*продано - unit_cost*принято
       - fixed_order_cost*[принято>0] - holding_cost*остаток_после
       - stockout_penalty*недостача``;
    5. день недели сдвигается на следующий, эпизод завершается после
       ``horizon`` шагов (``terminated=True``).
    """

    def __init__(self, cfg: InventoryConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self._stock = cfg.initial_stock
        self._day = 0
        self._step_index = 0
        self._finished = True

    # ------------------------------------------------------------------ свойства
    @property
    def n_states(self) -> int:
        """Число состояний среды (строк Q-таблицы)."""
        return self.cfg.n_states

    @property
    def n_actions(self) -> int:
        """Число действий среды (столбцов Q-таблицы)."""
        return self.cfg.n_actions

    @property
    def stock(self) -> int:
        """Текущий остаток на складе."""
        return self._stock

    @property
    def day(self) -> int:
        """Текущий день недели (0 — понедельник)."""
        return self._day

    @property
    def step_index(self) -> int:
        """Номер текущего шага внутри эпизода (0..horizon)."""
        return self._step_index

    @property
    def finished(self) -> bool:
        """Завершён ли текущий эпизод."""
        return self._finished

    # ------------------------------------------------------------------ API среды
    def reset(self) -> int:
        """Начать новый эпизод и вернуть стартовое состояние."""
        self._stock = self.cfg.initial_stock
        self._day = 0
        self._step_index = 0
        self._finished = False
        return self.cfg.state_index(self._stock, self._day)

    def current_state(self) -> int:
        """Номер текущего состояния без изменения среды."""
        return self.cfg.state_index(self._stock, self._day)

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """Выполнить действие ``action`` и перейти к следующему дню.

        Возвращает кортеж ``(следующее_состояние, награда, terminated, info)``.
        """
        if self._finished:
            raise RuntimeError("эпизод завершён: сначала вызовите reset()")
        cfg = self.cfg
        if not 0 <= action < cfg.n_actions:
            raise ValueError(f"действие {action} вне диапазона 0..{cfg.n_actions - 1}")

        # 1. приём заказа: сверх свободного места склад положить не может
        ordered = cfg.order_options[action]
        accepted = min(ordered, cfg.max_stock - self._stock)
        self._stock += accepted

        # 2. случайный суточный спрос
        demand = int(self._rng.poisson(cfg.demand_mean[self._day]))

        # 3. продажи и недостача
        sold = min(self._stock, demand)
        unmet = demand - sold
        self._stock -= sold

        # 4. экономический результат дня
        revenue = cfg.price * sold
        purchase_cost = cfg.unit_cost * accepted
        if accepted > 0:
            purchase_cost += cfg.fixed_order_cost
        holding_cost = cfg.holding_cost * self._stock
        stockout_cost = cfg.stockout_penalty * unmet
        reward = revenue - purchase_cost - holding_cost - stockout_cost

        info = {
            "day": self._day,
            "ordered": accepted,
            "demand": demand,
            "sold": sold,
            "unmet": unmet,
            "stock": self._stock,
            "revenue": revenue,
            "purchase_cost": purchase_cost,
            "holding_cost": holding_cost,
            "stockout_cost": stockout_cost,
        }

        # 5. переход к следующему дню
        self._day = (self._day + 1) % cfg.n_days
        self._step_index += 1
        terminated = self._step_index >= cfg.horizon
        self._finished = terminated

        return cfg.state_index(self._stock, self._day), float(reward), terminated, info
