"""Конфигурация прикладной задачи «управление запасами на складе».

Все параметры собраны в неизменяемые dataclass-и, чтобы эксперимент был
воспроизводим: модель среды (``InventoryConfig``), гиперпараметры агента
(``QLearningConfig``) и настройки запуска (``TrainConfig``).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Названия дней недели в порядке индексов (день 0 — понедельник).
DAY_NAMES: tuple[str, ...] = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")


@dataclass(frozen=True)
class InventoryConfig:
    """Параметры модели склада.

    Состояние агента — пара (остаток на складе, день недели).
    Действие агента — размер заказа из ``order_options``.

    Экономика шага (одного дня):
    ``R = price*продано - unit_cost*принято - fixed_order_cost*[принято>0]
    - holding_cost*остаток_после - stockout_penalty*недостача``.
    """

    # --- пространство состояний и действий -------------------------------
    max_stock: int = 20
    start_stock: int | None = None  # None -> max_stock // 2
    order_options: tuple[int, ...] = (0, 2, 5, 10)

    # --- горизонт эпизода ------------------------------------------------
    horizon: int = 30

    # --- стохастика спроса: средний суточный спрос по дням недели --------
    demand_mean: tuple[float, ...] = (3.0, 3.0, 3.0, 3.0, 4.0, 6.0, 6.0)

    # --- экономика -------------------------------------------------------
    price: float = 10.0            # цена продажи единицы товара
    unit_cost: float = 6.0         # закупочная цена единицы
    fixed_order_cost: float = 2.0  # фиксированные издержки оформления заказа
    holding_cost: float = 0.4      # хранение единицы остатка в сутки
    stockout_penalty: float = 6.0  # штраф за единицу неудовлетворённого спроса

    def __post_init__(self) -> None:
        if self.max_stock < 1:
            raise ValueError("max_stock должен быть >= 1")
        if not self.order_options:
            raise ValueError("order_options не может быть пустым")
        if self.order_options[0] != 0:
            raise ValueError("должен быть доступен нулевой заказ: order_options[0] == 0")
        if any(q < 0 for q in self.order_options):
            raise ValueError("размеры заказа не могут быть отрицательными")
        if len(set(self.order_options)) != len(self.order_options):
            raise ValueError("размеры заказа не должны повторяться")
        if self.horizon < 1:
            raise ValueError("horizon должен быть >= 1")
        if len(self.demand_mean) != len(DAY_NAMES):
            raise ValueError(
                f"demand_mean должен содержать {len(DAY_NAMES)} значений — по одному на день недели"
            )
        if any(m < 0 for m in self.demand_mean):
            raise ValueError("средний спрос не может быть отрицательным")
        costs = (
            self.price,
            self.unit_cost,
            self.fixed_order_cost,
            self.holding_cost,
            self.stockout_penalty,
        )
        if min(costs) < 0:
            raise ValueError("стоимостные параметры не могут быть отрицательными")
        if self.start_stock is not None and not 0 <= self.start_stock <= self.max_stock:
            raise ValueError("start_stock должен попадать в диапазон 0..max_stock")

    # --- производные величины -------------------------------------------
    @property
    def n_days(self) -> int:
        """Число дней недели (вторая размерность состояния)."""
        return len(self.demand_mean)

    @property
    def n_states(self) -> int:
        """Общее число дискретных состояний."""
        return (self.max_stock + 1) * self.n_days

    @property
    def n_actions(self) -> int:
        """Число доступных действий (вариантов заказа)."""
        return len(self.order_options)

    @property
    def initial_stock(self) -> int:
        """Остаток на складе в начале эпизода."""
        if self.start_stock is None:
            return self.max_stock // 2
        return self.start_stock

    # --- кодирование состояний ------------------------------------------
    def state_index(self, stock: int, day: int) -> int:
        """Перевод пары (остаток, день) в номер строки Q-таблицы."""
        if not 0 <= stock <= self.max_stock:
            raise ValueError(f"остаток {stock} вне диапазона 0..{self.max_stock}")
        if not 0 <= day < self.n_days:
            raise ValueError(f"день {day} вне диапазона 0..{self.n_days - 1}")
        return stock * self.n_days + day

    def split_state(self, state: int) -> tuple[int, int]:
        """Обратное преобразование номера состояния в (остаток, день)."""
        if not 0 <= state < self.n_states:
            raise ValueError(f"состояние {state} вне диапазона 0..{self.n_states - 1}")
        return divmod(state, self.n_days)


@dataclass(frozen=True)
class QLearningConfig:
    """Гиперпараметры табличного Q-обучения."""

    alpha: float = 0.1            # скорость обучения
    gamma: float = 0.95           # коэффициент дисконтирования
    epsilon_start: float = 1.0    # начальная доля случайных действий
    epsilon_min: float = 0.05     # нижняя граница epsilon
    epsilon_decay: float = 0.995  # множитель затухания за эпизод

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha должна попадать в (0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma должна попадать в [0, 1]")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start:
            raise ValueError("нужно 0 <= epsilon_min <= epsilon_start")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay должна попадать в (0, 1]")


@dataclass(frozen=True)
class TrainConfig:
    """Настройки эксперимента: обучение, оценка, сохранение результатов."""

    episodes: int = 10000
    log_every: int = 250
    eval_episodes: int = 200
    seed: int = 0
    eval_seed: int = 10_000
    window: int = 100
    output_dir: str = "outputs"

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes должен быть >= 1")
        if self.log_every < 1:
            raise ValueError("log_every должен быть >= 1")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes должен быть >= 1")
        if self.window < 1:
            raise ValueError("window должен быть >= 1")
        if not self.output_dir:
            raise ValueError("output_dir не может быть пустым")
