"""Вспомогательные функции: работа с путями, экспорт метрик, сглаживание."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Создать каталог (вместе с родительскими) и вернуть его как ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """Скользящее среднее по окну ``window`` (длина результата ``n - window + 1``)."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    window = max(1, min(int(window), array.size))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(array, kernel, mode="valid")


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Сохранить словарь в JSON-файл (кириллица — как есть, с отступами)."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return target


def format_metrics(title: str, metrics: dict[str, float]) -> str:
    """Красивый текстовый отчёт по метрикам для консоли."""
    lines = [title, "-" * len(title)]
    labels = {
        "mean_reward": "средняя награда за эпизод",
        "std_reward": "стандартное отклонение награды",
        "fill_rate": "доля удовлетворённого спроса",
        "stockout_rate": "доля дней с дефицитом",
        "avg_stock": "средний остаток на складе",
        "mean_unmet": "недостача за эпизод, шт.",
    }
    for key, value in metrics.items():
        lines.append(f"{labels.get(key, key):38s}: {value:10.3f}")
    return "\n".join(lines)
