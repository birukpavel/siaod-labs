#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 1, к разбору расхождения наклонов: три гипотезы, каждая с замером.

Наклоны на полном ряду размеров выходят ниже теоретических. Скрипт
проверяет три объяснения: накладные расходы на вызов, разные файлы данных
в разных точках, частота обновления максимума. Не часть решения — только
замеры для отчёта.

Запуск: python3 lab01/lab01_slope_check.py --variant 2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from lab01_complexity import (
    SIZES,
    array_max,
    array_sum,
    bench,
    check_variant,
    find_data_dir,
    load_array,
    log_log_slope,
)

#: Вызовов на замер: один вызов на пустом массиве неизмеримо мал.
EMPTY_CALLS = 100_000

#: Точки для замера на префиксах одного массива.
PREFIXES = (1_000, 10_000, 100_000)


def call_overhead() -> float:
    """Гипотеза 1: сколько стоит сам вызов. Замер на пустом массиве."""
    empty: list[int] = []

    def batch() -> None:
        for _ in range(EMPTY_CALLS):
            array_sum(empty)

    t = bench(batch) / EMPTY_CALLS
    print(f"1. Накладные расходы на вызов: {t * 1e9:.0f} нс")
    print(f"   при n = 1000 это {t / (1000 * 10e-9) * 100:.1f}% времени замера")
    return t


def prefixes(data_dir: Path) -> None:
    """Гипотеза 2: разные n — разные файлы. Тот же замер на одном массиве."""
    whole = load_array(data_dir, "random", 100_000)
    print("\n2. Префиксы одного массива против отдельных файлов:")
    print(f"{'n':>8}  {'sum, нс/эл.':>12}  {'max, нс/эл.':>12}")
    points_sum, points_max = [], []
    for n in PREFIXES:
        a = whole[:n]
        t_sum, t_max = bench(lambda a=a: array_sum(a)), bench(lambda a=a: array_max(a))
        points_sum.append((n, t_sum))
        points_max.append((n, t_max))
        print(f"{n:>8}  {t_sum / n * 1e9:>12.2f}  {t_max / n * 1e9:>12.2f}")
    print(
        f"   наклон на префиксах: sum {log_log_slope(points_sum):.3f}, "
        f"max {log_log_slope(points_max):.3f}"
    )


def max_updates(data_dir: Path) -> None:
    """Гипотеза 3: как часто array_max обновляет максимум (ожидается ~ln n)."""
    print("\n3. Обновлений максимума при проходе:")
    print(f"{'n':>8}  {'обновлений':>11}  {'ln n':>6}  {'доля элементов':>15}")
    for n in SIZES:
        a = load_array(data_dir, "random", n)
        best, updates = a[0], 0
        for value in a:
            if value > best:
                best, updates = value, updates + 1
        print(f"{n:>8}  {updates:>11}  {math.log(n):>6.1f}  {updates / n:>14.3%}")


def main() -> None:
    """Три замера подряд, вывод — в отчёт."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, default=2, help="номер варианта")
    ap.add_argument("--data", type=Path, default=None, help="каталог с данными")
    args = ap.parse_args()

    data_dir = find_data_dir(args.data)
    check_variant(data_dir, args.variant)

    call_overhead()
    prefixes(data_dir)
    max_updates(data_dir)


if __name__ == "__main__":
    main()
