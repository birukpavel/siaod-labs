#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 1, к пункту 5: почему замер ведётся по модулю.

Без модуля множители растут (x^n — около n*log10(x) цифр), умножение
перестаёт стоить O(1), и замер показывает длину чисел, а не число итераций.
Не часть решения, ничего не проверяет — только замер для отчёта.

Запуск: python3 lab01/lab01_longarith_demo.py
"""

from __future__ import annotations

import math

from lab01_complexity import POW_BASE, POW_MOD, binary_pow, bench, log_log_slope

#: Без модуля большие n считаются слишком долго.
EXPONENTS = (1_000, 3_000, 10_000, 30_000, 100_000)

#: По модулю одна операция неизмеримо мала — замеряем пачку.
MOD_CALLS = 200


def main() -> None:
    """Таблица времени по модулю и без него, наклоны в log-log."""
    print(
        f"Бинарное возведение в степень основания {POW_BASE}: "
        f"по модулю {POW_MOD} против длинной арифметики\n"
    )
    print(
        f"{'n':>8}  {'log2 n':>7}  {'по модулю, мкс':>15}  "
        f"{'без модуля, мкс':>16}  {'цифр в ответе':>14}"
    )

    with_mod: list[tuple[int, float]] = []
    without_mod: list[tuple[int, float]] = []
    for n in EXPONENTS:

        def modular(n: int = n) -> None:
            for _ in range(MOD_CALLS):
                binary_pow(POW_BASE, n, mod=POW_MOD)

        t_mod = bench(modular) / MOD_CALLS
        t_raw = bench(lambda n=n: binary_pow(POW_BASE, n))
        # По логарифму, а не len(str(...)): с 3.11 перевод int в строку
        # ограничен 4300 цифрами.
        digits = int(n * math.log10(POW_BASE)) + 1
        with_mod.append((n, t_mod))
        without_mod.append((n, t_raw))
        print(
            f"{n:>8}  {math.log2(n):>7.1f}  {t_mod * 1e6:>15.3f}  "
            f"{t_raw * 1e6:>16.1f}  {digits:>14}"
        )

    print("\nНаклон в осях log-log:")
    print(
        f"  по модулю   {log_log_slope(with_mod):+.3f} — почти горизонталь, "
        f"как и должно быть у логарифмического роста"
    )
    print(
        f"  без модуля  {log_log_slope(without_mod):+.3f} — степенной рост: "
        f"замер показывает длину чисел, а не число итераций"
    )
    growth_mod = with_mod[-1][1] / with_mod[0][1]
    growth_raw = without_mod[-1][1] / without_mod[0][1]
    print(
        f"\nПри росте n в 100 раз время выросло в {growth_mod:.2f} раза "
        f"(по модулю) и в {growth_raw:.0f} раз (без модуля)."
    )


if __name__ == "__main__":
    main()
