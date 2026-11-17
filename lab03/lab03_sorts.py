#!/usr/bin/env python3
"""ЛР 3: сортировки, проверки и воспроизводимые замеры варианта 2."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import random
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position


@dataclass
class Item:
    """Пара, сравниваемая только по ключу."""

    key: int
    label: str

    def __lt__(self, other):
        return self.key < other.key


def bubble_sort(values, rng=None):
    """Пузырёк: один обмен считается одной операцией."""
    del rng
    a = list(values)
    comparisons = moves = 0
    for end in range(len(a) - 1, 0, -1):
        changed = False
        for i in range(end):
            comparisons += 1
            if a[i + 1] < a[i]:
                a[i], a[i + 1] = a[i + 1], a[i]
                moves += 1
                changed = True
        if not changed:
            break
    return a, comparisons, moves


def insertion_sort(values, rng=None):
    """Вставки: перемещениями считаются сдвиги и запись элемента."""
    del rng
    a = list(values)
    comparisons = moves = 0
    for i in range(1, len(a)):
        value = a[i]
        j = i
        while j > 0:
            comparisons += 1
            if not value < a[j - 1]:
                break
            a[j] = a[j - 1]
            moves += 1
            j -= 1
        if j != i:
            a[j] = value
            moves += 1
    return a, comparisons, moves


def selection_sort(values, rng=None):
    """Выбор: сравнения ключей и обмены разных позиций."""
    del rng
    a = list(values)
    comparisons = moves = 0
    for i in range(len(a) - 1):
        minimum = i
        for j in range(i + 1, len(a)):
            comparisons += 1
            if a[j] < a[minimum]:
                minimum = j
        if minimum != i:
            a[i], a[minimum] = a[minimum], a[i]
            moves += 1
    return a, comparisons, moves


def quick_sort(values, rng=None):
    """Трёхпутевое разбиение, стек меньшей глубины и случайный опорный."""
    # pylint: disable=too-many-nested-blocks
    rng = rng if rng is not None else random.Random()
    a = list(values)
    comparisons = moves = 0
    stack = [(0, len(a))]
    while stack:
        lo, hi = stack.pop()
        while hi - lo > 1:
            pivot = a[rng.randrange(lo, hi)]
            lt, i, gt = lo, lo, hi
            while i < gt:
                comparisons += 1
                if a[i] < pivot:
                    if i != lt:
                        a[i], a[lt] = a[lt], a[i]
                        moves += 1
                    lt += 1
                    i += 1
                else:
                    comparisons += 1
                    if pivot < a[i]:
                        gt -= 1
                        if i != gt:
                            a[i], a[gt] = a[gt], a[i]
                            moves += 1
                    else:
                        i += 1
            left, right = (lo, lt), (gt, hi)
            if left[1] - left[0] < right[1] - right[0]:
                stack.append(right)
                lo, hi = left
            else:
                stack.append(left)
                lo, hi = right
    return a, comparisons, moves


ALGORITHMS = {
    "bubble": bubble_sort,
    "insertion": insertion_sort,
    "selection": selection_sort,
    "quick": quick_sort,
}
SIZES = (128, 256, 512, 1024, 2048)


def inputs(n, seed):
    """Генерация из заготовки курса: отдельный поток для размера."""
    rng = random.Random(seed + n)
    data = [rng.randint(-1_000_000, 1_000_000) for _ in range(n)]
    return {
        "sorted": sorted(data),
        "random": data,
        "reversed": sorted(data, reverse=True),
    }


def require(condition, message):
    """Проверка работает и при python -O."""
    if not condition:
        raise ValueError(message)


def verify():
    """Границы, 500 случайных массивов, счётчики и стабильность."""
    rng = random.Random(32)
    samples = [[], [7], [5] * 80, list(range(80)), list(range(80, 0, -1))]
    samples += [
        [rng.randint(-10, 10) for _ in range(rng.randrange(81))] for _ in range(500)
    ]
    for name, fn in ALGORITHMS.items():
        for data in samples:
            original = list(data)
            result, comparisons, moves = fn(data, random.Random(32))
            require(data == original, f"{name}: изменён вход")
            require(
                all(result[i] <= result[i + 1] for i in range(len(result) - 1)),
                f"{name}: нарушен порядок",
            )
            require(Counter(result) == Counter(data), f"{name}: не перестановка")
            require(result == sorted(data), f"{name}: расходится с sorted")
            require(comparisons >= 0 and moves >= 0, f"{name}: счётчики")
        print(f"{name}: 505 входов — OK")
    for n in (0, 1, 2, 10, 80):
        for data in (list(range(n)), list(range(n, 0, -1)), [1] * n):
            require(selection_sort(data)[1] == n * (n - 1) // 2, "счётчик выбора")
        if n:
            require(bubble_sort(list(range(n)))[1:] == (n - 1, 0), "пузырёк")
            require(insertion_sort(list(range(n)))[1:] == (n - 1, 0), "вставки")
    for name in ("bubble", "insertion"):
        for _ in range(100):
            items = [Item(rng.randrange(4), str(i)) for i in range(30)]
            result = ALGORITHMS[name](items, random.Random(32))[0]
            require(
                [x.label for x in result]
                == [x.label for x in sorted(items, key=lambda x: x.key)],
                f"{name}: нарушена стабильность",
            )
    # Все равные элементы не должны вызывать квадратичное разбиение.
    require(
        quick_sort([1] * 10000, random.Random(32))[1] == 20000,
        "QuickSort: обработка равных ключей",
    )
    print("Счётчики, стабильность и равные ключи — OK")


def demo():
    """Контрпример стабильности с фиксированным потоком опорных элементов."""
    items = [Item(0, "a"), Item(0, "b"), Item(1, "c"), Item(1, "d"), Item(0, "e")]
    result = {}
    for name, fn in ALGORITHMS.items():
        ordered = fn(items, random.Random(32))[0]
        result[name] = [[item.key, item.label] for item in ordered]
    return result


def run(seed, out):
    """Прогрев и медиана пяти повторов; каждый вызов сортирует копию."""
    # pylint: disable=too-many-locals
    records = []
    for n in SIZES:
        for kind, data in inputs(n, seed).items():
            for name, fn in ALGORITHMS.items():
                fn(data, random.Random(seed))
                times = []
                for _ in range(5):
                    rng = random.Random(seed)
                    start = time.perf_counter()
                    result, comparisons, moves = fn(data, rng)
                    times.append(time.perf_counter() - start)
                    require(result == sorted(data), "ошибка замера")
                record = {
                    "n": n,
                    "input": kind,
                    "algorithm": name,
                    "seconds": statistics.median(times),
                    "comparisons": comparisons,
                    "moves": moves,
                }
                records.append(record)
                print(f"{n:4} {kind:8} {name:9} {record['seconds']:.6f} с", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(
            {
                "variant": 2,
                "seed": seed,
                "sizes": SIZES,
                "repeats": 5,
                "stability": demo(),
                "measurements": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, kind in zip(axes, ("sorted", "random", "reversed")):
        for name in ALGORITHMS:
            rows = [r for r in records if r["input"] == kind and r["algorithm"] == name]
            ax.loglog(
                [r["n"] for r in rows],
                [r["seconds"] for r in rows],
                marker="o",
                label=name,
            )
        ax.set_title(kind)
        ax.set_xlabel("Размер n")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Время, с")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(out / "lab03_times.png", dpi=160)
    plt.close(fig)


def main():
    """Проверить вариант и запустить проверки и измерения."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("lab03/figures"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    require(args.variant == 2, "Работа выполнена для варианта 2")
    verify()
    print("Стабильность:", demo())
    if not args.check_only:
        run(30 + args.variant, args.out)


if __name__ == "__main__":
    main()
