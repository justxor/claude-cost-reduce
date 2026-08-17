"""
cost_meter.py — учёт токенов и денег для вызовов LLM.

Зачем: без учёта в момент вызова любая оптимизация стоимости делается вслепую.
Этот модуль не требует API-ключа и не делает сетевых вызовов: он считает деньги
по блоку usage, который вам вернул провайдер, и агрегирует их по задачам.

Запуск демонстрации (ключ не нужен):
    python code/cost_meter.py

Что внутри:
  * PRICES        — таблица цен (иллюстративная, вынесите в конфиг и обновляйте)
  * price()       — цена одного вызова по usage
  * CostMeter     — накопление расходов по задачам и сценариям
  * Budget/Meter  — бюджеты задачи: шаги, токены, деньги, время
  * report()      — сводка: cost_per_task, cost_per_success, p50/p95, cache_hit_ratio

Главное правило: считать деньги в момент вызова. Пересчёт по логам через месяц
всегда упирается в то, что тарифы уже изменились.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

# --------------------------------------------------------------------------
# 1. Цены. ИЛЛЮСТРАТИВНЫЕ значения за миллион токенов.
#    Держите их в конфиге с датой актуальности и сверяйте с официальным прайсом.
# --------------------------------------------------------------------------

PRICES: dict[str, dict[str, float]] = {
    "premium":  {"in": 15.00, "out": 75.00},
    "balanced": {"in": 3.00, "out": 15.00},
    "cheap":    {"in": 0.80, "out": 4.00},
}
PRICES_AS_OF = "2026-01-01"   # дата, на которую взяты цифры

CACHE_WRITE_MULT = 1.25   # надбавка за запись в кэш (короткий TTL)
CACHE_WRITE_LONG_MULT = 2.00  # надбавка за запись в кэш (длинный TTL)
CACHE_READ_MULT = 0.10    # чтение из кэша
BATCH_MULT = 0.50         # асинхронная очередь

MILLION = 1_000_000


def normalize_usage(usage: Any) -> dict[str, int]:
    """Приводит usage из SDK или словаря к плоскому словарю с нулями по умолчанию."""
    def get(name: str) -> int:
        if isinstance(usage, dict):
            return int(usage.get(name, 0) or 0)
        return int(getattr(usage, name, 0) or 0)

    return {
        "input_tokens": get("input_tokens"),
        "output_tokens": get("output_tokens"),
        "cache_read_input_tokens": get("cache_read_input_tokens"),
        "cache_creation_input_tokens": get("cache_creation_input_tokens"),
    }


def price(
    usage: Any,
    tier: str,
    batch: bool = False,
    long_cache: bool = False,
) -> float:
    """Стоимость одного вызова в долларах.

    usage      — блок использования токенов из ответа API (или совместимый словарь)
    tier       — ключ из PRICES: premium / balanced / cheap
    batch      — вызов ушёл в асинхронную очередь
    long_cache — использован длинный TTL кэша (запись дороже)
    """
    if tier not in PRICES:
        raise KeyError(f"неизвестный тариф: {tier}; известные: {sorted(PRICES)}")

    u = normalize_usage(usage)
    p = PRICES[tier]
    write_mult = CACHE_WRITE_LONG_MULT if long_cache else CACHE_WRITE_MULT

    tokens_cost = (
        u["input_tokens"] * p["in"]
        + u["cache_creation_input_tokens"] * p["in"] * write_mult
        + u["cache_read_input_tokens"] * p["in"] * CACHE_READ_MULT
        + u["output_tokens"] * p["out"]
    )
    multiplier = BATCH_MULT if batch else 1.0
    return tokens_cost * multiplier / MILLION


def price_naive(usage: Any, tier: str) -> float:
    """Сколько стоил бы тот же вызов БЕЗ кэша и БЕЗ батча.

    Нужно, чтобы честно считать экономию: разница с price() и есть эффект
    рычагов 1 и 5. Кэшированные токены считаются по полной цене входа.
    """
    u = normalize_usage(usage)
    p = PRICES[tier]
    total_in = (
        u["input_tokens"]
        + u["cache_creation_input_tokens"]
        + u["cache_read_input_tokens"]
    )
    return (total_in * p["in"] + u["output_tokens"] * p["out"]) / MILLION


# --------------------------------------------------------------------------
# 2. Бюджеты задачи. Не защита от ошибок, а часть контракта агента.
# --------------------------------------------------------------------------


@dataclass
class Budget:
    max_steps: int = 12
    max_tokens_total: int = 200_000
    max_cost_usd: float = 0.25
    max_wall_ms: int = 60_000


class BudgetExceeded(Exception):
    def __init__(self, kind: str, current: float, limit: float) -> None:
        super().__init__(f"budget:{kind} {current} > {limit}")
        self.kind, self.current, self.limit = kind, current, limit


class Meter:
    """Счётчик одной задачи. charge() вызывается после каждого вызова модели."""

    def __init__(self, budget: Budget | None = None) -> None:
        self.budget = budget or Budget()
        self.steps = 0
        self.tokens = 0
        self.cost = 0.0
        self.started_ms = int(time.time() * 1000)

    @property
    def elapsed_ms(self) -> int:
        return int(time.time() * 1000) - self.started_ms

    def charge(self, usage: Any, tier: str, batch: bool = False) -> float:
        u = normalize_usage(usage)
        cost = price(u, tier, batch=batch)
        self.steps += 1
        self.tokens += (
            u["input_tokens"]
            + u["cache_read_input_tokens"]
            + u["cache_creation_input_tokens"]
            + u["output_tokens"]
        )
        self.cost += cost
        self._check()
        return cost

    def _check(self) -> None:
        b = self.budget
        checks = (
            ("steps", self.steps, b.max_steps),
            ("tokens", self.tokens, b.max_tokens_total),
            ("cost", round(self.cost, 6), b.max_cost_usd),
            ("wall", self.elapsed_ms, b.max_wall_ms),
        )
        for kind, current, limit in checks:
            if current > limit:
                raise BudgetExceeded(kind, current, limit)

    def snapshot(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "cost_usd": round(self.cost, 6),
            "elapsed_ms": self.elapsed_ms,
        }


# --------------------------------------------------------------------------
# 3. Накопитель по задачам и сценариям.
# --------------------------------------------------------------------------


@dataclass
class CallRecord:
    task_id: str
    scenario: str
    tier: str
    step: int
    usage: dict[str, int]
    cost_usd: float
    naive_cost_usd: float
    batch: bool = False
    ok: bool = True
    stop_reason: str = "end_turn"
    latency_ms: int = 0
    route_reason: str = ""


@dataclass
class CostMeter:
    """Собирает вызовы и считает продуктовые метрики стоимости."""

    calls: list[CallRecord] = field(default_factory=list)
    task_success: dict[str, bool] = field(default_factory=dict)

    def record(
        self,
        task_id: str,
        scenario: str,
        tier: str,
        usage: Any,
        step: int = 1,
        batch: bool = False,
        ok: bool = True,
        stop_reason: str = "end_turn",
        latency_ms: int = 0,
        route_reason: str = "",
    ) -> CallRecord:
        u = normalize_usage(usage)
        rec = CallRecord(
            task_id=task_id,
            scenario=scenario,
            tier=tier,
            step=step,
            usage=u,
            cost_usd=price(u, tier, batch=batch),
            naive_cost_usd=price_naive(u, tier),
            batch=batch,
            ok=ok,
            stop_reason=stop_reason,
            latency_ms=latency_ms,
            route_reason=route_reason,
        )
        self.calls.append(rec)
        return rec

    def finish_task(self, task_id: str, success: bool) -> None:
        """Отметить итог задачи. Без этого нельзя посчитать cost_per_success."""
        self.task_success[task_id] = success

    # ---- агрегаты -------------------------------------------------------

    def task_costs(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in self.calls:
            out[c.task_id] = out.get(c.task_id, 0.0) + c.cost_usd
        return out

    def total(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def total_naive(self) -> float:
        return sum(c.naive_cost_usd for c in self.calls)

    def cache_hit_ratio(self) -> float:
        """Доля входных токенов, пришедших из кэша (по количеству токенов)."""
        read = sum(c.usage["cache_read_input_tokens"] for c in self.calls)
        total_in = sum(
            c.usage["input_tokens"]
            + c.usage["cache_read_input_tokens"]
            + c.usage["cache_creation_input_tokens"]
            for c in self.calls
        )
        return (read / total_in) if total_in else 0.0

    def out_in_ratio(self) -> float:
        out = sum(c.usage["output_tokens"] for c in self.calls)
        total_in = sum(
            c.usage["input_tokens"]
            + c.usage["cache_read_input_tokens"]
            + c.usage["cache_creation_input_tokens"]
            for c in self.calls
        )
        return (out / total_in) if total_in else 0.0

    def report(self) -> dict[str, Any]:
        costs = self.task_costs()
        values = sorted(costs.values())
        tasks = len(costs)
        successes = sum(1 for t in costs if self.task_success.get(t, False))
        steps_per_task = (len(self.calls) / tasks) if tasks else 0.0

        def pct(p: float) -> float:
            if not values:
                return 0.0
            idx = min(len(values) - 1, int(round((len(values) - 1) * p)))
            return values[idx]

        total = self.total()
        naive = self.total_naive()
        return {
            "prices_as_of": PRICES_AS_OF,
            "calls": len(self.calls),
            "tasks": tasks,
            "successes": successes,
            "pass_rate": round(successes / tasks, 4) if tasks else 0.0,
            "total_usd": round(total, 6),
            "total_without_cache_and_batch_usd": round(naive, 6),
            "saved_by_cache_and_batch_usd": round(naive - total, 6),
            "saved_share": round(1 - total / naive, 4) if naive else 0.0,
            "cost_per_task": round(total / tasks, 6) if tasks else 0.0,
            "cost_per_success": round(total / successes, 6) if successes else None,
            "median_task_cost": round(statistics.median(values), 6) if values else 0.0,
            "p95_task_cost": round(pct(0.95), 6),
            "p95_to_median": (
                round(pct(0.95) / statistics.median(values), 2)
                if values and statistics.median(values) else None
            ),
            "steps_per_task": round(steps_per_task, 2),
            "cache_hit_ratio": round(self.cache_hit_ratio(), 4),
            "out_in_ratio": round(self.out_in_ratio(), 4),
            "by_scenario": self.by_scenario(),
        }

    def by_scenario(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in self.calls:
            out[c.scenario] = round(out.get(c.scenario, 0.0) + c.cost_usd, 6)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def warnings(self) -> list[str]:
        """Автоматические подсказки: куда идти в гайде."""
        r = self.report()
        w: list[str] = []
        if r["cache_hit_ratio"] < 0.3:
            w.append("cache_hit_ratio < 0.3 -> модуль 01-prompt-caching")
        if r["out_in_ratio"] > 0.3:
            w.append("out/in > 0.3 -> модуль 04-output-tokens")
        if r["steps_per_task"] > 12:
            w.append("steps_per_task > 12 -> модуль 03-agent-loop")
        if r["p95_to_median"] and r["p95_to_median"] > 5:
            w.append("p95/median > 5 -> тяжёлый хвост, модуль 03 и 00.7")
        if r["cost_per_success"] and r["pass_rate"] < 0.8:
            w.append("pass_rate < 0.8 -> сначала качество, потом экономия")
        return w


# --------------------------------------------------------------------------
# 4. Демонстрация: три конфигурации на одном синтетическом наборе задач.
#    Ключ не нужен: usage мы задаём руками, как будто он пришёл из API.
# --------------------------------------------------------------------------


def _usage(inp: int, out: int, read: int = 0, write: int = 0) -> dict[str, int]:
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write,
    }


def _simulate(
    name: str,
    tier: str,
    header: int,
    steps: int,
    growth: int,
    out_tokens: int,
    use_cache: bool,
    batch: bool,
    tasks: int = 20,
    success_rate: float = 0.9,
) -> tuple[str, dict[str, Any]]:
    """Считает стоимость набора задач для одной конфигурации."""
    m = CostMeter()
    for i in range(tasks):
        task_id = f"{name}-task-{i:02d}"
        for step in range(1, steps + 1):
            history = growth * (step - 1)
            if use_cache:
                if step == 1:
                    u = _usage(inp=history, out=out_tokens, write=header)
                else:
                    u = _usage(inp=history, out=out_tokens, read=header)
            else:
                u = _usage(inp=header + history, out=out_tokens)
            m.record(
                task_id=task_id,
                scenario=name,
                tier=tier,
                usage=u,
                step=step,
                batch=batch,
                route_reason="demo",
            )
        # чуть-чуть детерминированной "неуспешности" для метрики cost_per_success
        m.finish_task(task_id, success=(i % 10) < int(success_rate * 10))
    return name, m.report()


def demo() -> None:
    print(f"Цены на дату: {PRICES_AS_OF} (иллюстративные, проверяйте прайс)")
    print()

    configs = [
        # name, tier, header, steps, growth, out, cache, batch, success
        ("A. premium без оптимизаций", "premium", 15_000, 5, 800, 300, False, False, 0.9),
        ("B. premium + кэш", "premium", 15_000, 5, 800, 300, True, False, 0.9),
        ("C. premium + кэш + короткий вывод", "premium", 15_000, 5, 800, 90, True, False, 0.9),
        ("D. balanced + кэш + короткий вывод", "balanced", 15_000, 5, 800, 90, True, False, 0.9),
        ("E. cheap + кэш + батч (фон)", "cheap", 15_000, 5, 800, 90, True, True, 0.7),
    ]

    rows = []
    for name, tier, header, steps, growth, out, cache, batch, sr in configs:
        _, rep = _simulate(
            name, tier, header, steps, growth, out, cache, batch, success_rate=sr
        )
        rows.append((name, rep))

    width = max(len(n) for n, _ in rows)
    print(
        "конфигурация".ljust(width),
        "cost/task".rjust(11),
        "cost/success".rjust(13),
        "pass".rjust(6),
        "cache".rjust(7),
        "out/in".rjust(7),
    )
    print("-" * (width + 50))
    for name, rep in rows:
        print(
            name.ljust(width),
            f"{rep['cost_per_task']:.5f}".rjust(11),
            (f"{rep['cost_per_success']:.5f}" if rep["cost_per_success"] else "-").rjust(13),
            f"{rep['pass_rate']:.2f}".rjust(6),
            f"{rep['cache_hit_ratio']:.2f}".rjust(7),
            f"{rep['out_in_ratio']:.3f}".rjust(7),
        )

    base = rows[0][1]["cost_per_task"]
    print()
    print("Во сколько раз дешевле конфигурации A:")
    for name, rep in rows[1:]:
        factor = base / rep["cost_per_task"] if rep["cost_per_task"] else 0
        print(f"  {name}: x{factor:.1f}")

    print()
    print("Обратите внимание на строку E: она дешевле всех, но pass_rate ниже.")
    print("Именно поэтому сравнивать конфигурации нужно по cost_per_success,")
    print("а решение о смене модели принимать последним (модуль 06).")

    print()
    print("Подсказки по последней конфигурации:")
    m = CostMeter()
    for step in range(1, 15):
        m.record(
            task_id="loop-demo",
            scenario="loop",
            tier="balanced",
            usage=_usage(inp=4_000 + 900 * step, out=1_500),
            step=step,
        )
    m.finish_task("loop-demo", success=True)
    for line in m.warnings():
        print("  !", line)
    print()
    print(json.dumps(m.report(), ensure_ascii=False, indent=2)[:600], "...")


def _self_test() -> None:
    """Мини-проверки: тест, который никогда не падал, — не тест."""
    u = _usage(inp=1_000, out=100)
    assert abs(price(u, "balanced") - (1_000 * 3 + 100 * 15) / MILLION) < 1e-12

    cached = _usage(inp=0, out=100, read=10_000)
    plain = _usage(inp=10_000, out=100)
    assert price(cached, "balanced") < price(plain, "balanced"), "кэш должен быть дешевле"

    assert price(plain, "balanced", batch=True) == price(plain, "balanced") * BATCH_MULT

    m = CostMeter()
    m.record("t1", "s", "cheap", _usage(500, 50), step=1)
    m.record("t1", "s", "cheap", _usage(700, 50), step=2)
    m.finish_task("t1", True)
    rep = m.report()
    assert rep["tasks"] == 1 and rep["calls"] == 2
    assert rep["steps_per_task"] == 2.0
    assert rep["cost_per_success"] == rep["cost_per_task"]

    try:
        meter = Meter(Budget(max_steps=2))
        for _ in range(5):
            meter.charge(_usage(100, 10), "cheap")
    except BudgetExceeded as e:
        assert e.kind == "steps"
    else:  # pragma: no cover
        raise AssertionError("бюджет по шагам не сработал")

    print("self-test: ок")


if __name__ == "__main__":
    _self_test()
    print()
    demo()
