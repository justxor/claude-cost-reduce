"""
router.py — маршрутизация задач по уровням моделей и каскад с эскалацией.

Идея: 80% маршрутизации закрывается детерминированными правилами по типу задачи.
Модель-маршрутизатор добавляет вызов и задержку, поэтому применяется последней
и только если доказано, что окупается.

Запуск демонстрации (ключ не нужен, LLM подменён заглушкой):
    python code/router.py

Содержимое:
  * TaskSpec / Decision   — что известно о задаче и что решил роутер
  * route()               — правила маршрутизации с объяснением (route_reason)
  * QualityGate           — дешёвый формальный контроль результата
  * cascade()             — дешёвая модель -> контроль -> эскалация
  * economics()           — расчёт цены каскада и точки его окупаемости
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

TIERS = ("cheap", "balanced", "premium")

# Иллюстративные цены за миллион токенов (см. code/cost_meter.py)
TIER_PRICE = {"cheap": (0.80, 4.00), "balanced": (3.00, 15.00), "premium": (15.00, 75.00)}

# Классы задач, для которых уровень известен заранее. Это главная таблица:
# она бесплатна, объяснима и покрывает большинство трафика.
TASK_TIER = {
    "classify": "cheap",
    "extract": "cheap",
    "route_intent": "cheap",
    "compact_history": "cheap",
    "judge_simple": "cheap",
    "answer_from_fragment": "balanced",
    "support_dialog": "balanced",
    "code_edit": "balanced",
    "plan": "premium",
    "incident_analysis": "premium",
    "legal_review": "premium",
    "finance_calc": "premium",
}

# Признаки в тексте, повышающие уровень. Грубая эвристика: используйте
# только вместе с замером, иначе она начнёт гнать всё на дорогую модель.
UPGRADE_PATTERNS = (
    r"\bпочему\b",
    r"\bсравни\b",
    r"\bдокажи\b",
    r"\bпротиворечи",
    r"\bразбери\s+инцидент",
)


@dataclass
class TaskSpec:
    kind: str                      # ключ из TASK_TIER или "unknown"
    text: str = ""
    input_tokens: int = 0
    attempt: int = 0               # 0 — первая попытка
    critical: bool = False         # продуктовая критичность сценария
    background: bool = False       # фоновая задача: можно вниз и в батч
    requires_planning: bool = False


@dataclass
class Decision:
    tier: str
    reason: str
    batch: bool = False
    thinking: bool = False

    def as_log(self) -> dict[str, Any]:
        return {
            "model_tier": self.tier,
            "route_reason": self.reason,
            "batch": self.batch,
            "thinking": self.thinking,
        }


def _bump(tier: str, steps: int = 1) -> str:
    idx = min(len(TIERS) - 1, TIERS.index(tier) + steps)
    return TIERS[idx]


def _drop(tier: str, steps: int = 1) -> str:
    idx = max(0, TIERS.index(tier) - steps)
    return TIERS[idx]


def route(task: TaskSpec) -> Decision:
    """Возвращает уровень модели и ОБЪЯСНЕНИЕ выбора.

    route_reason обязателен: без него отладить маршрутизацию невозможно,
    а дрейф маршрутизации — одна из частых причин внезапного роста счёта.
    """
    # 1. Повторная попытка: повторять то же самое той же моделью бессмысленно.
    if task.attempt > 0:
        base = TASK_TIER.get(task.kind, "balanced")
        return Decision(_bump(base), f"retry:attempt={task.attempt}", thinking=True)

    # 2. Критичный сценарий — вверх, без обсуждений.
    if task.critical:
        return Decision("premium", "critical:product_flag", thinking=True)

    # 3. Известный класс задачи.
    if task.kind in TASK_TIER:
        tier = TASK_TIER[task.kind]
        reason = f"rules:kind={task.kind}"
        thinking = tier == "premium" or task.requires_planning

        # 3a. Очень длинный вход: разрыв между уровнями растёт с длиной.
        if task.input_tokens > 60_000 and tier != "premium":
            tier, reason = _bump(tier), reason + "+long_input"

        # 3b. Фоновая задача: можно вниз и в батч.
        if task.background and tier != "premium":
            tier, reason = _drop(tier), reason + "+background"
            return Decision(tier, reason, batch=True, thinking=False)

        return Decision(tier, reason, thinking=thinking)

    # 4. Неизвестный класс: эвристика по тексту, затем середина.
    if any(re.search(p, task.text, re.IGNORECASE) for p in UPGRADE_PATTERNS):
        return Decision("premium", "heuristic:reasoning_words", thinking=True)
    if task.requires_planning:
        return Decision("premium", "heuristic:planning", thinking=True)
    return Decision("balanced", "default:unknown_kind")


# --------------------------------------------------------------------------
# Контроль качества: дешёвый, формальный, спроектированный под ошибки
# именно дешёвой модели. Без него каскад бессмыслен.
# --------------------------------------------------------------------------


@dataclass
class QualityGate:
    """Набор бесплатных проверок результата."""

    required_fields: tuple[str, ...] = ()
    allowed_values: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    must_cite_from: str | None = None      # текст источника для проверки цитат
    min_confidence: float | None = None

    def check(self, raw: str) -> tuple[bool, str]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return False, "невалидный JSON"
        if not isinstance(data, dict):
            return False, "ожидался объект"

        for f in self.required_fields:
            if f not in data:
                return False, f"нет обязательного поля: {f}"
        for f, allowed in self.allowed_values.items():
            if f in data and data[f] not in allowed:
                return False, f"недопустимое значение {f}={data[f]!r}"
        for f, (lo, hi) in self.numeric_ranges.items():
            if f in data:
                try:
                    val = float(data[f])
                except (TypeError, ValueError):
                    return False, f"поле {f} не число"
                if not lo <= val <= hi:
                    return False, f"{f}={val} вне диапазона [{lo}, {hi}]"
        if self.must_cite_from is not None:
            quote = str(data.get("quote", ""))
            if quote and quote not in self.must_cite_from:
                return False, "цитата отсутствует в источнике (галлюцинация)"
        if self.min_confidence is not None:
            conf = data.get("confidence")
            if conf is None:
                return False, "нет поля confidence"
            if float(conf) < self.min_confidence:
                return False, f"низкая уверенность: {conf}"
        if data.get("needs_review") is True:
            return False, "модель сама пометила needs_review"
        return True, "ок"


LLMCall = Callable[[str, str, bool], str]   # (tier, prompt, thinking) -> raw ответ


@dataclass
class CascadeResult:
    ok: bool
    raw: str
    tier_used: str
    escalated: bool
    attempts: int
    reasons: list[str]


def cascade(
    prompt: str,
    task: TaskSpec,
    gate: QualityGate,
    call: LLMCall,
) -> CascadeResult:
    """Дешёвая модель -> контроль -> при провале эскалация вверх.

    Каскад выгоден примерно с того момента, когда дешёвая модель проходит
    контроль хотя бы в 40% случаев. Иначе вы просто платите дважды.
    """
    reasons: list[str] = []
    decision = route(task)
    tier = decision.tier
    reasons.append(decision.reason)

    raw = call(tier, prompt, decision.thinking)
    ok, why = gate.check(raw)
    if ok:
        return CascadeResult(True, raw, tier, False, 1, reasons + ["gate:ok"])

    reasons.append(f"gate:fail:{why}")
    higher = _bump(tier)
    if higher == tier:
        return CascadeResult(False, raw, tier, False, 1, reasons + ["no_higher_tier"])

    raw2 = call(higher, prompt, True)
    ok2, why2 = gate.check(raw2)
    reasons.append("gate:ok" if ok2 else f"gate:fail:{why2}")
    return CascadeResult(ok2, raw2, higher, True, 2, reasons)


def economics(
    p_cheap_ok: float,
    cost_cheap: float,
    cost_expensive: float,
    manual_cost: float = 0.0,
    p_expensive_ok: float = 0.98,
) -> dict[str, float]:
    """Экономика каскада против чистых конфигураций.

    manual_cost — во сколько обходится провал (ручная доработка, эскалация
    к человеку, потерянный клиент). Именно это слагаемое обычно решает спор
    "дешёвая против дорогой", и именно его обычно забывают.
    """
    fail_cascade = (1 - p_cheap_ok) * (1 - p_expensive_ok)
    cascade_cost = cost_cheap + (1 - p_cheap_ok) * cost_expensive
    return {
        "cheap_only": cost_cheap + (1 - p_cheap_ok) * manual_cost,
        "expensive_only": cost_expensive + (1 - p_expensive_ok) * manual_cost,
        "cascade": cascade_cost + fail_cascade * manual_cost,
        "escalation_share": round(1 - p_cheap_ok, 4),
    }


# --------------------------------------------------------------------------
# Демонстрация: заглушка вместо LLM, чтобы всё работало без ключа.
# --------------------------------------------------------------------------


def fake_llm(tier: str, prompt: str, thinking: bool) -> str:
    """Дешёвый уровень иногда возвращает мусор, дорогой — валидный ответ."""
    if tier == "cheap":
        if "сложн" in prompt:
            return "Конечно! Вот ответ: категория billing, приоритет высокий."
        return json.dumps({"category": "billing", "priority": 2, "confidence": 0.91},
                          ensure_ascii=False)
    return json.dumps({"category": "billing", "priority": 2, "confidence": 0.97},
                      ensure_ascii=False)


def demo() -> None:
    gate = QualityGate(
        required_fields=("category", "priority"),
        allowed_values={"category": ("billing", "tech", "sales")},
        numeric_ranges={"priority": (1, 3)},
        min_confidence=0.6,
    )

    print("=== маршрутизация ===")
    samples = [
        TaskSpec(kind="classify", text="определи категорию обращения"),
        TaskSpec(kind="extract", text="извлеки поля", input_tokens=80_000),
        TaskSpec(kind="support_dialog", text="здравствуйте, где мой заказ"),
        TaskSpec(kind="support_dialog", text="повтор", attempt=1),
        TaskSpec(kind="classify", text="ночная разметка", background=True),
        TaskSpec(kind="unknown", text="почему счёт вырос втрое?"),
        TaskSpec(kind="legal_review", text="проверь договор", critical=True),
    ]
    for t in samples:
        d = route(t)
        print(f"  {t.kind:<18} -> {d.tier:<8} {d.reason}"
              + (" [batch]" if d.batch else "")
              + (" [thinking]" if d.thinking else ""))

    print()
    print("=== каскад ===")
    for prompt in ("простое обращение", "сложное обращение"):
        res = cascade(prompt, TaskSpec(kind="classify", text=prompt), gate, fake_llm)
        print(f"  {prompt:<20} ok={res.ok} tier={res.tier_used} "
              f"escalated={res.escalated} reasons={res.reasons}")

    print()
    print("=== экономика каскада (цена задачи, включая цену провала) ===")
    for p in (0.5, 0.7, 0.85, 0.95):
        e = economics(p_cheap_ok=p, cost_cheap=0.006, cost_expensive=0.041, manual_cost=4.0)
        print(f"  p_cheap_ok={p:.2f}: cheap_only={e['cheap_only']:.4f} "
              f"cascade={e['cascade']:.4f} expensive_only={e['expensive_only']:.4f}")
    print()
    print("Обратите внимание: при manual_cost=4 $ 'только дешёвая' почти всегда")
    print("самая дорогая конфигурация. Это и есть смысл метрики cost_per_success.")


def _self_test() -> None:
    assert route(TaskSpec(kind="classify")).tier == "cheap"
    assert route(TaskSpec(kind="classify", attempt=1)).tier == "balanced"
    assert route(TaskSpec(kind="extract", input_tokens=100_000)).tier == "balanced"
    assert route(TaskSpec(kind="classify", background=True)).batch is True
    assert route(TaskSpec(kind="unknown", text="почему так дорого")).tier == "premium"

    gate = QualityGate(required_fields=("a",), numeric_ranges={"a": (0, 10)})
    assert gate.check('{"a": 5}')[0] is True
    assert gate.check('{"a": 50}')[0] is False
    assert gate.check("не json")[0] is False
    assert gate.check('{"a": 1, "needs_review": true}')[0] is False

    e = economics(0.9, 0.006, 0.041)
    assert e["cascade"] < e["expensive_only"]
    print("self-test: ок")


if __name__ == "__main__":
    _self_test()
    print()
    demo()
