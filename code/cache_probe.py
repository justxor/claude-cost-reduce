"""
cache_probe.py — проверка, что кэширование промпта действительно работает.

Две части:
  1) offline-часть: проверяет СТАБИЛЬНОСТЬ сборки промпта. Ключ не нужен,
     и именно здесь находится 80% причин "кэш не работает".
  2) online-часть: три идентичных запроса и разбор блока usage. Нужен ключ,
     стоит центы. Запускается только с флагом --live.

Запуск:
    python code/cache_probe.py            # только offline-проверки
    python code/cache_probe.py --live     # плюс реальные вызовы (нужен ключ)

Что считается успехом:
    запрос 1: cache_creation большой, cache_read = 0
    запрос 2: cache_creation = 0,     cache_read примерно равен размеру шапки
    запрос 3: то же, что запрос 2
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any, Callable

# --------------------------------------------------------------------------
# 1. Сборка промпта. ОДНА функция на весь проект: если промпт собирается
#    в трёх местах по-разному, кэш будет попадать через раз.
# --------------------------------------------------------------------------

SYSTEM_VERSION = "2026-08-17.1"
TOOLS_VERSION = "2026-08-17.1"

POLICY = (
    "Ты ассистент службы поддержки. Отвечай кратко и по делу. "
    "Не выдумывай номера заказов и суммы: если данных нет, вызови инструмент. "
    "Если данных нет и после инструментов — скажи, чего именно не хватает."
)

# Длинный стабильный блок: справочник правил. В реальном проекте — из файла.
KNOWLEDGE = "\n".join(
    f"Правило {i:02d}. Условие возврата товара категории {i}: срок 14 дней, "
    f"чек не обязателен, упаковка должна быть целой."
    for i in range(1, 41)
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_orders",
        "description": "Найти заказы клиента. Возвращает сводки: id, дата, сумма, статус.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "например: последние 3 заказа"},
                "limit": {"type": "integer", "description": "по умолчанию 10"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_order",
        "description": "Подробности одного заказа по id. Используй после search_orders.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
]


def stable_json(obj: Any) -> str:
    """Единственный разрешённый способ сериализации в промпт.

    sort_keys=True — потому что порядок ключей в словарях Python не гарантирован
    между процессами и версиями кода, а любое изменение байтов убивает кэш.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def build_system(with_cache_marks: bool = True, broken: bool = False) -> list[dict[str, Any]]:
    """Собирает системный блок.

    broken=True воспроизводит самую частую ошибку: таймстемп в верхней части.
    """
    head = POLICY
    if broken:
        head = f"Текущее время: {time.time()}. " + head   # <-- убийца кэша №1

    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": head},
        {"type": "text", "text": "Справочник правил:\n" + KNOWLEDGE},
    ]
    if with_cache_marks:
        for b in blocks:
            b["cache_control"] = {"type": "ephemeral"}
    return blocks


def build_request(question: str, broken: bool = False) -> dict[str, Any]:
    return {
        "model": os.environ.get("PROBE_MODEL", "claude-sonnet-4-5"),
        "max_tokens": 64,
        "system": build_system(broken=broken),
        "tools": TOOLS,
        "messages": [{"role": "user", "content": question}],
    }


def prompt_fingerprint(req: dict[str, Any]) -> str:
    """Отпечаток кэшируемой части: system + tools + версии."""
    material = stable_json(
        {
            "system": req["system"],
            "tools": req["tools"],
            "system_version": SYSTEM_VERSION,
            "tools_version": TOOLS_VERSION,
        }
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# 2. Offline-проверки. Бесплатные, быстрые, ловят большинство проблем.
# --------------------------------------------------------------------------


def check_deterministic() -> tuple[bool, str]:
    """Дважды собранный промпт должен быть побайтово одинаковым."""
    a = prompt_fingerprint(build_request("вопрос"))
    b = prompt_fingerprint(build_request("другой вопрос"))
    ok = a == b
    return ok, (
        f"стабильность сборки: {'ок' if ok else 'ПРОВАЛ'} ({a} vs {b}); "
        "кэшируемая часть не должна зависеть от вопроса пользователя"
    )


def check_broken_detected() -> tuple[bool, str]:
    """Проверка самой проверки: сломанный промпт обязан отличаться."""
    a = prompt_fingerprint(build_request("вопрос", broken=True))
    time.sleep(0.01)
    b = prompt_fingerprint(build_request("вопрос", broken=True))
    ok = a != b
    return ok, (
        "детектор убийцы кэша: "
        + ("ок, таймстемп в шапке обнаружен" if ok else "ПРОВАЛ, детектор слепой")
    )


def check_marks_present() -> tuple[bool, str]:
    req = build_request("вопрос")
    marked = sum(1 for b in req["system"] if "cache_control" in b)
    ok = marked >= 1
    return ok, f"отметок кэша в system: {marked} ({'ок' if ok else 'нет отметок'})"


def check_variable_below() -> tuple[bool, str]:
    """Изменчивое (вопрос пользователя) не должно попадать в кэшируемую часть."""
    req = build_request("уникальный вопрос 12345")
    serialized = stable_json({"system": req["system"], "tools": req["tools"]})
    ok = "12345" not in serialized
    return ok, (
        "изменчивая часть ниже границы кэша: "
        + ("ок" if ok else "ПРОВАЛ, вопрос попал в кэшируемый блок")
    )


def estimate_tokens(text: str) -> int:
    """Очень грубая оценка (для порядка величины, не для биллинга).

    Русский текст даёт больше токенов на символ, чем английский, поэтому берём
    осторожный коэффициент. Точные числа — только из usage.
    """
    return max(1, len(text) // 3)


def check_prefix_size() -> tuple[bool, str]:
    req = build_request("вопрос")
    size = estimate_tokens(stable_json({"s": req["system"], "t": req["tools"]}))
    ok = size >= 1024
    return ok, (
        f"оценка размера кэшируемого префикса: примерно {size} токенов "
        + ("(достаточно)" if ok else "(возможно, ниже минимального порога кэша)")
    )


OFFLINE_CHECKS: list[Callable[[], tuple[bool, str]]] = [
    check_deterministic,
    check_broken_detected,
    check_marks_present,
    check_variable_below,
    check_prefix_size,
]


def run_offline() -> bool:
    print("=== offline-проверки (ключ не нужен) ===")
    all_ok = True
    for check in OFFLINE_CHECKS:
        ok, message = check()
        all_ok = all_ok and ok
        print(("  [ok]   " if ok else "  [FAIL] ") + message)
    print()
    if all_ok:
        print("Сборка промпта стабильна. Если в проде cache_read всё равно 0 —")
        print("значит промпт в проде собирается другим кодом. Ищите второе место сборки.")
    else:
        print("Сначала починить offline-проверки: без стабильного префикса кэш невозможен.")
    return all_ok


# --------------------------------------------------------------------------
# 3. Online-часть: три идентичных запроса, разбор usage.
# --------------------------------------------------------------------------


def _usage_row(n: int, usage: Any) -> dict[str, int]:
    def g(name: str) -> int:
        if isinstance(usage, dict):
            return int(usage.get(name, 0) or 0)
        return int(getattr(usage, name, 0) or 0)

    return {
        "n": n,
        "input": g("input_tokens"),
        "write": g("cache_creation_input_tokens"),
        "read": g("cache_read_input_tokens"),
        "out": g("output_tokens"),
    }


def diagnose(rows: list[dict[str, int]]) -> str:
    if not rows:
        return "нет данных"
    first, rest = rows[0], rows[1:]
    if not rest:
        return "нужно минимум два запроса"
    writes_always = all(r["write"] > 0 for r in rows)
    reads_after_first = all(r["read"] > 0 for r in rest)
    no_cache_at_all = all(r["write"] == 0 and r["read"] == 0 for r in rows)

    if no_cache_at_all:
        return (
            "кэш не задействован: отметки не проставлены или блок короче "
            "минимального порога -> см. 01.4"
        )
    if writes_always:
        return (
            "кэш пишется на каждом запросе: префикс меняется между вызовами "
            "-> ищите убийцу из 01.7 (таймстемп, порядок ключей, динамические инструменты)"
        )
    if reads_after_first and first["write"] > 0:
        share = rest[0]["read"] / max(1, first["write"])
        if share < 0.5:
            return (
                "кэш работает, но читается меньше половины записанного: "
                "отметка стоит слишком высоко -> опустите её ниже стабильных блоков"
            )
        return "кэш работает правильно"
    return "смешанная картина: часть запросов читает кэш, часть нет -> нестабильная сборка"


def run_live(repeats: int = 3, broken: bool = False) -> None:
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        print("Нужен пакет anthropic: pip install anthropic")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Нет ANTHROPIC_API_KEY в окружении. Положите ключ в .env и экспортируйте его.")
        print("Ключ никогда не хранится в коде и не передаётся в URL.")
        return

    client = anthropic.Anthropic()
    rows: list[dict[str, int]] = []
    print("=== live-проверка (несколько центов) ===")
    for i in range(1, repeats + 1):
        req = build_request("Скажи слово: тест.", broken=broken)
        resp = client.messages.create(**req)
        row = _usage_row(i, resp.usage)
        rows.append(row)
        print(
            f"  запрос {i}: input={row['input']:>6} write={row['write']:>6} "
            f"read={row['read']:>6} out={row['out']:>4}"
        )
        time.sleep(1)

    print()
    print("Диагноз:", diagnose(rows))
    print()
    print("Ожидаемая картина: write только в первом запросе, read во всех остальных.")


def main(argv: list[str]) -> int:
    ok = run_offline()
    if "--live" in argv:
        print()
        run_live(broken="--broken" in argv)
    else:
        print()
        print("Для проверки на реальном API: python code/cache_probe.py --live")
        print("Чтобы увидеть, как выглядит сломанный кэш: --live --broken")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
