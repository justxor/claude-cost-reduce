"""
batch_runner.py — возобновляемый пакетный прогон с дельта-обработкой.

Каркас, который не сгорит на 200 000 записях:
  * состояние в SQLite: pending / running / done / failed, попытки, отпечаток
  * дельта-обработка: неизменившиеся записи не обрабатываются повторно
  * пробная пачка перед большим прогоном
  * стоп-кран по доле сбоев (систематическая ошибка != 200 000 записей мусора)
  * идемпотентная запись результатов по custom_id
  * валидация каждого результата по схеме до сохранения

Запуск без ключа (заглушка вместо API):
    python code/batch_runner.py

Запуск на реальном API:
    python code/batch_runner.py --live
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

PROMPT_VERSION = "2026-08-17.1"
DB_PATH = os.environ.get("BATCH_DB", "batch_state.sqlite3")

SYSTEM_STABLE = (
    "Определи категорию обращения и приоритет. "
    "Верни только JSON: {\"category\": один из [billing, tech, sales], "
    "\"priority\": целое 1-3, \"needs_review\": bool}. Без пояснений."
)

SCHEMA_FIELDS = ("category", "priority")
ALLOWED_CATEGORIES = ("billing", "tech", "sales")

# Иллюстративные цены дешёвого тарифа за миллион токенов
PRICE_IN, PRICE_OUT = 0.80, 4.00
BATCH_MULT = 0.50


# --------------------------------------------------------------------------
# Состояние
# --------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS work (
    custom_id   TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    result      TEXT,
    error       TEXT,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS work_status ON work(status);
"""


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(DDL)
    return conn


def fingerprint(content: str) -> str:
    """Отпечаток содержимого вместе с версией промпта.

    Меняется либо документ, либо промпт — значит надо обработать заново.
    Не меняется ничего — значит платить второй раз не за что.
    """
    return hashlib.sha256((content + "|" + PROMPT_VERSION).encode("utf-8")).hexdigest()


@dataclass
class Item:
    custom_id: str
    content: str


def sync_items(conn: sqlite3.Connection, items: Iterable[Item]) -> dict[str, int]:
    """Заводит новые записи и сбрасывает в pending те, у которых изменился отпечаток."""
    stats = {"new": 0, "changed": 0, "skipped_delta": 0}
    now = time.time()
    for it in items:
        fp = fingerprint(it.content)
        row = conn.execute(
            "SELECT fingerprint, status FROM work WHERE custom_id = ?", (it.custom_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO work(custom_id, fingerprint, status, updated_at) "
                "VALUES(?, ?, 'pending', ?)",
                (it.custom_id, fp, now),
            )
            stats["new"] += 1
        elif row[0] != fp:
            conn.execute(
                "UPDATE work SET fingerprint = ?, status = 'pending', attempts = 0, "
                "error = NULL, updated_at = ? WHERE custom_id = ?",
                (fp, now, it.custom_id),
            )
            stats["changed"] += 1
        elif row[1] == "done":
            stats["skipped_delta"] += 1
    conn.commit()
    return stats


def take_pending(conn: sqlite3.Connection, limit: int, max_attempts: int = 3) -> list[str]:
    rows = conn.execute(
        "SELECT custom_id FROM work WHERE status IN ('pending', 'failed') "
        "AND attempts < ? ORDER BY updated_at LIMIT ?",
        (max_attempts, limit),
    ).fetchall()
    return [r[0] for r in rows]


def validate(raw: str) -> tuple[bool, Any, str]:
    """Формальная валидация результата. Ноль стоимости, ловит большинство сбоев."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, None, "invalid_json"
    if not isinstance(data, dict):
        return False, None, "not_an_object"
    for f in SCHEMA_FIELDS:
        if f not in data:
            return False, None, f"missing:{f}"
    if data["category"] not in ALLOWED_CATEGORIES:
        return False, None, "bad_category"
    try:
        pr = int(data["priority"])
    except (TypeError, ValueError):
        return False, None, "priority_not_int"
    if not 1 <= pr <= 3:
        return False, None, "priority_out_of_range"
    if data.get("needs_review") is True:
        return False, data, "needs_review"
    return True, data, "ok"


# --------------------------------------------------------------------------
# Прогон
# --------------------------------------------------------------------------

# Функция отправки пачки: (список запросов) -> список результатов
# результат: {"custom_id": str, "ok": bool, "raw": str, "usage": dict}
SendBatch = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


def build_request(custom_id: str, content: str) -> dict[str, Any]:
    """Одинаковая шапка на всю пачку -> кэш работает ВНУТРИ батча."""
    return {
        "custom_id": custom_id,
        "params": {
            "model": os.environ.get("BATCH_MODEL", "claude-haiku-4-5"),
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": SYSTEM_STABLE,
                 "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": content[:4000]}],
        },
    }


@dataclass
class RunStats:
    sent: int = 0
    done: int = 0
    failed: int = 0
    needs_review: int = 0
    cost_batch: float = 0.0
    cost_interactive_equivalent: float = 0.0

    def failure_share(self) -> float:
        return (self.failed / self.sent) if self.sent else 0.0


def absorb(conn: sqlite3.Connection, results: list[dict[str, Any]], stats: RunStats) -> None:
    """Идемпотентный разбор результатов: upsert по custom_id, а не insert."""
    now = time.time()
    for r in results:
        cid = r["custom_id"]
        if not r.get("ok"):
            conn.execute(
                "UPDATE work SET status='failed', attempts=attempts+1, error=?, "
                "updated_at=? WHERE custom_id=?",
                (str(r.get("error", "api_error")), now, cid),
            )
            stats.failed += 1
            continue

        ok, data, why = validate(r["raw"])
        u = r.get("usage", {})
        cost_full = (
            u.get("input_tokens", 0) * PRICE_IN
            + u.get("cache_read_input_tokens", 0) * PRICE_IN * 0.10
            + u.get("cache_creation_input_tokens", 0) * PRICE_IN * 1.25
            + u.get("output_tokens", 0) * PRICE_OUT
        ) / 1_000_000
        naive = (
            (u.get("input_tokens", 0)
             + u.get("cache_read_input_tokens", 0)
             + u.get("cache_creation_input_tokens", 0)) * PRICE_IN
            + u.get("output_tokens", 0) * PRICE_OUT
        ) / 1_000_000
        stats.cost_batch += cost_full * BATCH_MULT
        stats.cost_interactive_equivalent += naive

        if ok:
            conn.execute(
                "UPDATE work SET status='done', result=?, error=NULL, "
                "attempts=attempts+1, updated_at=? WHERE custom_id=?",
                (json.dumps(data, ensure_ascii=False), now, cid),
            )
            stats.done += 1
        else:
            conn.execute(
                "UPDATE work SET status='failed', attempts=attempts+1, error=?, "
                "updated_at=? WHERE custom_id=?",
                (why, now, cid),
            )
            stats.failed += 1
            if why == "needs_review":
                stats.needs_review += 1
    conn.commit()


def run(
    conn: sqlite3.Connection,
    items: list[Item],
    send: SendBatch,
    batch_size: int = 500,
    probe_size: int = 20,
    max_failure_share: float = 0.05,
) -> RunStats:
    sync = sync_items(conn, items)
    print(f"новых: {sync['new']}, изменившихся: {sync['changed']}, "
          f"пропущено дельтой: {sync['skipped_delta']}")

    by_id = {it.custom_id: it.content for it in items}
    stats = RunStats()

    # 1. Пробная пачка. Цена пробы — тысячная доля процента прогона,
    #    цена её отсутствия — весь прогон мусора.
    probe_ids = take_pending(conn, probe_size)
    if probe_ids:
        probe = [build_request(cid, by_id[cid]) for cid in probe_ids]
        stats.sent += len(probe)
        absorb(conn, send(probe), stats)
        share = stats.failure_share()
        print(f"пробная пачка: {len(probe)} записей, сбоев {share:.1%}")
        if share > max_failure_share:
            print("СТОП-КРАН: доля сбоев выше порога. Чинить промпт, а не повторять.")
            return stats

    # 2. Основной прогон пачками.
    while True:
        ids = take_pending(conn, batch_size)
        if not ids:
            break
        requests = [build_request(cid, by_id[cid]) for cid in ids]
        before_failed = stats.failed
        stats.sent += len(requests)
        absorb(conn, send(requests), stats)
        batch_fail_share = (stats.failed - before_failed) / len(requests)
        if batch_fail_share > max_failure_share:
            print(f"СТОП-КРАН на пачке: сбоев {batch_fail_share:.1%}")
            break

    return stats


# --------------------------------------------------------------------------
# Заглушка и живой отправитель
# --------------------------------------------------------------------------


def fake_send(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Имитация: 90% валидных ответов, 5% мусора, 5% needs_review."""
    out: list[dict[str, Any]] = []
    for i, req in enumerate(requests):
        cid = req["custom_id"]
        first_in_batch = i == 0
        usage = {
            "input_tokens": 60,
            "cache_creation_input_tokens": 1200 if first_in_batch else 0,
            "cache_read_input_tokens": 0 if first_in_batch else 1200,
            "output_tokens": 28,
        }
        mod = hash(cid) % 20
        if mod == 0:
            raw = "Конечно! Категория: billing."           # мусор вместо JSON
        elif mod == 1:
            raw = json.dumps({"category": "billing", "priority": 2, "needs_review": True})
        else:
            raw = json.dumps({"category": "tech", "priority": 2, "needs_review": False})
        out.append({"custom_id": cid, "ok": True, "raw": raw, "usage": usage})
    return out


def live_send(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Отправка через реальный пакетный API. Требует ключ в окружении."""
    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        time.sleep(10)   # опрос по расписанию, не в бесконечном цикле

    out: list[dict[str, Any]] = []
    for item in client.messages.batches.results(batch.id):
        if item.result.type == "succeeded":
            msg = item.result.message
            raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            out.append({
                "custom_id": item.custom_id,
                "ok": True,
                "raw": raw,
                "usage": {
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": msg.usage.output_tokens,
                    "cache_read_input_tokens": getattr(
                        msg.usage, "cache_read_input_tokens", 0),
                    "cache_creation_input_tokens": getattr(
                        msg.usage, "cache_creation_input_tokens", 0),
                },
            })
        else:
            out.append({"custom_id": item.custom_id, "ok": False,
                        "error": item.result.type, "raw": "", "usage": {}})
    return out


def demo(live: bool = False) -> None:
    items = [Item(f"doc-{i:04d}", f"Обращение номер {i}: не могу оплатить подписку.")
             for i in range(120)]
    conn = connect(":memory:")
    send = live_send if live else fake_send

    stats = run(conn, items, send, batch_size=50, probe_size=20)

    print()
    print(f"отправлено:      {stats.sent}")
    print(f"успешно:         {stats.done}")
    print(f"сбоев:           {stats.failed} (из них needs_review: {stats.needs_review})")
    print(f"цена батчем:     {stats.cost_batch:.6f} $")
    print(f"было бы интерактивно без кэша: {stats.cost_interactive_equivalent:.6f} $")
    if stats.cost_batch:
        factor = stats.cost_interactive_equivalent / stats.cost_batch
        print(f"экономия:        x{factor:.1f}")
    print()
    print("Повторный запуск на тех же данных обработает 0 записей:")
    stats2 = run(conn, items, send, batch_size=50, probe_size=20)
    print(f"отправлено при повторе: {stats2.sent} (дельта-обработка)")


def _self_test() -> None:
    assert fingerprint("a") == fingerprint("a")
    assert fingerprint("a") != fingerprint("b")

    ok, data, why = validate('{"category": "tech", "priority": 2}')
    assert ok and data["priority"] == 2 and why == "ok"
    assert validate("не json")[2] == "invalid_json"
    assert validate('{"category": "other", "priority": 1}')[2] == "bad_category"
    assert validate('{"category": "tech", "priority": 9}')[2] == "priority_out_of_range"
    assert validate('{"category": "tech", "priority": 1, "needs_review": true}')[2] == "needs_review"

    conn = connect(":memory:")
    items = [Item("a", "text"), Item("b", "text")]
    s1 = sync_items(conn, items)
    assert s1["new"] == 2
    assert len(take_pending(conn, 10)) == 2
    conn.execute("UPDATE work SET status='done' WHERE custom_id='a'")
    conn.commit()
    s2 = sync_items(conn, items)
    assert s2["skipped_delta"] == 1, s2
    s3 = sync_items(conn, [Item("a", "text-changed")])
    assert s3["changed"] == 1
    print("self-test: ок")


if __name__ == "__main__":
    _self_test()
    print()
    demo(live="--live" in sys.argv[1:])
