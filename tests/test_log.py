"""Журнал calls.jsonl (SPEC §4.3)."""

import json
import time
from types import SimpleNamespace

from openai.types import CompletionUsage

from kino import PRICE, log

FILMS = [{"id": "t-1", "title": "Фильм один", "description": "Длинное описание " * 50},
         {"id": "t-2", "title": "Фильм два", "description": "Ещё одно описание " * 50}]


def rows(tmp_path):
    lines = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def completion(prompt=1000, answer=100):
    usage = CompletionUsage(prompt_tokens=prompt, completion_tokens=answer, total_tokens=prompt + answer)
    return SimpleNamespace(id="chatcmpl-1", usage=usage)


def test_row_has_required_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log("gpt-6-luna", "choice", "доброе кино", FILMS, "ok", time.time(), completion())
    row = rows(tmp_path)[0]
    assert {"time", "model", "step", "request", "outcome", "seconds", "usage", "usd"} <= set(row)
    assert row["model"] == "gpt-6-luna" and row["step"] == "choice" and row["outcome"] == "ok"
    assert row["usage"]["prompt_tokens"] == 1000


def test_request_has_query_and_ids_but_not_the_catalog(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log("m", "choice", "доброе кино", FILMS, "ok", time.time())
    request = rows(tmp_path)[0]["request"]
    assert "доброе кино" in request and "t-1" in request and "t-2" in request
    assert "Длинное описание" not in request and "\\u" not in request


def test_usd_by_price_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log("gpt-6-luna", "choice", "q", FILMS, "ok", time.time(), completion(1000, 100))
    price_in, price_out = PRICE["gpt-6-luna"]
    assert rows(tmp_path)[0]["usd"] == (1000 * price_in + 100 * price_out) / 1_000_000


def test_local_model_costs_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log("qwen3.5:4b", "choice", "q", FILMS, "ok", time.time(), completion())
    assert rows(tmp_path)[0]["usd"] == 0


def test_invalid_has_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log("m", "choice", "q", FILMS, "invalid", time.time(), completion(), rejected=["f-999"])
    assert rows(tmp_path)[0]["rejected"] == ["f-999"]


def test_no_rejected_for_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log("m", "choice", "q", FILMS, "ok", time.time(), completion())
    assert "rejected" not in rows(tmp_path)[0]


def test_failure_without_answer_has_no_usage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log("m", "choice", "q", FILMS, "no_connection", time.time())
    row = rows(tmp_path)[0]
    assert "usage" not in row and row["outcome"] == "no_connection"


def test_rows_are_appended(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log("m", "filters", "q", FILMS, "ok", time.time())
    log("m", "choice", "q", FILMS, "ok", time.time())
    assert [r["step"] for r in rows(tmp_path)] == ["filters", "choice"]
