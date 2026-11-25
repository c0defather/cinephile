"""pick_big: модели уходят только кандидаты, проверки Common те же (SPEC §7, E1)."""

import json
from types import SimpleNamespace

import pytest
from openai.types import CompletionUsage

from kino import Choice, pick_big

# 200 фильмов: каждый пятый на корейском, остальные на английском
BIG = [{"id": f"b-{n}", "title": f"Большой фильм {n}", "year": 1990 + n % 30, "minutes": 80 + n % 100,
        "genres": ["драма"], "countries": ["США"], "languages": ["корейский" if n % 5 == 0 else "английский"],
        "description": f"Описание номер {n}."} for n in range(1, 201)]


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def answer(*ids, status="ok"):
    parsed = Choice(status=status, reason="проверка", picks=[{"film_id": i, "why": f"причина {i}"} for i in ids])
    usage = CompletionUsage(prompt_tokens=300, completion_tokens=30, total_tokens=330)
    message = SimpleNamespace(parsed=parsed, refusal=None, content="{}")
    return SimpleNamespace(id="chatcmpl-1", usage=usage, choices=[SimpleNamespace(message=message, finish_reason="stop")])


class Client:
    def __init__(self, result):
        self.result, self.calls = result, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def rows(tmp_path):
    return [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]


def run(result, query="фильм на корейском языке"):
    client = Client(result)
    return pick_big(client, "stub-model", query, BIG), client


def test_model_gets_only_the_candidates_not_the_catalog():
    (outcome, ids), client = run(answer("b-5"))
    sent = client.calls[0]["messages"][1]["content"]
    assert (outcome, ids) == ("ok", ["b-5"])
    assert "Большой фильм 5\"" in sent and "Большой фильм 7\"" not in sent  # корейские есть, английских нет
    assert sent.count('"id"') <= 30 and sent.count('"id"') / len(BIG) <= 0.2


def test_one_model_call_and_the_film_is_printed_from_the_catalog(capsys):
    _, client = run(answer("b-10"))
    assert len(client.calls) == 1
    out = capsys.readouterr().out
    assert "Большой фильм 10" in out and "Описание номер 10." in out and "причина b-10" in out


def test_alien_id_shows_nothing(capsys, tmp_path):
    (outcome, ids), _ = run(answer("b-5", "x-1"))
    assert (outcome, ids) == ("invalid", [])
    assert "Большой фильм" not in capsys.readouterr().out
    assert rows(tmp_path)[-1]["rejected"] == ["x-1"]


def test_real_film_that_was_not_offered_is_a_violation(capsys, tmp_path):
    (outcome, ids), _ = run(answer("b-5", "b-7"))  # b-7 есть в каталоге, но не среди кандидатов
    assert (outcome, ids) == ("invalid", [])
    assert "Большой фильм" not in capsys.readouterr().out
    row = rows(tmp_path)[-1]
    assert row["rejected"] == [] and row["violations"] == ["b-7"]


def test_no_candidates_means_no_match_without_a_call(capsys, tmp_path):
    outcome, client = run(answer("b-5"), query="фильм на арабском языке")
    assert outcome == ("no_match", []) and client.calls == []
    assert "Ничего не подходит" in capsys.readouterr().out
    assert [(r["step"], r["outcome"]) for r in rows(tmp_path)] == [("choice", "no_match")]


def test_model_can_say_no_match(capsys):
    (outcome, ids), _ = run(answer(status="no_match"))
    assert (outcome, ids) == ("no_match", [])
    assert "Ничего не подходит" in capsys.readouterr().out


def test_log_row_has_the_candidates_not_the_catalog(tmp_path):
    run(answer("b-5"))
    row = rows(tmp_path)[-1]
    assert row["step"] == "choice" and row["model"] == "stub-model"
    ids = json.loads(row["request"].split("каталог:")[1])
    assert 0 < len(ids) <= 30 and "b-7" not in ids


def test_failures_keep_the_common_outcomes():
    import httpx2
    import openai
    error = openai.APIConnectionError(request=httpx2.Request("POST", "http://127.0.0.1:9/v1"))
    (outcome, ids), _ = run(error)
    assert (outcome, ids) == ("no_connection", [])
