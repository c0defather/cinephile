"""Опыт с памятью модели (SPEC §5.7): без каталога, расхождения считает код."""

import json
from types import SimpleNamespace

import httpx2
import openai
import pytest
from openai.types import CompletionUsage

import memory_test as memory

FILMS = [{"id": f"t-{n}", "title": f"Фильм {n}", "year": 2000 + n, "minutes": 90 + n,
          "genres": ["драма"], "countries": ["США"], "languages": ["английский"],
          "description": f"Секретное описание {n}."} for n in range(1, 53)]


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def entries(films, shift_year=(), shift_minutes=()):
    return [memory.Entry(title=f["title"],
                         year=f["year"] + (1 if f["id"] in shift_year else 0),
                         minutes=f["minutes"] + (7 if f["id"] in shift_minutes else 0))
            for f in films]


class Client:
    def __init__(self, result):
        self.result, self.kwargs = result, None
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        usage = CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        message = SimpleNamespace(parsed=self.result, refusal=None)
        return SimpleNamespace(id="c-1", usage=usage,
                               choices=[SimpleNamespace(message=message, finish_reason="stop")])


def test_sample_is_ten_films_spread_over_the_catalog():
    chosen = memory.sample(FILMS)
    assert len(chosen) == 10
    assert len({f["id"] for f in chosen}) == 10
    assert chosen[0]["id"] == "t-1" and chosen[-1]["id"] != "t-10"


def test_compare_counts_every_difference():
    chosen = memory.sample(FILMS)
    answer = entries(chosen, shift_year={chosen[0]["id"], chosen[3]["id"]}, shift_minutes={chosen[3]["id"]})
    rows, wrong_year, wrong_minutes = memory.compare(chosen, answer)
    assert (wrong_year, wrong_minutes) == (2, 1)
    assert rows[3]["catalog_year"] == chosen[3]["year"]
    assert rows[3]["model_year"] == chosen[3]["year"] + 1


def test_missing_answer_counts_as_a_difference():
    chosen = memory.sample(FILMS)
    rows, wrong_year, wrong_minutes = memory.compare(chosen, entries(chosen)[:8])
    assert (wrong_year, wrong_minutes) == (2, 2)
    assert rows[9]["model_year"] is None


def test_ask_sends_titles_but_no_catalog():
    chosen = memory.sample(FILMS)
    client = Client(memory.Memory(films=entries(chosen)))
    memory.ask(client, "stub-model", chosen)
    sent = json.dumps(client.kwargs["messages"], ensure_ascii=False)
    assert all(f["title"] in sent for f in chosen)
    assert "Секретное описание" not in sent and "t-1" not in sent
    assert client.kwargs["model"] == "stub-model"
    assert client.kwargs["reasoning_effort"] == "none"
    assert client.kwargs["response_format"] is memory.Memory


def test_ask_returns_answers_and_writes_the_journal(tmp_path):
    chosen = memory.sample(FILMS)
    outcome, answer = memory.ask(Client(memory.Memory(films=entries(chosen))), "m", chosen)
    assert outcome == "ok" and len(answer) == 10
    row = json.loads((tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["step"] == "memory" and row["model"] == "m" and row["outcome"] == "ok"
    assert row["usage"]["prompt_tokens"] == 100


def test_ask_survives_connection_error(tmp_path):
    request = httpx2.Request("POST", "http://127.0.0.1:9/v1")
    chosen = memory.sample(FILMS)
    outcome, answer = memory.ask(Client(openai.APIConnectionError(request=request)), "m", chosen)
    assert (outcome, answer) == ("no_connection", [])
    assert json.loads((tmp_path / "calls.jsonl").read_text(encoding="utf-8"))["outcome"] == "no_connection"


def test_report_prints_totals(capsys):
    chosen = memory.sample(FILMS)
    rows, wrong_year, wrong_minutes = memory.compare(chosen, entries(chosen, shift_year={chosen[0]["id"]}))
    memory.report("m", rows, wrong_year, wrong_minutes)
    out = capsys.readouterr().out
    assert chosen[0]["title"] in out
    assert "Расхождений в годе: 1 из 10" in out
    assert "Расхождений в минутах: 0 из 10" in out
