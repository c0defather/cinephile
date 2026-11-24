"""pick с заглушкой вместо клиента, без сети (SPEC §4.1, §5.2, §5.3)."""

import json
from types import SimpleNamespace

import httpx2
import openai
import pytest
from openai.types import CompletionUsage
from pydantic import ValidationError

from kino import Choice, pick

FILMS = [{"id": f"t-{n}", "title": f"Проверочный фильм {n}", "year": 2000 + n, "minutes": 90 + n,
          "genres": ["драма"], "countries": ["Казахстан"], "languages": ["казахский"],
          "description": f"Описание номер {n}."} for n in range(1, 7)]
REQUEST = httpx2.Request("POST", "http://127.0.0.1:9/v1/chat/completions")


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def answer(*ids, status="ok", refusal=None, parsed="auto"):
    if parsed == "auto":
        parsed = Choice(status=status, reason="проверка",
                        picks=[{"film_id": i, "why": f"причина {i}"} for i in ids])
    usage = CompletionUsage(prompt_tokens=300, completion_tokens=30, total_tokens=330)
    message = SimpleNamespace(parsed=parsed, refusal=refusal, content="{}")
    return SimpleNamespace(id="chatcmpl-1", usage=usage,
                           choices=[SimpleNamespace(message=message, finish_reason="stop")])


class Client:
    def __init__(self, result):
        self.result, self.kwargs = result, None
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def run(result, query="доброе кино", films=FILMS, model="stub-model"):
    client = Client(result)
    return pick(client, model, query, films), client


def last_row(tmp_path):
    return json.loads((tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()[-1])


def test_good_answer_is_shown_from_catalog(capsys):
    (outcome, ids), _ = run(answer("t-2", "t-4"))
    printed = capsys.readouterr().out
    assert (outcome, ids) == ("ok", ["t-2", "t-4"])
    for film in (FILMS[1], FILMS[3]):
        assert film["title"] in printed and str(film["year"]) in printed
        assert str(film["minutes"]) in printed and film["description"] in printed
    assert "причина t-2" in printed


def test_alien_id_hides_everything(capsys, tmp_path):
    (outcome, ids), _ = run(answer("t-1", "x-999"))
    assert (outcome, ids) == ("invalid", [])
    assert FILMS[0]["title"] not in capsys.readouterr().out
    assert last_row(tmp_path)["rejected"] == ["x-999"]


def test_repeat_is_invalid(capsys):
    assert run(answer("t-1", "t-1"))[0] == ("invalid", [])
    assert FILMS[0]["title"] not in capsys.readouterr().out


FOUR = {"status": "ok", "reason": "", "picks": [{"film_id": f"t-{n}", "why": "причина"} for n in range(1, 5)]}


def test_schema_does_not_allow_four_films():
    with pytest.raises(ValidationError):
        Choice.model_validate(FOUR)


def test_schema_tells_the_model_about_the_limit():
    """Без maxItems qwen3.5:4b иногда перечисляет 20 фильмов и не укладывается в время."""
    assert Choice.model_json_schema()["properties"]["picks"]["maxItems"] == 3


def test_four_films_from_the_model_are_invalid(capsys):
    try:
        Choice.model_validate(FOUR)
    except ValidationError as error:
        assert run(error)[0] == ("invalid", [])
        assert FILMS[0]["title"] not in capsys.readouterr().out
    else:
        pytest.fail("схема приняла четыре фильма")


def test_empty_ok_is_invalid():
    assert run(answer())[0] == ("invalid", [])


def test_no_match(capsys):
    (outcome, ids), _ = run(answer(status="no_match"))
    assert (outcome, ids) == ("no_match", [])
    assert "Ничего не подходит" in capsys.readouterr().out


def test_no_match_with_picks_shows_nothing(capsys):
    assert run(answer("t-1", status="no_match"))[0] == ("no_match", [])
    assert FILMS[0]["title"] not in capsys.readouterr().out


def test_refusal():
    assert run(answer(parsed=None, refusal="Не могу."))[0] == ("refusal", [])


def test_truncated_answer():
    error = openai.LengthFinishReasonError(completion=answer())
    assert run(error)[0] == ("length", [])


def test_no_connection():
    assert run(openai.APIConnectionError(request=REQUEST))[0] == ("no_connection", [])


def test_timeout_is_no_connection():
    assert run(openai.APITimeoutError(request=REQUEST))[0] == ("no_connection", [])


@pytest.mark.parametrize("code", [429, 500, 503])
def test_api_error_has_code(code):
    response = httpx2.Response(code, request=REQUEST)
    error = openai.APIStatusError(f"Error code: {code}", response=response, body=None)
    assert run(error)[0] == (f"error_{code}", [])


def test_bad_schema_is_invalid():
    try:
        Choice.model_validate({"status": "maybe", "reason": "", "picks": []})
    except Exception as error:
        assert run(error)[0] == ("invalid", [])


def test_unparsed_answer_is_invalid():
    assert run(answer(parsed=None))[0] == ("invalid", [])


def test_works_with_given_catalog_and_model(capsys):
    other = [{"id": "z-1", "title": "Чужой каталог", "year": 1999, "minutes": 100, "genres": ["драма"],
              "countries": ["США"], "languages": ["английский"], "description": "Другое."}]
    (outcome, ids), client = run(answer("z-1"), films=other, model="my-model")
    assert (outcome, ids) == ("ok", ["z-1"])
    assert "Чужой каталог" in capsys.readouterr().out
    assert client.kwargs["model"] == "my-model"
    assert "Проверочный фильм" not in client.kwargs["messages"][1]["content"]


def test_call_arguments():
    _, client = run(answer("t-1"), query="грустное кино")
    assert client.kwargs["response_format"] is Choice
    assert client.kwargs["reasoning_effort"] == "none"
    assert "<запрос>грустное кино</запрос>" in client.kwargs["messages"][1]["content"]


def test_journal_row_for_every_outcome(tmp_path):
    run(answer("t-1"), model="m1")
    run(openai.APIConnectionError(request=REQUEST), model="m2")
    lines = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines]
    assert [(r["model"], r["outcome"], r["step"]) for r in rows] == [
        ("m1", "ok", "choice"), ("m2", "no_connection", "choice")]
    assert "доброе кино" in rows[0]["request"]
