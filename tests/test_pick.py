"""pick с заглушкой вместо клиента, без сети (SPEC §4.1, §5.2, §5.3)."""

import json
from types import SimpleNamespace

import httpx2
import openai
import pytest
from openai.types import CompletionUsage
from pydantic import ValidationError

from kino import Choice, Filters, pick

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


def filters_answer(filters):
    usage = CompletionUsage(prompt_tokens=200, completion_tokens=20, total_tokens=220)
    message = SimpleNamespace(parsed=filters, refusal=None, content="{}")
    return SimpleNamespace(id="chatcmpl-f", usage=usage,
                           choices=[SimpleNamespace(message=message, finish_reason="stop")])


class Client:
    """Заглушка: вызов со схемой Filters и вызов выбора отвечают по-разному."""

    def __init__(self, result, filters=None):
        self.result, self.kwargs, self.calls = result, None, []
        self.filters = Filters() if filters is None else filters
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["response_format"] is Filters:
            if isinstance(self.filters, Exception):
                raise self.filters
            return filters_answer(self.filters)
        self.kwargs = kwargs  # последний вызов выбора
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def run(result, query="доброе кино", films=FILMS, model="stub-model", filters=None):
    client = Client(result, filters)
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
    assert [(r["model"], r["step"], r["outcome"]) for r in rows] == [
        ("m1", "filters", "ok"), ("m1", "choice", "ok"),
        ("m2", "filters", "ok"), ("m2", "choice", "no_connection")]
    assert "доброе кино" in rows[0]["request"]


# ---------- Rare: первый вызов разбирает условия, код отбирает фильмы (SPEC §6, R1) ----------

def rows_of(tmp_path):
    lines = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_first_call_asks_for_filters_with_vocabulary_only():
    _, client = run(answer("t-1"))
    first = client.calls[0]
    assert first["response_format"] is Filters
    assert first["reasoning_effort"] == "none"
    sent = json.dumps(first["messages"], ensure_ascii=False)
    assert "<запрос>доброе кино</запрос>" in sent and "драма" in sent
    assert "Проверочный фильм" not in sent and "Описание номер" not in sent


def test_second_call_gets_only_the_films_that_pass_the_filters():
    _, client = run(answer("t-1"), filters=Filters(max_minutes=93))
    prompt = client.kwargs["messages"][1]["content"]
    for n in (1, 2, 3):
        assert f"Проверочный фильм {n}" in prompt
    for n in (4, 5, 6):
        assert f"Проверочный фильм {n}" not in prompt


def test_film_that_breaks_the_filters_is_cut_off(capsys, tmp_path):
    """R1: выбор длинного фильма при max_minutes: фильм не показан, исход invalid."""
    (outcome, ids), client = run(answer("t-6", "t-1"), filters=Filters(max_minutes=93))
    assert (outcome, ids) == ("invalid", [])
    printed = capsys.readouterr().out
    assert FILMS[5]["title"] not in printed and FILMS[0]["title"] not in printed
    assert FILMS[5]["title"] not in client.kwargs["messages"][1]["content"]  # в промпт выбора не ушёл
    row = last_row(tmp_path)
    assert row["outcome"] == "invalid" and row["violations"] == ["t-6"] and row["rejected"] == []


def test_alien_id_is_not_a_violation(tmp_path):
    run(answer("t-1", "x-999"), filters=Filters(max_minutes=93))
    row = last_row(tmp_path)
    assert row["rejected"] == ["x-999"] and "violations" not in row


def test_good_pick_inside_the_filters_is_shown(capsys):
    (outcome, ids), _ = run(answer("t-2"), filters=Filters(max_minutes=93))
    assert (outcome, ids) == ("ok", ["t-2"])
    assert FILMS[1]["title"] in capsys.readouterr().out


def test_nothing_passes_the_filters(capsys, tmp_path):
    (outcome, ids), client = run(answer("t-1"), filters=Filters(max_minutes=10))
    assert (outcome, ids) == ("no_match", [])
    assert "Ничего не подходит" in capsys.readouterr().out
    assert len(client.calls) == 1  # второй вызов не нужен
    # запрос кончается строкой choice, даже когда второй вызов не нужен: так его считают show.py и автопроверка
    rows = rows_of(tmp_path)
    assert [(r["step"], r["outcome"]) for r in rows] == [("filters", "ok"), ("choice", "no_match")]
    assert "usage" not in rows[1] and "t-1" not in rows[1]["request"].split("каталог:")[1]


def test_unknown_language_means_no_match():
    (outcome, ids), client = run(answer("t-1"), filters=Filters(languages=["хинди"]))
    assert (outcome, ids) == ("no_match", []) and len(client.calls) == 1


def test_filters_step_failure_ends_the_request(tmp_path):
    error = openai.APIConnectionError(request=REQUEST)
    (outcome, ids), client = run(answer("t-1"), filters=error)
    assert (outcome, ids) == ("no_connection", [])
    assert len(client.calls) == 1
    assert [(r["step"], r["outcome"]) for r in rows_of(tmp_path)] == [("filters", "no_connection")]


def test_bad_filters_schema_is_invalid(tmp_path):
    try:
        Filters.model_validate({"max_minutes": "много"})
    except ValidationError as error:
        (outcome, ids), _ = run(answer("t-1"), filters=error)
        assert (outcome, ids) == ("invalid", [])
        assert rows_of(tmp_path)[0]["rejected"] == []
    else:
        pytest.fail("схема приняла строку вместо числа")


def test_api_error_on_filters_step_has_code():
    response = httpx2.Response(503, request=REQUEST)
    error = openai.APIStatusError("Error code: 503", response=response, body=None)
    assert run(answer("t-1"), filters=error)[0] == ("error_503", [])


def test_both_steps_are_in_the_journal_with_usage(tmp_path):
    run(answer("t-2"), filters=Filters(max_minutes=93))
    first, second = rows_of(tmp_path)
    assert (first["step"], second["step"]) == ("filters", "choice")
    assert first["usage"]["prompt_tokens"] == 200 and second["usage"]["prompt_tokens"] == 300
    assert first["usd"] == 0 and "доброе кино" in first["request"] and "доброе кино" in second["request"]


def test_choice_row_lists_only_the_candidates(tmp_path):
    run(answer("t-1"), filters=Filters(max_minutes=92))
    first, second = rows_of(tmp_path)
    assert "t-6" in first["request"]  # фильтры разбираются по всему каталогу
    assert "t-2" in second["request"] and "t-6" not in second["request"]


def test_film_id_stays_a_plain_string():
    assert Choice.model_fields["picks"].annotation.__args__[0].model_fields["film_id"].annotation is str
