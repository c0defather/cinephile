"""Диалог: «ещё» без повторов, исключения до конца разговора (SPEC §8)."""

import pytest
from test_pick import FILMS, Client, answer, in_tmp  # noqa: F401

from kino import new_dialog, talk


def prompt(client):
    return " ".join(m["content"] for m in client.kwargs["messages"])


def catalog(client):
    """Только каталог из промпта: в тексте запроса название исключённого фильма может стоять."""
    return prompt(client).split("<каталог>")[-1].split("</каталог>")[0]


def say(client, state, text, films=FILMS):
    return talk(client, "stub-model", state, text, films)


def test_more_never_repeats_shown_films(capsys):
    state = new_dialog()
    first = Client(answer("t-1", "t-2"))
    assert say(first, state, "доброе кино") == ("ok", ["t-1", "t-2"])
    second = Client(answer("t-3"))
    assert say(second, state, "ещё") == ("ok", ["t-3"])
    sent = prompt(second)
    assert "Проверочный фильм 1" not in sent and "Проверочный фильм 2" not in sent
    assert "Проверочный фильм 3" in sent
    assert "доброе кино" in sent  # «ещё» это тот же запрос


def test_shown_stay_hidden_for_the_whole_conversation():
    state = new_dialog()
    say(Client(answer("t-1")), state, "кино")
    say(Client(answer("t-2")), state, "ещё")
    third = Client(answer("t-3"))
    say(third, state, "ещё")
    assert "Проверочный фильм 1" not in prompt(third) and "Проверочный фильм 2" not in prompt(third)


def test_model_repeating_a_shown_film_is_rejected(capsys):
    state = new_dialog()
    say(Client(answer("t-1")), state, "кино")
    outcome, ids = say(Client(answer("t-1")), state, "ещё")  # t-1 модели не показывали
    assert (outcome, ids) == ("invalid", [])


def test_exclusion_lasts_until_the_end(capsys):
    state = new_dialog()
    client = Client(answer("t-2"))
    say(client, state, "кино, уже смотрел Проверочный фильм 1")
    assert "Проверочный фильм 1" not in catalog(client)
    later = Client(answer("t-3"))
    say(later, state, "совсем другое кино")
    assert "Проверочный фильм 1" not in prompt(later)


def test_exclusion_alone_reruns_the_last_query():
    state = new_dialog()
    say(Client(answer("t-1")), state, "доброе кино")
    client = Client(answer("t-2"))
    assert say(client, state, "Проверочный фильм 3 уже видел") == ("ok", ["t-2"])
    assert "доброе кино" in prompt(client) and "Проверочный фильм 3" not in prompt(client)


def test_exclusion_without_a_query_asks_for_one(capsys):
    state = new_dialog()
    client = Client(answer("t-1"))
    assert say(client, state, "уже смотрел Проверочный фильм 1") == ("excluded", [])
    assert client.calls == []  # модель не звали
    assert "Учту" in capsys.readouterr().out


def test_more_without_a_query_asks_for_one(capsys):
    client = Client(answer("t-1"))
    assert say(client, new_dialog(), "ещё") == ("no_query", [])
    assert client.calls == []


def test_catalog_runs_out(capsys):
    state = new_dialog()
    for film in FILMS:
        say(Client(answer(film["id"])), state, "кино")
    client = Client(answer("t-1"))
    assert say(client, state, "ещё") == ("exhausted", [])
    assert client.calls == []
    assert "закончились" in capsys.readouterr().out


def test_clarify_answer_joins_the_query():
    state = new_dialog()
    assert say(Client(answer(status="clarify")), state, "что-нибудь")[0] == "clarify"
    client = Client(answer("t-1"))
    say(client, state, "грустное")
    assert "что-нибудь" in prompt(client) and "грустное" in prompt(client)


def test_failed_answer_shows_nothing_new_and_keeps_state():
    state = new_dialog()
    say(Client(answer("t-1")), state, "кино")
    assert say(Client(answer("t-9")), state, "ещё")[0] == "invalid"
    assert state["shown"] == ["t-1"]


@pytest.mark.parametrize("word", ["ещё", "еще", "Ещё!", "ЕЩЁ"])
def test_more_spellings(word):
    state = new_dialog()
    say(Client(answer("t-1")), state, "кино")
    assert say(Client(answer("t-2")), state, word) == ("ok", ["t-2"])


def test_first_request_may_clarify_but_the_answer_to_a_question_may_not():
    """Живой случай qwen3.5:4b: на «комедия» после вопроса модель спросила снова (замер шага 29)."""
    from kino import Choice, FinalChoice
    state = new_dialog()
    first = Client(answer(status="clarify"))
    say(first, state, "посоветуй фильм")
    assert first.kwargs["response_format"] is Choice
    second = Client(answer("t-1"))
    say(second, state, "комедия")
    assert second.kwargs["response_format"] is FinalChoice
    assert "Уточнять нельзя" in prompt(second)
    third = Client(answer("t-2"))
    say(third, state, "ещё")  # вопрос задан и отвечен: уточнять снова можно
    assert third.kwargs["response_format"] is Choice and "Уточнять нельзя" not in prompt(third)


def test_second_clarify_in_a_row_is_invalid_not_shown(capsys):
    state = new_dialog()
    say(Client(answer(status="clarify")), state, "посоветуй фильм")
    outcome, ids = say(Client(answer(status="clarify")), state, "комедия")
    assert (outcome, ids) == ("invalid", [])
    assert not state["asked"]  # вопросы по кругу не копятся


def test_final_choice_schema_has_no_clarify():
    from kino import FinalChoice
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        FinalChoice.model_validate({"status": "clarify", "reason": "?", "picks": []})
    assert FinalChoice.model_validate({"status": "no_match", "reason": "нет", "picks": []})
