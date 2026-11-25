"""Статус clarify: один уточняющий вопрос, фильмов нет (SPEC §8, L1)."""

import json

from test_pick import FILMS, answer, in_tmp, last_row, run  # noqa: F401

from kino import INSTRUCTIONS, Choice


def test_schema_accepts_clarify_without_picks():
    choice = Choice.model_validate({"status": "clarify", "reason": "Какое настроение?", "picks": []})
    assert choice.status == "clarify" and choice.picks == []


def test_clarify_prints_the_question_and_shows_no_film(capsys):
    (outcome, ids), _ = run(answer(status="clarify"), query="что-нибудь")
    assert (outcome, ids) == ("clarify", [])
    out = capsys.readouterr().out
    assert "проверка" in out  # вопрос это reason модели
    assert not any(film["title"] in out for film in FILMS)


def test_clarify_with_picks_is_invalid(capsys):
    (outcome, ids), _ = run(answer("t-1", status="clarify"))
    assert (outcome, ids) == ("invalid", [])
    assert FILMS[0]["title"] not in capsys.readouterr().out


def test_clarify_without_question_is_invalid():
    parsed = Choice(status="clarify", reason="  ", picks=[])
    assert run(answer(parsed=parsed))[0] == ("invalid", [])


def test_clarify_is_logged(tmp_path):
    run(answer(status="clarify"), model="m-1")
    row = last_row(tmp_path)
    assert row["step"] == "choice" and row["outcome"] == "clarify" and row["model"] == "m-1"


def test_instructions_tell_when_to_clarify():
    assert "clarify" in INSTRUCTIONS


def test_instructions_keep_no_match_for_a_film_that_is_not_in_the_catalog():
    """Без этого запрос «посоветуй Аватар 7» уходил в clarify вместо no_match (замер шага 28)."""
    assert "конкретный фильм" in INSTRUCTIONS and "status no_match" in INSTRUCTIONS.split("конкретный фильм")[1]
