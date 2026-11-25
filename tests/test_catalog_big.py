"""films_big.json: каталог больше промпта (SPEC §7)."""

import json

import pytest

import build_big
from build_catalog import GENRES

FIELDS = {"id", "title", "year", "minutes", "genres", "countries", "languages", "description", "source"}


@pytest.fixture(scope="module")
def big():
    with open("films_big.json", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="module")
def small():
    with open("films.json", encoding="utf-8") as file:
        return json.load(file)


def test_more_than_a_thousand_films(big):
    assert len(big) >= 1000


def test_fields_and_types(big):
    for film in big:
        assert set(film) == FIELDS
        assert isinstance(film["year"], int) and isinstance(film["minutes"], int)
        for name in ("genres", "countries", "languages"):
            assert film[name] and all(isinstance(x, str) for x in film[name])
        assert film["title"].strip() and film["description"].strip()


def test_ids_are_unique_strings(big):
    ids = [film["id"] for film in big]
    assert len(ids) == len(set(ids)) and all(isinstance(i, str) for i in ids)


def test_first_films_are_the_small_catalog(big, small):
    assert big[:len(small)] == small


def test_genres_use_the_catalog_words(big):
    words = {name for name, _ in GENRES}
    assert {g for film in big for g in film["genres"]} <= words


def test_values_look_sane(big):
    assert all(60 <= f["minutes"] <= 240 and 1920 <= f["year"] for f in big)
    assert all(f["source"].startswith("http") for f in big)
    assert all("́" not in f["description"] for f in big)  # без знаков ударения


def test_describe_keeps_one_or_two_sentences():
    text = "«Фильм» (англ. Film) — американская драма 2001 года. Режиссёр — Иван Иванов. Третье предложение.\nВторой абзац."
    assert build_big.describe(text) == "«Фильм» — американская драма 2001 года. Режиссёр — Иван Иванов."
    assert build_big.describe("Коротко.") == ""


def test_token_estimate_scales_with_the_catalog():
    one = build_big.estimate_tokens([{"id": "a", "title": "Название", "description": "Описание фильма."}], 0.5)
    two = build_big.estimate_tokens([{"id": "a", "title": "Название", "description": "Описание фильма."}] * 2, 0.5)
    assert one > 0 and two > one
