"""Каталог films.json соответствует SPEC §3."""

import json
from pathlib import Path

import pytest

FIELDS = {"id": str, "title": str, "year": int, "minutes": int, "genres": list,
          "countries": list, "languages": list, "description": str}


@pytest.fixture(scope="module")
def films():
    return json.loads(Path("films.json").read_text(encoding="utf-8"))


def test_size(films):
    assert 40 <= len(films) <= 60


def test_fields_and_types(films):
    for film in films:
        for field, kind in FIELDS.items():
            assert isinstance(film[field], kind), (film["id"], field)
            assert not isinstance(film[field], bool)
        for field in ("genres", "countries", "languages"):
            assert film[field] and all(isinstance(x, str) for x in film[field]), (film["id"], field)


def test_ids_are_unique_strings(films):
    ids = [film["id"] for film in films]
    assert len(ids) == len(set(ids))


def test_kazakhstan_films(films):
    kazakh = [f for f in films if "Казахстан" in f["countries"]]
    assert len(kazakh) >= 5
    assert any("казахский" in f["languages"] for f in kazakh)


def test_values_are_plausible(films):
    for film in films:
        assert 1900 <= film["year"] <= 2026, film["id"]
        assert 40 <= film["minutes"] <= 300, film["id"]
        assert film["description"].strip(), film["id"]


def test_vocabulary_is_russian_catalog_words(films):
    for film in films:
        for word in film["genres"] + film["languages"]:
            assert not word.endswith(" язык") and not word.endswith(" фильм"), (film["id"], word)
