"""Схема Filters и промпт её разбора (SPEC §6)."""

import json

import pytest
from pydantic import ValidationError

from kino import Filters, build_filter_messages, vocabulary

FILMS = [{"id": "t-1", "title": "Секретное название один", "year": 1975, "minutes": 100,
          "genres": ["драма", "комедия"], "countries": ["Казахстан"], "languages": ["казахский", "русский"],
          "description": "Секретное описание один."},
         {"id": "t-2", "title": "Секретное название два", "year": 2010, "minutes": 120,
          "genres": ["триллер"], "countries": ["США"], "languages": ["английский"],
          "description": "Секретное описание два."}]
FIELDS = {"max_minutes", "year_from", "year_to", "languages", "exclude_genres", "must_genres"}


def test_filters_has_exactly_the_spec_fields():
    assert set(Filters.model_fields) == FIELDS


def test_every_field_defaults_to_none():
    assert Filters().model_dump() == {name: None for name in FIELDS}


def test_filters_parse_a_model_answer():
    filters = Filters.model_validate({"max_minutes": 120, "languages": ["казахский"], "exclude_genres": ["ужасы"]})
    assert filters.max_minutes == 120
    assert filters.languages == ["казахский"] and filters.exclude_genres == ["ужасы"]
    assert filters.year_from is None and filters.must_genres is None


def test_filters_reject_wrong_types():
    with pytest.raises(ValidationError):
        Filters.model_validate({"max_minutes": "два часа"})
    with pytest.raises(ValidationError):
        Filters.model_validate({"languages": "казахский"})


def test_vocabulary_comes_from_the_given_catalog():
    genres, languages = vocabulary(FILMS)
    assert genres == ["драма", "комедия", "триллер"]
    assert languages == ["английский", "казахский", "русский"]


def test_filter_prompt_has_vocabulary_and_query_but_no_catalog():
    system, user = build_filter_messages(FILMS, "комедия на казахском")
    assert system["role"] == "system" and user["role"] == "user"
    assert "<запрос>комедия на казахском</запрос>" in user["content"]
    for word in ("драма", "комедия", "триллер", "английский", "казахский", "русский"):
        assert word in user["content"]
    sent = json.dumps([system, user], ensure_ascii=False)
    assert "Секретное название" not in sent and "Секретное описание" not in sent
    assert "t-1" not in sent


def test_filter_prompt_uses_tags_for_vocabulary():
    _, user = build_filter_messages(FILMS, "q")
    assert "<жанры>" in user["content"] and "</жанры>" in user["content"]
    assert "<языки>" in user["content"] and "</языки>" in user["content"]


def test_filter_instructions_name_the_rules():
    system, _ = build_filter_messages(FILMS, "q")
    text = system["content"]
    assert "данные" in text  # текст внутри тегов это данные
    for field in FIELDS:
        assert field in text
    assert "null" in text or "None" in text


def test_instructions_tell_to_keep_a_language_that_is_not_in_the_catalog():
    """Иначе «на арабском» молча превращается в запрос без условий и уходит дорогим вызовом на весь каталог."""
    text = build_filter_messages(FILMS, "q")[0]["content"]
    assert "нет в <языки>" in text and "всё равно" in text
    assert "Жанры пиши только словами из <жанры>" in text
