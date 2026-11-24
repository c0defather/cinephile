"""Схема ответа и промпт (SPEC §4.2, §5.1)."""

import json
import re

import pytest
from pydantic import ValidationError

from kino import INSTRUCTIONS, Choice, build_messages

FILMS = [
    {"id": "t-1", "title": "Проверочный фильм один", "year": 2001, "minutes": 100, "genres": ["драма"],
     "countries": ["Казахстан"], "languages": ["казахский"], "description": "Описание один.",
     "source": "https://example.org/1"},
    {"id": "t-2", "title": "Проверочный фильм два", "year": 2002, "minutes": 90, "genres": ["комедия"],
     "countries": ["США"], "languages": ["английский"], "description": "Описание два."},
]


def user_text(query="доброе кино"):
    return build_messages(FILMS, query)[1]["content"]


def test_catalog_in_tag_without_escapes():
    text = user_text()
    inside = re.search(r"<каталог>(.*?)</каталог>", text, re.S).group(1)
    assert json.loads(inside)[0]["id"] == "t-1"
    for film in FILMS:
        assert film["title"] in inside, "нужен ensure_ascii=False"
    assert "\\u" not in inside


def test_source_is_not_sent_to_model():
    assert "example.org" not in user_text()


def test_query_in_its_own_tag():
    assert re.search(r"<запрос>доброе кино</запрос>", user_text())


def test_system_has_data_rule_and_example():
    system = build_messages(FILMS, "q")[0]
    assert system["role"] == "system" and system["content"] == INSTRUCTIONS
    assert "данные" in INSTRUCTIONS and "<каталог>" in INSTRUCTIONS and "<запрос>" in INSTRUCTIONS
    assert INSTRUCTIONS.count("Пример") == 1


def test_example_does_not_use_catalog_titles():
    # пример не должен подсказывать модели настоящие фильмы каталога
    assert "Интерстеллар" not in INSTRUCTIONS


def test_schema_fields():
    assert set(Choice.model_fields) == {"status", "reason", "picks"}
    assert set(Choice.model_fields["picks"].annotation.__args__[0].model_fields) == {"film_id", "why"}


def test_schema_accepts_ok_and_no_match_only():
    Choice.model_validate({"status": "ok", "reason": "", "picks": [{"film_id": "t-1", "why": "подходит"}]})
    Choice.model_validate({"status": "no_match", "reason": "нет", "picks": []})
    with pytest.raises(ValidationError):
        Choice.model_validate({"status": "maybe", "reason": "", "picks": []})
