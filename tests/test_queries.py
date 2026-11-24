"""Набор запросов queries.json (SPEC §5.5)."""

import json
import re

import pytest

KAZAKH = re.compile(r"[әғқңөұүһі]", re.IGNORECASE)
NEGATION = re.compile(r"(^|\s)(не|без|кроме|емес|жоқ)(\s|$|,)", re.IGNORECASE)


@pytest.fixture(scope="module")
def queries():
    with open("queries.json", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="module")
def ids():
    with open("films.json", encoding="utf-8") as file:
        return {film["id"] for film in json.load(file)}


def expect(item):
    return item.get("expect") or {}


def test_format(queries):
    assert isinstance(queries, list)
    for item in queries:
        assert set(item) <= {"query", "expect"}
        assert isinstance(item["query"], str) and item["query"].strip()
        assert set(expect(item)) <= {"must", "forbid", "no_match"}


def test_queries_are_unique(queries):
    texts = [item["query"] for item in queries]
    assert len(texts) == len(set(texts))


def test_at_least_fifteen(queries):
    assert len(queries) >= 15


def test_no_match_queries(queries):
    assert sum(1 for q in queries if expect(q).get("no_match")) >= 3


def test_queries_with_expected_films(queries):
    assert sum(1 for q in queries if expect(q).get("must")) >= 3


def test_kazakh_queries(queries):
    assert sum(1 for q in queries if KAZAKH.search(q["query"])) >= 3


def test_queries_with_negation(queries):
    assert sum(1 for q in queries if NEGATION.search(q["query"])) >= 3


def test_provocation_query(queries):
    assert any("Игнорируй каталог" in q["query"] for q in queries)


def test_all_expected_ids_are_in_the_catalog(queries, ids):
    used = {i for q in queries for key in ("must", "forbid") for i in expect(q).get(key, [])}
    assert used <= ids


def test_expectations_do_not_contradict(queries):
    for item in queries:
        want = expect(item)
        assert not set(want.get("must", [])) & set(want.get("forbid", []))
        assert not (want.get("no_match") and want.get("must"))
