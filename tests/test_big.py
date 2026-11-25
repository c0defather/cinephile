"""big.py: два замера для Epic (SPEC §7)."""

import json

import pytest

import big
import kino

FILMS = [{"id": f"t-{n}", "title": f"Фильм {n}", "year": 2000 + n, "minutes": 80 + 10 * n, "genres": ["драма"],
          "countries": ["США"], "languages": ["английский"], "description": f"Описание {n}."} for n in range(1, 6)]


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_recall_counts_queries_whose_must_film_is_among_candidates():
    queries = [{"query": "фильм до 100 минут", "expect": {"must": ["t-1", "t-2"]}},  # t-1, t-2 влезают
               {"query": "фильм до 100 минут", "expect": {"must": ["t-5"]}},  # t-5 длинный: не в кандидатах
               {"query": "любой", "expect": {"no_match": True}},  # без must не считается
               {"query": "любой"}]
    hit, total, misses = big.recall(queries, FILMS, k=30)
    assert (hit, total) == (1, 2) and misses == ["фильм до 100 минут"]


def test_recall_respects_k():
    queries = [{"query": "фильм", "expect": {"must": ["t-5"]}}]
    assert big.recall(queries, FILMS, k=1)[:2] in {(0, 1), (1, 1)}
    assert big.recall(queries, FILMS, k=5)[:2] == (1, 1)


def test_summarize_variant_takes_only_new_rows(monkeypatch):
    kino.log("m", "choice", "старое", [{"id": "t-1"}], "ok", 0)
    before = big.journal_size()
    row = {"model": "m", "step": "choice", "request": "q", "outcome": "ok", "seconds": 2.0,
           "usage": {"prompt_tokens": 1000, "completion_tokens": 10}, "usd": 0.5}
    with open("calls.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n" + json.dumps(dict(row, seconds=4.0, usd=0.25)) + "\n")
    summary = big.since(before, requests=2)
    assert summary == {"tokens_in": 2000, "usd": 0.75, "seconds": 6.0, "per_request_tokens": 1000,
                       "per_request_usd": 0.375, "per_request_seconds": 3.0}


def test_variants_use_different_prompts(monkeypatch):
    seen = {}

    def fake_choose(client, model, query, candidates, films, filters, effort):
        seen.setdefault("sizes", []).append(len(candidates))
        return "no_match", []

    monkeypatch.setattr(kino, "choose", fake_choose)
    monkeypatch.setattr(big, "choose", fake_choose)
    big.pick_whole("c", "m", "q", FILMS)
    assert seen["sizes"] == [len(FILMS)]  # весь каталог целиком
