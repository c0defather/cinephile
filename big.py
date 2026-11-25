"""Два замера Epic на films_big.json.

    uv run big.py            # замер 1 без сети: в скольких запросах фильм из must попал в 30 кандидатов
    uv run big.py --run      # замер 2 на облачной модели: кандидаты против всего каталога в промпте
"""

import json
import os
import sys

from dotenv import load_dotenv

import check
from kino import CLOUD_MODEL, candidates, choose, make_clients, pick_big

JOURNAL = "calls.jsonl"


def recall(queries, films, k=30):
    """(в скольких запросах с must хотя бы один фильм из must в k кандидатах, запросов с must, непопавшие запросы)."""
    hit, total, misses = 0, 0, []
    for item in queries:
        must = (item.get("expect") or {}).get("must")
        if not must:
            continue
        total += 1
        if set(must) & set(candidates(item["query"], films, k)):
            hit += 1
        else:
            misses.append(item["query"])
    return hit, total, misses


def pick_whole(client, model, query, films, effort="none"):
    """Вариант для сравнения: весь каталог в промпте, без отбора кандидатов."""
    return choose(client, model, query, films, films, None, effort)


def journal_size():
    try:
        with open(JOURNAL, encoding="utf-8") as file:
            return sum(1 for line in file if line.strip())
    except FileNotFoundError:
        return 0


def since(before, requests):
    """Токены, цена и секунды по строкам журнала, добавленным после before."""
    with open(JOURNAL, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()][before:]
    tokens = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows)
    usd = sum(r.get("usd", 0) for r in rows)
    seconds = sum(r.get("seconds", 0) for r in rows)
    return {"tokens_in": tokens, "usd": usd, "seconds": seconds, "per_request_tokens": tokens / requests,
            "per_request_usd": usd / requests, "per_request_seconds": seconds / requests}


def run_variant(name, function, queries, films, client, model):
    before = journal_size()
    results = []
    for item in queries:
        outcome, ids = function(client, model, item["query"], films)
        results.append({"query": item["query"], "outcome": outcome, "ids": ids,
                        "passed": check.judge(item.get("expect"), outcome, ids)})
    judged = [r for r in results if r["passed"] is not None]
    passed = sum(1 for r in judged if r["passed"])
    summary = since(before, len(queries))
    print(f"\n{name}: пройдено {passed} из {len(judged)} ({100 * passed // len(judged)}%), "
          f"токенов на запрос {summary['per_request_tokens']:.0f}, цена запроса ${summary['per_request_usd']:.6f}, "
          f"секунд на запрос {summary['per_request_seconds']:.1f}")
    for r in judged:
        if not r["passed"]:
            print(f"  не пройден: {r['query']} -> {r['outcome']} {r['ids']}")
    return passed, len(judged), summary


if __name__ == "__main__":
    load_dotenv()
    with open("films_big.json", encoding="utf-8") as file:
        films = json.load(file)
    with open("queries.json", encoding="utf-8") as file:
        queries = json.load(file)
    hit, total, misses = recall(queries, films)
    print(f"Замер 1: фильм из must в 30 кандидатах в {hit} запросах из {total} ({100 * hit // total}%)")
    for query in misses:
        print(f"  мимо: {query}")
    if "--run" in sys.argv:
        cloud, _ = make_clients(os.environ)
        if cloud is None:
            sys.exit("Нет OPENAI_API_KEY: замер 2 идёт на облачной модели")
        run_variant("30 кандидатов", pick_big, queries, films, cloud, CLOUD_MODEL)
        run_variant("весь каталог в промпте", pick_whole, queries, films, cloud, CLOUD_MODEL)
