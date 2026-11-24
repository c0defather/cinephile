import json
import os
import sys

from dotenv import load_dotenv

import show
from kino import CLOUD_MODEL, LOCAL_MODEL, make_clients, pick


def judge(expect, outcome, ids):
    """Пройден ли запрос: True/False, None если ожиданий нет."""
    if not expect:
        return None
    if expect.get("no_match"):
        return outcome == "no_match"
    if outcome != "ok":
        return False
    if expect.get("must") and not set(expect["must"]) & set(ids):
        return False
    return not set(expect.get("forbid", [])) & set(ids)


def journal_rows():
    try:
        return show.load()
    except FileNotFoundError:
        return []


def run(queries, films, client, model):
    """Прогон набора через pick. Возвращает (результаты, чужие id, нарушения условий): числа из журнала."""
    before = len(journal_rows())
    results = []
    for item in queries:
        outcome, ids = pick(client, model, item["query"], films)
        results.append({"query": item["query"], "outcome": outcome, "ids": ids,
                        "passed": judge(item.get("expect"), outcome, ids)})
    new_rows = journal_rows()[before:]
    alien = sum(len(row.get("rejected", [])) for row in new_rows)
    violations = sum(len(row.get("violations", [])) for row in new_rows)
    return results, alien, violations


def report(results, alien, violations, model):
    print(f"\nМодель: {model}, запросов {len(results)}")
    outcomes = {}
    for item in results:
        outcomes[item["outcome"]] = outcomes.get(item["outcome"], 0) + 1
    for outcome, count in sorted(outcomes.items()):
        print(f"  {outcome}: {count}")
    print(f"Чужих id поймано: {alien}")
    print(f"Нарушений условий поймано: {violations}")
    judged = [item for item in results if item["passed"] is not None]
    if not judged:
        print("Подбор: ожиданий нет")
        return
    passed = sum(1 for item in judged if item["passed"])
    print(f"Подбор: пройдено {passed} из {len(judged)} запросов с ожиданием "
          f"({100 * passed // len(judged)}%)")
    for item in judged:
        if not item["passed"]:
            print(f"  не пройден: {item['query']} -> {item['outcome']} {item['ids']}")


def choose(cloud, local, local_only):
    """Облачная модель, если есть ключ и не просили --local; иначе локальная."""
    if cloud is not None and not local_only:
        return cloud, CLOUD_MODEL
    return local, LOCAL_MODEL


if __name__ == "__main__":
    load_dotenv()
    with open("films.json", encoding="utf-8") as file:
        films = json.load(file)
    with open("queries.json", encoding="utf-8") as file:
        queries = json.load(file)
    cloud, local = make_clients(os.environ)
    client, model = choose(cloud, local, "--local" in sys.argv)
    results, alien, violations = run(queries, films, client, model)
    report(results, alien, violations, model)
