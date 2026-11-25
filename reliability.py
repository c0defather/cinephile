"""Надёжность: набор queries.json несколько раз на одной модели.

    uv run reliability.py [--local] [--times 5]

Доля запросов, пройденных во всех прогонах, рядом с долей в одном прогоне. Итог в reliability.json,
вызовы в calls.jsonl.
"""

import json
import os
import sys

from dotenv import load_dotenv

from check import choose, run
from kino import make_clients

TIMES = 5


def measure(queries, films, client, model, times=TIMES):
    """times прогонов набора через pick; для каждого прогона список {query, outcome, ids, passed}."""
    runs = []
    for number in range(1, times + 1):
        print(f"Прогон {number} из {times}")
        results, _, _ = run(queries, films, client, model)
        runs.append(results)
    return runs


def summarize(runs):
    """Счёт по запросам с ожиданием: во всех прогонах, хотя бы в одном, средняя доля одного прогона."""
    judged = [item["query"] for item in runs[0] if item["passed"] is not None]
    passed = {q: [next(i for i in r if i["query"] == q)["passed"] for r in runs] for q in judged}
    per_run = [sum(1 for q in judged if passed[q][n]) for n in range(len(runs))]
    return {"queries": len(judged), "runs": len(runs), "in_all": sum(all(v) for v in passed.values()),
            "in_any": sum(any(v) for v in passed.values()), "per_run": per_run,
            "one_run_share": sum(per_run) / (len(judged) * len(runs)),
            "unstable": [q for q, v in passed.items() if any(v) and not all(v)]}


def report(summary, model):
    total, n = summary["queries"], summary["runs"]
    print(f"\nМодель: {model}, запросов с ожиданием {total}, прогонов {n}")
    print(f"Пройдено по прогонам: {summary['per_run']}")
    print(f"Доля в одном прогоне (в среднем): {100 * summary['one_run_share']:.0f}%")
    print(f"Доля во всех пяти: {summary['in_all']} из {total} ({100 * summary['in_all'] // total}%)"
          if n == 5 else f"Доля во всех {n}: {summary['in_all']} из {total} ({100 * summary['in_all'] // total}%)")
    print(f"Хотя бы в одном: {summary['in_any']} из {total}")
    for query in summary["unstable"]:
        print(f"  нестабильный: {query}")


if __name__ == "__main__":
    load_dotenv()
    argv = sys.argv[1:]
    times = int(argv[argv.index("--times") + 1]) if "--times" in argv else TIMES
    with open("films.json", encoding="utf-8") as file:
        films = json.load(file)
    with open("queries.json", encoding="utf-8") as file:
        queries = json.load(file)
    cloud, local = make_clients(os.environ)
    client, model = choose(cloud, local, "--local" in argv)
    summary = summarize(measure(queries, films, client, model, times))
    report(summary, model)
    with open("reliability.json", "w", encoding="utf-8") as file:
        json.dump({"model": model, **summary}, file, ensure_ascii=False, indent=1)
