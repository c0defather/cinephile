"""Надёжность: набор 5 раз на одной модели, доля «во всех пяти» и «в одном» (SPEC §8)."""

import json

import pytest
from test_pick import FILMS, Client, answer, in_tmp  # noqa: F401

import reliability

QUERIES = [{"query": "а", "expect": {"must": ["t-1"]}},
           {"query": "б", "expect": {"must": ["t-2"]}},
           {"query": "в", "expect": {"no_match": True}},
           {"query": "без ожиданий"}]


def run(passed_by_query):
    """Прогоны из таблицы: для запроса список из N булевых. Остальные поля не важны."""
    count = len(next(iter(passed_by_query.values())))
    return [[{"query": q, "passed": passed[i]} for q, passed in passed_by_query.items()] for i in range(count)]


def test_all_five_versus_one_run():
    runs = run({"а": [True] * 5, "б": [True, True, False, True, True], "в": [False] * 5})
    summary = reliability.summarize(runs)
    assert summary["queries"] == 3 and summary["runs"] == 5
    assert summary["in_all"] == 1  # только «а»
    assert summary["in_any"] == 2  # «а» и «б»
    assert summary["per_run"] == [2, 2, 1, 2, 2]  # сколько запросов пройдено в каждом прогоне
    assert summary["one_run_share"] == pytest.approx(9 / 15)
    assert summary["unstable"] == ["б"]


def test_queries_without_expectation_are_not_counted():
    runs = [[{"query": "x", "passed": None}, {"query": "а", "passed": True}]] * 5
    assert reliability.summarize(runs)["queries"] == 1


def test_measure_runs_the_set_n_times_and_logs_every_call(tmp_path):
    client = Client(answer("t-1"))
    runs = reliability.measure(QUERIES, FILMS, client, "stub-model", times=3)
    assert len(runs) == 3 and all(len(r) == len(QUERIES) for r in runs)
    assert runs[0][0]["passed"] is True and runs[0][1]["passed"] is False  # t-1 есть в выдаче «а», нет в «б»
    rows = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum(1 for r in rows if json.loads(r)["step"] == "choice") >= 3 * len(QUERIES)


def test_report_has_both_shares(capsys):
    summary = reliability.summarize(run({"а": [True] * 5, "б": [True, False, True, True, True]}))
    reliability.report(summary, "stub-model")
    out = capsys.readouterr().out
    assert "во всех пяти" in out and "в одном прогоне" in out and "stub-model" in out
