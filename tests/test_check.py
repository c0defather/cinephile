"""check.py: прогон queries.json через pick (SPEC §5.5) и флаг --local (SPEC §5.3)."""

import pytest

import check
import kino


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ---------- judge: пройден ли запрос ----------

def test_no_expectation_is_not_judged():
    assert check.judge(None, "ok", ["a"]) is None
    assert check.judge({}, "ok", ["a"]) is None


def test_no_match_expected():
    assert check.judge({"no_match": True}, "no_match", []) is True
    assert check.judge({"no_match": True}, "ok", ["a"]) is False


def test_must_needs_at_least_one_id():
    want = {"must": ["a", "b"]}
    assert check.judge(want, "ok", ["x", "b"]) is True
    assert check.judge(want, "ok", ["x"]) is False


def test_forbid_rejects_any_listed_id():
    want = {"must": ["a"], "forbid": ["z"]}
    assert check.judge(want, "ok", ["a", "z"]) is False
    assert check.judge({"forbid": ["z"]}, "ok", ["a"]) is True


def test_expected_films_need_outcome_ok():
    assert check.judge({"must": ["a"]}, "no_match", []) is False
    assert check.judge({"must": ["a"]}, "invalid", []) is False
    assert check.judge({"forbid": ["z"]}, "no_connection", []) is False


# ---------- run: прогон набора ----------

QUERIES = [{"query": "один", "expect": {"must": ["t-1"]}},
           {"query": "два", "expect": {"no_match": True}},
           {"query": "три"}]


def fake_pick(script):
    calls = []

    def pick(client, model, query, films):
        calls.append((client, model, query))
        outcome, ids = script[query]
        if outcome == "invalid":  # как настоящий pick: чужой id попал в журнал
            kino.log(model, "choice", query, films, "invalid", 0, rejected=["x-9"])
        return outcome, ids

    pick.calls = calls
    return pick


def test_run_judges_every_query(monkeypatch):
    fake = fake_pick({"один": ("ok", ["t-1"]), "два": ("ok", ["t-2"]), "три": ("no_match", [])})
    monkeypatch.setattr(check, "pick", fake)
    results, _ = check.run(QUERIES, [{"id": "t-1"}], "client", "model")
    assert [r["passed"] for r in results] == [True, False, None]
    assert [r["outcome"] for r in results] == ["ok", "ok", "no_match"]
    assert [c[:2] for c in fake.calls] == [("client", "model")] * 3


def test_run_counts_alien_ids_from_the_journal(monkeypatch):
    fake = fake_pick({"один": ("invalid", []), "два": ("invalid", []), "три": ("ok", ["t-1"])})
    monkeypatch.setattr(check, "pick", fake)
    _, alien = check.run(QUERIES, [{"id": "t-1"}], "client", "model")
    assert alien == 2


def test_old_journal_rows_are_not_counted(monkeypatch):
    kino.log("m", "choice", "старое", [{"id": "t-1"}], "invalid", 0, rejected=["x-1", "x-2"])
    fake = fake_pick({"один": ("ok", ["t-1"]), "два": ("no_match", []), "три": ("ok", ["t-1"])})
    monkeypatch.setattr(check, "pick", fake)
    assert check.run(QUERIES, [{"id": "t-1"}], "client", "model")[1] == 0


# ---------- report ----------

def test_report_counts_outcomes_and_picks_separately(capsys):
    results = [{"query": "один", "outcome": "ok", "ids": ["t-9"], "passed": False},
               {"query": "два", "outcome": "no_match", "ids": [], "passed": True},
               {"query": "три", "outcome": "ok", "ids": ["t-1"], "passed": None},
               {"query": "четыре", "outcome": "invalid", "ids": [], "passed": False}]
    check.report(results, alien=3, model="qwen")
    out = capsys.readouterr().out
    assert "qwen" in out
    assert "ok: 2" in out and "no_match: 1" in out and "invalid: 1" in out
    assert "Чужих id поймано: 3" in out
    assert "1 из 3" in out  # из трёх запросов с ожиданием пройден один
    assert "один" in out and "четыре" in out  # непройденные названы


def test_report_without_expectations(capsys):
    check.report([{"query": "а", "outcome": "ok", "ids": ["t-1"], "passed": None}], 0, "m")
    assert "ожиданий нет" in capsys.readouterr().out


# ---------- выбор модели ----------

def test_default_choice_is_cloud_when_key_exists():
    cloud, local = object(), object()
    assert check.choose(cloud, local, local_only=False) == (cloud, kino.CLOUD_MODEL)
    assert check.choose(None, local, local_only=False) == (local, kino.LOCAL_MODEL)
    assert check.choose(cloud, local, local_only=True) == (local, kino.LOCAL_MODEL)


def test_parse_args_of_kino():
    assert kino.parse_args(["запрос"]) == (False, "запрос")
    assert kino.parse_args(["--local", "запрос"]) == (True, "запрос")
    assert kino.parse_args(["запрос", "--local"]) == (True, "запрос")
    assert kino.parse_args(["--local"]) == (True, "")
    assert kino.parse_args([]) == (False, "")


def test_import_is_silent(capsys):
    import importlib
    importlib.reload(check)
    assert capsys.readouterr().out == ""
