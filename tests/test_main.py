"""Запуск и резерв на локальную модель (SPEC §5.3)."""

import pytest

import kino


class Recorder:
    """Подмена kino.pick: отвечает по сценарию и запоминает, куда ходили."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, client, model, query, films):
        self.calls.append((client, model))
        outcome = self.outcomes.pop(0)
        return outcome, (["t-1"] if outcome == "ok" else [])


CLOUD, LOCAL = object(), object()


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # pick пишет calls.jsonl в текущую папку


def test_cloud_ok_does_not_touch_local(monkeypatch):
    fake = Recorder("ok")
    monkeypatch.setattr(kino, "pick", fake)
    assert kino.answer("q", [], CLOUD, LOCAL) == ("ok", ["t-1"], kino.CLOUD_MODEL)
    assert fake.calls == [(CLOUD, kino.CLOUD_MODEL)]


@pytest.mark.parametrize("outcome", ["no_connection", "error_500", "error_502", "error_503"])
def test_cloud_failure_falls_back_to_local(monkeypatch, outcome):
    fake = Recorder(outcome, "ok")
    monkeypatch.setattr(kino, "pick", fake)
    assert kino.answer("q", [], CLOUD, LOCAL) == ("ok", ["t-1"], kino.LOCAL_MODEL)
    assert fake.calls == [(CLOUD, kino.CLOUD_MODEL), (LOCAL, kino.LOCAL_MODEL)]


@pytest.mark.parametrize("outcome", ["no_match", "invalid", "refusal", "length", "error_401", "error_429"])
def test_other_outcomes_stay_as_they_are(monkeypatch, outcome):
    fake = Recorder(outcome)
    monkeypatch.setattr(kino, "pick", fake)
    assert kino.answer("q", [], CLOUD, LOCAL)[0] == outcome
    assert len(fake.calls) == 1


def test_without_cloud_goes_straight_to_local(monkeypatch):
    fake = Recorder("ok")
    monkeypatch.setattr(kino, "pick", fake)
    assert kino.answer("q", [], None, LOCAL)[2] == kino.LOCAL_MODEL
    assert fake.calls == [(LOCAL, kino.LOCAL_MODEL)]


def test_pick_itself_does_not_fall_back():
    """Резерв живёт в answer, а pick возвращает no_connection как есть."""
    import httpx2
    import openai
    from types import SimpleNamespace

    def parse(**_):
        raise openai.APIConnectionError(request=httpx2.Request("POST", "http://127.0.0.1:9/v1"))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    assert kino.pick(client, "m", "q", [{"id": "t-1"}])[0] == "no_connection"


def test_clients_from_environment():
    cloud, local = kino.make_clients({"OPENAI_API_KEY": "test-key", "OLLAMA_BASE_URL": "http://host:1234/v1"})
    assert cloud is not None and cloud.api_key == "test-key"
    assert str(local.base_url).startswith("http://host:1234/v1")


def test_no_key_means_no_cloud_client():
    cloud, local = kino.make_clients({})
    assert cloud is None
    assert str(local.base_url).startswith("http://127.0.0.1:11434/v1")
