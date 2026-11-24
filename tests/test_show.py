"""show.py: итоги по журналу calls.jsonl (SPEC §5.4)."""

import json

import show


def row(model="m1", outcome="ok", step="choice", prompt=1000, answer=100, usd=0.001):
    result = {"model": model, "step": step, "outcome": outcome, "request": "q"}
    if prompt is not None:
        result["usage"] = {"prompt_tokens": prompt, "completion_tokens": answer,
                           "total_tokens": prompt + answer}
        result["usd"] = usd
    return result


def test_outcomes_are_counted_per_model():
    rows = [row("m1"), row("m1", "no_match"), row("m1"), row("m2", "invalid")]
    total = show.summarize(rows)
    assert total["m1"]["outcomes"] == {"ok": 2, "no_match": 1}
    assert total["m2"]["outcomes"] == {"invalid": 1}


def test_tokens_and_price_are_summed():
    total = show.summarize([row(prompt=1000, answer=100, usd=0.5), row(prompt=500, answer=50, usd=0.25)])
    assert total["m1"]["tokens_in"] == 1500
    assert total["m1"]["tokens_out"] == 150
    assert total["m1"]["usd"] == 0.75


def test_row_without_usage_counts_as_zero_tokens():
    total = show.summarize([row(outcome="no_connection", prompt=None)])
    assert total["m1"]["tokens_in"] == 0 and total["m1"]["usd"] == 0
    assert total["m1"]["outcomes"] == {"no_connection": 1}


def test_price_of_one_request_is_the_mean():
    total = show.summarize([row(usd=0.5), row(usd=0.25)])
    assert total["m1"]["usd_per_request"] == 0.375


def test_request_is_not_a_call():
    """Запрос Rare состоит из двух вызовов, но считается один раз (по step choice)."""
    rows = [row(step="filters"), row(step="choice"), row(step="filters"), row(step="choice")]
    total = show.summarize(rows)
    assert total["m1"]["requests"] == 2
    assert total["m1"]["calls"] == 4
    assert total["m1"]["outcomes"] == {"ok": 2}


def test_failed_filters_end_the_request():
    rows = [row(step="filters", outcome="invalid"), row(step="filters"), row(step="choice")]
    total = show.summarize(rows)
    assert total["m1"]["requests"] == 2
    assert total["m1"]["outcomes"] == {"invalid": 1, "ok": 1}


def test_price_includes_every_call_of_a_request():
    rows = [row(step="filters", usd=0.25), row(step="choice", usd=0.5)]
    total = show.summarize(rows)
    assert total["m1"]["usd"] == 0.75
    assert total["m1"]["usd_per_request"] == 0.75


def test_empty_journal():
    assert show.summarize([]) == {}


def test_load_skips_blank_lines(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps(row()) + "\n\n" + json.dumps(row("m2")) + "\n", encoding="utf-8")
    assert [r["model"] for r in show.load(path)] == ["m1", "m2"]


def test_print_report(capsys):
    show.report(show.summarize([row("gpt-6-luna", usd=0.0002), row("qwen3.5:4b", "no_match", usd=0)]))
    out = capsys.readouterr().out
    assert "gpt-6-luna" in out and "qwen3.5:4b" in out
    assert "ok" in out and "no_match" in out and "$" in out


def test_report_for_empty_journal(capsys):
    show.report({})
    assert "пуст" in capsys.readouterr().out


def test_import_is_silent(capsys):
    import importlib
    importlib.reload(show)
    assert capsys.readouterr().out == ""
