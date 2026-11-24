"""Импорт kino.py ничего не запускает и не печатает (SPEC §4.1)."""

import importlib


def test_import_is_silent(capsys):
    module = importlib.import_module("kino")
    assert capsys.readouterr().out == ""
    assert callable(module.pick)
