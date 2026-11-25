"""Процесс (SPEC §8): pre-commit и CI есть, запускают pytest и не требуют ключей."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_process_documents_exist():
    for name in ("SPEC.md", "AGENTS.md", "PLAN.md"):
        assert (ROOT / name).exists()


def test_pre_commit_runs_pytest_before_a_commit():
    text = read(".pre-commit-config.yaml")
    assert "uv run pytest -q" in text
    assert "always_run: true" in text and "pass_filenames: false" in text


def test_pre_commit_blocks_keys_and_env_file():
    text = read(".pre-commit-config.yaml")
    assert "sk-" in text and "language: pygrep" in text and "language: fail" in text
    assert r"(^|/)\.env$" in text


def test_ci_runs_pytest_on_push_and_pull_request():
    text = read(".github/workflows/ci.yml")
    assert "uv run pytest -q" in text
    assert "push:" in text and "pull_request:" in text
    assert "uv sync" in text and "--locked" in text  # uv.lock не должен разойтись с pyproject


def test_ci_has_no_real_keys_and_no_route_to_a_model():
    text = read(".github/workflows/ci.yml")
    assert "secrets." not in text  # у CI нет ключей: модель заменяют заглушки тестов
    assert "OLLAMA_BASE_URL" in text  # случайный настоящий вызов упадёт сразу, а не уйдёт в сеть


def test_ci_actions_are_pinned_to_a_version():
    for use in re.findall(r"uses:\s*(\S+)", read(".github/workflows/ci.yml")):
        assert re.search(r"@v\d+", use), use


def test_env_stays_out_of_git():
    assert ".env" in read(".gitignore").splitlines()
    assert (ROOT / ".env.example").exists()
