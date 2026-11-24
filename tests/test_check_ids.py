"""Проверка id кодом (SPEC §5.2)."""

from kino import Pick, check_ids

FILMS = [{"id": f"t-{n}", "title": f"Фильм {n}"} for n in range(1, 7)]


def picks(*ids):
    return [Pick(film_id=i, why="причина") for i in ids]


def test_good_answer():
    assert check_ids(picks("t-1", "t-2"), FILMS) == (None, [])


def test_one_to_three_films_are_fine():
    assert check_ids(picks("t-1"), FILMS)[0] is None
    assert check_ids(picks("t-1", "t-2", "t-3"), FILMS)[0] is None


def test_alien_id_is_rejected_and_listed():
    problem, rejected = check_ids(picks("t-1", "x-999"), FILMS)
    assert problem and rejected == ["x-999"]


def test_all_alien_ids_are_listed():
    assert check_ids(picks("a", "t-1", "b"), FILMS)[1] == ["a", "b"]


def test_repeat():
    problem, rejected = check_ids(picks("t-1", "t-1"), FILMS)
    assert problem and rejected == []


def test_more_than_three():
    problem, _ = check_ids(picks("t-1", "t-2", "t-3", "t-4"), FILMS)
    assert problem


def test_empty_picks_with_ok_is_a_problem():
    assert check_ids([], FILMS)[0]


def test_numeric_catalog_ids():
    films = [{"id": 1}, {"id": 2}]
    assert check_ids(picks("1", "2"), films)[0] is None
    assert check_ids(picks("3"), films) == ("фильма нет в каталоге", ["3"])
