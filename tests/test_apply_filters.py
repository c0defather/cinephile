"""Отбор по фильтрам и пересверка выбранного фильма (SPEC §6)."""

from kino import Filters, apply_filters, matches

FILMS = [
    {"id": "a", "year": 1975, "minutes": 80, "genres": ["комедия"], "languages": ["русский"]},
    {"id": "b", "year": 1990, "minutes": 100, "genres": ["драма", "комедия"], "languages": ["казахский", "русский"]},
    {"id": "c", "year": 2005, "minutes": 120, "genres": ["ужасы", "триллер"], "languages": ["английский"]},
    {"id": "d", "year": 2020, "minutes": 150, "genres": ["драма"], "languages": ["казахский"]},
]


def ids(films):
    return [film["id"] for film in films]


def test_empty_filters_keep_everything():
    assert ids(apply_filters(FILMS, Filters())) == ["a", "b", "c", "d"]


def test_max_minutes_is_inclusive():
    assert ids(apply_filters(FILMS, Filters(max_minutes=100))) == ["a", "b"]


def test_year_bounds_are_inclusive():
    assert ids(apply_filters(FILMS, Filters(year_from=1990, year_to=2005))) == ["b", "c"]
    assert ids(apply_filters(FILMS, Filters(year_from=2005))) == ["c", "d"]
    assert ids(apply_filters(FILMS, Filters(year_to=1975))) == ["a"]


def test_languages_match_any_of_the_asked():
    assert ids(apply_filters(FILMS, Filters(languages=["казахский"]))) == ["b", "d"]
    assert ids(apply_filters(FILMS, Filters(languages=["английский", "русский"]))) == ["a", "b", "c"]


def test_exclude_genres_drops_a_film_with_any_of_them():
    assert ids(apply_filters(FILMS, Filters(exclude_genres=["комедия"]))) == ["c", "d"]
    assert ids(apply_filters(FILMS, Filters(exclude_genres=["ужасы", "драма"]))) == ["a"]


def test_must_genres_need_every_asked_genre():
    assert ids(apply_filters(FILMS, Filters(must_genres=["комедия"]))) == ["a", "b"]
    assert ids(apply_filters(FILMS, Filters(must_genres=["драма", "комедия"]))) == ["b"]


def test_unknown_words_follow_their_meaning():
    """Жанра или языка нет в каталоге: «нужен» даёт пусто, «не нужен» ничего не меняет."""
    assert apply_filters(FILMS, Filters(languages=["хинди"])) == []
    assert apply_filters(FILMS, Filters(must_genres=["мультфильм"])) == []
    assert ids(apply_filters(FILMS, Filters(exclude_genres=["мультфильм"]))) == ["a", "b", "c", "d"]


def test_conditions_combine():
    wanted = Filters(max_minutes=120, languages=["казахский", "русский"], exclude_genres=["ужасы"])
    assert ids(apply_filters(FILMS, wanted)) == ["a", "b"]


def test_word_case_and_spaces_do_not_matter():
    assert ids(apply_filters(FILMS, Filters(languages=[" Казахский "]))) == ["b", "d"]
    assert ids(apply_filters(FILMS, Filters(exclude_genres=["КОМЕДИЯ"]))) == ["c", "d"]


def test_empty_lists_mean_no_condition():
    assert ids(apply_filters(FILMS, Filters(languages=[], exclude_genres=[], must_genres=[]))) == ["a", "b", "c", "d"]


def test_catalog_is_not_changed_and_order_is_kept():
    before = [dict(film) for film in FILMS]
    apply_filters(FILMS, Filters(max_minutes=100))
    assert FILMS == before


def test_matches_rechecks_one_film():
    long_film = FILMS[3]
    assert matches(long_film, Filters(max_minutes=100)) is False
    assert matches(long_film, Filters(max_minutes=150)) is True
    assert matches(long_film, Filters()) is True
