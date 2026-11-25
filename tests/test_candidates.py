"""candidates: до k id из каталога без вызова модели (SPEC §7)."""

import pytest

from kino import candidates

FILMS = [
    {"id": "a-1", "title": "Парк юрского периода", "year": 1993, "minutes": 127, "genres": ["фантастика", "приключения"],
     "countries": ["США"], "languages": ["английский"], "description": "Учёные оживляют динозавров на тропическом острове."},
    {"id": "a-2", "title": "Паразиты", "year": 2019, "minutes": 132, "genres": ["драма", "триллер"],
     "countries": ["Республика Корея"], "languages": ["корейский"], "description": "Бедная семья проникает в дом богатых."},
    {"id": "a-3", "title": "Тюльпан", "year": 2008, "minutes": 100, "genres": ["драма", "комедия"],
     "countries": ["Казахстан"], "languages": ["казахский", "русский"], "description": "Тракторист возвращается в степь."},
    {"id": "a-4", "title": "Иван Васильевич меняет профессию", "year": 1973, "minutes": 88, "genres": ["комедия", "фантастика"],
     "countries": ["СССР"], "languages": ["русский"], "description": "Машина времени отправляет управдома в XVI век."},
    {"id": "a-5", "title": "Гравитация", "year": 2013, "minutes": 91, "genres": ["драма", "фантастика", "триллер"],
     "countries": ["США"], "languages": ["английский"], "description": "Астронавты выживают в космосе после аварии."},
    {"id": "a-6", "title": "Амели", "year": 2001, "minutes": 122, "genres": ["комедия", "мелодрама"],
     "countries": ["Франция"], "languages": ["французский"], "description": "Парижская официантка устраивает чужие судьбы."},
    {"id": "a-7", "title": "Сталкер", "year": 1979, "minutes": 163, "genres": ["драма", "фантастика"],
     "countries": ["СССР"], "languages": ["русский"], "description": "Проводник ведёт писателя через Зону."},
]
IDS = {f["id"] for f in FILMS}


def test_returns_known_ids_without_a_model():
    ids = candidates("любой фильм", FILMS)
    assert ids and set(ids) <= IDS and len(ids) == len(set(ids))


def test_at_most_k():
    assert len(candidates("фильм", FILMS, k=3)) <= 3
    many = [dict(FILMS[0], id=f"m-{n}") for n in range(100)]
    assert len(candidates("фильм", many)) == 30


def test_max_minutes_is_a_hard_filter():
    assert set(candidates("фильм до полутора часов", FILMS)) == {"a-4"}
    assert set(candidates("не длиннее ста минут", FILMS)) == {"a-3", "a-4", "a-5"}
    assert set(candidates("короче 95 минут", FILMS)) == {"a-4", "a-5"}


def test_decade_and_years():
    assert set(candidates("фантастика семидесятых годов", FILMS)) == {"a-4", "a-7"}
    assert set(candidates("фильм после 2010 года", FILMS)) == {"a-2", "a-5"}
    assert set(candidates("кино до 2000", FILMS)) == {"a-1", "a-4", "a-7"}


def test_language_is_a_hard_filter():
    assert candidates("фильм на корейском языке", FILMS) == ["a-2"]
    assert set(candidates("на русском языке", FILMS)) == {"a-3", "a-4", "a-7"}


def test_language_not_in_the_catalog_gives_nothing():
    assert candidates("фильм на арабском языке", FILMS) == []


def test_kazakh_query_means_kazakh_language():
    assert candidates("Қазақ тілінде түсірілген фильм", FILMS) == ["a-3"]


def test_negated_genre_is_excluded():
    ids = candidates("только не комедия", FILMS)
    assert ids and not set(ids) & {"a-3", "a-4", "a-6"}
    ids = candidates("драма без триллеров", FILMS)
    assert "a-2" not in ids and "a-3" in ids


def test_wanted_genre_comes_first():
    assert candidates("комедия", FILMS)[:3] and set(candidates("комедия", FILMS)[:3]) == {"a-3", "a-4", "a-6"}


def test_description_words_rank_first():
    assert candidates("про динозавров", FILMS)[0] == "a-1"
    assert candidates("кто-то в космосе", FILMS)[0] == "a-5"


def test_negated_topic_goes_down():
    assert candidates("фантастика не про космос", FILMS).index("a-5") > candidates("фантастика", FILMS).index("a-5")


def test_country_words_help():
    assert candidates("французское кино", FILMS)[0] == "a-6"
    assert set(candidates("про СССР", FILMS)[:2]) <= {"a-4", "a-7"}


def test_does_not_touch_the_catalog_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # пустая папка: ни films.json, ни сети
    assert candidates("фильм", FILMS)


def test_country_words_in_kazakh_and_soviet():
    assert candidates("Қазақстанда түсірілген фильм", FILMS)[0] == "a-3"
    assert candidates("Советское кино", FILMS)[0] in {"a-4", "a-7"}


def test_negated_country_goes_down():
    ids = candidates("комедия, Голливуд емес", FILMS)
    assert ids.index("a-3") < ids.index("a-1")  # США ниже казахстанской комедии
    ids = candidates("фильм не американский", FILMS)
    assert ids.index("a-6") < ids.index("a-1")
