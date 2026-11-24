"""Сборка films.json из Wikidata.

    uv run build_catalog.py

Год, длительность, страны, языки и жанры берутся только из Wikidata (SPARQL).
Список фильмов (название, год) задан здесь вручную. Описания лежат в descriptions.json:
они написаны по вступлению статьи в русской Википедии, остальные поля модель не трогала.
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
AGENT = "film-catalog-student-project/0.1"

# (название по-русски как в Wikidata, год выхода)
WANTED = [
    ("Интерстеллар", 2014), ("Начало", 2010), ("Матрица", 1999), ("Форрест Гамп", 1994),
    ("Побег из Шоушенка", 1994), ("Крёстный отец", 1972), ("Леон", 1994), ("Один дома", 1990),
    ("Назад в будущее", 1985), ("Шоу Трумана", 1998), ("Зелёная миля", 1999), ("Гравитация", 2013),
    ("Марсианин", 2015), ("Паразиты", 2019), ("Амели", 2001), ("Жизнь прекрасна", 1997),
    ("Ла-Ла Ленд", 2016), ("Король Лев", 1994), ("Тайна Коко", 2017), ("Шрек", 2001),
    ("12 разгневанных мужчин", 1957), ("Психо", 1960), ("Сияние", 1980), ("Чужой", 1979),
    ("Бегущий по лезвию", 1982), ("Семь самураев", 1954), ("В джазе только девушки", 1959),
    ("Титаник", 1997), ("Касабланка", 1942), ("Хороший, плохой, злой", 1966), ("Брат", 1997),
    ("Иван Васильевич меняет профессию", 1973), ("Белое солнце пустыни", 1970),
    ("Москва слезам не верит", 1980), ("Сталкер", 1979), ("Летят журавли", 1957),
    ("Служебный роман", 1977), ("Тёмный рыцарь", 2008), ("Список Шиндлера", 1993),
    ("Молчание ягнят", 1991),
    # казахстанские
    ("Рэкетир", 2007), ("Кочевник", 2005), ("Тюльпан", 2008), ("Айка", 2018), ("Аманат", 2016),
    ("Подарок Сталину", 2008), ("Старик", 2012), ("Томирис", 2019), ("Жаужурек мын бала", 2011),
    ("Прощай, Гульсары!", 2008), ("Монгол", 2007), ("Бизнес по-казахски", 2016),
]

GENRES = [  # (слово каталога, части названия жанра в Wikidata)
    ("драма", ["драм"]), ("комедия", ["комед"]), ("боевик", ["боевик", "экшен"]),
    ("приключения", ["приключен"]), ("фантастика", ["фантаст", "киберпанк"]),
    ("фэнтези", ["фэнтез"]), ("триллер", ["триллер", "саспенс"]), ("ужасы", ["ужас"]),
    ("криминал", ["криминал", "гангстер", "нуар"]), ("мелодрама", ["мелодрам", "романтическ"]),
    ("военный", ["военн"]), ("исторический", ["истор"]), ("биография", ["биограф"]),
    ("мультфильм", ["мультфильм", "анимацион"]), ("семейный", ["семейн", "детск"]),
    ("детектив", ["детектив"]), ("вестерн", ["вестерн"]), ("мюзикл", ["мюзикл"]),
]
EXAMPLE = Path("data/films_kz_example.json")  # пример курса: запись берётся, если в Wikidata нет длительности
LANGUAGES = {"севернокитайский": "китайский", "путунхуа": "китайский"}


def query(wanted):
    values = " ".join(f'("{title}"@ru {year})' for title, year in wanted)
    return f"""
SELECT ?film ?label ?year ?article (MIN(?d) AS ?minutes)
  (GROUP_CONCAT(DISTINCT ?cL; separator="|") AS ?countries)
  (GROUP_CONCAT(DISTINCT ?lL; separator="|") AS ?languages)
  (GROUP_CONCAT(DISTINCT ?gL; separator="|") AS ?genres)
WHERE {{
  VALUES (?label ?year) {{ {values} }}
  ?film rdfs:label ?label; wdt:P31/wdt:P279* wd:Q11424; wdt:P577 ?date; wdt:P2047 ?d.
  FILTER(YEAR(?date) = ?year && ?d >= 40 && ?d <= 300)
  OPTIONAL {{ ?film wdt:P495 ?c. ?c rdfs:label ?cL FILTER(LANG(?cL) = "ru") }}
  OPTIONAL {{ ?film wdt:P364 ?l. ?l rdfs:label ?lL FILTER(LANG(?lL) = "ru") }}
  OPTIONAL {{ ?film wdt:P136 ?g. ?g rdfs:label ?gL FILTER(LANG(?gL) = "ru") }}
  OPTIONAL {{ ?article schema:about ?film; schema:isPartOf <https://ru.wikipedia.org/> }}
}}
GROUP BY ?film ?label ?year ?article"""


def fetch(wanted):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query(wanted), "format": "json"})
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=120) as answer:
        return json.load(answer)["results"]["bindings"]


def words(text, mapping=None):
    result = []
    for word in filter(None, text.split("|")):
        word = word.removesuffix(" язык")
        word = (mapping or {}).get(word, word)
        if word not in result:
            result.append(word)
    return result


def genres(text):
    labels = [x.lower() for x in text.split("|")]
    return [name for name, parts in GENRES if any(p in label for label in labels for p in parts)]


def normalize(row):
    return {
        "title": row["label"]["value"], "year": int(row["year"]["value"]),
        "minutes": int(float(row["minutes"]["value"])),
        "genres": genres(row["genres"]["value"]),
        "countries": words(row["countries"]["value"]),
        "languages": words(row["languages"]["value"], LANGUAGES),
        "source": row["film"]["value"], "article": row.get("article", {}).get("value", ""),
    }


def main():
    rows = {}
    for row in fetch(WANTED):
        rows.setdefault((row["label"]["value"], int(row["year"]["value"])), normalize(row))
    example = {(f["title"], f["year"]): f for f in json.loads(EXAMPLE.read_text(encoding="utf-8"))}
    for key in [x for x in WANTED if x not in rows and x in example]:
        film = example[key]
        rows[key] = {**{k: film[k] for k in ("title", "year", "minutes", "genres", "countries", "languages")},
                     "source": film["source"], "article": ""}
    missing = [x for x in WANTED if x not in rows]
    if missing:
        print("Нет в Wikidata (или без длительности):", missing)
    Path("data").mkdir(exist_ok=True)
    found = [rows[x] for x in WANTED if x in rows]
    Path("data/wikidata.json").write_text(
        json.dumps(found, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    descriptions = json.loads(Path("descriptions.json").read_text(encoding="utf-8")) \
        if Path("descriptions.json").exists() else {}
    films, lacking = [], []
    for film in found:
        text = descriptions.get(f"{film['title']} ({film['year']})")
        problems = [name for name in ("genres", "countries", "languages") if not film[name]]
        if text is None or problems:
            lacking.append((film["title"], film["year"], "нет описания" if text is None else problems))
            continue
        films.append({"id": f"f-{len(films) + 1:03d}", "title": film["title"], "year": film["year"],
                      "minutes": film["minutes"], "genres": film["genres"],
                      "countries": film["countries"], "languages": film["languages"],
                      "description": text, "source": film["source"]})
    if lacking:
        print("Не вошли в films.json:", lacking)
    Path("films.json").write_text(
        json.dumps(films, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"films.json: {len(films)} фильмов")
    return 1 if missing or lacking else 0


if __name__ == "__main__":
    sys.exit(main())
