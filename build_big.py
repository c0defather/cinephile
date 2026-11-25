"""Сборка films_big.json: 1000+ фильмов из Wikidata и русской Википедии.

    uv run build_big.py

Поля (год, минуты, страны, языки, жанры) берутся только из Wikidata (CC0), описание это начало статьи
русской Википедии (CC BY-SA 4.0). В начало идут 52 фильма из films.json с теми же id, дальше фильмы Wikidata
с наибольшим числом языковых версий статьи: самые известные. Жанры, страны и языки приводятся к словам
films.json таблицей из build_catalog.py. Ответы сервисов лежат в data/, повторный запуск идёт без сети.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from build_catalog import AGENT, ENDPOINT, LANGUAGES, genres, words

BIG = 1100  # сколько фильмов в итоговом каталоге (вместе с 52 из films.json)
TEXTS = Path("data/wikipedia_big.json")
WIKI = "https://ru.wikipedia.org/w/api.php"

MIN_LINKS = 40  # фильм известен: у статьи не меньше стольких языковых версий
PERIODS = [(1920, 1949), (1950, 1969), (1970, 1979), (1980, 1989), (1990, 1999), (2000, 2004), (2005, 2009),
           (2010, 2014), (2015, 2019), (2020, 2024), (2025, 2030)]


def query(first, last):
    """Один запрос на несколько лет: за всё время Wikidata не успевает ответить."""
    return f"""
SELECT ?film ?label ?year ?article ?links (MIN(?d) AS ?minutes)
  (GROUP_CONCAT(DISTINCT ?cL; separator="|") AS ?countries)
  (GROUP_CONCAT(DISTINCT ?lL; separator="|") AS ?languages)
  (GROUP_CONCAT(DISTINCT ?gL; separator="|") AS ?genres)
WHERE {{
  ?film wdt:P31 wd:Q11424; wikibase:sitelinks ?links; wdt:P2047 ?d; wdt:P577 ?date.
  FILTER(?links >= {MIN_LINKS} && ?d >= 60 && ?d <= 240 && YEAR(?date) >= {first} && YEAR(?date) <= {last})
  BIND(YEAR(?date) AS ?year)
  ?film rdfs:label ?label FILTER(LANG(?label) = "ru").
  ?article schema:about ?film; schema:isPartOf <https://ru.wikipedia.org/>.
  OPTIONAL {{ ?film wdt:P495 ?c. ?c rdfs:label ?cL FILTER(LANG(?cL) = "ru") }}
  OPTIONAL {{ ?film wdt:P364 ?l. ?l rdfs:label ?lL FILTER(LANG(?lL) = "ru") }}
  OPTIONAL {{ ?film wdt:P136 ?g. ?g rdfs:label ?gL FILTER(LANG(?gL) = "ru") }}
}}
GROUP BY ?film ?label ?year ?article ?links"""


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    for attempt in range(5):  # Wikidata изредка отвечает 502/504
        try:
            with urllib.request.urlopen(request, timeout=180) as answer:
                return json.load(answer)
        except urllib.error.HTTPError as error:
            if attempt == 4:
                raise
            time.sleep(90 if error.code == 429 else 15)


def fetch_films():
    rows = []
    Path("data").mkdir(exist_ok=True)
    for first, last in PERIODS:
        part = Path(f"data/wikidata_big_{first}.json")  # кэш по периодам: после сбоя продолжаем с места
        if not part.exists():
            url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query(first, last), "format": "json"})
            part.write_text(json.dumps(get_json(url)["results"]["bindings"], ensure_ascii=False), encoding="utf-8")
            time.sleep(20)
        rows += json.loads(part.read_text(encoding="utf-8"))
    films = {}
    for row in rows:  # у фильма может быть несколько дат выхода: берём первую строку
        films.setdefault(row["film"]["value"], {
            "title": row["label"]["value"], "year": int(row["year"]["value"]),
            "minutes": int(float(row["minutes"]["value"])),
            "genres": row["genres"]["value"], "countries": row["countries"]["value"],
            "languages": row["languages"]["value"], "source": row["film"]["value"],
            "article": row["article"]["value"], "links": int(row["links"]["value"])})
    films = sorted(films.values(), key=lambda f: -f["links"])
    return films


def fetch_texts(films):
    """Начало статьи русской Википедии по заголовку, по 20 статей за запрос."""
    texts = json.loads(TEXTS.read_text(encoding="utf-8")) if TEXTS.exists() else {}
    titles = [urllib.parse.unquote(f["article"].rsplit("/", 1)[1]).replace("_", " ") for f in films]
    todo = [t for t in titles if t not in texts]
    for start in range(0, len(todo), 20):
        batch = todo[start:start + 20]
        params = {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": 20,
                  "redirects": 1, "titles": "|".join(batch), "format": "json"}
        answer = get_json(WIKI + "?" + urllib.parse.urlencode(params))["query"]
        back = {item["to"]: item["from"] for item in answer.get("redirects", [])}
        for page in answer["pages"].values():
            texts[back.get(page["title"], page["title"])] = page.get("extract", "")
        for title in batch:
            texts.setdefault(title, "")
    TEXTS.write_text(json.dumps(texts, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return texts


def describe(text, limit=300):
    """Одно-два первых предложения без скобок и без служебных строк, не длиннее limit."""
    text = re.sub(r"\s*\([^()]*\)", "", text.split("\n")[0]).replace("\xa0", " ").replace("\u0301", "").strip()  # без знаков ударения
    sentences = re.split(r"(?<=[.!?])\s+(?=[«\"A-ZА-ЯЁ])", text)
    result = ""
    for sentence in sentences[:2]:
        if len(result) + len(sentence) > limit and result:
            break
        result = f"{result} {sentence}".strip()
    return result if 40 <= len(result) <= limit + 150 else ""


def estimate_tokens(films, per_char):
    """Сколько токенов займёт каталог в промпте: символы каталога (без source) на токены на символ."""
    catalog = [{k: v for k, v in film.items() if k != "source"} for film in films]
    return round(len(json.dumps(catalog, ensure_ascii=False)) * per_char)


def tokens_per_char(small):
    """Токенов на символ по настоящим вызовам из calls.jsonl, где весь films.json шёл в промпт (Common)."""
    from kino import build_messages
    rows = [json.loads(line) for line in Path("calls.jsonl").read_text(encoding="utf-8").splitlines()]
    used = [r["usage"]["prompt_tokens"] for r in rows
            if r.get("step") == "choice" and r.get("usage") and r["request"].split("каталог:")[1].count("f-") == len(small)]
    chars = sum(len(m["content"]) for m in build_messages(small, ""))
    return sum(used) / len(used) / chars, len(used)


def main():
    small = json.loads(Path("films.json").read_text(encoding="utf-8"))
    known = {f["source"] for f in small}
    candidates = [f for f in fetch_films() if f["source"] not in known]
    texts = fetch_texts(candidates)
    films, lacking = list(small), 0
    for film in candidates:
        title = urllib.parse.unquote(film["article"].rsplit("/", 1)[1]).replace("_", " ")
        row = {"title": film["title"], "year": film["year"], "minutes": film["minutes"],
               "genres": genres(film["genres"]), "countries": words(film["countries"]),
               "languages": words(film["languages"], LANGUAGES), "description": describe(texts.get(title, "")),
               "source": film["source"]}
        if not row["description"] or not row["genres"] or not row["countries"] or not row["languages"]:
            lacking += 1
            continue
        if len(films) >= BIG:
            break
        films.append({"id": f"f-{len(films) + 1:03d}", **row})
    Path("films_big.json").write_text(json.dumps(films, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    chars = len(json.dumps(films, ensure_ascii=False))
    per_char, runs = tokens_per_char(small)
    print(f"films_big.json: {len(films)} фильмов, не вошли {lacking}, {chars} символов")
    print(f"Каталог в промпте: около {estimate_tokens(films, per_char)} токенов "
          f"(films.json: {estimate_tokens(small, per_char)}; оценка по {runs} вызовам из calls.jsonl)")


if __name__ == "__main__":
    main()
