import json
import os
import sys
import time

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from check import choose
from kino import LIMIT, log, make_clients

COUNT = 10
INSTRUCTIONS = """Ответь по памяти, без каталога и поиска. Для каждого фильма из <фильмы>
назови год выхода и длительность в минутах. Если не уверен, всё равно назови своё лучшее число.
Названия в ответе пиши так же, как в списке."""


class Entry(BaseModel):
    title: str
    year: int
    minutes: int


class Memory(BaseModel):
    films: list[Entry]


def sample(films):
    """10 фильмов через равные промежутки каталога, без случайности."""
    return films[::5][:COUNT]


def ask(client, model, films):
    """Один вызов без каталога: в промпт идут только названия. Возвращает (исход, ответы)."""
    started = time.time()
    query = f"память: год и минуты {len(films)} фильмов"
    titles = json.dumps([film["title"] for film in films], ensure_ascii=False)
    messages = [{"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": f"<фильмы>{titles}</фильмы>"}]

    def end(outcome, completion=None, answers=()):
        log(model, "memory", query, films, outcome, started, completion)
        return outcome, list(answers)

    try:
        completion = client.chat.completions.parse(
            model=model, messages=messages, response_format=Memory,
            max_completion_tokens=LIMIT, reasoning_effort="none")
    except openai.APIConnectionError:
        print("Нет связи с сервером")
        return end("no_connection")
    except openai.APIStatusError as error:
        print("Ошибка API:", error.message)
        return end(f"error_{error.status_code}")
    except openai.LengthFinishReasonError as error:
        print("Ответ обрезан на лимите токенов")
        return end("length", error.completion)
    except ValidationError:
        print("Ответ модели не по схеме")
        return end("invalid")
    parsed = completion.choices[0].message.parsed if completion.choices else None
    if parsed is None:
        print("Ответ модели не по схеме")
        return end("invalid", completion)
    return end("ok", completion, parsed.films)


def compare(films, answers):
    """Сверка с каталогом. Нет ответа на фильм считается расхождением."""
    by_title = {answer.title: answer for answer in answers}
    rows, wrong_year, wrong_minutes = [], 0, 0
    for film in films:
        answer = by_title.get(film["title"])
        row = {"title": film["title"],
               "catalog_year": film["year"], "model_year": answer.year if answer else None,
               "catalog_minutes": film["minutes"], "model_minutes": answer.minutes if answer else None}
        wrong_year += row["model_year"] != row["catalog_year"]
        wrong_minutes += row["model_minutes"] != row["catalog_minutes"]
        rows.append(row)
    return rows, wrong_year, wrong_minutes


def report(model, rows, wrong_year, wrong_minutes):
    print(f"\nМодель: {model}, фильмов {len(rows)}")
    for row in rows:
        print(f"  {row['title']}: год {row['model_year']} (каталог {row['catalog_year']}), "
              f"минут {row['model_minutes']} (каталог {row['catalog_minutes']})")
    print(f"Расхождений в годе: {wrong_year} из {len(rows)}")
    print(f"Расхождений в минутах: {wrong_minutes} из {len(rows)}")


if __name__ == "__main__":
    load_dotenv()
    with open("films.json", encoding="utf-8") as file:
        films = sample(json.load(file))
    cloud, local = make_clients(os.environ)
    client, model = choose(cloud, local, "--local" in sys.argv)
    outcome, answers = ask(client, model, films)
    if outcome == "ok":
        report(model, *compare(films, answers))
