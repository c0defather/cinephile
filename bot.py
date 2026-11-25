"""Telegram-бот «Киноманьяк»: тот же подбор, что в kino.py, и диалог из talk().

    uv run bot.py [--local]

Токен только в .env (TELEGRAM_BOT_TOKEN). Библиотек нет: Bot API по HTTPS, getUpdates с долгим ожиданием.
"""

import contextlib
import copy
import io
import json
import os
import sys
import time
import urllib.request

from dotenv import load_dotenv

from kino import CLOUD_MODEL, LOCAL_MODEL, cloud_failed, make_clients, new_dialog, talk

LIMIT_CHARS = 4096  # предел одного сообщения Telegram
WAIT = 30  # секунд долгого ожидания getUpdates
HELP = ("Опишите, какой фильм хотите: жанр, язык, длительность, настроение.\n"
        "«ещё» покажет другие фильмы по тому же запросу, «уже смотрел <название>» уберёт фильм до конца разговора.\n"
        "/reset начинает разговор заново.")


def make_api(token):
    """Вызов метода Bot API: api("sendMessage", chat_id=1, text="…") → result."""
    def api(method, **params):
        request = urllib.request.Request(f"https://api.telegram.org/bot{token}/{method}",
                                         data=json.dumps(params).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=WAIT + 10) as answer:
            return json.load(answer)["result"]
    return api


class Bot:
    def __init__(self, films, cloud, local):
        self.films, self.cloud, self.local, self.state = films, cloud, local, {}

    def talk(self, chat, text):
        """Реплика в диалоге: сначала облачная модель, при сбое облака та же реплика на локальной."""
        state = self.state.setdefault(chat, new_dialog())
        before = copy.deepcopy(state)
        if self.cloud is not None:
            outcome, _ = talk(self.cloud, CLOUD_MODEL, state, text, self.films)
            if not cloud_failed(outcome):
                return
            print("Облако недоступно, пробуем локальную модель", file=sys.stderr)
            state.clear()
            state.update(before)  # реплика не должна учесться дважды
        talk(self.local, LOCAL_MODEL, state, text, self.films)

    def reply(self, chat, text):
        """Текст ответа в чат. pick печатает сам, печать перехватывается и уходит в сообщение."""
        text = text.strip()
        if text in ("/start", "/help"):
            return HELP
        if text == "/reset":
            self.state.pop(chat, None)
            return "Разговор начат заново."
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.talk(chat, text)
        return (out.getvalue().strip() or "Не получилось ответить, попробуйте ещё раз.")[:LIMIT_CHARS]

    def poll(self, api, offset):
        """Один круг: забрать обновления, ответить, вернуть новый offset. Сбой сети offset не двигает."""
        try:
            updates = api("getUpdates", offset=offset, timeout=WAIT)
            for item in updates:
                offset = item["update_id"] + 1
                message = item.get("message") or {}
                if message.get("text"):
                    chat = message["chat"]["id"]
                    api("sendMessage", chat_id=chat, text=self.reply(chat, message["text"]))
        except Exception as error:  # в тексте ошибки сети может быть адрес с токеном: печатаем только тип
            print("Сбой связи с Telegram:", type(error).__name__, file=sys.stderr)
            time.sleep(2)
        return offset


def main(argv):
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Нет TELEGRAM_BOT_TOKEN: впишите токен из @BotFather в .env")
        sys.exit(1)
    with open("films.json", encoding="utf-8") as file:
        films = json.load(file)
    cloud, local = make_clients(os.environ)
    bot = Bot(films, None if "--local" in argv else cloud, local)
    api, offset = make_api(token), 0
    print("Бот запущен")
    while True:
        offset = bot.poll(api, offset)


if __name__ == "__main__":
    main(sys.argv[1:])
