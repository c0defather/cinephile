"""Telegram-бот: хендлеры без сети, API Telegram и модели подменены (SPEC §8)."""

import json
import urllib.error

import pytest
from test_pick import FILMS, Client, answer, in_tmp  # noqa: F401

import bot

TOKEN = "123456:SECRET-TOKEN-VALUE"


class Api:
    """Заглушка API Telegram: хранит отправленные сообщения, отдаёт заранее заданные обновления."""

    def __init__(self, updates=()):
        self.updates, self.sent = list(updates), []

    def __call__(self, method, **params):
        if method == "sendMessage":
            self.sent.append((params["chat_id"], params["text"]))
            return {}
        if method == "getUpdates":
            batch, self.updates = self.updates, []
            return batch
        raise AssertionError(method)


def update(number, chat, text):
    return {"update_id": number, "message": {"chat": {"id": chat}, "text": text}}


def make(results, local=None):
    return bot.Bot(FILMS, Client(results), local)


def test_query_gets_films_from_the_catalog_as_text():
    bot_ = make(answer("t-1", "t-2"))
    text = bot_.reply(1, "доброе кино")
    assert "Проверочный фильм 1" in text and "Проверочный фильм 2" in text
    assert "Описание номер 1." in text


def test_clarify_goes_to_the_chat_as_a_question():
    text = make(answer(status="clarify")).reply(1, "что-нибудь")
    assert "Уточните запрос" in text and "Проверочный фильм" not in text


def test_chats_have_separate_dialogs():
    bot_ = make(answer("t-1"))
    bot_.reply(1, "кино")
    assert bot_.reply(2, "ещё").startswith("Сначала опишите")  # у чата 2 запроса ещё не было


def test_more_in_the_same_chat_continues():
    bot_ = make(answer("t-1"))
    bot_.reply(1, "кино")
    bot_.cloud = Client(answer("t-2"))
    assert "Проверочный фильм 2" in bot_.reply(1, "ещё")


def test_start_and_help_do_not_call_the_model():
    bot_ = make(answer("t-1"))
    for command in ("/start", "/help"):
        assert "ещё" in bot_.reply(1, command)
    assert bot_.cloud.calls == []


def test_reset_forgets_the_dialog():
    bot_ = make(answer("t-1"))
    bot_.reply(1, "кино")
    bot_.reply(1, "/reset")
    assert bot_.reply(1, "ещё").startswith("Сначала опишите")


def test_cloud_down_falls_back_to_local():
    import httpx2
    import openai
    down = openai.APIConnectionError(request=httpx2.Request("POST", "http://x"))
    bot_ = bot.Bot(FILMS, Client(down), Client(answer("t-3")))
    assert "Проверочный фильм 3" in bot_.reply(1, "кино")
    assert bot_.state[1]["shown"] == ["t-3"]


def test_fallback_does_not_double_the_clarify_answer():
    import httpx2
    import openai
    down = openai.APIConnectionError(request=httpx2.Request("POST", "http://x"))
    local = Client(answer("t-1"))
    bot_ = bot.Bot(FILMS, Client(answer(status="clarify")), local)
    bot_.reply(1, "что-нибудь")
    bot_.cloud = Client(down)
    bot_.reply(1, "грустное")
    sent = " ".join(m["content"] for m in local.kwargs["messages"])
    assert sent.count("грустное") == 1


def test_long_reply_is_cut_to_the_telegram_limit():
    films = [dict(FILMS[0], description="я" * 5000)]
    text = bot.Bot(films, Client(answer("t-1")), None).reply(1, "кино")
    assert len(text) <= 4096


def test_poll_sends_replies_and_moves_the_offset():
    api = Api([update(10, 5, "кино"), update(11, 5, "ещё")])
    bot_ = make(answer("t-1"))
    assert bot_.poll(api, 0) == 12
    assert [chat for chat, _ in api.sent] == [5, 5]


def test_poll_skips_updates_without_text():
    api = Api([{"update_id": 3, "message": {"chat": {"id": 1}, "sticker": {}}}, {"update_id": 4}])
    bot_ = make(answer("t-1"))
    assert bot_.poll(api, 0) == 5
    assert api.sent == [] and bot_.cloud.calls == []


def test_poll_survives_a_network_error_and_keeps_the_offset(capsys):
    def broken(method, **params):
        raise urllib.error.URLError(f"https://api.telegram.org/bot{TOKEN}/getUpdates")
    assert make(answer("t-1")).poll(broken, 7) == 7
    assert TOKEN not in capsys.readouterr().out  # в адресе ошибки токен: его не печатаем


def test_api_call_builds_the_url_with_token_and_json(monkeypatch):
    seen = {}

    class Reply:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"ok": True, "result": [1]}).encode()

    def fake(request, timeout):
        seen["url"], seen["body"] = request.full_url, json.loads(request.data)
        return Reply()
    monkeypatch.setattr(bot.urllib.request, "urlopen", fake)
    assert bot.make_api(TOKEN)("getUpdates", offset=5) == [1]
    assert seen["url"] == f"https://api.telegram.org/bot{TOKEN}/getUpdates" and seen["body"] == {"offset": 5}


def test_main_without_a_token_stops_with_a_hint(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(bot, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        bot.main([])
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().out
