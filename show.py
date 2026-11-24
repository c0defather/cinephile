import json

JOURNAL = "calls.jsonl"


def load(path=JOURNAL):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def summarize(rows):
    """По моделям: исходы, токены, цена. Запрос считается по choice, а не по вызову."""
    total = {}
    for row in rows:
        item = total.setdefault(row["model"], {
            "outcomes": {}, "requests": 0, "calls": 0,
            "tokens_in": 0, "tokens_out": 0, "usd": 0})
        usage = row.get("usage") or {}
        item["calls"] += 1
        item["tokens_in"] += usage.get("prompt_tokens", 0)
        item["tokens_out"] += usage.get("completion_tokens", 0)
        item["usd"] += row.get("usd", 0)
        # запрос кончается на choice; сорвавшийся filters тоже конец запроса;
        # опыт с памятью (step memory) не запрос
        if row["step"] == "choice" or (row["step"] == "filters" and row["outcome"] != "ok"):
            item["requests"] += 1
            item["outcomes"][row["outcome"]] = item["outcomes"].get(row["outcome"], 0) + 1
    for item in total.values():
        item["usd_per_request"] = item["usd"] / item["requests"] if item["requests"] else 0
    return total


def report(total):
    if not total:
        print("Журнал пуст")
        return
    for model, item in total.items():
        print(f"{model}: запросов {item['requests']}, вызовов {item['calls']}")
        for outcome, count in sorted(item["outcomes"].items()):
            print(f"  {outcome}: {count}")
        print(f"  токены: вход {item['tokens_in']}, выход {item['tokens_out']}")
        print(f"  цена: всего ${item['usd']:.6f}, один запрос ${item['usd_per_request']:.6f}")


if __name__ == "__main__":
    try:
        rows = load()
    except FileNotFoundError:
        rows = []
    report(summarize(rows))
