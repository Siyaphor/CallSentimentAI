import json
from pathlib import Path

from mongo_history import import_history_entries


HISTORY_PATH = Path("call_history.json")


def main():
    if not HISTORY_PATH.exists():
        print("No call_history.json file found.")
        return

    entries = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    inserted_count = import_history_entries(entries)
    print(f"Imported {inserted_count} new call history record(s) into MongoDB.")


if __name__ == "__main__":
    main()
