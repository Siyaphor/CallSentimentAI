import os
from datetime import datetime
from typing import Dict, List, Optional

from pymongo import DESCENDING, MongoClient, UpdateOne


DEFAULT_DB_NAME = "voiceiq"
DEFAULT_COLLECTION_NAME = "call_history"

_client: Optional[MongoClient] = None


def _mongo_uri() -> str:
    uri = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
    if not uri:
        raise RuntimeError("Set MONGODB_URI to save and read call history from MongoDB.")
    return uri


def _collection():
    global _client
    if _client is None:
        _client = MongoClient(_mongo_uri(), serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")

    db_name = os.environ.get("MONGODB_DATABASE", DEFAULT_DB_NAME)
    collection_name = os.environ.get("MONGODB_COLLECTION", DEFAULT_COLLECTION_NAME)
    collection = _client[db_name][collection_name]
    collection.create_index([("created_at", DESCENDING)])
    return collection


def _serialize(entry: Dict) -> Dict:
    entry = dict(entry)
    entry.pop("_id", None)
    return entry


def load_history(limit: int = 100) -> List[Dict]:
    cursor = _collection().find({}, {"_id": 0}).sort("created_at", DESCENDING).limit(limit)
    return [_serialize(entry) for entry in cursor]


def add_history_entry(filename: str, transcript: str, sentiment_label: str, confidence: float) -> List[Dict]:
    created_at = datetime.utcnow().isoformat()
    entry = {
        "id": created_at,
        "filename": filename,
        "transcript": transcript,
        "sentiment": sentiment_label,
        "confidence": confidence,
        "created_at": created_at,
    }
    _collection().insert_one(entry)
    return load_history()


def import_history_entries(entries: List[Dict]) -> int:
    if not entries:
        return 0

    operations = []
    for entry in entries:
        if not entry.get("id"):
            entry["id"] = entry.get("created_at") or datetime.utcnow().isoformat()
        if not entry.get("created_at"):
            entry["created_at"] = entry["id"]
        operations.append(
            UpdateOne(
                {"id": entry["id"]},
                {"$setOnInsert": entry},
                upsert=True,
            )
        )

    result = _collection().bulk_write(operations, ordered=False)
    return result.upserted_count
