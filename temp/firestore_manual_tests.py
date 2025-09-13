#!/usr/bin/env python3
import json
import os
import re
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google.api_core.exceptions import FailedPrecondition  # type: ignore
from google.cloud import firestore  # type: ignore

# ----------------------------
# CONFIG
# ----------------------------
PROJECT_ID = "ad3-sam"
DATABASE_ID = "ad3gem-email"
QUESTIONS_PATH = "/Users/craigcharity/dev/ad3gem/questions.md"
AD3GEM_SQL_PATH = "/Users/craigcharity/dev/ad3gem/ad3gem-sql.py"
MAX_DOCS_PER_QUERY = int(os.getenv("MAX_DOCS_PER_QUERY", "10"))


# ----------------------------
# HELPERS
# ----------------------------
def load_questions(path: str) -> List[str]:
    """Extract natural-language questions from questions.md."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"questions.md not found at {path}")

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as fh:
        lines = [line.rstrip("\n") for line in fh.readlines()]
    questions: List[str] = []
    # Pattern matches lines like: \t1. “Show me ...” or 1. "Show me ..."
    pattern = re.compile(r"^\s*\d+\.\s*[\“\"']?(.*?)[\”\"']?$")
    for line in lines:
        m = pattern.match(line)
        if m:
            q = m.group(1).strip()
            if q:
                questions.append(q)
    return questions


def import_ad3gem_sql(module_path: str):
    """Dynamically import ad3gem-sql.py as a module and return it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ad3gem_sql", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load ad3gem-sql module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_openai_key_present():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Please export it in your environment "
            "before running."
        )


def print_document(doc_data: Dict[str, Any]):
    print("\n--- MATCHING DOCUMENT ---")
    print(f"Message ID: {doc_data.get('messageId')}")
    print(f"Sent At:    {doc_data.get('sentAt')}")
    print(f"From:       {doc_data.get('fromEmail', doc_data.get('fromFull'))}")
    print(f"To:         {doc_data.get('toEmails', [])}")
    print(f"Subject:    {doc_data.get('subject')}")
    print(f"Snippet:    {doc_data.get('snippet')}")
    print("All Emails: ", doc_data.get("allEmails", []))
    print("Thread ID:   ", doc_data.get("threadId"))


def execute_translated_sdk_code(db: firestore.Client, code: str) -> List[Any]:
    """Exec the model-generated Firestore SDK code and collect results.

    We try to find one of the following in the exec namespace after running
    `code`:
      - a list named `docs` or `results` containing DocumentSnapshot objects
      - a query-like object named `q` or `query` that has a .stream() method
    If none found, return empty list.
    """
    local_ns: Dict[str, Any] = {
        "db": db,
        "firestore": firestore,
        "FailedPrecondition": FailedPrecondition,
        "datetime": datetime,
        "timezone": timezone,
    }
    try:
        exec(code, {}, local_ns)
    except Exception:
        # Return empty if the snippet failed
        return []

    # 1) If snippet produced a list of docs
    for key in ("docs", "results"):
        if key in local_ns and isinstance(local_ns[key], list):
            return local_ns[key][:MAX_DOCS_PER_QUERY]

    # 2) If snippet produced a query object; detect by having a stream() attr
    for key in ("q", "query"):
        obj = local_ns.get(key)
        if obj is None:
            continue
        if hasattr(obj, "stream") and callable(getattr(obj, "stream")):
            try:
                docs = list(obj.stream())
                return docs[:MAX_DOCS_PER_QUERY]
            except Exception:
                return []

    return []


def main():
    ensure_openai_key_present()

    # Named-database Firestore client to match production layout
    db = firestore.Client(project=PROJECT_ID, database=DATABASE_ID)

    # Load inputs and generator
    questions = load_questions(QUESTIONS_PATH)
    ad3_sql = import_ad3gem_sql(AD3GEM_SQL_PATH)

    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")

    # Aggregate SDK outputs for one big JSON file
    aggregated_items: List[Dict[str, Any]] = []
    generated_at_iso = datetime.now(timezone.utc).isoformat()

    for idx, q in enumerate(questions, start=1):
        print("\n" + "-" * 80)
        print(f"Q{idx}: {q}")

        try:
            combined = ad3_sql.generate_json_from_nl(q)
        except Exception as e:
            print(f"Generator error: {e}")
            traceback.print_exc()
            continue

        # Persist raw combined JSON for inspection (optional)
        try:
            safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", f"q{idx}")
            out_dir = "/Users/craigcharity/dev/ad3gem/temp/sdk_runs"
            os.makedirs(out_dir, exist_ok=True)
            with open(
                os.path.join(out_dir, f"{safe_name}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        sdk_json = combined.get("sdk_json", {}) if isinstance(combined, dict) else {}
        translated_code: Optional[str] = sdk_json.get("translated_sdk_code")
        if not translated_code:
            print("No translated_sdk_code returned by generator. Skipping.")
            continue

        # Add to aggregated outputs
        aggregated_items.append(
            {
                "index": idx,
                "question": q,
                "sdk_json": sdk_json,
            }
        )

        # Execute the Firestore query code and print results like
        # firestore_test.py
        docs = execute_translated_sdk_code(db, translated_code)

        if not docs:
            print("No documents returned.")
            continue

        # Normalize DocumentSnapshot to dicts if needed
        normalized: List[Dict[str, Any]] = []
        for d in docs:
            if hasattr(d, "to_dict") and callable(getattr(d, "to_dict")):
                try:
                    normalized.append(d.to_dict() or {})
                except Exception:
                    continue
            elif isinstance(d, dict):
                normalized.append(d)

        print(f"Found {len(normalized)} documents:")
        for doc in normalized:
            print_document(doc)

    # Write aggregated SDK outputs
    try:
        out_dir = "/Users/craigcharity/dev/ad3gem/temp/sdk_runs"
        os.makedirs(out_dir, exist_ok=True)
        agg_path = os.path.join(out_dir, "all_sdk_outputs.json")
        payload = {
            "generated_at": generated_at_iso,
            "count": len(aggregated_items),
            "items": aggregated_items,
        }
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved aggregated SDK outputs to: {agg_path}")
    except Exception as e:
        print(f"Failed to write aggregated SDK outputs: {e}")


if __name__ == "__main__":
    main()
