#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nl2json_only.py — Python 3.12

Interactive CLI that:
- Prompts you in terminal for your natural-language question.
- Loads your complete schema from ./firestore_collections_structure.md.
- Uses GPT-4o-mini to produce ONE JSON object containing BOTH:
    - sdk_command (exact command string for an SDK-style query; no libs executed)
    - sql_command (exact BigQuery Standard SQL string; no libs executed)
- Writes JSON to ./nl2queries_output.json

No Firestore or BigQuery dependencies. It ONLY creates JSON.

Setup:
  pip install openai python-dotenv
  export OPENAI_API_KEY="sk-..."
Run:
  python nl2json_only.py
"""

import json
import os
import sys
from typing import Any, Dict

# --------------------
# Paths / configuration
# --------------------
SCHEMA_PATH = os.path.abspath("./firestore_collections_structure.md")
OUTPUT_PATH = os.path.abspath("./nl2queries_output.json")

SYSTEM_PROMPT = """You are a senior data engineer.
Your job is to translate ambiguous natural-language requests into a single JSON object
that contains BOTH an SDK-style query command string and a BigQuery Standard SQL string.

Absolute rules:
- Output MUST be a single valid JSON object with keys: query_intent, sdk_command, sql_command, notes.
- sdk_command MUST be an object: { "description": str, "command": str }
  - 'command' is a single-line string that looks like an SDK query (e.g., Firestore Node/Python chain or a FHIR REST path).
  - Do NOT include placeholders like <PROJECT>; instead, use supplied project/dataset/collection values when provided.
- sql_command MUST be an object: { "description": str, "dialect": "standard", "command": str }
  - 'command' is a single-line Standard SQL string with fully-qualified tables in the form `PROJECT.DATASET.TABLE`.
- Prefer fields/collections present in the provided SCHEMA. If unsure, choose the closest match and explain in 'notes'.
- Convert relative time windows to concrete YYYY-MM-DD using Africa/Johannesburg 'today' (assume today is current date).
- Include ORDER BY and LIMIT unless the user explicitly asks for full results.
- No extra commentary outside the JSON.
"""

USER_PROMPT_TEMPLATE = """User question:
{question}

Target context:
- BigQuery project: {bq_project}
- BigQuery dataset: {bq_dataset}
- Default collection (hint, may be empty): {default_collection}

Authoritative schema (use these fields/collections when possible):
---
{schema}
---

Return EXACTLY this JSON shape (no markdown fences):
{{
  "query_intent": "string",
  "sdk_command": {{
    "description": "string",
    "command": "string"
  }},
  "sql_command": {{
    "description": "string",
    "dialect": "standard",
    "command": "string"
  }},
  "notes": "string"
}}
"""


def load_schema_text(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Schema file not found at {path}. "
            "Place your complete firestore_collections_structure.md next to this script."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def prompt(text: str, default: str = "") -> str:
    val = input(f"{text} [{default}]: ").strip()
    return val or default


def call_llm(
    question: str,
    bq_project: str,
    bq_dataset: str,
    default_collection: str,
    schema_text: str,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI  # pip install openai
    except Exception as e:
        raise RuntimeError("Install OpenAI SDK: pip install openai") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")

    client = OpenAI(api_key=api_key)

    user_block = USER_PROMPT_TEMPLATE.format(
        question=question.strip(),
        bq_project=bq_project.strip(),
        bq_dataset=bq_dataset.strip(),
        default_collection=default_collection.strip(),
        schema=schema_text.strip(),
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.15,
        response_format={"type": "json_object"},  # enforce valid JSON
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_block},
        ],
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model returned invalid JSON: {content}") from e

    # minimal validation
    for k in ("query_intent", "sdk_command", "sql_command"):
        if k not in data:
            raise RuntimeError(f"Missing key '{k}' in model response.")
    if (
        not isinstance(data["sdk_command"], dict)
        or "command" not in data["sdk_command"]
    ):
        raise RuntimeError("sdk_command must contain 'command'.")
    if (
        not isinstance(data["sql_command"], dict)
        or "command" not in data["sql_command"]
    ):
        raise RuntimeError("sql_command must contain 'command'.")
    return data


def main():
    print("🧠  Natural Language → JSON (sdk_command + sql_command)")
    print("-------------------------------------------------------")

    question = prompt("Enter your question")
    if not question:
        print("No question provided. Exiting.")
        sys.exit(1)

    # Ask for identifiers so the LLM can produce non-ambiguous, fully-qualified SQL.
    bq_project = prompt("BigQuery Project ID", "my-gcp-project")
    bq_dataset = prompt("BigQuery Dataset", "my_dataset")
    default_collection = prompt("Default Firestore collection (optional hint)", "")

    try:
        schema_text = load_schema_text(SCHEMA_PATH)
    except Exception as e:
        print(f"ERROR loading schema: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        result = call_llm(
            question, bq_project, bq_dataset, default_collection, schema_text
        )
    except Exception as e:
        print(f"ERROR from LLM: {e}", file=sys.stderr)
        sys.exit(3)

    # Write single JSON file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Wrote JSON to: {OUTPUT_PATH}")
    # Optional: also print to stdout
    print("\n--- JSON ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
