"""Add Chinese helper translations to a human-review CSV.

The translated columns are annotation aids only. Training data should keep the
original English context and QA.

Usage:
    python -m scripts.translate_review_csv \
        --input outputs/runs/no_vote_500_newdef_qwen3_32b/human_review_100.csv \
        --output outputs/runs/no_vote_500_newdef_qwen3_32b/human_review_100_zh.csv \
        --model deepseek-ai/DeepSeek-V3 --batch_size 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.utils.api_client import call_openai_compatible
from dcqg.utils.config import get_api_config


DIFFICULTY_HINT_ZH = {
    "Easy": "Easy：答案可以直接在文本中找到，并且只需要一个必要证据句。",
    "Medium": (
        "Medium：答案不能直接在文本中找到，但只需要一个必要证据句并进行简单推理；"
        "或者答案可以直接找到，但需要综合多个必要证据句。"
    ),
    "Hard": (
        "Hard：答案不能直接在文本中找到，需要综合多个必要证据句，"
        "并且至少进行一次推理。"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate review CSV helper fields into Chinese.",
    )
    parser.add_argument("--input", required=True, help="Input review CSV.")
    parser.add_argument("--output", required=True, help="Output translated CSV.")
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI-compatible model. Defaults to deepseek-ai/DeepSeek-V3.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_batch_chars",
        type=int,
        default=12000,
        help="Maximum source-character budget per API batch.",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries before splitting a failed batch.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max output tokens. Defaults to 12000 with full context.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip_full_context",
        action="store_true",
        help="Only translate suggested evidence and QA, not the full story section.",
    )
    parser.add_argument(
        "--no_difficulty_hint",
        action="store_true",
        help="Do not add the difficulty_hint_zh column.",
    )
    return parser.parse_args()


def _extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[-1].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def build_prompt(rows: list[dict], include_full_context: bool) -> str:
    items = []
    for row in rows:
        item = {
            "sample_id": row["sample_id"],
            "evidence_context": row.get("evidence_context", ""),
            "qa": row.get("qa", ""),
        }
        if include_full_context:
            item["full_context"] = (
                row.get("full_context_numbered")
                or row.get("full_context_qa")
                or ""
            )
        items.append(item)
    full_context_schema = ""
    if include_full_context:
        full_context_schema = ',\n      "full_context_zh": "..."'
    return f"""Translate the following fairy-tale reading-comprehension annotation fields from English into accurate Simplified Chinese.

Requirements:
1. Preserve all sentence markers exactly, such as [S0], [S12].
2. Preserve Q: and A: labels exactly.
3. Translate faithfully and literally enough for evidence annotation.
4. Do not add explanations, interpretations, or missing information.
5. Keep names and story-specific terms understandable; transliterate names only when necessary.
6. If full_context is present, translate the whole numbered story section faithfully.
7. Return only valid JSON with this schema:
{{
  "translations": [
    {{
      "sample_id": "EV001",
      "evidence_context_zh": "...",
      "qa_zh": "Q: ...\\nA: ..."{full_context_schema}
    }}
  ]
}}

Items:
{json.dumps(items, ensure_ascii=False, indent=2)}
"""


def translate_batch(
    rows: list[dict],
    api_url: str,
    api_key: str,
    model: str,
    timeout: int,
    max_tokens: int,
    include_full_context: bool,
) -> dict[str, dict]:
    prompt = build_prompt(rows, include_full_context=include_full_context)
    raw = call_openai_compatible(
        prompt,
        api_url=api_url,
        api_key=api_key,
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
        json_mode=False,
        system=(
            "/no_think\n"
            "You are a careful English-to-Chinese translator for academic data annotation. "
            "Return only valid JSON."
        ),
        timeout=timeout,
    )
    data = _extract_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("translations"), list):
        raise ValueError(f"Could not parse translation JSON: {raw[:500]}")
    output = {}
    for item in data["translations"]:
        if not isinstance(item, dict):
            continue
        sid = item.get("sample_id")
        if not sid:
            continue
        output[sid] = {
            "evidence_context_zh": item.get("evidence_context_zh", ""),
            "qa_zh": item.get("qa_zh", ""),
        }
        if include_full_context:
            output[sid]["full_context_zh"] = item.get("full_context_zh", "")
    required_fields = ["evidence_context_zh", "qa_zh"]
    if include_full_context:
        required_fields.append("full_context_zh")
    missing = []
    for row in rows:
        sid = row["sample_id"]
        translation = output.get(sid)
        if not translation:
            missing.append(f"{sid}:missing_item")
            continue
        empty_fields = [field for field in required_fields if not translation.get(field)]
        if empty_fields:
            missing.append(f"{sid}:{','.join(empty_fields)}")
    if missing:
        raise ValueError(f"Missing translation fields: {missing[:5]}")
    return output


def translate_batch_with_fallback(
    rows: list[dict],
    api_url: str,
    api_key: str,
    model: str,
    timeout: int,
    max_tokens: int,
    include_full_context: bool,
    retries: int,
) -> dict[str, dict]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            return translate_batch(
                rows,
                api_url=api_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                include_full_context=include_full_context,
            )
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                print(
                    f"Batch of {len(rows)} failed on attempt {attempt}; retrying: {exc}",
                    flush=True,
                )
                time.sleep(min(10, 2 * attempt))
    if len(rows) == 1:
        assert last_error is not None
        raise last_error
    mid = len(rows) // 2
    print(
        f"Batch of {len(rows)} failed after retries; splitting into "
        f"{mid} and {len(rows) - mid}.",
        flush=True,
    )
    output = translate_batch_with_fallback(
        rows[:mid],
        api_url=api_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
        include_full_context=include_full_context,
        retries=retries,
    )
    output.update(
        translate_batch_with_fallback(
            rows[mid:],
            api_url=api_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            include_full_context=include_full_context,
            retries=retries,
        )
    )
    return output


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _insert_after(fieldnames: list[str], anchor: str, new_fields: list[str]) -> list[str]:
    output = [f for f in fieldnames if f not in new_fields]
    if anchor not in output:
        return output + new_fields
    idx = output.index(anchor) + 1
    return output[:idx] + new_fields + output[idx:]


def _move_to_back(fieldnames: list[str], back_fields: list[str]) -> list[str]:
    back = [field for field in back_fields if field in fieldnames]
    return [field for field in fieldnames if field not in back] + back


def _translation_source_len(row: dict, include_full_context: bool) -> int:
    total = len(row.get("evidence_context", "")) + len(row.get("qa", ""))
    if include_full_context:
        total += len(
            row.get("full_context_numbered")
            or row.get("full_context_qa")
            or ""
        )
    return total


def _iter_batches(
    rows: list[dict],
    batch_size: int,
    max_batch_chars: int,
    include_full_context: bool,
) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for row in rows:
        row_chars = _translation_source_len(row, include_full_context)
        would_exceed_size = len(current) >= batch_size
        would_exceed_chars = (
            bool(current)
            and current_chars + row_chars > max_batch_chars
        )
        if would_exceed_size or would_exceed_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += row_chars
    if current:
        batches.append(current)
    return batches


def main() -> None:
    args = parse_args()
    cfg = get_api_config()
    api_url = cfg["SILICONFLOW_API_URL"]
    api_key = cfg["SILICONFLOW_API_KEY"]
    model = args.model or "deepseek-ai/DeepSeek-V3"
    include_full_context = not args.skip_full_context
    max_tokens = args.max_tokens or (12000 if include_full_context else 6000)

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = read_csv(input_path)
    if not rows:
        raise SystemExit("No rows found in input CSV.")

    if args.resume and output_path.exists():
        existing = read_csv(output_path)
        by_id = {row.get("sample_id"): row for row in existing}
        for row in rows:
            old = by_id.get(row.get("sample_id"))
            if old:
                row["evidence_context_zh"] = old.get("evidence_context_zh", "")
                row["qa_zh"] = old.get("qa_zh", "")
                if include_full_context:
                    row["full_context_zh"] = old.get("full_context_zh", "")

    if args.no_difficulty_hint:
        for row in rows:
            row.pop("difficulty_hint_zh", None)
    else:
        for row in rows:
            row["difficulty_hint_zh"] = DIFFICULTY_HINT_ZH.get(
                row.get("suggested_difficulty_label", ""),
                "",
            )

    fieldnames = list(rows[0].keys())
    if include_full_context:
        fieldnames = _insert_after(fieldnames, "full_context_numbered", ["full_context_zh"])
    fieldnames = _insert_after(fieldnames, "evidence_context", ["evidence_context_zh"])
    fieldnames = _insert_after(fieldnames, "qa", ["qa_zh"])
    if args.no_difficulty_hint:
        fieldnames = [field for field in fieldnames if field != "difficulty_hint_zh"]
    else:
        fieldnames = _insert_after(
            fieldnames,
            "suggested_difficulty_label",
            ["difficulty_hint_zh"],
        )
    fieldnames = _move_to_back(fieldnames, ["evidence_context", "qa"])

    required_translation_fields = ["evidence_context_zh", "qa_zh"]
    if include_full_context:
        required_translation_fields.append("full_context_zh")
    pending = [
        row for row in rows
        if any(not row.get(field) for field in required_translation_fields)
    ]
    print(f"Loaded rows: {len(rows)}")
    print(f"Pending translations: {len(pending)}")
    print(f"Model: {model}")
    print(f"Include full context: {include_full_context}")
    print(f"Max tokens: {max_tokens}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max batch source chars: {args.max_batch_chars}")

    batches = _iter_batches(
        pending,
        batch_size=args.batch_size,
        max_batch_chars=args.max_batch_chars,
        include_full_context=include_full_context,
    )
    translated = 0
    for batch_index, batch in enumerate(batches, start=1):
        translations = translate_batch_with_fallback(
            batch,
            api_url=api_url,
            api_key=api_key,
            model=model,
            timeout=args.timeout,
            max_tokens=max_tokens,
            include_full_context=include_full_context,
            retries=args.retries,
        )
        for row in batch:
            trans = translations.get(row["sample_id"], {})
            row["evidence_context_zh"] = trans.get("evidence_context_zh", "")
            row["qa_zh"] = trans.get("qa_zh", "")
            if include_full_context:
                row["full_context_zh"] = trans.get("full_context_zh", "")
            missing_fields = [
                field for field in required_translation_fields if not row.get(field)
            ]
            if missing_fields:
                raise ValueError(
                    f"Missing translation for {row['sample_id']}: {missing_fields}"
                )
        translated += len(batch)
        write_csv(output_path, rows, fieldnames)
        print(
            f"Translated {translated}/{len(pending)} "
            f"(batch {batch_index}/{len(batches)}, rows {len(batch)})"
        )
        time.sleep(0.2)

    write_csv(output_path, rows, fieldnames)
    print(f"Saved translated CSV: {output_path}")


if __name__ == "__main__":
    main()
