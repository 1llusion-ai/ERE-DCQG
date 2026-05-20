"""Translate selector evidence labels to Chinese for human annotation.

Reads labels_implicit.jsonl (output of run_evidence_no_vote_pilot.py),
converts to a review-friendly format with numbered sentences, and
translates key fields into Simplified Chinese.

Usage:
    python -m scripts.translate_evidence_labels \
        --input outputs/runs/no_vote_pilot/labels_implicit.jsonl \
        --output outputs/runs/no_vote_pilot/labels_implicit_zh.jsonl \
        --model deepseek-ai/DeepSeek-V3 --batch_size 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.utils.api_client import call_openai_compatible
from dcqg.utils.config import get_api_config
from dcqg.utils.jsonl import read_jsonl, write_jsonl


DIFFICULTY_HINT_ZH = {
    "Easy": "Easy：答案可以直接在文本中找到，并且最小证据集只有一个证据句。",
    "Medium": (
        "Medium：答案不能直接在文本中找到，但最小证据集只有一个证据句并需要简单推理；"
        "或者答案可以直接找到，但最小证据集包含多个证据句。"
    ),
    "Hard": (
        "Hard：答案不能直接在文本中找到，最小证据集包含多个证据句，"
        "并且需要至少一次推理。"
    ),
}


def _split_sentences(text):
    from dcqg.utils.text import split_sentences
    return split_sentences(text)


def _numbered_sentences(section):
    sentences = _split_sentences(section)
    return "\n".join(f"[S{i}] {s}" for i, s in enumerate(sentences))


def _evidence_highlight(section, evidence_ids):
    sentences = _split_sentences(section)
    lines = []
    evidence_set = set(evidence_ids or [])
    for i, s in enumerate(sentences):
        if i in evidence_set:
            lines.append(f"[S{i}] {s}")
    return "\n".join(lines)


def _extract_json(raw):
    if not raw:
        return None
    if "<think>" in raw:
        raw = raw.rsplit("<think>", 1)[-1].strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError, TypeError):
        return None


def build_review_record(record, idx):
    section = record.get("story_section", "")
    required = record.get("required_evidence_sentences", [])
    selected = record.get("selected_evidence_sentences", [])
    evidence_ids = required if required else selected

    return {
        "sample_id": f"EV{idx:04d}",
        "story_name": record.get("story_name", ""),
        "question": record.get("question", ""),
        "answer": record.get("answer1", "") or record.get("answer", ""),
        "full_context_numbered": _numbered_sentences(section),
        "evidence_context": _evidence_highlight(section, evidence_ids),
        "difficulty_label": record.get("difficulty_label", "Invalid"),
        "difficulty_hint_zh": DIFFICULTY_HINT_ZH.get(
            record.get("difficulty_label", ""), ""
        ),
        "answer_directly_found": record.get("final_answer_directly_found", ""),
        "reasoning_level": record.get("final_reasoning_level", ""),
        "num_required_sentences": record.get("num_required_sentences", 0),
        "evidence_reason": record.get("evidence_reason", ""),
        "sufficiency_reason": record.get("sufficiency_reason", ""),
        "verification_status": record.get("verification_status", ""),
    }


def build_translation_prompt(items):
    return f"""Translate the following fairy-tale reading-comprehension annotation fields from English into accurate Simplified Chinese.

Requirements:
1. Preserve all sentence markers exactly, such as [S0], [S12]. Keep ★ markers.
2. Translate faithfully and literally enough for evidence annotation.
3. Do not add explanations, interpretations, or missing information.
4. Keep names understandable; transliterate only when necessary.
5. Return only valid JSON with this schema:
{{
  "translations": [
    {{
      "sample_id": "EV0001",
      "question_zh": "...",
      "answer_zh": "...",
      "full_context_zh": "...",
      "evidence_context_zh": "..."
    }}
  ]
}}

Items:
{json.dumps(items, ensure_ascii=False, indent=2)}
"""


def translate_batch(items, api_url, api_key, model, timeout, max_tokens):
    prompt = build_translation_prompt(items)
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
            "question_zh": item.get("question_zh", ""),
            "answer_zh": item.get("answer_zh", ""),
            "full_context_zh": item.get("full_context_zh", ""),
            "evidence_context_zh": item.get("evidence_context_zh", ""),
        }
    return output


def translate_batch_with_fallback(items, api_url, api_key, model, timeout,
                                  max_tokens, retries):
    last_error = None
    for attempt in range(1, retries + 2):
        try:
            return translate_batch(items, api_url, api_key, model, timeout,
                                   max_tokens)
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                print(f"  Batch of {len(items)} failed attempt {attempt}: {exc}")
                time.sleep(min(10, 2 * attempt))
    if len(items) == 1:
        raise last_error
    mid = len(items) // 2
    print(f"  Batch of {len(items)} failed; splitting into {mid} + {len(items) - mid}")
    output = translate_batch_with_fallback(
        items[:mid], api_url, api_key, model, timeout, max_tokens, retries
    )
    output.update(translate_batch_with_fallback(
        items[mid:], api_url, api_key, model, timeout, max_tokens, retries
    ))
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Translate evidence labels to Chinese for annotation"
    )
    parser.add_argument("--input", required=True, help="Input labels JSONL")
    parser.add_argument("--output", required=True, help="Output translated JSONL")
    parser.add_argument("--model", default=None,
                        help="Model for translation (default: deepseek-ai/DeepSeek-V3)")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = get_api_config()
    api_url = cfg["SILICONFLOW_API_URL"]
    api_key = cfg["SILICONFLOW_API_KEY"]
    model = args.model or "deepseek-ai/DeepSeek-V3"

    records = read_jsonl(args.input)
    if not records:
        print("No records found.")
        return

    # Filter out Invalid labels
    valid = [r for r in records if r.get("difficulty_label") != "Invalid"]
    print(f"Total records: {len(records)}, valid labels: {len(valid)}")
    print(f"Model: {model}")

    # Build review records
    reviews = [build_review_record(r, i) for i, r in enumerate(valid)]

    # Load existing translations for resume
    existing = {}
    if args.resume:
        try:
            for row in read_jsonl(args.output):
                sid = row.get("sample_id")
                if sid and row.get("question_zh"):
                    existing[sid] = row
        except FileNotFoundError:
            pass
        print(f"Existing translations: {len(existing)}")

    # Apply existing translations
    for review in reviews:
        old = existing.get(review["sample_id"])
        if old:
            for key in ["question_zh", "answer_zh", "full_context_zh",
                        "evidence_context_zh"]:
                review[key] = old.get(key, "")

    # Find pending
    pending = [r for r in reviews if not r.get("question_zh")]
    print(f"Pending translations: {len(pending)}")

    # Translate in batches
    translated = 0
    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start:batch_start + args.batch_size]
        trans_items = [
            {
                "sample_id": r["sample_id"],
                "question": r["question"],
                "answer": r["answer"],
                "full_context": r["full_context_numbered"],
                "evidence_context": r["evidence_context"],
            }
            for r in batch
        ]
        t0 = time.time()
        translations = translate_batch_with_fallback(
            trans_items, api_url, api_key, model, args.timeout,
            args.max_tokens, args.retries
        )
        elapsed = time.time() - t0

        for review in batch:
            trans = translations.get(review["sample_id"], {})
            review["question_zh"] = trans.get("question_zh", "")
            review["answer_zh"] = trans.get("answer_zh", "")
            review["full_context_zh"] = trans.get("full_context_zh", "")
            review["evidence_context_zh"] = trans.get("evidence_context_zh", "")

        translated += len(batch)
        # Save after each batch
        write_jsonl(args.output, reviews)
        print(f"  Translated {translated}/{len(pending)} ({elapsed:.1f}s)")
        time.sleep(0.2)

    write_jsonl(args.output, reviews)
    print(f"\nSaved: {args.output}")

    # Print sample
    if reviews:
        r = reviews[0]
        print(f"\n=== Sample ({r['sample_id']}) ===")
        print(f"Q: {r['question']}")
        print(f"Q(中): {r.get('question_zh', '')}")
        print(f"A: {r['answer']}")
        print(f"A(中): {r.get('answer_zh', '')}")
        print(f"Difficulty: {r['difficulty_label']} | {r.get('difficulty_hint_zh', '')}")


if __name__ == "__main__":
    main()
