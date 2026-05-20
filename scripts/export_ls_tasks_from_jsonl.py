"""Export Label Studio tasks from translated evidence labels JSONL.

Reads the output of translate_evidence_labels.py and creates a JSON
file importable into Label Studio.

Usage:
    python -m scripts.export_ls_tasks_from_jsonl \
        --input outputs/runs/no_vote_100_skipverify/labels_implicit_zh.jsonl \
        --output outputs/runs/no_vote_100_skipverify/label_studio_tasks.json
"""

import argparse
import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.utils.jsonl import read_jsonl


def _extract_evidence_sentences(numbered_text, evidence_ids):
    """Extract only evidence sentences from numbered text by sentence IDs."""
    if not numbered_text or not evidence_ids:
        return ""
    id_set = set(evidence_ids)
    lines = []
    for line in numbered_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Parse [S<num>] prefix
        if line.startswith("[S") and "]" in line:
            try:
                sid = int(line[2:line.index("]")])
                if sid in id_set:
                    lines.append(line)
            except ValueError:
                pass
    return "\n".join(lines)


def _evidence_only_from_highlight(text):
    """Extract only ★-marked evidence lines, removing the ★ marker."""
    if not text:
        return ""
    lines = []
    for line in text.split("\n"):
        if "★" in line:
            lines.append(line.replace(" ★", "").replace("★ ", "").replace("★", ""))
    return "\n".join(lines)


def to_task(row):
    full_ctx = row.get("full_context_numbered", "")
    full_ctx_zh = row.get("full_context_zh", "")
    evidence_en = row.get("evidence_context", "")
    evidence_zh = row.get("evidence_context_zh", "")

    data = {
        "sample_id": row.get("sample_id", ""),
        "story_name": row.get("story_name", ""),
        "question": row.get("question", ""),
        "question_zh": row.get("question_zh", ""),
        "answer": row.get("answer", ""),
        "answer_zh": row.get("answer_zh", ""),
        "full_context": full_ctx,
        "full_context_zh": full_ctx_zh,
        "evidence_context": evidence_en,
        "evidence_context_zh": evidence_zh,
        "qa_zh": f"Q: {row.get('question_zh', '')}\nA: {row.get('answer_zh', '')}",
        "difficulty_label": row.get("difficulty_label", ""),
        "difficulty_hint_zh": row.get("difficulty_hint_zh", ""),
        "answer_directly_found": row.get("answer_directly_found", ""),
        "reasoning_level": row.get("reasoning_level", ""),
        "num_required_sentences": row.get("num_required_sentences", 0),
        "evidence_reason": row.get("evidence_reason", ""),
    }
    # Model suggestion summary for annotators
    parts = [
        f"样本ID: {data['sample_id']}",
        f"故事: {data['story_name']}",
        f"模型判定难度: {data['difficulty_label']}",
        f"答案是否直接找到: {data['answer_directly_found']}",
        f"推理级别: {data['reasoning_level']}",
        f"证据句数: {data['num_required_sentences']}",
    ]
    if data["evidence_reason"]:
        parts.append(f"模型理由: {data['evidence_reason']}")
    data["model_suggestion"] = "\n".join(parts)
    return {"data": data}


def main():
    parser = argparse.ArgumentParser(
        description="Export Label Studio tasks from translated JSONL"
    )
    parser.add_argument("--input", required=True, help="Input translated JSONL")
    parser.add_argument("--output", required=True, help="Output Label Studio JSON")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    valid = [r for r in records if r.get("difficulty_label") != "Invalid"]
    tasks = [to_task(r) for r in valid]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"Loaded {len(records)} records, {len(valid)} valid")
    print(f"Wrote {len(tasks)} Label Studio tasks to {output_path}")


if __name__ == "__main__":
    main()
