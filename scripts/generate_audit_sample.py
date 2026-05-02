"""Generate MANUAL_AUDIT_SAMPLE.md from path filter outputs.

Usage:
    python -m scripts.generate_audit_sample --input_dir outputs/runs/path_filter_strict_pilot
"""
import argparse
import random
from pathlib import Path

from dcqg.utils.jsonl import read_jsonl


def fmt_item(x, n=150):
    events = x.get("events", [])
    path_str = " -> ".join(f"{e.get('trigger','')}/{e.get('type','')}" for e in events)
    rels = ", ".join(x.get("relation_subtypes", []))
    phrase = x.get("gold_answer_phrase", "")
    sentence = x.get("gold_answer_sentence", "")
    supporting = x.get("supporting_sentences", [])
    judge = x.get("llm_path_judge", {})
    pd = judge.get("can_write_path_dependent_question", "?")
    pq = judge.get("path_questionable", "?")
    risk = judge.get("single_sentence_risk", "?")
    steps = judge.get("expected_required_steps", "?")
    rec = judge.get("recommended_difficulty", "?")
    reason = judge.get("reason", "")
    strict_r = x.get("policy_strict_reason", "")
    relaxed_r = x.get("policy_relaxed_reason", "")

    lines = []
    lines.append(f"- **doc_id:** `{x.get('doc_id', '')}`")
    lines.append(f"- **title:** {x.get('title', '')}")
    lines.append(f"- **difficulty:** {x.get('difficulty', '?')}")
    lines.append(f"- **event path:** {path_str}")
    lines.append(f"- **relation sequence:** {rels}")
    lines.append(f"- **gold_answer_phrase:** `{phrase}`")
    lines.append(f"- **answer_sentence:** {sentence[:n]}")
    lines.append(f"- **supporting_sentences:**")
    for s in supporting[:5]:
        sid = s[0] if isinstance(s, (list, tuple)) else ""
        txt = s[1] if isinstance(s, (list, tuple)) else s
        lines.append(f"  - [S{sid}] {txt[:n]}")
    lines.append(f"- **LLM judge:** pq={pq} risk={risk} steps={steps} rec={rec} path_dep={pd}")
    lines.append(f"- **judge reason:** {reason}")
    lines.append(f"- **strict_reason:** {strict_r}")
    lines.append(f"- **relaxed_reason:** {relaxed_r}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate manual audit sample.")
    parser.add_argument("--input_dir", default="outputs/runs/path_filter_strict_pilot")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n_easy", type=int, default=5)
    parser.add_argument("--n_medium", type=int, default=5)
    parser.add_argument("--n_hard_strict", type=int, default=10)
    parser.add_argument("--n_hard_relaxed_only", type=int, default=10)
    parser.add_argument("--n_hard_rejected", type=int, default=10)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rng = random.Random(args.seed)

    strict = read_jsonl(input_dir / "paths.filtered.strict.jsonl")
    relaxed = read_jsonl(input_dir / "paths.filtered.relaxed.jsonl")
    rejected = read_jsonl(input_dir / "paths.rejected.jsonl")

    # Separate by difficulty
    strict_easy = [x for x in strict if x.get("difficulty") == "Easy"]
    strict_medium = [x for x in strict if x.get("difficulty") == "Medium"]
    strict_hard = [x for x in strict if x.get("difficulty") == "Hard"]

    # Relaxed-only Hard (in relaxed but not in strict)
    strict_keys = {x.get("dedup_key", "") for x in strict}
    relaxed_only_hard = [x for x in relaxed
                         if x.get("difficulty") == "Hard"
                         and x.get("dedup_key", "") not in strict_keys]

    rejected_hard = [x for x in rejected if x.get("difficulty") == "Hard"]

    def sample(pool, n):
        if len(pool) <= n:
            return pool
        return rng.sample(pool, n)

    lines = []
    lines.append("# Manual Audit Sample\n")
    lines.append(f"**Source:** `{input_dir}`\n")
    lines.append(f"**Strict total:** {len(strict)} (Easy {len(strict_easy)}, Medium {len(strict_medium)}, Hard {len(strict_hard)})")
    lines.append(f"**Relaxed-only Hard:** {len(relaxed_only_hard)}")
    lines.append(f"**Rejected Hard:** {len(rejected_hard)}\n")

    # Easy strict
    lines.append(f"## Easy Strict Kept ({args.n_easy} samples)\n")
    for i, x in enumerate(sample(strict_easy, args.n_easy), 1):
        lines.append(f"### Easy #{i}\n")
        lines.append(fmt_item(x))
        lines.append("")

    # Medium strict
    lines.append(f"## Medium Strict Kept ({args.n_medium} samples)\n")
    for i, x in enumerate(sample(strict_medium, args.n_medium), 1):
        lines.append(f"### Medium #{i}\n")
        lines.append(fmt_item(x))
        lines.append("")

    # Hard strict
    lines.append(f"## Hard Strict Kept ({args.n_hard_strict} samples)\n")
    for i, x in enumerate(sample(strict_hard, args.n_hard_strict), 1):
        lines.append(f"### Hard Strict #{i}\n")
        lines.append(fmt_item(x))
        lines.append("")

    # Hard relaxed-only
    lines.append(f"## Hard Relaxed-Only (partial) ({args.n_hard_relaxed_only} samples)\n")
    lines.append("These are Hard paths with `can_write_path_dependent_question=partial`.\n")
    for i, x in enumerate(sample(relaxed_only_hard, args.n_hard_relaxed_only), 1):
        lines.append(f"### Hard Partial #{i}\n")
        lines.append(fmt_item(x))
        lines.append("")

    # Hard rejected
    lines.append(f"## Hard Rejected ({args.n_hard_rejected} samples)\n")
    for i, x in enumerate(sample(rejected_hard, args.n_hard_rejected), 1):
        lines.append(f"### Hard Rejected #{i}\n")
        lines.append(fmt_item(x))
        lines.append("")

    out_path = input_dir / "MANUAL_AUDIT_SAMPLE.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
