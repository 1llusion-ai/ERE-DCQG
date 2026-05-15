"""Run LLM judges on test samples for difficulty assessment.

Uses two LLM judges (GPT-4o-mini via AIHUBMIX, Qwen-32B via SiliconFlow),
each called n_runs times with temperature=0.0.  Takes majority vote per judge.

Usage:
    python -m scripts.run_llm_judge_difficulty \
        --samples_path outputs/eval/test_200.jsonl \
        --output_path outputs/eval/llm_judge_results.jsonl \
        --n_runs 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.utils.config import get_api_config
from dcqg.utils.api_client import call_openai_compatible
from dcqg.difficulty.definitions import difficulty_definitions_block

VALID_DIFFICULTIES = {"Easy", "Medium", "Hard"}

# ── Counters ──
api_call_count = 0
parse_error_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run LLM judges on test samples for difficulty assessment.",
    )
    p.add_argument(
        "--samples_path",
        type=str,
        default="outputs/eval/test_200.jsonl",
        help="Path to test samples JSONL.",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="outputs/eval/llm_judge_results.jsonl",
        help="Output path for judge results.",
    )
    p.add_argument("--n_runs", type=int, default=3, help="Number of runs per judge per sample.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt & Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def build_difficulty_prompt(item: dict) -> str:
    """Build a prompt that shows story section + question + answer and asks
    the judge to classify difficulty as Easy / Medium / Hard."""
    story = item.get("story_section") or item.get("context") or item.get("story", "")
    question = item.get("question") or item.get("generated_question", "")
    answer = (
        item.get("answer")
        or item.get("gold_answer_phrase")
        or item.get("gold_answer_trigger", "")
    )

    return f"""You are an expert reading comprehension evaluator.

## Story Section
{story}

## Question
{question}

## Answer
{answer}

## Difficulty Definitions

{difficulty_definitions_block()}

## Task

Classify the difficulty of this question into exactly one of: Easy, Medium, or Hard,
following the difficulty definitions above.

Reply as a JSON object with exactly these fields:
{{"difficulty": "Easy|Medium|Hard", "reasoning": "one sentence explanation"}}"""


def parse_difficulty_response(resp: str | None) -> str | None:
    """Extract a difficulty label from the judge response. Returns None on failure."""
    if not resp:
        return None

    # Try JSON parse
    try:
        obj = json.loads(resp)
        d = obj.get("difficulty", "")
        if d in VALID_DIFFICULTIES:
            return d
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try extracting JSON from markdown fences
    try:
        s = resp.index("{")
        e = resp.rindex("}") + 1
        obj = json.loads(resp[s:e])
        d = obj.get("difficulty", "")
        if d in VALID_DIFFICULTIES:
            return d
    except (ValueError, json.JSONDecodeError):
        pass

    # Regex fallback
    m = re.search(r'"difficulty"\s*:\s*"(Easy|Medium|Hard)"', resp, re.IGNORECASE)
    if m:
        label = m.group(1).capitalize()
        if label in VALID_DIFFICULTIES:
            return label

    # Last resort: check if any label appears
    for label in VALID_DIFFICULTIES:
        if label.lower() in resp.lower():
            return label

    return None


def majority_vote(votes: list[str]) -> str:
    """Return the most common label. Ties broken by Easy < Medium < Hard ordering."""
    if not votes:
        return "Medium"
    counts = Counter(votes)
    max_count = max(counts.values())
    # Among labels with max count, pick the one earliest in difficulty order
    for label in ["Easy", "Medium", "Hard"]:
        if counts.get(label, 0) == max_count:
            return label
    return "Medium"


# ═══════════════════════════════════════════════════════════════════════════════
# Judge call
# ═══════════════════════════════════════════════════════════════════════════════

def call_judge_once(
    prompt: str,
    api_url: str,
    api_key: str,
    model: str,
) -> str | None:
    """Call a single LLM judge once. Returns parsed difficulty or None."""
    global api_call_count, parse_error_count
    try:
        resp = call_openai_compatible(
            prompt,
            api_url=api_url,
            api_key=api_key,
            model=model,
            temperature=0.0,
            max_tokens=200,
            json_mode=True,
            system="You are a strict JSON-only difficulty evaluator.",
        )
        api_call_count += 1
        label = parse_difficulty_response(resp)
        if label is None:
            parse_error_count += 1
        return label
    except Exception as exc:
        api_call_count += 1
        print(f"    API error: {type(exc).__name__}: {exc}")
        return None


def run_judge_n_times(
    prompt: str,
    api_url: str,
    api_key: str,
    model: str,
    n_runs: int,
) -> tuple[list[str], str]:
    """Run the judge n_runs times, collect votes, return (votes, majority)."""
    votes: list[str] = []
    for _ in range(n_runs):
        label = call_judge_once(prompt, api_url, api_key, model)
        if label is not None:
            votes.append(label)
        time.sleep(0.1)  # rate-limit courtesy
    maj = majority_vote(votes)
    return votes, maj


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global api_call_count, parse_error_count
    api_call_count = 0
    parse_error_count = 0

    args = parse_args()

    print(f"Loading samples from: {args.samples_path}")
    samples = read_jsonl(args.samples_path)
    print(f"Loaded {len(samples)} samples, n_runs={args.n_runs}")

    cfg = get_api_config()

    # Judge 1: GPT-4o-mini via AIHUBMIX
    gpt_url = cfg["AIHUBMIX_API_URL"]
    gpt_key = cfg["AIHUBMIX_API_KEY"]
    gpt_model = cfg["AIHUBMIX_MODEL"]

    # Judge 2: Qwen-32B via SiliconFlow
    qwen_url = cfg["SILICONFLOW_API_URL"]
    qwen_key = cfg["SILICONFLOW_API_KEY"]
    qwen_model = cfg.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")

    print(f"Judge 1: {gpt_model} (AIHUBMIX)")
    print(f"Judge 2: {qwen_model} (SiliconFlow)")
    print(f"Estimated API calls: {len(samples) * args.n_runs * 2}")
    print("=" * 60)

    results: list[dict] = []

    for i, item in enumerate(samples):
        story_name = item.get("story_name") or item.get("doc_id", "?")
        question = (item.get("question") or item.get("generated_question", ""))[:60]
        ground_truth = (
            item.get("difficulty_label")
            or item.get("difficulty")
            or item.get("label", "?")
        )

        print(f"[{i + 1}/{len(samples)}] {story_name} | {ground_truth} | {question}...")

        prompt = build_difficulty_prompt(item)

        # Judge 1: GPT-4o-mini
        gpt_votes, gpt_maj = run_judge_n_times(
            prompt, gpt_url, gpt_key, gpt_model, args.n_runs,
        )

        # Judge 2: Qwen-32B
        qwen_votes, qwen_maj = run_judge_n_times(
            prompt, qwen_url, qwen_key, qwen_model, args.n_runs,
        )

        result = {
            "story_name": story_name,
            "question": item.get("question") or item.get("generated_question", ""),
            "ground_truth": ground_truth,
            "gpt4omini_votes": gpt_votes,
            "gpt4omini_majority": gpt_maj,
            "qwen32b_votes": qwen_votes,
            "qwen32b_majority": qwen_maj,
        }
        results.append(result)

        print(
            f"  GPT-4o-mini: {gpt_votes} -> {gpt_maj} | "
            f"Qwen-32B: {qwen_votes} -> {qwen_maj}"
        )

    write_jsonl(args.output_path, results)
    print(f"\nSaved {len(results)} results to: {args.output_path}")
    print(f"Total API calls: {api_call_count}")
    print(f"Parse errors: {parse_error_count}")


if __name__ == "__main__":
    main()
