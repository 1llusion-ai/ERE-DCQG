"""Generate K=5 candidates per item x method for reranking evaluation.

Reads a balanced sample JSONL (50 per difficulty level = 150 items), generates
K candidates per item per method, and saves results for downstream reranking.

Usage::

    python -m scripts.run_k5_generation \
        --sample_path outputs/runs/crossqg_sample.jsonl \
        --output_dir outputs/runs/k5_generation/ \
        --K 5 --temperature 0.7 \
        --methods direct icl self_refine ours
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dcqg.generation.fairytale_qg import (
    generate_direct,
    generate_icl,
    generate_ours,
    generate_self_refine,
)
from dcqg.graph.narrative_graph import NarrativeGraphExtractor

logger = logging.getLogger(__name__)

METHOD_NAMES = ["direct", "icl", "self_refine", "ours"]


# ---------------------------------------------------------------------------
# Graph extraction (cached per story section)
# ---------------------------------------------------------------------------

_graph_cache: dict[str, dict] = {}


def _get_graph(candidate: dict) -> dict:
    """Extract or retrieve cached graph for a candidate."""
    story = candidate.get("story_section", "")
    key = story[:200]  # use prefix as cache key
    if key in _graph_cache:
        return _graph_cache[key]

    extractor = NarrativeGraphExtractor()
    difficulty = candidate.get("target_difficulty", "Hard")
    try:
        record = extractor.extract(candidate, difficulty=difficulty)
    except Exception as e:
        logger.warning("Graph extraction failed: %s", e)
        record = {
            "nodes": [],
            "edges": [],
            "graph_valid": False,
            "graph_validation_reason": f"extraction_error: {e}",
        }

    _graph_cache[key] = record
    return record


# ---------------------------------------------------------------------------
# Per-method generation wrappers
# ---------------------------------------------------------------------------

def _generate_candidates_direct(
    candidate: dict, K: int, temperature: float
) -> list[dict]:
    """Generate K candidates using Direct QG."""
    story = candidate["story_section"]
    answer = candidate.get("answer1", "") or candidate.get("answer", "")
    diff = candidate.get("target_difficulty", "Hard")

    results = []
    for k in range(K):
        try:
            result, attempts = generate_direct(story, answer, diff, max_retries=2)
        except Exception as e:
            result = {
                "generated_question": "",
                "method": "Direct",
                "parse_ok": False,
                "generation_error": str(e),
            }
            attempts = 1
        result["candidate_index"] = k
        result["candidate_attempts"] = attempts
        results.append(result)
    return results


def _generate_candidates_icl(
    candidate: dict, K: int, temperature: float
) -> list[dict]:
    """Generate K candidates using ICL QG."""
    story = candidate["story_section"]
    answer = candidate.get("answer1", "") or candidate.get("answer", "")
    diff = candidate.get("target_difficulty", "Hard")

    results = []
    for k in range(K):
        try:
            result, attempts = generate_icl(story, answer, diff, max_retries=2)
        except Exception as e:
            result = {
                "generated_question": "",
                "method": "ICL",
                "parse_ok": False,
                "generation_error": str(e),
            }
            attempts = 1
        result["candidate_index"] = k
        result["candidate_attempts"] = attempts
        results.append(result)
    return results


def _generate_candidates_self_refine(
    candidate: dict, K: int, temperature: float
) -> list[dict]:
    """Generate K candidates using SelfRefine QG."""
    story = candidate["story_section"]
    answer = candidate.get("answer1", "") or candidate.get("answer", "")
    diff = candidate.get("target_difficulty", "Hard")

    results = []
    for k in range(K):
        try:
            result, attempts = generate_self_refine(story, answer, diff, max_retries=1)
        except Exception as e:
            result = {
                "generated_question": "",
                "method": "SelfRefine",
                "parse_ok": False,
                "generation_error": str(e),
            }
            attempts = 1
        result["candidate_index"] = k
        result["candidate_attempts"] = attempts
        results.append(result)
    return results


def _generate_candidates_ours(
    candidate: dict, K: int, temperature: float
) -> list[dict]:
    """Generate K candidates using Ours (narrative evidence graph guided) QG."""
    story = candidate["story_section"]
    answer = candidate.get("answer1", "") or candidate.get("answer", "")
    diff = candidate.get("target_difficulty", "Hard")

    graph_data = _get_graph(candidate)
    if not graph_data.get("graph_valid", False):
        # Return K empty candidates when graph is invalid
        return [
            {
                "generated_question": "",
                "method": "Ours",
                "parse_ok": False,
                "generation_error": "graph_invalid",
                "candidate_index": k,
                "candidate_attempts": 0,
            }
            for k in range(K)
        ]

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    required_ev = candidate.get("required_evidence_sentences", [])
    bridge_ids = candidate.get("bridge_sentence_ids", [])
    reasoning_op = candidate.get("reasoning_operation", "")
    necessity = candidate.get("necessity_type", "")

    results = []
    for k in range(K):
        try:
            result, attempts = generate_ours(
                story, answer, diff,
                nodes=nodes, edges=edges,
                required_evidence_sentences=required_ev,
                bridge_sentence_ids=bridge_ids,
                reasoning_operation=reasoning_op,
                necessity_type=necessity,
                max_retries=3,
            )
        except Exception as e:
            result = {
                "generated_question": "",
                "method": "Ours",
                "parse_ok": False,
                "generation_error": str(e),
            }
            attempts = 1
        result["candidate_index"] = k
        result["candidate_attempts"] = attempts
        results.append(result)
    return results


GENERATOR_MAP = {
    "direct": ("Direct", _generate_candidates_direct),
    "icl": ("ICL", _generate_candidates_icl),
    "self_refine": ("SelfRefine", _generate_candidates_self_refine),
    "ours": ("Ours", _generate_candidates_ours),
}


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _load_existing(output_path: Path) -> set[str]:
    """Load already-processed item keys from an existing output file.

    Returns a set of ``story_name::question`` keys.
    """
    done: set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = f"{obj.get('story_name', '')}::{obj.get('question', '')}"
                done.add(key)
            except json.JSONDecodeError:
                continue
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate K candidates per item per method for reranking."
    )
    parser.add_argument(
        "--sample_path", required=True,
        help="Path to the balanced sample JSONL (e.g. 150 items, 50 per difficulty).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for per-method K-candidate files.",
    )
    parser.add_argument(
        "--K", type=int, default=5,
        help="Number of candidates to generate per item (default: 5).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Generation temperature (default: 0.7).",
    )
    parser.add_argument(
        "--methods", nargs="+", default=METHOD_NAMES,
        choices=METHOD_NAMES,
        help="Which methods to run (default: all 4).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output files, skipping already-processed items.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load sample
    sample_path = Path(args.sample_path)
    if not sample_path.exists():
        print(f"ERROR: sample not found at {sample_path}", file=sys.stderr)
        sys.exit(1)

    items: list[dict] = []
    with open(sample_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if not items:
        print("ERROR: no items loaded from sample", file=sys.stderr)
        sys.exit(1)

    # Difficulty distribution summary
    diff_counts = {}
    for it in items:
        d = it.get("target_difficulty", "?")
        diff_counts[d] = diff_counts.get(d, 0) + 1
    print(f"Loaded {len(items)} items: {diff_counts}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    K = args.K
    temperature = args.temperature

    # Run each method
    for method_key in args.methods:
        method_label, gen_fn = GENERATOR_MAP[method_key]
        output_path = output_dir / f"{method_key}_k{K}.jsonl"

        # Resume support
        done_keys: set[str] = set()
        if args.resume:
            done_keys = _load_existing(output_path)
            if done_keys:
                print(f"  [{method_label}] Resuming: {len(done_keys)} items already done")

        mode = "a" if args.resume and done_keys else "w"

        print(f"\n{'='*60}")
        print(f"Method: {method_label}  K={K}  items={len(items)}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")

        t0 = time.time()
        n_done = 0
        n_skipped = 0

        with open(output_path, mode, encoding="utf-8") as fout:
            for i, item in enumerate(items):
                item_key = f"{item.get('story_name', '')}::{item.get('question', '')}"
                if item_key in done_keys:
                    n_skipped += 1
                    continue

                candidates = gen_fn(item, K, temperature)

                # Build output record
                record = {
                    "story_name": item.get("story_name", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer1", "") or item.get("answer", ""),
                    "target_difficulty": item.get("target_difficulty", ""),
                    "story_section": item.get("story_section", ""),
                    "required_evidence_sentences": item.get("required_evidence_sentences", []),
                    "bridge_sentence_ids": item.get("bridge_sentence_ids", []),
                    "reasoning_operation": item.get("reasoning_operation", ""),
                    "necessity_type": item.get("necessity_type", ""),
                    "method": method_label,
                    "K": K,
                    "candidates": candidates,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

                n_done += 1
                if (n_done) % 10 == 0 or n_done == 1:
                    elapsed = time.time() - t0
                    rate = elapsed / n_done if n_done else 0
                    remaining = rate * (len(items) - n_skipped - n_done)
                    print(
                        f"  [{method_label}] {n_done}/{len(items) - n_skipped} "
                        f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
                    )

        elapsed = time.time() - t0
        print(
            f"  [{method_label}] Done: {n_done} generated, {n_skipped} skipped "
            f"({elapsed:.1f}s total)"
        )

    print(f"\nAll methods complete. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
