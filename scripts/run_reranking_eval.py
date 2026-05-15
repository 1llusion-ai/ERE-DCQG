"""Apply reranking conditions and compute Table 2 metrics.

Reads K=5 candidate files and K=1 baseline results, applies four reranking
conditions (K=1, K=5+classifier, K=5+LLM judge, K=5+random), and produces
a full metrics table with bootstrap 95% CIs.

Usage::

    python -m scripts.run_reranking_eval \
        --k5_dir outputs/runs/k5_generation/ \
        --k1_dir outputs/runs/fairytale_qg_crossqg_eval_20260511_v1/ \
        --classifier_path outputs/models/multitask_deberta_v1/fold_0/model/ \
        --output_dir outputs/runs/reranking_eval/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

METHODS = ["Direct", "ICL", "SelfRefine", "Ours"]
METHOD_FILE_MAP = {
    "Direct": "direct",
    "ICL": "icl",
    "SelfRefine": "self_refine",
    "Ours": "ours",
}
DIFFICULTIES = ["Easy", "Medium", "Hard"]
DIFF_ORD = {"Easy": 0, "Medium": 1, "Hard": 2}
CONDITIONS = ["K=1", "K=5+classifier", "K=5+LLM", "K=5+random"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_k5(k5_dir: Path, method_file_key: str, K: int) -> list[dict]:
    """Load K-candidate JSONL for a method."""
    path = k5_dir / f"{method_file_key}_k{K}.jsonl"
    if not path.exists():
        logger.warning("K5 file not found: %s", path)
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_k1(k1_dir: Path) -> dict[tuple[str, str, str, str], dict]:
    """Load K=1 judged results, keyed by (story_name, question, answer, method).

    Reads generations.judged.full.jsonl if available, else generations.judged.jsonl.
    """
    for fname in ("generations.judged.full.jsonl", "generations.judged.jsonl"):
        path = k1_dir / fname
        if path.exists():
            break
    else:
        logger.error("No judged JSONL found in %s", k1_dir)
        return {}

    lookup: dict[tuple[str, str, str, str], dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (
                obj.get("story_name", ""),
                obj.get("question", ""),
                obj.get("answer", ""),
                obj.get("method", ""),
            )
            lookup[key] = obj
    logger.info("Loaded %d K=1 results from %s", len(lookup), path)
    return lookup


# ---------------------------------------------------------------------------
# Judging helpers
# ---------------------------------------------------------------------------

def _judge_quality(question: str, story_section: str, target_answer: str,
                   difficulty: str) -> dict:
    """Quality gate using LLM judge."""
    from dcqg.generation.fairytale_qg import quality_judge
    return quality_judge(question, story_section, target_answer, difficulty)


def _judge_difficulty(question: str, story_section: str, target_answer: str,
                      difficulty: str) -> dict:
    """Difficulty assessment using LLM judge."""
    from dcqg.generation.fairytale_qg import difficulty_evidence_judge
    return difficulty_evidence_judge(question, story_section, target_answer, difficulty)


# ---------------------------------------------------------------------------
# Reranking conditions
# ---------------------------------------------------------------------------

def _apply_k1(item_k1: dict | None) -> dict[str, Any]:
    """Extract metrics from K=1 baseline result."""
    if item_k1 is None:
        return {
            "condition": "K=1",
            "selected_question": "",
            "quality_pass": False,
            "predicted_difficulty": "N/A",
            "target_difficulty": "N/A",
            "difficulty_match": False,
        }

    return {
        "condition": "K=1",
        "selected_question": item_k1.get("generated_question", ""),
        "quality_pass": item_k1.get("quality_pass", False),
        "predicted_difficulty": item_k1.get("predicted_difficulty", "N/A"),
        "target_difficulty": item_k1.get("target_difficulty", "N/A"),
        "difficulty_match": (
            item_k1.get("predicted_difficulty") == item_k1.get("target_difficulty")
        ),
    }


def _apply_classifier_rerank(
    candidates: list[dict], story_section: str, target_difficulty: str,
    target_answer: str, reranker: Any,
) -> dict[str, Any]:
    """K=5 + classifier reranking."""
    best, scored = reranker.rerank(
        candidates, story_section, target_difficulty, target_answer,
    )

    if best is None:
        return {
            "condition": "K=5+classifier",
            "selected_question": "",
            "quality_pass": False,
            "predicted_difficulty": "N/A",
            "target_difficulty": target_difficulty,
            "difficulty_match": False,
            "classifier_target_prob": 0.0,
            "classifier_predicted": "N/A",
        }

    return {
        "condition": "K=5+classifier",
        "selected_question": best.get("generated_question", ""),
        "quality_pass": True,
        "predicted_difficulty": best.get("classifier_predicted_difficulty", "N/A"),
        "target_difficulty": target_difficulty,
        "difficulty_match": (
            best.get("classifier_predicted_difficulty") == target_difficulty
        ),
        "classifier_target_prob": best.get("target_prob", 0.0),
        "classifier_predicted": best.get("classifier_predicted_difficulty", "N/A"),
    }


def _apply_llm_rerank(
    candidates: list[dict], story_section: str, target_difficulty: str,
    target_answer: str,
) -> dict[str, Any]:
    """K=5 + LLM judge reranking."""
    from dcqg.difficulty.reranker import llm_rerank

    # First apply quality gate to all candidates
    for c in candidates:
        q = c.get("generated_question", "")
        if not q or not q.strip():
            c["quality_pass"] = False
            continue
        if "quality_pass" not in c:
            qj = _judge_quality(q, story_section, target_answer, target_difficulty)
            c["quality_pass"] = qj.get("quality_pass", False)

    best, scored = llm_rerank(candidates, story_section, target_difficulty, target_answer)

    if best is None:
        return {
            "condition": "K=5+LLM",
            "selected_question": "",
            "quality_pass": False,
            "predicted_difficulty": "N/A",
            "target_difficulty": target_difficulty,
            "difficulty_match": False,
        }

    pred = best.get("llm_predicted_difficulty", "N/A")
    return {
        "condition": "K=5+LLM",
        "selected_question": best.get("generated_question", ""),
        "quality_pass": True,
        "predicted_difficulty": pred,
        "target_difficulty": target_difficulty,
        "difficulty_match": pred == target_difficulty,
    }


def _apply_random_rerank(
    candidates: list[dict], story_section: str, target_difficulty: str,
    target_answer: str, seed: int,
) -> dict[str, Any]:
    """K=5 + random selection among quality-passing."""
    from dcqg.difficulty.reranker import random_rerank

    # Ensure quality gate
    for c in candidates:
        q = c.get("generated_question", "")
        if not q or not q.strip():
            c["quality_pass"] = False
            continue
        if "quality_pass" not in c:
            qj = _judge_quality(q, story_section, target_answer, target_difficulty)
            c["quality_pass"] = qj.get("quality_pass", False)

    best, scored = random_rerank(candidates, seed=seed)

    if best is None:
        return {
            "condition": "K=5+random",
            "selected_question": "",
            "quality_pass": False,
            "predicted_difficulty": "N/A",
            "target_difficulty": target_difficulty,
            "difficulty_match": False,
        }

    # Judge difficulty of randomly selected candidate
    q = best.get("generated_question", "")
    dj = _judge_difficulty(q, story_section, target_answer, target_difficulty)
    pred = dj.get("predicted_difficulty", "N/A")

    return {
        "condition": "K=5+random",
        "selected_question": q,
        "quality_pass": True,
        "predicted_difficulty": pred,
        "target_difficulty": target_difficulty,
        "difficulty_match": pred == target_difficulty,
    }


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _accuracy(results: list[dict]) -> float:
    """Overall difficulty accuracy: fraction of items where predicted == target."""
    valid = [r for r in results if r.get("quality_pass")]
    if not valid:
        return 0.0
    return sum(1 for r in valid if r["difficulty_match"]) / len(valid)


def _per_class_accuracy(results: list[dict]) -> dict[str, float]:
    """Per-class difficulty accuracy."""
    acc = {}
    for diff in DIFFICULTIES:
        subset = [r for r in results if r.get("target_difficulty") == diff and r.get("quality_pass")]
        if subset:
            acc[diff] = sum(1 for r in subset if r["difficulty_match"]) / len(subset)
        else:
            acc[diff] = 0.0
    return acc


def _macro_accuracy(results: list[dict]) -> float:
    """Macro accuracy: mean of per-class accuracies."""
    pca = _per_class_accuracy(results)
    vals = [v for v in pca.values() if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _quality_rate(results: list[dict]) -> float:
    """Fraction of items that pass quality gate."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("quality_pass")) / len(results)


def _bootstrap_ci(
    results: list[dict],
    metric_fn,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for a metric function.

    Returns (observed, ci_low, ci_high).
    """
    rng = random.Random(seed)
    n = len(results)
    if n == 0:
        return 0.0, 0.0, 0.0

    observed = metric_fn(results)
    boot_values = []

    for _ in range(n_bootstrap):
        sample = [results[rng.randint(0, n - 1)] for _ in range(n)]
        boot_values.append(metric_fn(sample))

    boot_values.sort()
    ci_low = boot_values[max(0, int(0.025 * n_bootstrap))]
    ci_high = boot_values[min(n_bootstrap - 1, int(0.975 * n_bootstrap))]
    return observed, ci_low, ci_high


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _format_pct(val: float, ci_low: float = None, ci_high: float = None) -> str:
    """Format percentage with optional CI."""
    if ci_low is not None and ci_high is not None:
        return f"{100*val:.1f}% [{100*ci_low:.1f}, {100*ci_high:.1f}]"
    return f"{100*val:.1f}%"


def _build_table2(all_metrics: dict) -> str:
    """Build formatted Table 2 text.

    Structure: rows = methods, columns = conditions, cells = metrics.
    """
    lines = []
    lines.append("=" * 100)
    lines.append("Table 2: Reranking Evaluation Results")
    lines.append("=" * 100)
    lines.append("")

    # Header
    header = f"{'Method':<14}"
    for cond in CONDITIONS:
        header += f" | {cond:<24}"
    lines.append(header)
    lines.append("-" * len(header))

    # Difficulty accuracy rows
    lines.append("")
    lines.append("--- Difficulty Accuracy (overall) ---")
    for method in METHODS:
        row = f"{method:<14}"
        for cond in CONDITIONS:
            key = (method, cond)
            m = all_metrics.get(key, {})
            val = m.get("diff_acc", 0.0)
            ci_lo = m.get("diff_acc_ci_lo", 0.0)
            ci_hi = m.get("diff_acc_ci_hi", 0.0)
            row += f" | {_format_pct(val, ci_lo, ci_hi):<24}"
        lines.append(row)

    # Macro accuracy rows
    lines.append("")
    lines.append("--- Macro Accuracy (mean of per-class) ---")
    for method in METHODS:
        row = f"{method:<14}"
        for cond in CONDITIONS:
            key = (method, cond)
            m = all_metrics.get(key, {})
            val = m.get("macro_acc", 0.0)
            ci_lo = m.get("macro_acc_ci_lo", 0.0)
            ci_hi = m.get("macro_acc_ci_hi", 0.0)
            row += f" | {_format_pct(val, ci_lo, ci_hi):<24}"
        lines.append(row)

    # Per-class breakdown
    for diff in DIFFICULTIES:
        lines.append("")
        lines.append(f"--- {diff} Accuracy ---")
        for method in METHODS:
            row = f"{method:<14}"
            for cond in CONDITIONS:
                key = (method, cond)
                m = all_metrics.get(key, {})
                val = m.get(f"acc_{diff}", 0.0)
                row += f" | {_format_pct(val):<24}"
            lines.append(row)

    # Quality pass rate
    lines.append("")
    lines.append("--- Quality Pass Rate ---")
    for method in METHODS:
        row = f"{method:<14}"
        for cond in CONDITIONS:
            key = (method, cond)
            m = all_metrics.get(key, {})
            val = m.get("quality_rate", 0.0)
            row += f" | {_format_pct(val):<24}"
        lines.append(row)

    # Sample sizes
    lines.append("")
    lines.append("--- N (quality-passing) ---")
    for method in METHODS:
        row = f"{method:<14}"
        for cond in CONDITIONS:
            key = (method, cond)
            m = all_metrics.get(key, {})
            n = m.get("n_quality_pass", 0)
            n_total = m.get("n_total", 0)
            row += f" | {n}/{n_total:<22}"
        lines.append(row)

    lines.append("")
    lines.append("=" * 100)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    k5_dir: Path,
    k1_dir: Path,
    classifier_path: str,
    output_dir: Path,
    K: int = 5,
    n_bootstrap: int = 1000,
    skip_llm: bool = False,
):
    """Run the full reranking evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load K=1 results
    print("Loading K=1 results...")
    k1_lookup = _load_k1(k1_dir)
    if not k1_lookup:
        print("WARNING: No K=1 results loaded. K=1 condition will be empty.",
              file=sys.stderr)

    # Load classifier for reranking
    print(f"Loading classifier from {classifier_path}...")
    from dcqg.difficulty.reranker import DifficultyReranker
    reranker = DifficultyReranker(classifier_path)

    # Process each method
    all_metrics: dict[tuple[str, str], dict] = {}
    all_detailed: list[dict] = []

    for method in METHODS:
        method_file_key = METHOD_FILE_MAP[method]
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        # Load K=5 candidates
        k5_items = _load_k5(k5_dir, method_file_key, K)
        if not k5_items:
            print(f"  WARNING: No K=5 data for {method}, skipping")
            for cond in CONDITIONS:
                all_metrics[(method, cond)] = {
                    "diff_acc": 0.0, "macro_acc": 0.0, "quality_rate": 0.0,
                    "n_total": 0, "n_quality_pass": 0,
                }
            continue

        print(f"  Loaded {len(k5_items)} items with K={K} candidates each")

        # Collect results per condition
        cond_results: dict[str, list[dict]] = {c: [] for c in CONDITIONS}

        for idx, item in enumerate(k5_items):
            story_name = item.get("story_name", "")
            question = item.get("question", "")
            answer = item.get("answer", "")
            target_diff = item.get("target_difficulty", "")
            story_section = item.get("story_section", "")
            candidates = item.get("candidates", [])

            # Condition 1: K=1 (from pre-existing results)
            k1_key = (story_name, question, answer, method)
            k1_result = k1_lookup.get(k1_key)
            cond_results["K=1"].append(_apply_k1(k1_result))

            # Make deep copies of candidates for each condition to avoid
            # cross-contamination of quality_pass / rerank fields
            import copy

            # Condition 2: K=5 + classifier
            cands_clf = copy.deepcopy(candidates)
            clf_result = _apply_classifier_rerank(
                cands_clf, story_section, target_diff, answer, reranker,
            )
            cond_results["K=5+classifier"].append(clf_result)

            # Condition 3: K=5 + LLM judge
            if not skip_llm:
                cands_llm = copy.deepcopy(candidates)
                llm_result = _apply_llm_rerank(
                    cands_llm, story_section, target_diff, answer,
                )
                cond_results["K=5+LLM"].append(llm_result)
            else:
                cond_results["K=5+LLM"].append({
                    "condition": "K=5+LLM",
                    "selected_question": "",
                    "quality_pass": False,
                    "predicted_difficulty": "skipped",
                    "target_difficulty": target_diff,
                    "difficulty_match": False,
                })

            # Condition 4: K=5 + random
            cands_rnd = copy.deepcopy(candidates)
            rnd_result = _apply_random_rerank(
                cands_rnd, story_section, target_diff, answer,
                seed=42 + idx,
            )
            cond_results["K=5+random"].append(rnd_result)

            # Save detailed record
            detailed = {
                "story_name": story_name,
                "question": question,
                "answer": answer,
                "target_difficulty": target_diff,
                "method": method,
                "k1": cond_results["K=1"][-1],
                "k5_classifier": cond_results["K=5+classifier"][-1],
                "k5_llm": cond_results["K=5+LLM"][-1],
                "k5_random": cond_results["K=5+random"][-1],
            }
            all_detailed.append(detailed)

            if (idx + 1) % 25 == 0:
                print(f"  [{method}] Processed {idx+1}/{len(k5_items)}")

        # Compute metrics for each condition
        for cond in CONDITIONS:
            results = cond_results[cond]
            n_total = len(results)
            n_qp = sum(1 for r in results if r.get("quality_pass"))

            obs_acc, ci_lo_acc, ci_hi_acc = _bootstrap_ci(
                results, _accuracy, n_bootstrap=n_bootstrap
            )
            obs_mac, ci_lo_mac, ci_hi_mac = _bootstrap_ci(
                results, _macro_accuracy, n_bootstrap=n_bootstrap
            )

            pca = _per_class_accuracy(results)
            qr = _quality_rate(results)

            metrics = {
                "diff_acc": obs_acc,
                "diff_acc_ci_lo": ci_lo_acc,
                "diff_acc_ci_hi": ci_hi_acc,
                "macro_acc": obs_mac,
                "macro_acc_ci_lo": ci_lo_mac,
                "macro_acc_ci_hi": ci_hi_mac,
                "quality_rate": qr,
                "n_total": n_total,
                "n_quality_pass": n_qp,
            }
            for diff in DIFFICULTIES:
                metrics[f"acc_{diff}"] = pca.get(diff, 0.0)

            all_metrics[(method, cond)] = metrics

            print(
                f"  [{method}][{cond}] acc={100*obs_acc:.1f}% "
                f"macro={100*obs_mac:.1f}% quality={100*qr:.1f}% "
                f"(n={n_qp}/{n_total})"
            )

    # Save results
    print(f"\nSaving results to {output_dir}/")

    # Table 2 JSON (serializable keys)
    metrics_json = {}
    for (method, cond), m in all_metrics.items():
        metrics_json[f"{method}::{cond}"] = m

    with open(output_dir / "table2_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    # Table 2 formatted text
    table_text = _build_table2(all_metrics)
    with open(output_dir / "table2.txt", "w", encoding="utf-8") as f:
        f.write(table_text)
    print(f"\n{table_text}")

    # Detailed results
    with open(output_dir / "detailed_results.jsonl", "w", encoding="utf-8") as f:
        for rec in all_detailed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. Files written:")
    print(f"  {output_dir / 'table2_results.json'}")
    print(f"  {output_dir / 'table2.txt'}")
    print(f"  {output_dir / 'detailed_results.jsonl'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply reranking conditions and compute Table 2 metrics."
    )
    parser.add_argument(
        "--k5_dir", required=True,
        help="Directory containing K=5 candidate files (from run_k5_generation).",
    )
    parser.add_argument(
        "--k1_dir", required=True,
        help="Directory containing K=1 judged results (from run_crossqg_eval).",
    )
    parser.add_argument(
        "--classifier_path", required=True,
        help="Path to the trained MultiTaskDifficultyClassifier model directory.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--K", type=int, default=5,
        help="Number of candidates per item (must match k5 files; default: 5).",
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=1000,
        help="Number of bootstrap resamples for CIs (default: 1000).",
    )
    parser.add_argument(
        "--skip_llm", action="store_true",
        help="Skip K=5+LLM condition (saves API calls).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_evaluation(
        k5_dir=Path(args.k5_dir),
        k1_dir=Path(args.k1_dir),
        classifier_path=args.classifier_path,
        output_dir=Path(args.output_dir),
        K=args.K,
        n_bootstrap=args.n_bootstrap,
        skip_llm=args.skip_llm,
    )


if __name__ == "__main__":
    main()
