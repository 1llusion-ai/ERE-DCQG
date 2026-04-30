"""
LLM path-quality judge for DCQG.

This script runs a small pilot over sampled/prefiltered event paths and asks an
OpenAI-compatible model to judge whether each path can support a good question.

Default target: aihubmix + gpt-4o-mini.

Usage:
    python event_qg/src/path_llm_judge.py \
        --input event_qg/outputs/prefiltered_paths.jsonl \
        --output_dir event_qg/outputs/path_judge_pilot_gpt4omini \
        --sample_per_level 30
"""
import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


def load_env():
    """Load event_qg/.env without requiring python-dotenv."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

DEFAULT_API_URL = os.environ.get("AIHUBMIX_API_URL", "https://aihubmix.com/v1/chat/completions")
DEFAULT_API_KEY = os.environ.get("AIHUBMIX_API_KEY", "")
DEFAULT_MODEL = os.environ.get("AIHUBMIX_MODEL", "gpt-4o-mini")


DIFFICULTY_TO_STEPS = {
    "Easy": "1",
    "Medium": "2",
    "Hard": "3+",
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_supporting_sentences(item, max_sentences=12):
    lines = []
    for s in item.get("supporting_sentences", [])[:max_sentences]:
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            lines.append(f"[S{s[0]}] {s[1]}")
        elif isinstance(s, str):
            lines.append(s)
    return "\n".join(lines)


def format_event_path(item):
    events = item.get("events", [])
    parts = []
    for i, e in enumerate(events, start=1):
        trigger = e.get("trigger", "")
        event_type = e.get("type", "")
        sent_id = e.get("sent_id", "")
        parts.append(f"{i}. trigger=\"{trigger}\" type={event_type} sent=S{sent_id}")
    return "\n".join(parts)


def format_relations(item):
    rels = item.get("relation_subtypes", [])
    if not rels:
        return "NONE"
    return "\n".join(f"{i}. {r}" for i, r in enumerate(rels, start=1))


def get_final_event(item):
    events = item.get("events", [])
    return events[-1] if events else {}


def build_path_judge_prompt(item):
    difficulty = item.get("difficulty", "")
    final_event = get_final_event(item)
    final_trigger = final_event.get("trigger", item.get("answer_trigger", ""))
    final_type = final_event.get("type", item.get("gold_event_type", ""))
    answer_phrase = item.get("gold_answer_phrase", "")
    answer_sentence = item.get("gold_answer_sentence", "")

    rule_features = {
        "original_difficulty": difficulty,
        "target_steps_from_hop": DIFFICULTY_TO_STEPS.get(difficulty, ""),
        "relation_group": item.get("relation_group", item.get("relation_distribution", "")),
        "non_temporal_count": item.get("non_temporal_count", None),
        "support_span": item.get("support_span", item.get("num_supporting_sentences", None)),
        "rule_single_sentence_risk": item.get("rule_single_sentence_risk", ""),
        "prefilter_pass": item.get("prefilter_pass", None),
        "prefilter_reason": item.get("prefilter_reason", ""),
        "answer_phrase_pass": item.get("answer_phrase_pass", None),
        "answer_phrase_reason": item.get("answer_phrase_reason", ""),
    }

    return f"""You are judging event paths for controllable question generation.

Goal:
Determine whether the event path can support a natural question whose answer is the FINAL event.

Important:
- The original target difficulty below is only metadata from a rule-based sampler.
- Do NOT copy it automatically.
- Judge the actual path and context independently.
- A path should be "yes" only if it can naturally produce a clear, answerable question about the final event.
- If the proposed answer phrase is truncated, unnatural, or not a complete answer, mark path_questionable as "partial" or "no".
- If the final answer can be obtained from the answer sentence alone, single_sentence_risk should be "high".

Original target difficulty:
{difficulty}

Rule features:
{json.dumps(rule_features, ensure_ascii=False, indent=2)}

Event path:
{format_event_path(item)}

Relation sequence:
{format_relations(item)}

Final event:
- trigger: "{final_trigger}"
- type: {final_type}
- proposed answer phrase: "{answer_phrase}"
- answer sentence: "{answer_sentence}"

Supporting context:
{format_supporting_sentences(item)}

Judge these criteria:
1. path_questionable:
   - "yes": the path can support a good question about the final event.
   - "partial": usable but noisy, weak, or likely needs difficulty adjustment.
   - "no": unsuitable for question generation.
2. expected_required_steps:
   - "1": answer likely found from one sentence or one event.
   - "2": question should connect two events/sentences.
   - "3+": question can naturally require three or more event steps.
3. single_sentence_risk:
   - "low": unlikely answerable from one sentence alone.
   - "medium": possible shortcut.
   - "high": likely answerable from one sentence.
4. recommended_difficulty:
   - "easy", "medium", or "hard" based on the actual path/question potential.
   - choose "hard" only when the answer cannot be found from a single sentence and the path genuinely needs 3+ event steps.

Return ONLY one JSON object with this exact schema:
{{
  "path_questionable": "yes|partial|no",
  "expected_required_steps": "1|2|3+",
  "single_sentence_risk": "low|medium|high",
  "recommended_difficulty": "easy|medium|hard",
  "reason": "one short sentence"
}}"""


def parse_json_object(text):
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_label(value, allowed, default):
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in allowed:
        return v
    if allowed == {"1", "2", "3+"}:
        if "3" in v:
            return "3+"
        if "2" in v:
            return "2"
        if "1" in v:
            return "1"
    return default


def normalize_judge(parsed):
    if not isinstance(parsed, dict):
        return None
    return {
        "path_questionable": normalize_label(
            parsed.get("path_questionable"), {"yes", "partial", "no"}, "no"
        ),
        "expected_required_steps": normalize_label(
            parsed.get("expected_required_steps"), {"1", "2", "3+"}, "1"
        ),
        "single_sentence_risk": normalize_label(
            parsed.get("single_sentence_risk"), {"low", "medium", "high"}, "high"
        ),
        "recommended_difficulty": normalize_label(
            parsed.get("recommended_difficulty"), {"easy", "medium", "hard"}, "easy"
        ),
        "reason": str(parsed.get("reason", "")).strip()[:500],
    }


def call_openai_compatible(prompt, api_url, api_key, model, max_tokens=300, temperature=0.0,
                           timeout=90, json_mode=True):
    if not api_key:
        raise RuntimeError("AIHUBMIX_API_KEY is empty. Add it to event_qg/.env first.")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict JSON-only evaluator for event-path question generation.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def choose_keep_policy(item, judge):
    original = item.get("difficulty", "")
    pq = judge.get("path_questionable")
    risk = judge.get("single_sentence_risk")
    recommended = judge.get("recommended_difficulty")

    reasons = []
    keep = True

    if pq == "no":
        keep = False
        reasons.append("path_questionable=no")
    if original == "Hard" and risk == "high":
        keep = False
        reasons.append("hard_single_sentence_risk=high")
    if not recommended:
        keep = False
        reasons.append("missing_recommended_difficulty")

    if not reasons:
        reasons.append("keep")
    return keep, "; ".join(reasons)


def sample_items(items, sample_per_level, seed, include_failed_prefilter=False, limit=0):
    if not include_failed_prefilter:
        items = [x for x in items if x.get("prefilter_pass", True)]

    if limit and limit > 0:
        return items[:limit]

    if sample_per_level <= 0:
        return items

    rng = random.Random(seed)
    by_level = defaultdict(list)
    for item in items:
        by_level[item.get("difficulty", "Easy")].append(item)

    sampled = []
    for level in ["Easy", "Medium", "Hard"]:
        pool = list(by_level.get(level, []))
        rng.shuffle(pool)
        sampled.extend(pool[:sample_per_level])
    rng.shuffle(sampled)
    return sampled


def judge_paths(items, args):
    judged = []
    traces = []

    for i, item in enumerate(items, start=1):
        prompt = build_path_judge_prompt(item)
        raw = ""
        parsed = None
        normalized = None
        error = ""
        judge_status = "ok"

        if args.dry_run:
            raw = '{"path_questionable":"partial","expected_required_steps":"2","single_sentence_risk":"medium","recommended_difficulty":"medium","reason":"dry run"}'
        else:
            for attempt in range(1, args.retries + 1):
                try:
                    raw = call_openai_compatible(
                        prompt=prompt,
                        api_url=args.api_url,
                        api_key=args.api_key,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        timeout=args.timeout,
                        json_mode=not args.no_json_mode,
                    )
                    parsed = parse_json_object(raw)
                    normalized = normalize_judge(parsed)
                    if normalized:
                        error = ""
                        break
                    error = f"parse_failed attempt={attempt}"
                    judge_status = "parse_error"
                except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    judge_status = "api_error"
                time.sleep(args.sleep)

        if normalized is None:
            parsed = parse_json_object(raw)
            normalized = normalize_judge(parsed)
        if normalized is None:
            normalized = {
                "path_questionable": "unknown",
                "expected_required_steps": "unknown",
                "single_sentence_risk": "unknown",
                "recommended_difficulty": "unknown",
                "reason": f"judge_error: {error or 'parse_failed'}",
            }
            parse_ok = False
            # Keep judge_status as set above (api_error or parse_error)
        else:
            parse_ok = True
            judge_status = "ok"

        keep, keep_reason = choose_keep_policy(item, normalized)

        # Don't drop paths on API/parse errors — keep them for review
        if judge_status != "ok":
            keep = True
            keep_reason = f"kept_despite_{judge_status}: {error}"

        out = dict(item)
        out["llm_path_judge"] = normalized
        out["llm_path_judge_parse_ok"] = parse_ok
        out["llm_path_judge_status"] = judge_status
        out["llm_path_keep"] = keep
        out["llm_path_keep_reason"] = keep_reason
        out["llm_path_judge_model"] = args.model

        trace = {
            "index": i,
            "doc_id": item.get("doc_id", ""),
            "title": item.get("title", ""),
            "difficulty": item.get("difficulty", ""),
            "path": [
                {
                    "trigger": e.get("trigger", ""),
                    "type": e.get("type", ""),
                    "sent_id": e.get("sent_id", ""),
                }
                for e in item.get("events", [])
            ],
            "prompt": prompt,
            "raw_response": raw,
            "parsed": parsed,
            "normalized": normalized,
            "parse_ok": parse_ok,
            "keep": keep,
            "keep_reason": keep_reason,
            "error": error,
        }

        judged.append(out)
        traces.append(trace)

        if i % args.progress_every == 0 or i == len(items):
            kept = sum(1 for x in judged if x["llm_path_keep"])
            print(f"[{i}/{len(items)}] kept={kept} parse_ok={sum(1 for x in judged if x['llm_path_judge_parse_ok'])}")

        if not args.dry_run:
            time.sleep(args.sleep)

    return judged, traces


def agreement(original, recommended):
    return str(original).strip().lower() == str(recommended).strip().lower()


def build_report(judged, input_count, args):
    total = len(judged)
    kept = [x for x in judged if x.get("llm_path_keep")]
    parse_ok = [x for x in judged if x.get("llm_path_judge_parse_ok")]

    by_level = defaultdict(list)
    for x in judged:
        by_level[x.get("difficulty", "")].append(x)

    report = {
        "model": args.model,
        "input_count": input_count,
        "judged_count": total,
        "parse_ok_count": len(parse_ok),
        "parse_ok_rate": round(len(parse_ok) / total, 4) if total else 0,
        "kept_count": len(kept),
        "kept_rate": round(len(kept) / total, 4) if total else 0,
        "llm_path_judge_status_distribution": dict(Counter(x.get("llm_path_judge_status", "ok") for x in judged)),
        "path_questionable_distribution": dict(Counter(x["llm_path_judge"]["path_questionable"] for x in judged)),
        "expected_required_steps_distribution": dict(Counter(x["llm_path_judge"]["expected_required_steps"] for x in judged)),
        "single_sentence_risk_distribution": dict(Counter(x["llm_path_judge"]["single_sentence_risk"] for x in judged)),
        "recommended_difficulty_distribution": dict(Counter(x["llm_path_judge"]["recommended_difficulty"] for x in judged)),
        "per_level": {},
        "examples": {
            "not_kept": [],
            "disagreement": [],
            "hard_high_risk": [],
        },
    }

    for level in ["Easy", "Medium", "Hard"]:
        xs = by_level.get(level, [])
        if not xs:
            report["per_level"][level] = {"total": 0}
            continue
        kept_l = [x for x in xs if x.get("llm_path_keep")]
        agree_l = [
            x for x in xs
            if agreement(x.get("difficulty", ""), x["llm_path_judge"].get("recommended_difficulty", ""))
        ]
        high_risk_l = [x for x in xs if x["llm_path_judge"].get("single_sentence_risk") == "high"]
        report["per_level"][level] = {
            "total": len(xs),
            "kept": len(kept_l),
            "kept_rate": round(len(kept_l) / len(xs), 4),
            "difficulty_agreement": len(agree_l),
            "difficulty_agreement_rate": round(len(agree_l) / len(xs), 4),
            "single_sentence_high_risk": len(high_risk_l),
            "single_sentence_high_risk_rate": round(len(high_risk_l) / len(xs), 4),
            "recommended_difficulty_distribution": dict(Counter(x["llm_path_judge"]["recommended_difficulty"] for x in xs)),
            "expected_steps_distribution": dict(Counter(x["llm_path_judge"]["expected_required_steps"] for x in xs)),
        }

    def slim_example(x):
        events = x.get("events", [])
        return {
            "difficulty": x.get("difficulty", ""),
            "title": x.get("title", ""),
            "path": " -> ".join(f"{e.get('trigger','')}/{e.get('type','')}" for e in events),
            "relations": x.get("relation_subtypes", []),
            "answer_phrase": x.get("gold_answer_phrase", ""),
            "judge": x.get("llm_path_judge", {}),
            "keep_reason": x.get("llm_path_keep_reason", ""),
        }

    for x in judged:
        if not x.get("llm_path_keep") and len(report["examples"]["not_kept"]) < 5:
            report["examples"]["not_kept"].append(slim_example(x))
        if not agreement(x.get("difficulty", ""), x["llm_path_judge"].get("recommended_difficulty", "")) and len(report["examples"]["disagreement"]) < 5:
            report["examples"]["disagreement"].append(slim_example(x))
        if x.get("difficulty") == "Hard" and x["llm_path_judge"].get("single_sentence_risk") == "high" and len(report["examples"]["hard_high_risk"]) < 5:
            report["examples"]["hard_high_risk"].append(slim_example(x))

    return report


def write_report_md(report, path):
    lines = []
    lines.append("# LLM Path Judge Pilot Report\n")
    lines.append(f"**Model:** {report['model']}")
    lines.append(f"**Judged:** {report['judged_count']} / input {report['input_count']}")
    lines.append(f"**Parse OK:** {report['parse_ok_count']} ({report['parse_ok_rate']*100:.1f}%)")
    lines.append(f"**Kept:** {report['kept_count']} ({report['kept_rate']*100:.1f}%)\n")

    lines.append("## Overall Distributions\n")
    for key in [
        "path_questionable_distribution",
        "expected_required_steps_distribution",
        "single_sentence_risk_distribution",
        "recommended_difficulty_distribution",
    ]:
        lines.append(f"### {key}\n")
        lines.append("| Label | Count |")
        lines.append("|---|---:|")
        for label, count in report[key].items():
            lines.append(f"| {label} | {count} |")
        lines.append("")

    lines.append("## Per-Level Summary\n")
    lines.append("| Level | Total | Kept | Kept% | Diff Agree% | High Single-Sent Risk% | Recommended Difficulty | Expected Steps |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for level in ["Easy", "Medium", "Hard"]:
        s = report["per_level"].get(level, {"total": 0})
        if not s.get("total"):
            lines.append(f"| {level} | 0 | 0 | 0.0% | 0.0% | 0.0% | - | - |")
            continue
        rec = ", ".join(f"{k}:{v}" for k, v in s["recommended_difficulty_distribution"].items())
        steps = ", ".join(f"{k}:{v}" for k, v in s["expected_steps_distribution"].items())
        lines.append(
            f"| {level} | {s['total']} | {s['kept']} | {s['kept_rate']*100:.1f}% | "
            f"{s['difficulty_agreement_rate']*100:.1f}% | {s['single_sentence_high_risk_rate']*100:.1f}% | {rec} | {steps} |"
        )

    for name, examples in report["examples"].items():
        lines.append(f"\n## Examples: {name}\n")
        if not examples:
            lines.append("(none)\n")
            continue
        for ex in examples:
            lines.append(f"- [{ex['difficulty']}] {ex['title']}")
            lines.append(f"  - Path: {ex['path']}")
            lines.append(f"  - Answer phrase: {ex['answer_phrase']}")
            lines.append(f"  - Judge: {json.dumps(ex['judge'], ensure_ascii=False)}")
            lines.append(f"  - Keep reason: {ex['keep_reason']}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run LLM path-quality judge pilot.")
    parser.add_argument("--input", default="event_qg/outputs/prefiltered_paths.jsonl")
    parser.add_argument("--output_dir", default="event_qg/outputs/path_judge_pilot_gpt4omini")
    parser.add_argument("--sample_per_level", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Use first N items after filtering; overrides stratified sampling.")
    parser.add_argument("--include_failed_prefilter", action="store_true")
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument("--api_key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_json_mode", action="store_true")
    args = parser.parse_args()

    all_items = read_jsonl(args.input)
    items = sample_items(
        all_items,
        sample_per_level=args.sample_per_level,
        seed=args.seed,
        include_failed_prefilter=args.include_failed_prefilter,
        limit=args.limit,
    )

    print(f"Loaded {len(all_items)} paths from {args.input}")
    print(f"Judging {len(items)} paths with model={args.model}")
    if not args.dry_run and not args.api_key:
        raise SystemExit("AIHUBMIX_API_KEY is empty. Add it to event_qg/.env or pass --api_key.")

    judged, traces = judge_paths(items, args)
    report = build_report(judged, len(all_items), args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "judged_paths.jsonl", judged)
    write_jsonl(output_dir / "path_judge_trace.jsonl", traces)

    with open(output_dir / "path_judge_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_report_md(report, output_dir / "path_judge_report.md")

    print("\nDone.")
    print(f"  judged_paths: {output_dir / 'judged_paths.jsonl'}")
    print(f"  trace:        {output_dir / 'path_judge_trace.jsonl'}")
    print(f"  report_json:  {output_dir / 'path_judge_report.json'}")
    print(f"  report_md:    {output_dir / 'path_judge_report.md'}")
    print(f"  kept:         {report['kept_count']}/{report['judged_count']} ({report['kept_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()
