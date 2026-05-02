"""
LLM path-quality judge for DCQG.

Runs a pilot over sampled/prefiltered event paths and asks an
OpenAI-compatible model to judge whether each path can support a good question.

Default target: aihubmix + gpt-4o-mini.
"""
import json
import random
import re
import time
import urllib.error
from collections import Counter, defaultdict
from pathlib import Path

from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.utils.api_client import call_openai_compatible
from dcqg.utils.config import get_api_config

_cfg = get_api_config()
DEFAULT_API_URL = _cfg["AIHUBMIX_API_URL"]
DEFAULT_API_KEY = _cfg["AIHUBMIX_API_KEY"]
DEFAULT_MODEL = _cfg["AIHUBMIX_MODEL"]


DIFFICULTY_TO_STEPS = {
    "Easy": "1",
    "Medium": "2",
    "Hard": "3+",
}


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
5. can_write_path_dependent_question:
   - "yes": even if the answer is in one sentence, you can write a question that REQUIRES knowing about earlier path events to understand or answer correctly.
   - "partial": the question could mention path events but they are not strictly necessary.
   - "no": no meaningful question can be written that depends on the path; the answer can be found without any path context.

Return ONLY one JSON object with this exact schema:
{{
  "path_questionable": "yes|partial|no",
  "expected_required_steps": "1|2|3+",
  "single_sentence_risk": "low|medium|high",
  "recommended_difficulty": "easy|medium|hard",
  "can_write_path_dependent_question": "yes|partial|no",
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
        "can_write_path_dependent_question": normalize_label(
            parsed.get("can_write_path_dependent_question"), {"yes", "partial", "no"}, "no"
        ),
        "reason": str(parsed.get("reason", "")).strip()[:500],
    }


def choose_keep_policy(item, judge):
    """Return a dict with strict and relaxed keep decisions.

    Keys:  strict_keep, relaxed_keep, strict_reason, relaxed_reason, risk_note
    """
    original = item.get("difficulty", "")
    pq = judge.get("path_questionable")
    risk = judge.get("single_sentence_risk")
    recommended = judge.get("recommended_difficulty")
    path_dep = judge.get("can_write_path_dependent_question")
    ap_pass = item.get("answer_phrase_pass", True)

    strict_reasons = []
    relaxed_reasons = []
    risk_notes = []
    strict_keep = True
    relaxed_keep = True

    # --- shared: answer_phrase_pass must be True ---
    if not ap_pass:
        strict_keep = False
        relaxed_keep = False
        strict_reasons.append("answer_phrase_fail")
        relaxed_reasons.append("answer_phrase_fail")

    # --- shared: path_questionable=no always rejects ---
    if pq == "no":
        strict_keep = False
        relaxed_keep = False
        strict_reasons.append("path_questionable=no")
        relaxed_reasons.append("path_questionable=no")

    # --- shared: missing recommended difficulty ---
    if not recommended:
        strict_keep = False
        relaxed_keep = False
        strict_reasons.append("missing_recommended_difficulty")
        relaxed_reasons.append("missing_recommended_difficulty")

    # --- Hard: strict requires path_dep=yes, relaxed accepts yes+partial ---
    if original == "Hard":
        if path_dep == "yes":
            # Hard path_dep=yes: passes both strict and relaxed
            if not strict_reasons and not relaxed_reasons:
                strict_reasons.append("keep_path_dep_yes")
                relaxed_reasons.append("keep_path_dep_yes")
        elif path_dep == "partial":
            # Hard path_dep=partial: fails strict, passes relaxed
            strict_keep = False
            strict_reasons.append("hard_path_dep=partial")
            if not relaxed_reasons:
                relaxed_reasons.append("keep_path_dep_partial")
        else:
            # path_dep=no or unknown: fails both (unless already failed above)
            strict_keep = False
            relaxed_keep = False
            if not any("path_questionable" in r for r in strict_reasons):
                strict_reasons.append(f"hard_path_dep={path_dep}")
            if not any("path_questionable" in r for r in relaxed_reasons):
                relaxed_reasons.append(f"hard_path_dep={path_dep}")

    # --- risk_note: single_sentence_risk=high is informational, not a filter ---
    if risk == "high":
        risk_notes.append("single_sentence_risk=high")

    if not strict_reasons:
        strict_reasons.append("keep")
    if not relaxed_reasons:
        relaxed_reasons.append("keep")

    return {
        "strict_keep": strict_keep,
        "relaxed_keep": relaxed_keep,
        "strict_reason": "; ".join(strict_reasons),
        "relaxed_reason": "; ".join(relaxed_reasons),
        "risk_note": "; ".join(risk_notes) if risk_notes else "",
    }


def apply_policy(judged_items):
    """Apply strict/relaxed policy and add policy fields to each item.

    Returns the items with policy fields added (in-place on copies).
    """
    out = []
    for item in judged_items:
        judge = item.get("llm_path_judge", {})
        policy = choose_keep_policy(item, judge)
        row = dict(item)
        row["policy_strict_keep"] = policy["strict_keep"]
        row["policy_relaxed_keep"] = policy["relaxed_keep"]
        row["policy_strict_reason"] = policy["strict_reason"]
        row["policy_relaxed_reason"] = policy["relaxed_reason"]
        row["risk_note"] = policy["risk_note"]
        out.append(row)
    return out


def deduplicate(items):
    """Deduplicate by doc_id + answer_event_id, then by doc_id + normalized phrase.

    Returns (deduped_items, removed_items) with dedup fields added.
    """
    seen_keys = set()
    kept = []
    removed = []

    for item in items:
        doc_id = item.get("doc_id", "")
        event_id = item.get("answer_event_id", "")
        phrase = item.get("gold_answer_phrase", "").lower().strip()

        # Primary key: doc_id + answer_event_id
        key = f"{doc_id}::{event_id}"
        if key in seen_keys:
            row = dict(item)
            row["dedup_key"] = key
            row["dedup_removed"] = True
            row["dedup_reason"] = "duplicate doc_id+answer_event_id"
            removed.append(row)
            continue

        # Fallback: doc_id + normalized phrase
        norm_phrase = " ".join(phrase.split())
        fallback_key = f"{doc_id}::phrase::{norm_phrase}"
        if fallback_key in seen_keys:
            row = dict(item)
            row["dedup_key"] = fallback_key
            row["dedup_removed"] = True
            row["dedup_reason"] = "duplicate doc_id+answer_phrase"
            removed.append(row)
            continue

        seen_keys.add(key)
        seen_keys.add(fallback_key)
        row = dict(item)
        row["dedup_key"] = key
        row["dedup_removed"] = False
        row["dedup_reason"] = ""
        kept.append(row)

    return kept, removed


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
            raw = '{"path_questionable":"partial","expected_required_steps":"2","single_sentence_risk":"medium","recommended_difficulty":"medium","can_write_path_dependent_question":"partial","reason":"dry run"}'
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
                except Exception as exc:
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
                "can_write_path_dependent_question": "unknown",
                "reason": f"judge_error: {error or 'parse_failed'}",
            }
            parse_ok = False
            # Keep judge_status as set above (api_error or parse_error)
        else:
            parse_ok = True
            judge_status = "ok"

        policy = choose_keep_policy(item, normalized)
        # For backward compat in judge_paths, use relaxed_keep as the main keep
        keep = policy["relaxed_keep"]
        keep_reason = policy["relaxed_reason"]

        # Don't drop paths on API/parse errors -- keep them for review
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
        out["llm_path_judge_prompt"] = prompt
        out["llm_path_judge_raw_response"] = raw

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
            "policy_strict_keep": policy["strict_keep"],
            "policy_strict_reason": policy["strict_reason"],
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
        "can_write_path_dependent_question_distribution": dict(Counter(x["llm_path_judge"].get("can_write_path_dependent_question", "unknown") for x in judged)),
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
        "can_write_path_dependent_question_distribution",
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


def generate_filter_report(all_items, strict_items, relaxed_items, rejected_items,
                            dedup_removed_strict, dedup_removed_relaxed,
                            raw_count, prefiltered_count, report_path):
    """Generate PATH_FILTER_REPORT.md with comprehensive statistics."""

    def by_level(items):
        d = defaultdict(list)
        for x in items:
            d[x.get("difficulty", "?")].append(x)
        return d

    all_by = by_level(all_items)
    strict_by = by_level(strict_items)
    relaxed_by = by_level(relaxed_items)
    rej_by = by_level(rejected_items)
    dedup_strict_by = by_level(dedup_removed_strict)
    dedup_relaxed_by = by_level(dedup_removed_relaxed)

    # Hard analysis
    hard_all = all_by.get("Hard", [])
    hard_strict_final = strict_by.get("Hard", [])
    hard_relaxed_only = [x for x in relaxed_by.get("Hard", [])
                         if x.get("dedup_key") not in {y.get("dedup_key") for y in strict_items if y.get("difficulty") == "Hard"}]
    hard_rejected = rej_by.get("Hard", [])

    # Strict candidates before dedup = final strict + dedup-removed strict
    hard_strict_candidates = len(hard_strict_final) + len(dedup_strict_by.get("Hard", []))

    # path_dep distribution for Hard (from all judged)
    pd_dist = Counter()
    for h in hard_all:
        judge = h.get("llm_path_judge", {})
        pd_dist[judge.get("can_write_path_dependent_question", "?")] += 1

    # relation group for Hard strict vs partial
    hard_strict_rel = Counter(x.get("relation_group", "?") for x in hard_strict_final)
    hard_partial = [x for x in hard_all
                    if x.get("llm_path_judge", {}).get("can_write_path_dependent_question") == "partial"]
    hard_partial_rel = Counter(x.get("relation_group", "?") for x in hard_partial)

    # Top reject reasons
    rej_reasons = Counter()
    for x in rejected_items:
        reason = x.get("policy_relaxed_reason", x.get("policy_strict_reason", ""))
        for part in reason.split("; "):
            rej_reasons[part.strip()] += 1

    # Top weak triggers
    weak_triggers = Counter()
    for x in all_items:
        wt = x.get("weak_trigger_type", "none")
        if wt != "none":
            weak_triggers[wt] += 1

    def fmt_example(x, n=120):
        events = x.get("events", [])
        path_str = " -> ".join(e.get("trigger", "") for e in events)
        phrase = x.get("gold_answer_phrase", "")[:n]
        sentence = x.get("gold_answer_sentence", "")[:n]
        judge = x.get("llm_path_judge", {})
        pd = judge.get("can_write_path_dependent_question", "?")
        pq = judge.get("path_questionable", "?")
        risk = judge.get("single_sentence_risk", "?")
        strict_r = x.get("policy_strict_reason", "")
        relaxed_r = x.get("policy_relaxed_reason", "")
        risk_note = x.get("risk_note", "")
        lines = []
        lines.append(f"- **{x.get('title', '')[:50]}** [{x.get('difficulty', '?')}]")
        lines.append(f"  - doc_id: `{x.get('doc_id', '')}`")
        lines.append(f"  - path: {path_str}")
        lines.append(f"  - relations: {', '.join(x.get('relation_subtypes', []))}")
        lines.append(f"  - answer_phrase: `{phrase}`")
        lines.append(f"  - answer_sentence: {sentence}")
        lines.append(f"  - judge: pq={pq} risk={risk} path_dep={pd}")
        lines.append(f"  - strict_reason: {strict_r}")
        lines.append(f"  - relaxed_reason: {relaxed_r}")
        if risk_note:
            lines.append(f"  - risk_note: {risk_note}")
        return "\n".join(lines)

    lines = []
    lines.append("# Path Filter Report\n")

    lines.append("## Pipeline Summary\n")
    lines.append("| Stage | Easy | Medium | Hard | Total |")
    lines.append("|-------|-----:|-------:|-----:|------:|")
    lines.append(f"| Raw paths | - | - | - | {raw_count} |")
    lines.append(f"| Prefiltered | - | - | - | {prefiltered_count} |")
    lines.append(f"| LLM judged | {len(all_by.get('Easy',[]))} | {len(all_by.get('Medium',[]))} | {len(all_by.get('Hard',[]))} | {len(all_items)} |")
    # Strict pipeline: candidates → dedup → final
    strict_cand_by = defaultdict(int)
    for k, v in strict_by.items():
        strict_cand_by[k] += v.__len__() if hasattr(v, '__len__') else 0
    for k, v in dedup_strict_by.items():
        strict_cand_by[k] += v.__len__() if hasattr(v, '__len__') else 0
    lines.append(f"| Strict candidates | {strict_cand_by.get('Easy',0)} | {strict_cand_by.get('Medium',0)} | {strict_cand_by.get('Hard',0)} | {sum(strict_cand_by.values())} |")
    lines.append(f"| Strict dedup removed | {len(dedup_strict_by.get('Easy',[]))} | {len(dedup_strict_by.get('Medium',[]))} | {len(dedup_strict_by.get('Hard',[]))} | {len(dedup_removed_strict)} |")
    lines.append(f"| **Strict final** | {len(strict_by.get('Easy',[]))} | {len(strict_by.get('Medium',[]))} | {len(strict_by.get('Hard',[]))} | {len(strict_items)} |")
    lines.append(f"| Relaxed final | {len(relaxed_by.get('Easy',[]))} | {len(relaxed_by.get('Medium',[]))} | {len(relaxed_by.get('Hard',[]))} | {len(relaxed_items)} |")
    lines.append(f"| Rejected | {len(rej_by.get('Easy',[]))} | {len(rej_by.get('Medium',[]))} | {len(rej_by.get('Hard',[]))} | {len(rejected_items)} |")

    lines.append("\n## Policy\n")
    lines.append("- **Easy/Medium:** keep if `path_questionable in {yes, partial}` and `answer_phrase_pass=True`")
    lines.append("- **Hard strict:** `path_questionable in {yes, partial}` AND `can_write_path_dependent_question=yes` AND `answer_phrase_pass=True`")
    lines.append("- **Hard relaxed:** same but accepts `can_write_path_dependent_question in {yes, partial}`")
    lines.append("- Dedup: `doc_id + answer_event_id`, fallback `doc_id + normalized(answer_phrase)`\n")

    lines.append("## Hard Path Analysis\n")
    lines.append("### can_write_path_dependent_question Distribution\n")
    lines.append("| Label | Count | % of Hard judged |")
    lines.append("|-------|------:|-----------------:|")
    for label in ["yes", "partial", "no"]:
        c = pd_dist.get(label, 0)
        pct = c / len(hard_all) * 100 if hard_all else 0
        lines.append(f"| {label} | {c} | {pct:.1f}% |")

    lines.append("\n### Hard Strict Pipeline\n")
    lines.append(f"- LLM judged Hard path_dep=yes: **{pd_dist.get('yes', 0)}/{len(hard_all)}** ({pd_dist.get('yes',0)/len(hard_all)*100:.0f}% of judged)" if hard_all else "")
    lines.append(f"- Hard strict candidates before dedup: **{hard_strict_candidates}**")
    lines.append(f"- Hard strict removed by dedup: **{len(dedup_strict_by.get('Hard',[]))}**")
    lines.append(f"- Hard strict final (deduped): **{len(hard_strict_final)}**")
    lines.append("")
    lines.append("LLM judged a significant portion of Hard paths as path-dependent, but many")
    lines.append("share the same final answer event. Deduplication by `doc_id + answer_event_id`")
    lines.append("reduces strict Hard paths to unique final-answer items for QG.\n")

    lines.append("### Hard Strict vs Partial — Relation Group\n")
    lines.append("| Relation | Strict (path_dep=yes) | Partial (path_dep=partial) |")
    lines.append("|----------|----------------------:|---------------------------:|")
    all_rels = set(list(hard_strict_rel.keys()) + list(hard_partial_rel.keys()))
    for rel in sorted(all_rels):
        lines.append(f"| {rel} | {hard_strict_rel.get(rel, 0)} | {hard_partial_rel.get(rel, 0)} |")

    lines.append("\n## Top Reject Reasons\n")
    lines.append("| Reason | Count |")
    lines.append("|--------|------:|")
    for reason, count in rej_reasons.most_common(10):
        lines.append(f"| {reason} | {count} |")

    if weak_triggers:
        lines.append("\n## Top Weak Trigger Types\n")
        lines.append("| Type | Count |")
        lines.append("|------|------:|")
        for wt, count in weak_triggers.most_common(10):
            lines.append(f"| {wt} | {count} |")

    lines.append("\n## Examples: Hard Strict Kept (5)\n")
    for x in hard_strict_final[:5]:
        lines.append(fmt_example(x))

    lines.append("\n## Examples: Hard Relaxed-Only Partial (5)\n")
    for x in hard_relaxed_only[:5]:
        lines.append(fmt_example(x))

    lines.append("\n## Examples: Hard Rejected (5)\n")
    for x in hard_rejected[:5]:
        lines.append(fmt_example(x))

    lines.append("\n## Conclusion\n")
    lines.append(f"- Strict paths ready for QG: **{len(strict_items)}** (Easy {len(strict_by.get('Easy',[]))}, Medium {len(strict_by.get('Medium',[]))}, Hard {len(hard_strict_final)})")
    lines.append(f"- Relaxed-only Hard partial candidates: **{len(hard_relaxed_only)}** (for future analysis)")
    if len(hard_strict_final) >= 20:
        lines.append(f"- Hard strict count ({len(hard_strict_final)}) is sufficient for QG pilot.")
    else:
        lines.append(f"- Hard strict count ({len(hard_strict_final)}) is limited. Consider running on more documents for Hard QG pilot.")

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
