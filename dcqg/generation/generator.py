"""PathQG-HardAware generator with retry and validation.

generate_with_retry_hardaware: main generation function with difficulty-aware
prompts, grammar filtering, path binding check, and repair retries.
"""
from dcqg.question_filter.grammar import grammar_filter
from dcqg.path.direction import check_path_binding, validate_hard_question
from dcqg.generation.prompts import prompt_pathqg_easy, prompt_pathqg_medium, prompt_pathqg_hard, prompt_pathqg_hard_implicit
from dcqg.generation.parser import generate_one
from dcqg.generation.repair import build_repair_prompt, REPAIRABLE_REASONS
from dcqg.question_filter.hard_implicitness import count_explicit_prior_triggers


def generate_with_retry_hardaware(item, max_attempts=5):
    """Generate with difficulty-aware prompt + hard post-validation.
    Retries up to max_attempts times on empty/parse-fail. Checks path binding.
    Returns (result_dict, num_attempts).
    """
    diff = item["difficulty"]
    gold = item.get("answer_trigger", "")
    events = item.get("events", [])

    if diff == "Easy":
        prompt_fn = prompt_pathqg_easy
    elif diff == "Medium":
        prompt_fn = prompt_pathqg_medium
    else:
        prompt_fn = prompt_pathqg_hard_implicit

    question = ""
    rt = "error"
    g_ok, g_reason = False, "not attempted"
    covered_indices = []
    attempts = 0
    generation_error = False
    all_attempt_prompts = []
    all_attempt_raws = []

    for attempt in range(max_attempts):
        attempts = attempt + 1

        if attempt == 0:
            prompt = prompt_fn(item)
            temp = 0.1
        else:
            prompt = build_repair_prompt(item, question, g_reason, diff, covered_indices)
            temp = 0.1 + min(attempt * 0.1, 0.3)

        gen, raw = generate_one(prompt, temperature=temp)
        all_attempt_prompts.append(prompt)
        all_attempt_raws.append(raw or "")

        if gen is None:
            question = ""
            rt = "error"
            g_ok, g_reason = False, "parse error"
            continue

        question = gen.get("question", "") if isinstance(gen, dict) else ""
        rt = gen.get("reasoning_type", "unknown") if isinstance(gen, dict) else "error"

        # Validate hidden_path_events for implicit chain
        if diff == "Hard" and isinstance(gen, dict) and "hidden_path_events" in gen:
            valid_ids = {e.get("id", "") for e in events}
            raw_ids = gen.get("hidden_path_events", [])
            if isinstance(raw_ids, list):
                valid_hidden = [eid for eid in raw_ids if eid in valid_ids]
            else:
                valid_hidden = []
            gen["hidden_path_events"] = valid_hidden

        if not question:
            g_ok, g_reason = False, "empty"
            continue

        g_ok, g_reason = grammar_filter(question)

        if g_ok and gold and gold.lower() in question.lower():
            g_ok, g_reason = False, "trigger leakage"

        # Hard implicitness check at generation time: only catch extremely over-explicit (3+)
        # The filter pipeline enforces the real constraint (max 1)
        if g_ok and diff == "Hard":
            explicit_count = count_explicit_prior_triggers(question, events)
            if explicit_count >= 3:
                g_ok, g_reason = False, f"too_explicit: {explicit_count} prior triggers in question"

        if g_ok and diff == "Hard":
            g_ok, g_reason = validate_hard_question(question, events, gold)

        if g_ok:
            # For Hard (implicit chain), relax path_binding to 1 trigger
            # since the design intentionally avoids naming triggers.
            # The LLM path_coverage_judge in the filter pipeline enforces real coverage.
            effective_diff = diff
            if diff == "Hard":
                effective_diff = "Medium"  # Medium requires 1 prior event
            pb_ok, covered_indices, pb_reason = check_path_binding(question, events, effective_diff)
            if not pb_ok:
                g_ok, g_reason = False, f"path_binding: {pb_reason}"

        if g_ok:
            break

        if g_reason not in REPAIRABLE_REASONS:
            if not any(g_reason.startswith(r) for r in ["only ", "banned", "path_binding"]):
                break

    if not g_ok and not question:
        generation_error = True

    pb_method = "lexical_pass" if g_ok else "fail"

    return {
        "item_id": item.get("_item_id", 0),
        "doc_id": item.get("doc_id", ""),
        "difficulty": diff,
        "method": "PathQG-HardAware",
        "generated_question": question,
        "gold_answer_trigger": gold,
        "gold_answer_phrase": item.get("gold_answer_phrase", ""),
        "gold_answer_sentence": item.get("gold_answer_sentence", ""),
        "gold_event_type": item.get("gold_event_type", ""),
        "answer_phrase_status": item.get("answer_phrase_status", "unknown"),
        "reasoning_type": rt,
        "grammar_pass": g_ok,
        "grammar_reason": g_reason,
        "retry_attempts": attempts,
        "generation_error": generation_error,
        "covered_event_indices": covered_indices,
        "path_binding_method": pb_method,
        "events": events,
        "supporting_sentences": item.get("supporting_sentences", []),
        "relation_subtypes": item.get("relation_subtypes", []),
        "generation_prompts": all_attempt_prompts,
        "generation_raw_responses": all_attempt_raws,
        "hidden_path_events": gen.get("hidden_path_events", []) if isinstance(gen, dict) else [],
        "expected_steps": gen.get("expected_steps", "") if isinstance(gen, dict) else "",
    }, attempts
