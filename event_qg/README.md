# DCQG — Event-based Question Generation

## Quality Pilot

Run the 90-item quality pilot:

```bash
python event_qg/src/quality_pilot.py
```

### Options

| Flag | Description |
|------|-------------|
| `--skip_generation` | Load existing generated questions, skip generation |
| `--skip_llm_filters` | Skip LLM-based filters (consistency, coverage, degraded) |
| `--debug_trace` | Enable debug trace logging |
| `--trace_dir PATH` | Custom trace output directory |
| `--pilot_n N` | Override N_PER_LEVEL for quick testing (e.g., `--pilot_n 3` = 9 items) |

### Debug Trace Logging

Enable with `--debug_trace` to get full visibility into every pipeline step:

```bash
# Full 90-item run with traces
python event_qg/src/quality_pilot.py --debug_trace

# Quick 9-item test with traces
python event_qg/src/quality_pilot.py --pilot_n 3 --debug_trace

# Custom trace directory
python event_qg/src/quality_pilot.py --debug_trace --trace_dir my_traces/
```

Trace files are written to `{output_dir}/debug_traces/` by default:

- **`full_trace.jsonl`** — one JSON line per item, no truncation. Contains:
  - Input: events, relation_subtypes, supporting_sentences
  - Target: final event trigger/sentence/answer_phrase/event_type
  - Generation: all prompts, raw responses, parsed question, retry attempts
  - Filters: grammar, weak_trigger, answer_phrase, path_coverage (with raw LLM responses)
  - Judges: answer_consistency (all raw responses), asks_target_event, hard_degraded (raw response)
  - Final: pass/fail and reason

- **`readable_trace.md`** — human-readable log containing:
  - All failure cases with full context
  - All judge_error cases
  - 3 random pass samples per difficulty level
  - Each case shows: path, relations, target event, question, generation attempts (expandable), filter results, judge raw responses, supporting sentences

### Example trace output

```json
{
  "item_id": 0,
  "doc_id": "84ce009a...",
  "difficulty": "Medium",
  "target_final_event": {
    "trigger": "lifted",
    "answer_phrase": "the President lifted the state of emergency"
  },
  "generation": {
    "prompts": ["Medium questions require..."],
    "raw_responses": ["{\"question\": \"What action did the President take...\"}"],
    "parsed_question": "What action did the President take regarding the state of emergency?",
    "retry_attempts": 1
  },
  "judges": {
    "answer_consistency": {
      "label": "judge_error",
      "raw_responses": ["garbled output...", "garbled output...", "{\"asks_target\":\"yes\"...}"]
    }
  },
  "final_pass": true,
  "final_reason": "all checks passed"
}
```

## Baselines

Run baselines and ablations:

```bash
python event_qg/src/baselines.py [--skip_generation] [--skip_evaluation]
```

## Hard-Aware Generation

Run PathQG-HardAware comparison:

```bash
python event_qg/src/compare_hardaware.py [--pilot 9]
```
