# Hard Rescue Pilot Report (Quality Filter Edition)

**Date:** 2026-05-09 22:36
**Paths:** 5
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 2
**Total candidates generated:** 20
**API calls:** generation=57, filter=20, judge=60

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 20 |
| Generation errors | 0 |
| Grammar pass | 17 |
| Drift check failures | 15 |
| Drift repaired | 1 |
| Too direct answer-type cue | 2 |
| **Quality filter pass** | **1** |

### Quality Filter Fail Reasons

| Reason | Count |
|--------|------:|
| answer_consistency=no | 12 |
| double_question | 9 |
| answer_type=invalid | 8 |
| answer_phrase=skipped (early exit) | 3 |
| alignment_asks=no | 3 |
| target_drift=yes | 3 |
| trigger_leakage | 2 |
| grammar=repeat_token_pattern: on on | 1 |
| grammar=base: word repetition: followed | 1 |
| fec=no | 1 |
| grammar=base: word repetition: march | 1 |

## 2. Quality-Pass Difficulty Distribution

*(Computed over quality_filter_pass candidates only)*

| Metric | Count | Rate |
|--------|------:|-----:|
| Quality-pass (judged) | 1 | — |
| Blind Pred Easy | 1 | 100.0% |
| Blind Pred Medium | 0 | 0.0% |
| Blind Pred Hard | 0 | 0.0% |

### Required Steps (Blind Judge)

| Steps | Count | Rate |
|------:|------:|-----:|
| 1 | 1 | 100.0% |

### Single Sentence Answerable (Blind Judge)

| SSA | Count | Rate |
|-----|------:|-----:|
| yes | 1 | 100.0% |

## 3. Quality-Pass Structural Metrics

### Path Dependency

| Level | Count | Rate |
|-------|------:|-----:|
| none | 1 | 100.0% |

### Shortcut Without Path

| Value | Count | Rate |
|-------|------:|-----:|
| no | 1 | 100.0% |

## 4. Quality Metrics (Among Quality-Pass)

| Metric | Value |
|--------|------:|
| Answerable (yes/partial) | 1/1 (100.0%) |
| Final-Event Consistent (yes/partial) | 1/1 (100.0%) |
| Alignment asks (yes/partial) | 1/1 (100.0%) |
| Target drift != yes | 1/1 (100.0%) |
| Answer consistency != no | 1/1 (100.0%) |

## 5. Per Answer Type (Quality-Pass Only)

| Hard Answer Type | N QP | Blind Easy | Blind Med | Blind Hard | FEC yes/partial | SSA=no | PathDep strong |
|-----------------|-----:|----------:|---------:|----------:|----------------|-------:|---------------:|
| casualty_damage | 1 | 1 | 0 | 0 | 1 (100%) | 0 | 0 |

## 6. Per Strategy

| Strategy | N judged | QP Rate | Blind Hard | Blind Med | Blind Easy | FEC% | SSA=no% | PathDep strong% |
|----------|--------:|--------:|----------:|---------:|----------:|-----:|--------:|----------------:|
| hidden_endpoint | 10 | 0% | 0 (0%) | 2 (20%) | 8 (80%) | 100% | 0% | 20% |
| relation_composition | 10 | 10% | 0 (0%) | 4 (40%) | 6 (60%) | 90% | 0% | 30% |

## 7. Quality-Pass Samples by Predicted Difficulty

### Easy Samples (top 3)

### Easy #1 [relation_composition]

- **Question:** What damage resulted from The Myyrmanni's crowded on October 11, 2002?
- **Answer:** 66 victims required hospitalization with the remainder treated
- **Event path:** crowded -> come to -> took place -> treated
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes | **Shortcut:** no
- **Quality Filter Pass:** True
- **Quality Reason:** all checks passed
- **Blind Judge Reason:** The question can be answered by directly referencing the information in a single sentence about the number of victims requiring hospitalization.

### Medium Samples (top 3)

*No quality-pass medium samples.*

### Hard Samples (top 3)

*No quality-pass hard samples.*

## 8. Quality-Pass Easy Diagnostic

*Why are quality-pass candidates judged Easy by the blind judge?*

| # | Strategy | Answer Type | Ans Sent ID | SSA | Blind Reason (truncated) | PathDep | Question (truncated) |
|--:|----------|------------|-------------|-----|--------------------------|---------|---------------------|
| 1 | relation_composition | casualty_damage | S4 | yes | The question can be answered by directly referencing the information in a single | none | What damage resulted from The Myyrmanni's crowded  |

## 9. Path-Level Diagnostic (Selected Hard Paths)

*For each selected hard path, show path-level risk factors.*

### Single Sentence Risk Distribution

| Risk Level | Count | Rate |
|------------|------:|-----:|
| high | 4 | 80.0% |
| unknown | 1 | 20.0% |

### Path Detail

| Doc ID | Candidates | QP | Best Blind Pred | SSR |
|--------|----------:|---:|----------------:|-----|
|  | 0 | 0 | Easy | high |
|  | 0 | 0 | Easy | high |
|  | 0 | 0 | Easy | high |
|  | 0 | 0 | Easy | high |
| 94189a357036 | 20 | 1 | Medium | unknown |

## 10. Oracle Top-1 Diagnostic

> **NOT USED FOR MAIN METRICS. Diagnostic only.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 1 | — |
| Oracle Blind Hard | 0 | 0.0% |
| Oracle SSA=no | 0 | 0.0% |
| Oracle PathDep strong | 1 | 100.0% |
| Oracle quality-filter-pass | 0 | 0.0% |

| # | Strategy | Blind Pred | SSA | PathDep | QP | Question (truncated) |
|--:|----------|-----------:|-----|---------|----|---------------------|
| 1 | relation_composition | Medium | partial | strong | N | What final harm resulted from Merah's wounded of F |

## Success / Readiness Criteria

- Quality filter pass rate: 1/20 (5.0%)
- [PASS] FEC among quality-pass >= 80% (100.0%)
- [PASS] Alignment (asks_expected_answer) among quality-pass >= 80% (100.0%)
- [INFO] Difficulty distribution (quality-pass): Easy=[{'item_id': 0, 'doc_id': 'e253b7fd1109bd5f87966022eea7762f', 'difficulty': 'Hard', 'method': 'PathQG-HardAware', 'hard_strategy': 'relation_composition', 'generated_question': "What damage resulted from The Myyrmanni's crowded on October 11, 2002?", 'gold_answer_trigger': 'treated', 'gold_answer_phrase': '66 victims required hospitalization with the remainder treated', 'gold_answer_sentence': '66 victims required hospitalization with the remainder treated and released at the scene.', 'gold_event_type': 'Cure', 'answer_phrase_status': 'complete', 'inferred_answer_type': 'death_injury_damage', 'hard_answer_type': 'casualty_damage', 'template_family': 'casualty', 'reasoning_type': 'relation_composition', 'grammar_pass': True, 'grammar_reason': 'pass', 'retry_attempts': 2, 'generation_error': False, 'covered_event_indices': [0], 'path_binding_method': 'lexical_pass', 'drift_check_fail': 1, 'drift_repaired': 0, 'too_direct_cue': 0, 'events': [{'id': 'EVENT_444b1dbbf8ecd3779bfca4f2c88fba55', 'type': 'Come_together', 'trigger': 'crowded', 'sent_id': 5, 'offset': [5, 6]}, {'id': 'EVENT_63ef5f68e3d1284577348ceb08d92ad8', 'type': 'Motion', 'trigger': 'come to', 'sent_id': 5, 'offset': [16, 18]}, {'id': 'EVENT_8f660c6c0cacbe7843d1fb6ee4a70c4f', 'type': 'Process_start', 'trigger': 'took place', 'sent_id': 0, 'offset': [3, 5]}, {'id': 'EVENT_1ea6764130aab9d53b1e1bd4af6266df', 'type': 'Cure', 'trigger': 'treated', 'sent_id': 4, 'offset': [7, 8]}], 'supporting_sentences': [[0, 'The Myyrmanni bombing took place on October 11, 2002, in Myyrmäki, Vantaa, Finland, in Greater Helsinki, at the Myyrmanni shopping mall.'], [1, 'A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.'], [3, '166 people were injured, including 10 children.'], [4, '66 victims required hospitalization with the remainder treated and released at the scene.'], [5, 'The shopping center was especially crowded, with 1,000–2,000 people, including many children who had come to see a clown performance.'], [6, 'The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.']], 'relation_subtypes': ['TEMPORAL/OVERLAP', 'TEMPORAL/BEFORE', 'CAUSE/PRECONDITION'], 'generation_prompts': ['Generate a HARD question about the composed effect of a multi-step event chain.\n\nContext:\n[S0] The Myyrmanni bombing took place on October 11, 2002, in Myyrmäki, Vantaa, Finland, in Greater Helsinki, at the Myyrmanni shopping mall.\n[S1] A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.\n[S3] 166 people were injured, including 10 children.\n[S4] 66 victims required hospitalization with the remainder treated and released at the scene.\n[S5] The shopping center was especially crowded, with 1,000–2,000 people, including many children who had come to see a clown performance.\n[S6] The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.\n\nExpected answer: "66 victims required hospitalization with the remainder treated"\nStart event (you MAY mention): "crowded"\nDo NOT mention: "come to", "took place", "treated"\n\nOBLIQUE CHAIN openings (do NOT use "harm", "damage", "casualties", "toll" directly):\n  - "What was ultimately reported after the unrest connected to crowded?"\n  - "Which later consequence resulted from the chain of events beginning with crowded?"\n  - "What emerged from the events set in motion by crowded?"\nDo NOT copy answer words. Do NOT mention the final event. The solver must trace the chain.\nThe expected answer is: "66 victims required hospitalization with the remainder treated"\n\nRULES:\n1. Mention ONLY the start event "crowded". Do NOT mention intermediate or final events.\n2. Ask about the FINAL result/action/outcome — NOT intermediate events.\n3. Solver must trace 4 events across 4 context sentences.\n4. Include entity from: The Myyrmanni, October, Myyrm, Vantaa, Finland\n5. Use "What" question. Do NOT use "Why". End with "?".\n6. Do NOT copy the answer phrase into the question.\n7. The natural answer MUST be "66 victims required hospitalization with the remainder treated".\n\nGOOD example: "What [result] resulted from [entity]\'s crowded?"\n\nOutput ONLY one JSON object: {"question": "..."}', 'Your question drifted from the expected answer.\nRejected: "What emerged from the events set in motion by The Myyrmanni\'s crowded on October 11, 2002?"\nProblem: head_mismatch: question does not start with allowed head for casualty_damage\n\nRewrite so the natural answer is: "66 victims required hospitalization with the remainder treated"\nMust begin with:   - "what final harm"\n  - "what casualties"\n  - "what damage"\n  - "what was the toll"\n  - "what destruction"\n\nContext:\n[S0] The Myyrmanni bombing took place on October 11, 2002, in Myyrmäki, Vantaa, Finland, in Greater Helsinki, at the Myyrmanni shopping mall.\n[S1] A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.\n[S3] 166 people were injured, including 10 children.\n[S4] 66 victims required hospitalization with the remainder treated and released at the scene.\n[S5] The shopping center was especially crowded, with 1,000–2,000 people, including many children who had come to see a clown performance.\n[S6] The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.\n\nRULES:\n1. Mention ONLY "crowded". Do NOT mention: "come to", "took place", "treated"\n2. Ask about FINAL result — NOT intermediate causes.\n3. Use a SINGLE "What" question. End with "?".\n4. Do NOT copy the answer phrase.\n5. BANNED: outcry, inquiry, campaign, influence, response, reason, why, how did, led to\n\nGOOD example: "What [result] resulted from [entity]\'s crowded?"\n\nOutput ONLY one JSON object: {"question": "..."}'], 'generation_raw_responses': ['{"question": "What emerged from the events set in motion by The Myyrmanni\'s crowded on October 11, 2002?"}', '{"question": "What damage resulted from The Myyrmanni\'s crowded on October 11, 2002?"}'], 'hidden_path_events': [], 'expected_steps': '', 'answer_alignment_check': 'aligned', '_path_idx': 1, '_candidate_idx': 0, 'dedup_key': '', 'relation_group': 'CAUSE', 'support_span': 6, 'non_temporal_count': 1, 'question_word_count': 11, 'answer_event_id': 'EVENT_1ea6764130aab9d53b1e1bd4af6266df', 'llm_answer_phrase': 'the remainder treated and released at the', 'llm_answer_phrase_pass': True, 'llm_answer_phrase_reason': 'local_extraction_after_parse_fail', 'answer_type': 'event_phrase', 'answer_phrase_pass': True, 'answer_phrase_reason': 'upstream_answer_phrase', 'answer_phrase_raw': '{"phrase": "treated with the remainder released and released the scene.",", "answer_type": "treated"}', 'weak_trigger_flag': False, 'weak_trigger_type': 'none', 'weak_trigger_pass': True, 'weak_trigger_reason': 'not a weak trigger', 'expected_answer_type': 'unknown', 'expected_answer_summary': '', 'answer_consistency_label': 'judge_error', 'answer_consistency_reason': 'judge_error after 3 attempts: {"ask":"yes","answer":"yes","consistent":"yes","reason":"The damage resulted fro', 'answer_consistency_pass': True, 'asks_target_event': None, 'judge_answerable': None, 'consistency_judge_raw': ['{"ask":"yes","answer":"yes","consistent":"yes","reason":"The damage resulted from the crowded Myyrmanni shopping mall on October 2 on  on ." "} "The incident occurred the context is described " " "', '{"ask":"yes","answer":"yes","consistent":"yes","reason":"The damage resulted from the crowded Myyrmanni shopping mall on October 2 on,, ," "} with "no"/"on" " " consistent', '{"ask":"yes","answer":"yes","consistent":"yes","reason":"The damage resulted from the crowded Myyrmanni shopping mall on October 2 on,, ," "} with "no"/"on" " " consistent'], 'path_coverage_count': 1, 'path_coverage_prior_count': 1, 'path_coverage_all_count': 1, 'path_covered_events': ['EVENT_444b1dbbf8ecd3779bfca4f2c88fba55'], 'path_covered_prior_events': ['EVENT_444b1dbbf8ecd3779bfca4f2c88fba55'], 'path_coverage_pass': False, 'path_coverage_reason': 'covers 1 prior events, need >= 2 [FAIL]', 'path_coverage_method': 'lexical_fallback', 'path_coverage_details': [{'event_id': 'EVENT_444b1dbbf8ecd3779bfca4f2c88fba55', 'trigger': 'crowded', 'is_prior': True, 'covered': True, 'evidence': '', 'match_type': 'exact'}, {'event_id': 'EVENT_63ef5f68e3d1284577348ceb08d92ad8', 'trigger': 'come to', 'is_prior': True, 'covered': False, 'evidence': '', 'match_type': 'not_covered'}, {'event_id': 'EVENT_8f660c6c0cacbe7843d1fb6ee4a70c4f', 'trigger': 'took place', 'is_prior': True, 'covered': False, 'evidence': '', 'match_type': 'not_covered'}, {'event_id': 'EVENT_1ea6764130aab9d53b1e1bd4af6266df', 'trigger': 'treated', 'is_prior': False, 'covered': False, 'evidence': '', 'match_type': 'not_covered'}], 'path_coverage_raw': 'To address the given question in the form format, provided,,",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",on",on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"on"', 'shortcut_without_path': 'no', 'needs_prior_events_to_identify_answer': 'yes', 'shortcut_sentence_id': 'N/A', 'shortcut_reason': '', 'can_answer_from_single_sentence': 'no', 'single_sentence_id': 'N/A', 'need_intermediate_events': 'yes', 'evidence_hops_used': 3, 'hard_degraded': False, 'hard_degraded_reason': 'not degraded: shortcut=no, needs_prior=yes', 'hard_degraded_raw': '"shortcut_without_path"no","needs_prior_events on_identify_answer":"yes","shortcut_sentence":"N/A","reason "brief explanation"}', 'hard_explicit_prior_count': 1, 'hard_implicit_chain_pass': True, 'hard_implicit_chain_reason': '1 prior triggers in question (max 2 allowed, pass)', 'final_filter_pass': False, 'final_filter_reason': 'path_coverage=covers 1 prior events, need >= 2 [FAIL]', 'blind_difficulty_judge_prompt': 'You are an expert difficulty evaluator for reading comprehension questions.\n\n## Context\n[S0] The Myyrmanni bombing took place on October 11, 2002, in Myyrmäki, Vantaa, Finland, in Greater Helsinki, at the Myyrmanni shopping mall.\n[S1] A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.\n[S3] 166 people were injured, including 10 children.\n[S4] 66 victims required hospitalization with the remainder treated and released at the scene.\n[S5] The shopping center was especially crowded, with 1,000–2,000 people, including many children who had come to see a clown performance.\n[S6] The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.\n\n## Question\n"What damage resulted from The Myyrmanni\'s crowded on October 11, 2002?"\n\n## Expected answer\n"66 victims required hospitalization with the remainder treated"\n\n## Task\nEvaluate the REASONING DIFFICULTY of this question from the solver\'s perspective.\n\nThe solver does NOT know the answer. They must:\n1. Understand what the question asks\n2. Find relevant information across the context\n3. Trace through a chain of events/facts to reach the answer\n\nThe key question: HOW MANY sequential reasoning steps must the solver make to answer this question, given ONLY the context above?\n\nReply as a single JSON object:\n{\n  "predicted_difficulty": "Easy",\n  "required_steps": "1",\n  "single_sentence_answerable": "yes",\n  "answerable": "yes",\n  "final_event_consistent": "yes",\n  "reason": "short explanation"\n}\n\nGuidelines:\n- predicted_difficulty: Easy, Medium, or Hard\n- required_steps: "1", "2", or "3+"\n- single_sentence_answerable: can the answer be found in a single sentence? "yes", "partial", or "no"\n- answerable: is the question answerable from the context? "yes", "partial", or "no"\n- final_event_consistent: does the question ask for the expected answer? "yes", "partial", or "no"\n\nDifficulty definitions — judge ONLY from what the question and context require:\n\n- Easy (1 step): The solver reads ONE sentence and extracts the answer directly. No chain reasoning.\n  Example: "What did the army do?" → read the sentence about the army.\n\n- Medium (2 steps): The solver connects 2 pieces of information from 2 different sentences. Simple A→B link.\n  Example: "What happened after X?" → find X in sentence 1, find result in sentence 2.\n\n- Hard (3+ steps): The question and context REQUIRE the solver to trace a CHAIN of 3+ events/facts where each step depends on the previous. The solver cannot answer correctly without following the full intermediate chain.', 'blind_difficulty_judge_raw': '{\n  "predicted_difficulty": "Easy",\n  "required_steps": "1",\n  "single_sentence_answerable": "yes",\n  "answerable": "yes",\n  "final_event_consistent": "yes",\n  "reason": "The question can be answered by directly referencing the information in a single sentence about the number of victims requiring hospitalization."\n}', 'blind_difficulty_judge': {'predicted_difficulty': 'Easy', 'required_steps': '1', 'single_sentence_answerable': 'yes', 'answerable': 'yes', 'final_event_consistent': 'yes', 'reason': 'The question can be answered by directly referencing the information in a single sentence about the number of victims requiring hospitalization.'}, 'blind_difficulty_judge_status': 'ok', 'blind_context_contains_answer_sentence': True, 'path_dependency_judge_prompt': 'You are an expert evaluator for event-path question generation.\n\n## Event path\n  1. id=EVENT_444b1dbbf8ecd3779bfca4f2c88fba55 trigger="crowded" type=Come_together role=PRIOR\n  2. id=EVENT_63ef5f68e3d1284577348ceb08d92ad8 trigger="come to" type=Motion role=PRIOR\n  3. id=EVENT_8f660c6c0cacbe7843d1fb6ee4a70c4f trigger="took place" type=Process_start role=PRIOR\n  4. id=EVENT_1ea6764130aab9d53b1e1bd4af6266df trigger="treated" type=Cure role=FINAL\n\n## Relation sequence\nTEMPORAL/OVERLAP -> TEMPORAL/BEFORE -> CAUSE/PRECONDITION\n\n## Context sentences\n[S0] The Myyrmanni bombing took place on October 11, 2002, in Myyrmäki, Vantaa, Finland, in Greater Helsinki, at the Myyrmanni shopping mall.\n[S1] A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.\n[S3] 166 people were injured, including 10 children.\n[S4] 66 victims required hospitalization with the remainder treated and released at the scene.\n[S5] The shopping center was especially crowded, with 1,000–2,000 people, including many children who had come to see a clown performance.\n[S6] The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.\n\n## Question\n"What damage resulted from The Myyrmanni\'s crowded on October 11, 2002?"\n\n## Expected answer\n"66 victims required hospitalization with the remainder treated"\n\n## Task\nEvaluate whether this question requires understanding the EVENT PATH (the chain of prior events leading to the final event). Consider:\n\n1. Does the question IMPLICITLY require knowing about prior events?\n   - Even if the question doesn\'t name prior events, does the solver need to trace\n     the chain to understand what the question is asking about?\n   - Example: "What formal resolution ended the conflict after X?" requires knowing\n     that X led to Y led to Z led to the resolution — the solver must trace the chain.\n\n2. Could someone answer this question by reading ONLY the sentence containing the final event?\n   - If the question asks about the CONSEQUENCE of a chain, reading only the final\n     sentence gives the answer text but doesn\'t tell the solver that THIS is the\n     answer they\'re looking for. They need the chain to identify it.\n\n3. How many prior events must be understood to answer correctly?\n\nReply as a single JSON object with exactly these fields:\n{\n  "path_dependency": "none",\n  "covered_prior_events": [],\n  "num_required_prior_events": 0,\n  "can_answer_without_path": "yes",\n  "reason": "short explanation"\n}\n\nGuidelines:\n- path_dependency: "none", "partial", or "strong"\n  * "strong": the solver MUST trace the event chain to find/identify the answer\n  * "partial": the chain helps but isn\'t strictly necessary\n  * "none": the answer is directly findable from one sentence with no chain needed\n- covered_prior_events: list of prior event IDs that the question references or requires\n- num_required_prior_events: how many prior events are needed to answer\n- can_answer_without_path: "yes", "partial", or "no"\n- reason: one sentence explanation', 'path_dependency_judge_raw': '{\n  "path_dependency": "none",\n  "covered_prior_events": [],\n  "num_required_prior_events": 0,\n  "can_answer_without_path": "yes",\n  "reason": "The question can be answered directly from the context sentence about the treatment of victims without needing to trace prior events."\n}', 'path_dependency_judge': {'path_dependency': 'none', 'covered_prior_events': [], 'num_required_prior_events': 0, 'can_answer_without_path': 'yes', 'reason': 'The question can be answered directly from the context sentence about the treatment of victims without needing to trace prior events.'}, 'path_dependency_judge_status': 'ok', 'hard_alignment_prompt': 'You are an expert evaluator for reading comprehension questions.\n\n## Context\n[S0] The Myyrmanni bombing took place on October 11, 2002, in Myyrmäki, Vantaa, Finland, in Greater Helsinki, at the Myyrmanni shopping mall.\n[S1] A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.\n[S3] 166 people were injured, including 10 children.\n[S4] 66 victims required hospitalization with the remainder treated and released at the scene.\n[S5] The shopping center was especially crowded, with 1,000–2,000 people, including many children who had come to see a clown performance.\n[S6] The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.\n\n## Question\n"What damage resulted from The Myyrmanni\'s crowded on October 11, 2002?"\n\n## Expected answer\n"66 victims required hospitalization with the remainder treated"\n\n## Task\nEvaluate whether this question is well-aligned with its expected answer.\n\n1. asks_expected_answer: Does the question ask for information that the expected answer provides?\n   - "yes": the question clearly asks for what the answer gives\n   - "partial": the question is somewhat aligned but indirect or vague\n   - "no": the question asks for something fundamentally different from the answer\n\n2. expected_answer_natural: Would a reader, seeing only the question, naturally expect an answer like "66 victims required hospitalization with the remainder treated"?\n   - "yes": the answer type matches what the question asks for\n   - "partial": the answer could fit but isn\'t the most natural type\n   - "no": the answer type doesn\'t match (e.g., question asks "why" but answer is a date)\n\n3. target_drift: Has the question drifted away from the target answer to ask about something else?\n   - "yes": the question asks about a different event/concept than the answer addresses\n   - "no": the question stays on target\n\nReply as a single JSON object:\n{"asks_expected_answer":"yes","expected_answer_natural":"yes","target_drift":"no","reason":"brief explanation"}', 'hard_alignment_raw': '{"asks_expected_answer":"yes","expected_answer_natural":"yes","target_drift":"no","reason":"The question directly asks about the damage resulting from the crowd at The Myyrmanni, which aligns with the expected answer regarding the number of victims requiring hospitalization and treatment."}', 'hard_alignment': {'asks_expected_answer': 'yes', 'expected_answer_natural': 'yes', 'target_drift': 'no', 'reason': 'The question directly asks about the damage resulting from the crowd at The Myyrmanni, which aligns with the expected answer regarding the number of victims requiring hospitalization and treatment.'}, 'hard_alignment_status': 'ok', 'quality_filter_pass': True, 'quality_filter_reason': 'all checks passed'}], Medium=0, Hard=0
- [NOTE] No difficulty metric used to select main evaluation set
