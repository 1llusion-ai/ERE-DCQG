# Manual QG Audit — Strict 29 v2

**Source:** `outputs/runs/path_filter_strict_pilot/paths.filtered.strict.jsonl`
**Selected paths:** `outputs/runs/qg_pilot_strict_29/selected_paths.jsonl` (same as v1)
**Total:** 29


## Easy (10 items)

### Easy #1

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **title:** Myyrmanni bombing
- **difficulty:** Easy
- **event_path:** took place/Process_start -> released/Releasing
- **relations:** CAUSE/PRECONDITION
- **gold_answer_phrase:** `released at the scene`
- **gold_answer_sentence:** 66 victims required hospitalization with the remainder treated and released at the scene.
- **generated_question:** What happened to the victims after they were treated?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** judge_error (judge_error after 3 attempts: "yes")
- **path_coverage:** count=0 pass=False (covers 0 all events in the path, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 all events in the path, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** None
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 3

### Easy #2

- **doc_id:** `04a82d4eac379a98efcd87ebdba0b0ce`
- **title:** Death and state funeral of Raúl Alfonsín
- **difficulty:** Easy
- **event_path:** died/Death -> funeral/Rite
- **relations:** CAUSE/PRECONDITION
- **gold_answer_phrase:** `a state funeral at the Palace of the Argentine National Congress`
- **gold_answer_sentence:** Vice president Julio Cobos, the acting president at the time, arranged three days of national mourning and a state funeral at the Palace of the Argent
- **generated_question:** What significant event was organized by Vice President Julio Cobos after Raúl Alfonsín's death?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events in the path, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** onosons the Palace of the Argentine Congress.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.717
- **item_id:** 6

### Easy #3

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **title:** King David Hotel bombing
- **difficulty:** Easy
- **event_path:** Disguised/Wearing -> operation/Military_operation
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `this had been cancelled by the time the operation was carried out`
- **gold_answer_sentence:** When planned, the attack had the approval of the Haganah, the principal Jewish paramilitary group in Palestine, though, unbeknownst to the Irgun, this
- **generated_question:** What happened after the Irgun members had disguised themselves?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (The question does not specifically ask about the target final event, but rather about the actions of the Irgun members after they had disguised themselves.)
- **path_coverage:** count=1 pass=True (covers 1 all events in the path, need >= 1 [PASS])
- **final_filter_pass:** False
- **final_filter_reason:** answer_consistency=no: The question does not specifically ask about the target final event, but rather about the actions of the Irgun members after they had disguised themselves.
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 7

### Easy #4

- **doc_id:** `dd2a791aa826766cf0d05dc8102f5c8e`
- **title:** Death of Joy Gardner
- **difficulty:** Easy
- **event_path:** attention/Perception_active -> inquest/Criminal_investigation
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `no coroner's inquest or public inquiry into the circumstances of Gardner's death has been held`
- **gold_answer_sentence:** Despite continuing pressure by campaigners, no coroner's inquest or public inquiry into the circumstances of Gardner's death has been held.
- **generated_question:** What has not been conducted despite the case becoming a cause célèbre?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events in the path, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** No coroner inquest was conducted.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 1.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.917
- **item_id:** 9

### Easy #5

- **doc_id:** `8af040b8bfa2eec89d1197392f2fdac5`
- **title:** Cyclone Forrest
- **difficulty:** Easy
- **event_path:** striking/Attack -> losses/Earnings_and_losses
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `Agricultural losses in the hardest hit areas were estimated at ฿800 million (US$32 million)`
- **gold_answer_sentence:** Agricultural losses in the hardest hit areas were estimated at ฿800 million (US$32 million).
- **generated_question:** What were the estimated agricultural losses in Thailand due to the storm?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events in the path, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** ฿8 billion
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.33
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.633
- **item_id:** 12

### Easy #6

- **doc_id:** `8af040b8bfa2eec89d1197392f2fdac5`
- **title:** Cyclone Forrest
- **difficulty:** Easy
- **event_path:** reached/Arriving -> produced/Creating
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `the system produced significant storm`
- **gold_answer_sentence:** In Thailand, the system produced significant storm surge, damaged or destroyed 1,700 homes, and killed two people.
- **generated_question:** What did the system do in Thailand after reaching its peak intensity?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** judge_error (judge_error after 3 attempts: {"asks_target":"yes","answer_on:"yes"+"consistent:"yes""})
- **path_coverage:** count=1 pass=True (covers 1 all events in the path, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Produced significant storm surge.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 1.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.33
- **quality_difficulty_alignment:** 0.33
- **composite:** 0.833
- **item_id:** 16

### Easy #7

- **doc_id:** `c0c67db40cd5e2e03645ff1116fafcfc`
- **title:** Cherry Valley massacre
- **difficulty:** Easy
- **event_path:** ordered/Arranging -> minimize/Cause_change_of_position_on_a_scale
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `he actively sought to minimize the atrocities`
- **gold_answer_sentence:** He was not present at Wyoming — although many thought he was — and he actively sought to minimize the atrocities that took place at Cherry Valley.
- **generated_question:** 
- **generation_error:** True
- **retry_attempts:** 3
- **grammar_pass:** False (parse error)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** ? ()
- **path_coverage:** count=? pass=? ()
- **final_filter_pass:** False
- **final_filter_reason:** generation_error
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** ?
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 19

### Easy #8

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **title:** King David Hotel bombing
- **difficulty:** Easy
- **event_path:** planned/Arranging -> planted/Placing
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `members of the Irgun planted a bomb in the basement of the main building of the`
- **gold_answer_sentence:** Disguised as Arab workmen and as hotel waiters, members of the Irgun planted a bomb in the basement of the main building of the hotel, whose southern 
- **generated_question:** What did members of the Irgun do in the basement of the hotel's main building, as described in the context provided? ?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** False (repeat_question_mark)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (skipped (early exit))
- **path_coverage:** count=0 pass=False (skipped (early exit))
- **final_filter_pass:** False
- **final_filter_reason:** grammar=repeat_question_mark; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** None
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 23

### Easy #9

- **doc_id:** `28a13a10cb57f8245b1f98270bad9860`
- **title:** Territorial era of Minnesota
- **difficulty:** Easy
- **event_path:** diminished/Cause_change_of_position_on_a_scale -> establish/Building
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `the United States began to establish a firm presence in what would become Minnesota`
- **gold_answer_sentence:** Though there was a long history of European presence in the area before 19th century, it was during the 19th century that the United States began to e
- **generated_question:** What process replaced the diminishing trade in furs as the primary economic activity in Minnesota during the early 19th century?
- **generation_error:** False
- **retry_attempts:** 2
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** judge_error (judge_error after 3 attempts: No)
- **path_coverage:** count=1 pass=True (covers 1 all events in the path, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** The demand for furs f the primary economic activity in Minnesota gradually declined during the 1 1 century as.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.717
- **item_id:** 25

### Easy #10

- **doc_id:** `c0c67db40cd5e2e03645ff1116fafcfc`
- **title:** Cherry Valley massacre
- **difficulty:** Easy
- **event_path:** permitted/Preventing_or_letting -> restrain/Hindering
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `he was powerless to restrain the Seneca`
- **gold_answer_sentence:** Butler repeatedly maintained, against accusations that he permitted the atrocities to take place, that he was powerless to restrain the Seneca.
- **generated_question:** What did Butler claim he was unable to do during the campaigns of 1778 regarding the Seneca?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 all events in the path, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 all events in the path, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 28


## Medium (10 items)

### Medium #1

- **doc_id:** `84ce009a07b987d60a79d92bc4d45744`
- **title:** 2006 state of emergency in the Philippines
- **difficulty:** Medium
- **event_path:** occurred/Presence -> revocation/Process_end -> lifted/Process_end
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `the President lifted the state of emergency`
- **gold_answer_sentence:** 1021, the President lifted the state of emergency.
- **generated_question:** What action did the government take on March 3, 2006, one week after proclaiming the state of emergency, and what immediate effects did this have on public activities and demonstrations?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (extracted from text: asks=no ans=yes cons=no)
- **path_coverage:** count=0 pass=False (covers 0 PRIOR events (all events EXCEPT the last one), need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** answer_consistency=no: extracted from text: asks=no ans=yes cons=no; path_coverage=covers 0 PRIOR events (all events EXCEPT the last one), need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 0

### Medium #2

- **doc_id:** `3dcfd60153822a6a8f6a516f161fc506`
- **title:** Battle of Malacca (1641)
- **difficulty:** Medium
- **event_path:** capture/Conquering -> casualties/Catastrophe -> rallied/Filling
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `the battle rallied the forces of the Sultanate of Johor in an alliance with the Dutch`
- **gold_answer_sentence:** Although the Dutch were routed, the Portuguese fleet of Martim Afonso de Castro, the Viceroy of Portuguese India, suffered heavier casualties and the 
- **generated_question:** What event brought the Sultanate of Johor and the Dutch together in a strategic alliance? 
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 PRIOR events (all events EXCEPT the last one), need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 PRIOR events (all events EXCEPT the last one), need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 2

### Medium #3

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **title:** Battle of Ciołków
- **difficulty:** Medium
- **event_path:** refused/Agree_or_refuse_to_act -> fight/Hostile_encounter -> lost/Earnings_and_losses
- **relations:** TEMPORAL/BEFORE, CAUSE/CAUSE
- **gold_answer_phrase:** `lost his eye`
- **gold_answer_sentence:** Polish losses were negligible, but the Polish commander was wounded and lost his eye.
- **generated_question:** What happened to the Polish commander after he refused to negotiate with the Russians?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 PRIOR events (all events EXCEPT the last one), need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 PRIOR events (all events EXCEPT the last one), need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 4

### Medium #4

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **title:** Operation Deny Flight
- **difficulty:** Medium
- **event_path:** began/Process_start -> expanded/Expansion -> bombed/Attack
- **relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `NATO aircraft first bombed ground targets in an operation near Goražde`
- **gold_answer_sentence:** The operation included the first combat engagement in NATO's history, a 28 February 1994 air battle over Banja Luka, and in April 1994, NATO aircraft 
- **generated_question:** What was the first significant action taken by NATO aircraft during Operation Deny Flight after the mission's scope was expanded?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 PRIOR events (all events EXCEPT the last one), need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Bombing ground targets near Goražde.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 1.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.967
- **item_id:** 5

### Medium #5

- **doc_id:** `6dabade56742b6040cda6a5838176f6c`
- **title:** Who's That Girl World Tour
- **difficulty:** Medium
- **event_path:** trained/Education_teaching -> addressing/Expressing_publicly -> wearing/Wearing
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `wearing a conical`
- **gold_answer_sentence:** A statue of Madonna, wearing a conical bra, was erected in her name at the center of the town of Pacentro in Italy, where her ancestors used to live.
- **generated_question:** What kind of attire did the statue of Madonna, erected in Pacentro, feature, and how did her physical training influence this choice?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** partial (brief)
- **path_coverage:** count=1 pass=True (covers 1 PRIOR events (all events EXCEPT the last one), need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** The statue of of Madonna wore a bra.
- **solver_eval_status:** ok
- **judge_answerable:** 0.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 0.33
- **quality_path_relevance:** 0.33
- **quality_difficulty_alignment:** 0.33
- **composite:** 0.283
- **item_id:** 8

### Medium #6

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **title:** Operation Deny Flight
- **difficulty:** Medium
- **event_path:** expanded/Expansion -> bombed/Attack -> helped/Assistance
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `These engagements helped show`
- **gold_answer_sentence:** These engagements helped show that NATO had adapted to the post-Cold War era and could operate in environments other than a major force on force engag
- **generated_question:** What did the engagements during Operation Deny Flight, such as the air battle over Banja Luka and the bombing of ground targets near Goražde, demonstrate about NATO's capabilities in the post-Cold War era?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** judge_error (judge_error after 3 attempts: Yesyes Gconsistent Gyes","answer":"yes Gconsistent Gyes","
 on with the target t)
- **path_coverage:** count=2 pass=True (covers 2 PRIOR events (all events EXCEPT the last one), need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** NATO could operate globally.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 11

### Medium #7

- **doc_id:** `37d153abeafe0477ce6a6398e26eb442`
- **title:** Defense of Sihang Warehouse
- **difficulty:** Medium
- **event_path:** defense/Defending -> invasion/Attack -> aftermath/Catastrophe
- **relations:** TEMPORAL/BEFORE, CAUSE/CAUSE
- **gold_answer_phrase:** `people in the demoralizing aftermath of the Japanese invasion of Shanghai`
- **gold_answer_sentence:** The successful defense of the warehouse provided a morale-lifting consolation to the Chinese army and people in the demoralizing aftermath of the Japa
- **generated_question:** What provided a morale-lifting consolation to the Chinese people during the difficult period following the Japanese invasion of Shanghai?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (asks=yes ans=yes cons=yes)
- **path_coverage:** count=1 pass=True (covers 1 PRIOR events (all events EXCEPT the last one), need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** The defense of Sihang Warehouse.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 14

### Medium #8

- **doc_id:** `f83fc49c020ec542b16d463b3f7c2c14`
- **title:** Sinjar massacre
- **difficulty:** Medium
- **event_path:** reacted/Response -> airstrikes/Attack -> started/Process_start
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `YPG forces started the December 2014 Sinjar offensive with the support of US airstrikes`
- **gold_answer_sentence:** On 17 December 2014, the Kurdish Peshmerga, PKK and YPG forces started the December 2014 Sinjar offensive with the support of US airstrikes.
- **generated_question:** What offensive did the Kurdish forces begin after the United States launched airstrikes against ISIL in northern Iraq?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (weak trigger but valid phrase: 'YPG forces started the December 2014 Sinjar offensive with the support of US airstrikes')
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 PRIOR events (all events EXCEPT the last one), need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 PRIOR events (all events EXCEPT the last one), need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 15

### Medium #9

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **title:** Operation Deny Flight
- **difficulty:** Medium
- **event_path:** spanned/Self_motion -> providing/Supply -> taken/Conquering
- **relations:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE
- **gold_answer_phrase:** `UN peacekeepers were taken as hostages in response to NATO bombing`
- **gold_answer_sentence:** Most notably, significant tension arose between the two after UN peacekeepers were taken as hostages in response to NATO bombing.
- **generated_question:** What significant event occurred during the span of Deny Flight that led to increased tension between UN and NATO forces?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 PRIOR events (all events EXCEPT the last one), need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** UN peacekeepers taken as hostages.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 1.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.917
- **item_id:** 17

### Medium #10

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **title:** Crimean War
- **difficulty:** Medium
- **event_path:** changes/Change -> quipped/Statement -> proved/Convincing
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `The Crimean War proved to be the moment of truth for Nikolaevan Russia`
- **gold_answer_sentence:** The Crimean War proved to be the moment of truth for Nikolaevan Russia.
- **generated_question:** 
- **generation_error:** True
- **retry_attempts:** 3
- **grammar_pass:** False (parse error)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** ? ()
- **path_coverage:** count=? pass=? ()
- **final_filter_pass:** False
- **final_filter_reason:** generation_error
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** ?
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 27


## Hard (9 items)

### Hard #1

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **title:** Battle of Ciołków
- **difficulty:** Hard
- **event_path:** uprising/Change_of_leadership -> approached/Arriving -> refused/Agree_or_refuse_to_act -> ordered/Arranging
- **relations:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `ordered a charge of the Russians`
- **gold_answer_sentence:** However, Rogaliński refused to negotiate and ordered a charge of the Russians.
- **generated_question:** During the uprising, when the Russians approached the Polish quarters and the commander tried to negotiate, what did Rogaliński decide to do?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=False (covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 1

### Hard #2

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **title:** Crimean War
- **difficulty:** Hard
- **event_path:** stopped/Preventing_or_letting -> destroyed/Destroying -> arriving/Arriving -> signed on/Sign_agreement
- **relations:** TEMPORAL/BEFORE, CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `signed on 30 March`
- **gold_answer_sentence:** The Treaty of Paris, signed on 30 March 1856, ended the war.
- **generated_question:** After the Russians were stopped at Silistra and their fleet destroyed at Sinop, when did France and Britain act to welcome the development of the war ending?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (extracted from text: asks=yes ans=no cons=no)
- **path_coverage:** count=1 pass=False (covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL])
- **shortcut_without_path:** ?
- **needs_prior_events:** ?
- **hard_degraded:** ? ()
- **final_filter_pass:** False
- **final_filter_reason:** filter_exception: AttributeError: 'str' object has no attribute 'get'
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** False
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 10

### Hard #3

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **title:** King David Hotel bombing
- **difficulty:** Hard
- **event_path:** warnings/Warning -> called/Communication -> carried out/Attack -> killed/Killing
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `91 people of various nationalities were killed`
- **gold_answer_sentence:** 91 people of various nationalities were killed, and 46 were injured.
- **generated_question:** What was the outcome after warnings were sent and calls were made but no evacuation was ordered?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (extracted from text: asks=no ans=yes cons=no)
- **path_coverage:** count=1 pass=False (covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** False
- **final_filter_reason:** answer_consistency=no: extracted from text: asks=no ans=yes cons=no; path_coverage=covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 13

### Hard #4

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **title:** Myyrmanni bombing
- **difficulty:** Hard
- **event_path:** crowded/Come_together -> took place/Process_start -> investigated/Criminal_investigation -> closed/Self_motion
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION
- **gold_answer_phrase:** `closed in January 2003 without any indictments as Gerdt was the sole suspect`
- **gold_answer_sentence:** The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.
- **generated_question:** After the shopping center was especially crowded with many children present, and the bombing took place, what was the outcome of the investigation?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=False (covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 18

### Hard #5

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **title:** Battle of Ciołków
- **difficulty:** Hard
- **event_path:** uprising/Change_of_leadership -> negotiate/Communication -> refused/Agree_or_refuse_to_act -> killed/Killing
- **relations:** TEMPORAL/CONTAINS, CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `After a short hand-to-hand fight the Russian commander was killed`
- **gold_answer_sentence:** After a short hand-to-hand fight (the Polish unit had only two pieces of firearms and was mostly equipped with sabres, war scythes and improvised weap
- **generated_question:** When the uprising began and the Polish refused to negotiate, the Russian commander did what?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=False (covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 20

### Hard #6

- **doc_id:** `db50381e7d1dd4a41fb4ac60eaebe3a4`
- **title:** Battle of Orthez
- **difficulty:** Hard
- **event_path:** Moving/Motion -> surrounded/Surrounding -> pushed/Motion -> battle/Hostile_encounter
- **relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `the French marshal offered battle`
- **gold_answer_sentence:** Resuming their eastward drive, the remaining two Allied corps pushed Soult's army back to Orthez where the French marshal offered battle.
- **generated_question:** After the Allied corps surrounded Bayonne and then pushed Soult's army back, what did the French marshal do?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (weak trigger but valid phrase: 'the French marshal offered battle')
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 21

### Hard #7

- **doc_id:** `3dcfd60153822a6a8f6a516f161fc506`
- **title:** Battle of Malacca (1641)
- **difficulty:** Hard
- **event_path:** began/Process_start -> launching/Military_operation -> took/Conquering -> agreed/Agree_or_refuse_to_act
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION
- **gold_answer_phrase:** `agreed not to seek territories or wage war with the Malay kingdoms`
- **gold_answer_sentence:** In line with the agreement with Johor in 1606, the Dutch took control of Malacca and agreed not to seek territories or wage war with the Malay kingdom
- **generated_question:** After the Dutch began their campaign to destroy Portuguese power and launched small incursions against the Portuguese, what did they agree to do with the Malay kingdoms?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** partial (The question does not align with the target event trigger and meaning. The target event is about an agreement not to seek territories or wage war with Malay kingdoms, which is not mentioned in the context provided.)
- **path_coverage:** count=1 pass=False (covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** False
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 22

### Hard #8

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **title:** Operation Deny Flight
- **difficulty:** Hard
- **event_path:** began/Process_start -> providing/Supply -> helped/Assistance -> adapted/Coming_to_be
- **relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE, TEMPORAL/CONTAINS
- **gold_answer_phrase:** `NATO had adapted to the post-Cold War era and could operate in environments other than a major force on force engagement on the plains of Central Europe`
- **gold_answer_sentence:** These engagements helped show that NATO had adapted to the post-Cold War era and could operate in environments other than a major force on force engag
- **generated_question:** What did Operation Deny Flight help show about NATO's ability to operate in the post-Cold War era after providing close air support and engaging in combat over Banja Luka and Goražde?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=False (covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 24

### Hard #9

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **title:** Crimean War
- **difficulty:** Hard
- **event_path:** stopped/Preventing_or_letting -> destroyed/Destroying -> rushed/Motion -> forbade/Preventing_or_letting
- **relations:** TEMPORAL/BEFORE, CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `It forbade Russia from basing warships in the Black Sea`
- **gold_answer_sentence:** It forbade Russia from basing warships in the Black Sea.
- **generated_question:** After the Ottomans stopped the Russian advance at Silistra and their fleet destroyed a Russian attempt to reinforce Kars, what did France and Britain do to prevent further Russian aggression?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (extracted from text: asks=no ans=no cons=no)
- **path_coverage:** count=2 pass=True (covers 2 PRIOR events (all events EXCEPT the last one), need >= 2 [PASS])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** False
- **final_filter_reason:** answer_consistency=no: extracted from text: asks=no ans=no cons=no
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** False
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 26
