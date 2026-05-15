# Manual QG Audit — Strict 29 pilot_100

**Source:** `outputs/runs/path_filter_strict_pilot/paths.filtered.strict.jsonl`
**Selected paths:** `outputs/runs/qg_pilot_strict_29/selected_paths.jsonl` (same as v1)
**Total:** 76


## Easy (32 items)

### Easy #1

- **doc_id:** `06f91ced00b41867979f3d5dc6996da2`
- **title:** Operation Vengeance
- **difficulty:** Easy
- **event_path:** attack/Attack -> blamed/Judgment_communication
- **relations:** CAUSE/PRECONDITION
- **gold_answer_phrase:** `blamed Yamamoto for the attack on Pearl Harbor`
- **gold_answer_sentence:** The death of Yamamoto reportedly damaged the morale of Japanese naval personnel, raised the morale of the Allied forces, and was intended as revenge b
- **generated_question:** What was the intended outcome of the U.S. leaders' actions during the mission to kill Yamamoto?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 all events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
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

### Easy #2

- **doc_id:** `84ce009a07b987d60a79d92bc4d45744`
- **title:** 2006 state of emergency in the Philippines
- **difficulty:** Easy
- **event_path:** announced/Expressing_publicly -> detain/Preventing_or_letting
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `the government was allowed at the moment to detain anyone indefinitely without the privilege of the writ of habeas corpus`
- **gold_answer_sentence:** Under the provisions of the 1987 Constitution, the government was allowed at the moment to detain anyone indefinitely without the privilege of the wri
- **generated_question:** What action was the government permitted to take immediately after the state of emergency was announced?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** The privilege of the writ of habeas corpus wasong- suspendedmediately
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.33
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.683
- **item_id:** 1

### Easy #3

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **title:** Battle of Ciołków
- **difficulty:** Easy
- **event_path:** ordered/Arranging -> wounded/Bodily_harm
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `the Polish commander was wounded`
- **gold_answer_sentence:** Polish losses were negligible, but the Polish commander was wounded and lost his eye.
- **generated_question:** What after the Russian commander refused to negotiate and ordered a charge of the Russians, , what the outcome of the engagement was ? 
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** False (broken_grammar: What after)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (skipped (early exit))
- **path_coverage:** count=0 pass=False (skipped (early exit))
- **final_filter_pass:** False
- **final_filter_reason:** grammar=broken_grammar: What after; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** None
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 2

### Easy #4

- **doc_id:** `db50381e7d1dd4a41fb4ac60eaebe3a4`
- **title:** Battle of Orthez
- **difficulty:** Easy
- **event_path:** Moving/Motion -> attack/Attack
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `The Battle of Orthez saw the Anglo-Portuguese Army under Field Marshal Arthur Wellesley, Marquess of Wellington attack an Imperial French army led by Marshal Nicolas Soult in southern France`
- **gold_answer_sentence:** The Battle of Orthez (27 February 1814) saw the Anglo-Portuguese Army under Field Marshal Arthur Wellesley, Marquess of Wellington attack an Imperial 
- **generated_question:** What action did the Anglo-Portuguese Army take against the Imperial French army during the Battle of Orthez in February 1814?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 all events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 5

### Easy #5

- **doc_id:** `28a13a10cb57f8245b1f98270bad9860`
- **title:** Territorial era of Minnesota
- **difficulty:** Easy
- **event_path:** diminished/Cause_change_of_position_on_a_scale -> establish/Building
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `the United States began to establish a firm presence in what would become Minnesota`
- **gold_answer_sentence:** Though there was a long history of European presence in the area before 19th century, it was during the 19th century that the United States began to e
- **generated_question:** What economic resource replaced furs as the key economic activity in the area during the early 19th century?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (brief)
- **path_coverage:** count=0 pass=False (covers 0 all events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** answer_consistency=no: brief; path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 6

### Easy #6

- **doc_id:** `6dabade56742b6040cda6a5838176f6c`
- **title:** Who's That Girl World Tour
- **difficulty:** Easy
- **event_path:** supported/Supporting -> commending/Judgment_communication
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `with reviewers commending the extravagant nature of the concert and Madonna as a performer`
- **gold_answer_sentence:** The tour was critically appreciated, with reviewers commending the extravagant nature of the concert and Madonna as a performer.
- **generated_question:** What did the reviewers think about the extravagant nature of the concert and Madonna's performance during the Who's That Girl World Tour?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 all events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
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

### Easy #7

- **doc_id:** `37d153abeafe0477ce6a6398e26eb442`
- **title:** Defense of Sihang Warehouse
- **difficulty:** Easy
- **event_path:** defense/Defending -> reveal/Reveal_secret
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `commander Xie Jinyuan not wanting to reveal their true strength to the`
- **gold_answer_sentence:** In Chinese, the 423 defenders are known as the Eight Hundred Heroes, because commander Xie Jinyuan not wanting to reveal their true strength to the Ja
- **generated_question:** What did commander Xie Jinyuan do to the Japanese forces during the defense of Sihang Warehouse?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Provided exaggerated numbers.
- **solver_eval_status:** ok
- **judge_answerable:** 0.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.517
- **item_id:** 8

### Easy #8

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **title:** King David Hotel bombing
- **difficulty:** Easy
- **event_path:** Disguised/Wearing -> operation/Military_operation
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `this had been cancelled by the time the operation was carried out`
- **gold_answer_sentence:** When planned, the attack had the approval of the Haganah, the principal Jewish paramilitary group in Palestine, though, unbeknownst to the Irgun, this
- **generated_question:** What was the name of the event where disguised Irgun members planted a bomb in the basement of the hotel's main building, causing the collapse of the western half of the southern wing? ?
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
- **item_id:** 9

### Easy #9

- **doc_id:** `84ce009a07b987d60a79d92bc4d45744`
- **title:** 2006 state of emergency in the Philippines
- **difficulty:** Easy
- **event_path:** declared/Expressing_publicly -> foiled/Removing
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `it foiled an alleged coup d'état attempt against the rule of President Gloria Macapagal-Arroyo earlier`
- **gold_answer_sentence:** This occurred after the government claimed that it foiled an alleged coup d'état attempt against the rule of President Gloria Macapagal-Arroyo earlier
- **generated_question:** What did the government claim to have done on the same day it declared a state of emergency?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** The government claimed to have foiled a coup d'état attempt that day and to have on a state manh on (state of emergency due that " clear " " clear " to one a "
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 16

### Easy #10

- **doc_id:** `c0c67db40cd5e2e03645ff1116fafcfc`
- **title:** Cherry Valley massacre
- **difficulty:** Easy
- **event_path:** permitted/Preventing_or_letting -> restrain/Hindering
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `he was powerless to restrain the Seneca`
- **gold_answer_sentence:** Butler repeatedly maintained, against accusations that he permitted the atrocities to take place, that he was powerless to restrain the Seneca.
- **generated_question:** What was Butler claimed to be unable to do during the campaigns of 1778 regarding the Seneca?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 all events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 19

### Easy #11

- **doc_id:** `f28bce270df5a122c09365002d247e76`
- **title:** United States occupation of Nicaragua
- **difficulty:** Easy
- **event_path:** began/Process_start -> assumed/Choosing
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `Nicaragua assumed a quasi-protectorate status under the 1916 Bryan–Chamorro Treaty`
- **gold_answer_sentence:** Nicaragua assumed a quasi-protectorate status under the 1916 Bryan–Chamorro Treaty.
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
- **item_id:** 21

### Easy #12

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
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** onosons the Palace of the Argentine Congress.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.33
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.633
- **item_id:** 23

### Easy #13

- **doc_id:** `06f91ced00b41867979f3d5dc6996da2`
- **title:** Operation Vengeance
- **difficulty:** Easy
- **event_path:** Vengeance/Revenge -> shot down/Use_firearm
- **relations:** SUBEVENT
- **gold_answer_phrase:** `his transport bomber aircraft was shot down by United States Army Air Forces fighter aircraft operating from Kukum Field on Guadalcanal`
- **gold_answer_sentence:** Isoroku Yamamoto, commander of the Combined Fleet of the Imperial Japanese Navy, was killed on Bougainville Island when his transport bomber aircraft 
- **generated_question:** How did Admiral Isoroku Yamamoto die during Operation Vengeance in 1943?
- **generation_error:** False
- **retry_attempts:** 2
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Yes on
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 0.33
- **quality_path_relevance:** 0.33
- **quality_difficulty_alignment:** 0.33
- **composite:** 0.483
- **item_id:** 24

### Easy #14

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
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
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
- **item_id:** 29

### Easy #15

- **doc_id:** `dd2a791aa826766cf0d05dc8102f5c8e`
- **title:** Death of Joy Gardner
- **difficulty:** Easy
- **event_path:** raid/Attack -> wrapped/Filling
- **relations:** TEMPORAL/CONTAINS
- **gold_answer_phrase:** `gagged with a 13-foot length of adhesive "Elastoplast" tape wrapped around her head`
- **gold_answer_sentence:** During a police raid on her home in Crouch End, London, she was restrained with handcuffs and leather straps and gagged with a 13-foot length of adhes
- **generated_question:** What method did police use to restrain her during the raid in Crouch End, London?
- **generation_error:** False
- **retry_attempts:** 2
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** handcuffs and leather straps on
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 30

### Easy #16

- **doc_id:** `dd2a791aa826766cf0d05dc8102f5c8e`
- **title:** Death of Joy Gardner
- **difficulty:** Easy
- **event_path:** placed/Placing -> involved/Cause_to_be_included
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `three of the police officers involved stood trial for Gardner's`
- **gold_answer_sentence:** In 1995, three of the police officers involved stood trial for Gardner's manslaughter, but were acquitted.
- **generated_question:** In 1995, which action did the three police officers take regarding the case? ?
- **generation_error:** False
- **retry_attempts:** 2
- **grammar_pass:** False (base: bad start: in)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (skipped (early exit))
- **path_coverage:** count=0 pass=False (skipped (early exit))
- **final_filter_pass:** False
- **final_filter_reason:** grammar=base: bad start: in; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** None
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 32

### Easy #17

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **title:** Myyrmanni bombing
- **difficulty:** Easy
- **event_path:** crowded/Come_together -> exploded/Attack
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five`
- **gold_answer_sentence:** A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.
- **generated_question:** What immediately followed the detonation of the bomb at the crowded Myyrmanni shopping mall?
- **generation_error:** False
- **retry_attempts:** 2
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Five died immediately.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.33
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.633
- **item_id:** 34

### Easy #18

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **title:** King David Hotel bombing
- **difficulty:** Easy
- **event_path:** planned/Arranging -> planted/Placing
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `members of the Irgun planted a bomb in the basement of the main building of the`
- **gold_answer_sentence:** Disguised as Arab workmen and as hotel waiters, members of the Irgun planted a bomb in the basement of the main building of the hotel, whose southern 
- **generated_question:** What did the members of the Irgun do after they disguised themselves as Arab workers and hotel waiters?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** judge_error (judge_error after 3 attempts: {"asks_target":"yes","answer_target":"members of the Irgun planted a bomb in the)
- **path_coverage:** count=0 pass=False (covers 0 all events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** None
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 35

### Easy #19

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **title:** King David Hotel bombing
- **difficulty:** Easy
- **event_path:** Disguised/Wearing -> carried out/Attack
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `a bomb search had already been carried out, it appears`
- **gold_answer_sentence:** From the fact that a bomb search had already been carried out, it appears that a hoax call or tip-off had been received at the hotel earlier that day.
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
- **item_id:** 36

### Easy #20

- **doc_id:** `8af040b8bfa2eec89d1197392f2fdac5`
- **title:** Cyclone Forrest
- **difficulty:** Easy
- **event_path:** classified/Check -> damaged/Damaging
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `Martin's Island were damaged`
- **gold_answer_sentence:** Only two deaths were recorded and overall damage was light, though half of all homes on St. Martin's Island were damaged.
- **generated_question:** What happened to St. Martin's Island after Cyclone Forrest turned eastward?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Cyclone Forrest turned St. Martin's Island on turned an eastward turn.
- **solver_eval_status:** ok
- **judge_answerable:** 0.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.467
- **item_id:** 38

### Easy #21

- **doc_id:** `f83fc49c020ec542b16d463b3f7c2c14`
- **title:** Sinjar massacre
- **difficulty:** Easy
- **event_path:** started/Process_start -> killing/Killing
- **relations:** CAUSE/PRECONDITION
- **gold_answer_phrase:** `The Sinjar massacre was the genocidal killing`
- **gold_answer_sentence:** The Sinjar massacre was the genocidal killing and abduction of thousands of Yazidi men in Sinjar ( "Şingal") city and Sinjar District in Iraq's Nineve
- **generated_question:** What did ISIL's attack on Sinjar and neighboring towns on August 3, 2014, lead to?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 all events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 45

### Easy #22

- **doc_id:** `8af040b8bfa2eec89d1197392f2fdac5`
- **title:** Cyclone Forrest
- **difficulty:** Easy
- **event_path:** striking/Attack -> losses/Earnings_and_losses
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `Agricultural losses in the hardest hit areas were estimated at ฿800 million (US$32 million)`
- **gold_answer_sentence:** Agricultural losses in the hardest hit areas were estimated at ฿800 million (US$32 million).
- **generated_question:** What were the estimated agricultural damages in the hardest hit areas of Thailand due to Forrest?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 all events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 46

### Easy #23

- **doc_id:** `37d153abeafe0477ce6a6398e26eb442`
- **title:** Defense of Sihang Warehouse
- **difficulty:** Easy
- **event_path:** defense/Defending -> Battle/Hostile_encounter
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `marked the beginning of the end of the three-month Battle of Shanghai in the opening phase of the Second Sino-Japanese War`
- **gold_answer_sentence:** The defense of Sihang Warehouse took place from October 26 to November 1, 1937, and marked the beginning of the end of the three-month Battle of Shang
- **generated_question:** What was the broader conflict during which the defense of Sihang Warehouse took place?
- **generation_error:** False
- **retry_attempts:** 2
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (weak trigger but valid phrase: 'marked the beginning of the end of the three-month Battle of Shanghai in the opening phase of the Second Sino-Japanese War')
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Second Sino-Japanese War
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 47

### Easy #24

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **title:** Myyrmanni bombing
- **difficulty:** Easy
- **event_path:** investigated/Criminal_investigation -> determined/Deciding
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `His motive was not determined`
- **gold_answer_sentence:** His motive was not determined.
- **generated_question:** What was not established regarding Gerdt's motive for the incident?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 all events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 48

### Easy #25

- **doc_id:** `c0c67db40cd5e2e03645ff1116fafcfc`
- **title:** Cherry Valley massacre
- **difficulty:** Easy
- **event_path:** ordered/Arranging -> minimize/Cause_change_of_position_on_a_scale
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `he actively sought to minimize the atrocities`
- **gold_answer_sentence:** He was not present at Wyoming — although many thought he was — and he actively sought to minimize the atrocities that took place at Cherry Valley.
- **generated_question:** What did Brant do regarding the atrocities at Cherry Valley?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 all events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 50

### Easy #26

- **doc_id:** `dd2a791aa826766cf0d05dc8102f5c8e`
- **title:** Death of Joy Gardner
- **difficulty:** Easy
- **event_path:** attention/Perception_active -> inquest/Criminal_investigation
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `no coroner's inquest or public inquiry into the circumstances of Gardner's death has been held`
- **gold_answer_sentence:** Despite continuing pressure by campaigners, no coroner's inquest or public inquiry into the circumstances of Gardner's death has been held.
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
- **item_id:** 52

### Easy #27

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **title:** Battle of Ciołków
- **difficulty:** Easy
- **event_path:** killed/Killing -> dispersed/Dispersal
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `After a short hand-to-hand fight the Russian commander was killed and his unit dispersed`
- **gold_answer_sentence:** After a short hand-to-hand fight (the Polish unit had only two pieces of firearms and was mostly equipped with sabres, war scythes and improvised weap
- **generated_question:** What happened to the Russian unit after the hand-to-hand fight?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 all events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 58

### Easy #28

- **doc_id:** `db50381e7d1dd4a41fb4ac60eaebe3a4`
- **title:** Battle of Orthez
- **difficulty:** Easy
- **event_path:** pause/Change_event_time -> Battle/Hostile_encounter
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `The next action was the Battle of Toulouse`
- **gold_answer_sentence:** The next action was the Battle of Toulouse.
- **generated_question:** What was the next action after the Allied corps paused in the campaign?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (weak trigger but valid phrase: 'The next action was the Battle of Toulouse')
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Resuming eastwardwardward march
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 67

### Easy #29

- **doc_id:** `04a82d4eac379a98efcd87ebdba0b0ce`
- **title:** Death and state funeral of Raúl Alfonsín
- **difficulty:** Easy
- **event_path:** arranged/Arranging -> mourning/Rite
- **relations:** CAUSE/PRECONDITION
- **gold_answer_phrase:** `arranged three days of national mourning`
- **gold_answer_sentence:** Vice president Julio Cobos, the acting president at the time, arranged three days of national mourning and a state funeral at the Palace of the Argent
- **generated_question:** What did Vice President Julio Cobos arrange after Alfonsín's death?
- **generation_error:** False
- **retry_attempts:** 2
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** three days of national mourning and a state state funeral at
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.33
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.633
- **item_id:** 69

### Easy #30

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **title:** Myyrmanni bombing
- **difficulty:** Easy
- **event_path:** took place/Process_start -> released/Releasing
- **relations:** CAUSE/PRECONDITION
- **gold_answer_phrase:** `released at the scene`
- **gold_answer_sentence:** 66 victims required hospitalization with the remainder treated and released at the scene.
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
- **item_id:** 70

### Easy #31

- **doc_id:** `6dabade56742b6040cda6a5838176f6c`
- **title:** Who's That Girl World Tour
- **difficulty:** Easy
- **event_path:** trained/Education_teaching -> collaborated/Collaboration
- **relations:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `she collaborated with designer Marlene`
- **gold_answer_sentence:** For the costumes, she collaborated with designer Marlene Stewart, expanding on the idea of bringing her popular video characters to life onstage, rewo
- **generated_question:** What did Madonna do after she trained for the choreography?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (The question does not ask about the target final event, which is 'collaborated.' It asks about Madonna's training for choreography, which is not directly related to the target event.)
- **path_coverage:** count=1 pass=True (covers 1 all events, need >= 1 [PASS])
- **final_filter_pass:** False
- **final_filter_reason:** answer_consistency=no: The question does not ask about the target final event, which is 'collaborated.' It asks about Madonna's training for choreography, which is not directly related to the target event.
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** False
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 72

### Easy #32

- **doc_id:** `f28bce270df5a122c09365002d247e76`
- **title:** United States occupation of Nicaragua
- **difficulty:** Easy
- **event_path:** assumed/Choosing -> opposed/Agree_or_refuse_to_act
- **relations:** CAUSE/PRECONDITION
- **gold_answer_phrase:** `President Herbert Hoover (1929–1933) opposed the relationship`
- **gold_answer_sentence:** President Herbert Hoover (1929–1933) opposed the relationship.
- **generated_question:** How Herbert Hoover oppose the United of Nicaragua and the United States as in the context of American military interventions in Nicaragua to prevent a Nicaraguan Canal? 
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=2 pass=True (covers 2 all events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Opposed 1916 treaty.
- **solver_eval_status:** ok
- **judge_answerable:** 0.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.467
- **item_id:** 75


## Medium (35 items)

### Medium #1

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **title:** Crimean War
- **difficulty:** Medium
- **event_path:** prepared/GetReady -> granted/Preventing_or_letting -> recognize/Know
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `to recognize the need for fundamental transformations aimed at modernizing`
- **gold_answer_sentence:** The humiliation forced Russia's educated elites to identify the Empire's problems and to recognize the need for fundamental transformations aimed at m
- **generated_question:** What significant realization did Russia's educated elites come to after the Crimean War, prompting a push for fundamental transformations? 
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** partial (brief)
- **path_coverage:** count=0 pass=False (covers 0 prior events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** False
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 3

### Medium #2

- **doc_id:** `8af040b8bfa2eec89d1197392f2fdac5`
- **title:** Cyclone Forrest
- **difficulty:** Medium
- **event_path:** Originating/Coming_to_be -> Tracking/Scrutiny -> storm/Catastrophe
- **relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `also referred to as Tropical Storm`
- **gold_answer_sentence:** Cyclone Forrest, also referred to as Tropical Storm Forrest, was a powerful tropical cyclone that prompted the evacuation of 600,000 people in Banglad
- **generated_question:** What was the name of the tropical cyclone that prompted the evacuation of 600,000 people in Bangladesh in late November 1992, and how did it develop over time?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=2 pass=True (covers 2 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Cyclone Forrest, developed from disturbed weather.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 1.0
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.75
- **item_id:** 12

### Medium #3

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
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
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
- **item_id:** 13

### Medium #4

- **doc_id:** `6dabade56742b6040cda6a5838176f6c`
- **title:** Who's That Girl World Tour
- **difficulty:** Medium
- **event_path:** trained/Education_teaching -> addressing/Expressing_publicly -> wearing/Wearing
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `wearing a conical`
- **gold_answer_sentence:** A statue of Madonna, wearing a conical bra, was erected in her name at the center of the town of Pacentro in Italy, where her ancestors used to live.
- **generated_question:** What kind of attire did the statue of Madonna in Pacentro feature, and how did her physical training contribute to this choice?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (extracted from text: asks=yes ans=no cons=no)
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** False
- **final_filter_reason:** answer_consistency=no: extracted from text: asks=yes ans=no cons=no
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** False
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 14

### Medium #5

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **title:** Operation Deny Flight
- **difficulty:** Medium
- **event_path:** expanded/Expansion -> bombed/Attack -> helped/Assistance
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `These engagements helped show`
- **gold_answer_sentence:** These engagements helped show that NATO had adapted to the post-Cold War era and could operate in environments other than a major force on force engag
- **generated_question:** What did these operations demonstrate about NATO's capabilities after expanding the mission and carrying out bombing raids?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** judge_error (judge_error after 3 attempts: No, the target did the did did did did did did did did did did did did did did d)
- **path_coverage:** count=2 pass=True (covers 2 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** NATO aircraft bombed ground
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 15

### Medium #6

- **doc_id:** `c0c67db40cd5e2e03645ff1116fafcfc`
- **title:** Cherry Valley massacre
- **difficulty:** Medium
- **event_path:** restrain/Hindering -> took place/Process_start -> descended on/Motion_directional
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `Mohawks descended on Cherry`
- **gold_answer_sentence:** A mixed force of Loyalists, British soldiers, Seneca and Mohawks descended on Cherry Valley, whose defenders, despite warnings, were unprepared for th
- **generated_question:** What actions did the Loyalists, British soldiers, Seneca, and Mohawks take that led to the unpreparedness of Cherry Valley's defenders during the attack?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** partial (brief)
- **path_coverage:** count=0 pass=False (covers 0 prior events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** False
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 17

### Medium #7

- **doc_id:** `3dcfd60153822a6a8f6a516f161fc506`
- **title:** Battle of Malacca (1641)
- **difficulty:** Medium
- **event_path:** capture/Conquering -> combined/Cause_to_amalgamate -> removing/Removing
- **relations:** TEMPORAL/BEFORE, CAUSE/CAUSE
- **gold_answer_phrase:** `removing their influence in the Malay archipelago`
- **gold_answer_sentence:** This combined Dutch-Johor effort effectively destroyed the last bastion of Portuguese power, removing their influence in the Malay archipelago.
- **generated_question:** What was the outcome of the combined Dutch-Johor effort that followed the successful capture of Malacca from the Portuguese?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=2 pass=True (covers 2 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Dutchutch destroyed Portuguese power influence.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 18

### Medium #8

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **title:** Operation Deny Flight
- **difficulty:** Medium
- **event_path:** began/Process_start -> combat/Hostile_encounter -> end/Process_end
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `by its end on 20 December`
- **gold_answer_sentence:** Twelve NATO members contributed forces to the operation and, by its end on 20 December 1995, NATO pilots had flown 100,420 sorties.
- **generated_question:** What was the final outcome of Operation Deny Flight, and how many sorties were flown by NATO pilots during the operation?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** judge_error (judge_error after 3 attempts: "asks_target":"yes","answeron":"no","consistent":"no",")
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** 14,40
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 0.33
- **quality_path_relevance:** 0.33
- **quality_difficulty_alignment:** 0.33
- **composite:** 0.483
- **item_id:** 20

### Medium #9

- **doc_id:** `275eb0bc9caacb30e4f58d85469458d1`
- **title:** Survivor Series (1999)
- **difficulty:** Medium
- **event_path:** delivered/Sending -> take/Creating -> recover/Recovering
- **relations:** TEMPORAL/BEFORE, TEMPORAL/CONTAINS
- **gold_answer_phrase:** `had been run down by a car earlier in the night (which was an angle for Austin to take time to recover from his injuries)`
- **gold_answer_sentence:** He was a replacement for Stone Cold Steve Austin, who had been run down by a car earlier in the night (which was an angle for Austin to take time to r
- **generated_question:** Why was Stone Cold Steve Austin replaced by Big Show, and what was the wrestling company's angle for this change?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 prior events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 22

### Medium #10

- **doc_id:** `db50381e7d1dd4a41fb4ac60eaebe3a4`
- **title:** Battle of Orthez
- **difficulty:** Medium
- **event_path:** offered/Giving -> conducted/Action -> fall/Motion_directional
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `fall back east toward Toulouse`
- **gold_answer_sentence:** In subsequent operations, Soult decided to abandon the large western port of Bordeaux and fall back east toward Toulouse.
- **generated_question:** What decision did Soult make after abandoning Bordeaux in the context of the Peninsular War?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 prior events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 25

### Medium #11

- **doc_id:** `f83fc49c020ec542b16d463b3f7c2c14`
- **title:** Sinjar massacre
- **difficulty:** Medium
- **event_path:** started/Process_start -> enabled/Preventing_or_letting -> evacuated/Emptying
- **relations:** TEMPORAL/BEFORE, CAUSE/PRECONDITION
- **gold_answer_phrase:** `fled into the Sinjar Mountains to be evacuated`
- **gold_answer_sentence:** The assistance of PKK and YPG enabled the majority of the 50,000 Yazidis who fled into the Sinjar Mountains to be evacuated.
- **generated_question:** What assistance allowed the majority of the 50,000 Yazidis who fled into the Sinjar Mountains to be helped out? 
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 prior events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 26

### Medium #12

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **title:** Crimean War
- **difficulty:** Medium
- **event_path:** changes/Change -> quipped/Statement -> proved/Convincing
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `The Crimean War proved to be the moment of truth for Nikolaevan Russia`
- **gold_answer_sentence:** The Crimean War proved to be the moment of truth for Nikolaevan Russia.
- **generated_question:** What significant event did the Crimean War represent for Nikolaevan Russia, considering Karl Marx's comment and the subsequent demands for action from the public?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 prior events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 27

### Medium #13

- **doc_id:** `8af040b8bfa2eec89d1197392f2fdac5`
- **title:** Cyclone Forrest
- **difficulty:** Medium
- **event_path:** enacted/Statement -> turned/Becoming -> evacuation/Emptying
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `prompted the evacuation of 600,000 people in Bangladesh in late November 1992`
- **gold_answer_sentence:** Cyclone Forrest, also referred to as Tropical Storm Forrest, was a powerful tropical cyclone that prompted the evacuation of 600,000 people in Banglad
- **generated_question:** What event was successfully carried out, leading to the relocation of 600,000 people and preventing a potential disaster during Cyclone Forrest's approach to Bangladesh in 1992?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=2 pass=True (covers 2 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Mass evacuation plans
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 1.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.917
- **item_id:** 28

### Medium #14

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
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** UN peacekeepers taken as hostages.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 1.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.967
- **item_id:** 31

### Medium #15

- **doc_id:** `6dabade56742b6040cda6a5838176f6c`
- **title:** Who's That Girl World Tour
- **difficulty:** Medium
- **event_path:** looked/Perception_active -> playing/Competition -> presenting/Presence
- **relations:** TEMPORAL/BEFORE, TEMPORAL/CONTAINS
- **gold_answer_phrase:** `presenting her older songs for the show`
- **gold_answer_sentence:** Patrick Leonard, who was the music director, encouraged Madonna to go with the idea of remixing and presenting her older songs for the show.
- **generated_question:** What did Madonna decide to do with her older songs during the rehearsals according to Patrick Leonard's encouragement? 
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 prior events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 33

### Medium #16

- **doc_id:** `db50381e7d1dd4a41fb4ac60eaebe3a4`
- **title:** Battle of Orthez
- **difficulty:** Medium
- **event_path:** isolated/Having_or_lacking_access -> overcome/Conquering -> decided/Deciding
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `Soult decided to abandon the large western port of Bordeaux and fall back east toward Toulouse`
- **gold_answer_sentence:** In subsequent operations, Soult decided to abandon the large western port of Bordeaux and fall back east toward Toulouse.
- **generated_question:** What decision did Marshal Soult make after his army was isolated and forced to retreat during the subsequent operations following the Battle of Orthez?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** partial (partial)
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Abandon Bordeaux, retreat east.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 1.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.917
- **item_id:** 39

### Medium #17

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
- **answer_consistency:** judge_error (judge_error after 3 attempts: {"asks_target":" :yes" , "answer " "yes" , "consistent " "yes" , "reason " "brie)
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** The Polish commander was and pérdida wasonestly suffered as wasansible
 pérdida one de dos dos dos dos dos dos dos pérdida one eye
icílos dos dos dos dos dos dos dos dos dos dos
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 41

### Medium #18

- **doc_id:** `3dcfd60153822a6a8f6a516f161fc506`
- **title:** Battle of Malacca (1641)
- **difficulty:** Medium
- **event_path:** capture/Conquering -> began/Process_start -> destroyed/Destroying
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `This combined Dutch-Johor effort effectively destroyed the last bastion of Portuguese`
- **gold_answer_sentence:** This combined Dutch-Johor effort effectively destroyed the last bastion of Portuguese power, removing their influence in the Malay archipelago.
- **generated_question:** What significant action did the Dutch and their allies take that marked the end of Portuguese power in the Malay archipelago after they began their campaign?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=2 pass=True (covers 2 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Captured Malacca in January 1641.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.717
- **item_id:** 43

### Medium #19

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **title:** Operation Deny Flight
- **difficulty:** Medium
- **event_path:** began/Process_start -> expanded/Expansion -> bombed/Attack
- **relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `NATO aircraft first bombed ground targets in an operation near Goražde`
- **gold_answer_sentence:** The operation included the first combat engagement in NATO's history, a 28 February 1994 air battle over Banja Luka, and in April 1994, NATO aircraft 
- **generated_question:** What was the first significant action taken by NATO aircraft during Operation Deny Flight after the mission scope was expanded?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** judge_error (judge_error after 3 attempts: {"asks_target': 'yes',
'answer': 'yes',
'consistent': 'yes',
're':: "brief"})
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** NATO aircraft provided air-ground support and conducted coercive strikes on targets kukos in Bosnia. kukos the mission scope was expanded kukos kukos kuk kukos kuk kukos kuk kukos kuk
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.717
- **item_id:** 44

### Medium #20

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **title:** Crimean War
- **difficulty:** Medium
- **event_path:** prepared/GetReady -> forbade/Preventing_or_letting -> restoring/Recovering
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `restoring Russia's position in the ranks of European powers`
- **gold_answer_sentence:** The humiliation forced Russia's educated elites to identify the Empire's problems and to recognize the need for fundamental transformations aimed at m
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
- **item_id:** 49

### Medium #21

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **title:** King David Hotel bombing
- **difficulty:** Medium
- **event_path:** occurred/Coming_to_be -> carried out/Attack -> arisen/Coming_to_be
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `Controversy has arisen over the timing`
- **gold_answer_sentence:** Controversy has arisen over the timing and adequacy of the warnings, and the reasons why, given that warnings were made, the hotel was not evacuated.
- **generated_question:** What controversy arose regarding the timing and adequacy of the warnings, and the reasons why the hotel was evacuated evacuated before before the explosion?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** False (base: word repetition: evacuated)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (skipped (early exit))
- **path_coverage:** count=0 pass=False (skipped (early exit))
- **final_filter_pass:** False
- **final_filter_reason:** grammar=base: word repetition: evacuated; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** None
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 51

### Medium #22

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **title:** Myyrmanni bombing
- **difficulty:** Medium
- **event_path:** come to/Motion -> exploded/Attack -> injured/Bodily_harm
- **relations:** TEMPORAL/BEFORE, CAUSE/CAUSE
- **gold_answer_phrase:** `166 people were injured, including 10 children`
- **gold_answer_sentence:** 166 people were injured, including 10 children.
- **generated_question:** What happened to the people who were at the shopping center when the bomb exploded, and how many of them were children?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** 7 died, two were children.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 53

### Medium #23

- **doc_id:** `3dcfd60153822a6a8f6a516f161fc506`
- **title:** Battle of Malacca (1641)
- **difficulty:** Medium
- **event_path:** launching/Military_operation -> assaulted/Attack -> control/Control
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `the Dutch took control of Malacca and agreed not to seek territories or wage war with the Malay kingdoms`
- **gold_answer_sentence:** In line with the agreement with Johor in 1606, the Dutch took control of Malacca and agreed not to seek territories or wage war with the Malay kingdom
- **generated_question:** What was the ultimate result of the combined Dutch and Johor forces' efforts against the Portuguese in Malacca in 1641?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (weak trigger but valid phrase: 'the Dutch took control of Malacca and agreed not to seek territories or wage war with the Malay kingdoms')
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 prior events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 54

### Medium #24

- **doc_id:** `6fc6538bd2c19d34942f7f36274d83ae`
- **title:** Cyclone Winifred
- **difficulty:** Medium
- **event_path:** persisted/Wearing -> aid/Assistance -> relief/Assistance
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `$150 million program to provide relief to damaged areas`
- **gold_answer_sentence:** Meanwhile, the Commonwealth of Australia initiated a three-year, $150 million program to provide relief to damaged areas.
- **generated_question:** What kind of assistance did the Commonwealth of Australia provide to help with the aftermath of Winifred's damage?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** partial (brief)
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Three-year, $150 million relief program.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 55

### Medium #25

- **doc_id:** `84ce009a07b987d60a79d92bc4d45744`
- **title:** 2006 state of emergency in the Philippines
- **difficulty:** Medium
- **event_path:** claimed/Statement -> occurred/Presence -> revocation/Process_end
- **relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `an immediate revocation on all licenses and permits to hold demonstrations and protests`
- **gold_answer_sentence:** The state of national emergency also led to a temporary suspension of lower-level education classes and an immediate revocation on all licenses and pe
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
- **item_id:** 56

### Medium #26

- **doc_id:** `06f91ced00b41867979f3d5dc6996da2`
- **title:** Operation Vengeance
- **difficulty:** Medium
- **event_path:** killed/Killing -> revenge/Revenge -> claimed/Statement
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `pilots claimed to have shot down three twin-engined bombers and two fighters during the`
- **gold_answer_sentence:** The U.S. pilots claimed to have shot down three twin-engined bombers and two fighters during the mission, but Japanese sources show only two bombers w
- **generated_question:** What did the U.S. pilots claim to have achieved during their mission to kill Yamamoto, and how does this claim compare to Japanese sources?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** judge_error (judge_error after 3 attempts: Does the target target did the did_target: Does the did_target did did the did_e)
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** U.S. pilots claim to shoot down two bombers and two fighters., Japanese sources only confirm two bombers two shot down.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.717
- **item_id:** 57

### Medium #27

- **doc_id:** `3dcfd60153822a6a8f6a516f161fc506`
- **title:** Battle of Malacca (1641)
- **difficulty:** Medium
- **event_path:** capture/Conquering -> casualties/Catastrophe -> rallied/Filling
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `the battle rallied the forces of the Sultanate of Johor in an alliance with the Dutch`
- **gold_answer_sentence:** Although the Dutch were routed, the Portuguese fleet of Martim Afonso de Castro, the Viceroy of Portuguese India, suffered heavier casualties and the 
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
- **item_id:** 59

### Medium #28

- **doc_id:** `28a13a10cb57f8245b1f98270bad9860`
- **title:** Territorial era of Minnesota
- **difficulty:** Medium
- **event_path:** transition/Change -> replacing/Change_of_leadership -> formed/Coming_to_be
- **relations:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE
- **gold_answer_phrase:** `The Minnesota Territory itself was formed only in 1849`
- **gold_answer_sentence:** The Minnesota Territory itself was formed only in 1849 but the area had a rich history well before this.
- **generated_question:** What significant change occurred in the economic resources of the area before the formation of the Minnesota Territory in 1849?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Furs declined, lumber rose.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 60

### Medium #29

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
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** December 2014 Sinjar offensive
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.667
- **item_id:** 61

### Medium #30

- **doc_id:** `f83fc49c020ec542b16d463b3f7c2c14`
- **title:** Sinjar massacre
- **difficulty:** Medium
- **event_path:** started/Process_start -> reacted/Response -> broke/Destroying
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `This offensive broke ISIL's troop transport routes and supply lines between Mosul and`
- **gold_answer_sentence:** This offensive broke ISIL's troop transport routes and supply lines between Mosul and Raqqa, the largest ISIL-controlled cities in Iraq and Syria at t
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
- **item_id:** 62

### Medium #31

- **doc_id:** `c0c67db40cd5e2e03645ff1116fafcfc`
- **title:** Cherry Valley massacre
- **difficulty:** Medium
- **event_path:** descended on/Motion_directional -> minimize/Cause_change_of_position_on_a_scale -> drove/Motion
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `drove the Iroquois out of western New York`
- **gold_answer_sentence:** The massacre contributed to calls for reprisals, leading to the 1779 Sullivan Expedition which drove the Iroquois out of western New York.
- **generated_question:** What action ultimately led to the Iroquois being forced out of western New York in 1779?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 prior events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 64

### Medium #32

- **doc_id:** `37d153abeafe0477ce6a6398e26eb442`
- **title:** Defense of Sihang Warehouse
- **difficulty:** Medium
- **event_path:** shot/Use_firearm -> dare/Risk -> keep/Temporary_stay
- **relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `whom the Japanese wanted to keep out of the war`
- **gold_answer_sentence:** It was across from the foreign concessions in Shanghai, and the Japanese did not dare to call naval artillery strikes on the area, since a stray shot 
- **generated_question:** Why the Japanese avoid using naval artillery strikes and mustard gas in the area where the battle took took take place? 
- **generation_error:** False
- **retry_attempts:** 2
- **grammar_pass:** False (base: word repetition: took)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (skipped (early exit))
- **path_coverage:** count=0 pass=False (skipped (early exit))
- **final_filter_pass:** False
- **final_filter_reason:** grammar=base: word repetition: took; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** None
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 65

### Medium #33

- **doc_id:** `84ce009a07b987d60a79d92bc4d45744`
- **title:** 2006 state of emergency in the Philippines
- **difficulty:** Medium
- **event_path:** occurred/Presence -> revocation/Process_end -> lifted/Process_end
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `the President lifted the state of emergency`
- **gold_answer_sentence:** 1021, the President lifted the state of emergency.
- **generated_question:** What action did the government take on March 3, 2006, regarding the state of emergency?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=0 pass=False (covers 0 prior events, need >= 1 [FAIL])
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 71

### Medium #34

- **doc_id:** `04a82d4eac379a98efcd87ebdba0b0ce`
- **title:** Death and state funeral of Raúl Alfonsín
- **difficulty:** Medium
- **event_path:** died/Death -> took/Bringing -> left/Placing
- **relations:** CAUSE/PRECONDITION, CAUSE/PRECONDITION
- **gold_answer_phrase:** `left him at the pantheon for the veterans of the Revolution of the Park`
- **gold_answer_sentence:** A military escort took his coffin to the La Recoleta Cemetery, and left him at the pantheon for the veterans of the Revolution of the Park.
- **generated_question:** What final resting place was given to Raúl Alfonsín after his death?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** judge_error (judge_error after 3 attempts: {"yes")
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** La Recoleta Cemetery
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 0.33
- **quality_path_relevance:** 0.33
- **quality_difficulty_alignment:** 0.33
- **composite:** 0.483
- **item_id:** 73

### Medium #35

- **doc_id:** `84ce009a07b987d60a79d92bc4d45744`
- **title:** 2006 state of emergency in the Philippines
- **difficulty:** Medium
- **event_path:** allowed/Preventing_or_letting -> detain/Preventing_or_letting -> lifted/Process_end
- **relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `President Arroyo lifted the state of emergency on March`
- **gold_answer_sentence:** President Arroyo lifted the state of emergency on March 3, 2006 by the virtue of Proclamation No.
- **generated_question:** What action did President Arroyo take regarding detentions after the state of emergency was allowed on March 3, 2006?
- **generation_error:** False
- **retry_attempts:** 2
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=True (covers 1 prior events, need >= 1 [PASS])
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Ar the on
- **solver_eval_status:** ok
- **judge_answerable:** 0.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 0.33
- **quality_path_relevance:** 0.33
- **quality_difficulty_alignment:** 0.33
- **composite:** 0.283
- **item_id:** 74


## Hard (9 items)

### Hard #1

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **title:** Crimean War
- **difficulty:** Hard
- **event_path:** stopped/Preventing_or_letting -> destroyed/Destroying -> rushed/Motion -> forbade/Preventing_or_letting
- **relations:** TEMPORAL/BEFORE, CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `It forbade Russia from basing warships in the Black Sea`
- **gold_answer_sentence:** It forbade Russia from basing warships in the Black Sea.
- **generated_question:** After the Ottomans stopped the Russian advance at Silistra and their fleet destroyed a Russian reinforcement attempt at Sinop, what did France and Britain do to prevent further Russian aggression?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** no (extracted from text: asks=no ans=no cons=no)
- **path_coverage:** count=2 pass=True (covers 2 prior events, need >= 2 [PASS])
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
- **item_id:** 4

### Hard #2

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **title:** King David Hotel bombing
- **difficulty:** Hard
- **event_path:** warnings/Warning -> called/Communication -> carried out/Attack -> killed/Killing
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `91 people of various nationalities were killed`
- **gold_answer_sentence:** 91 people of various nationalities were killed, and 46 were injured.
- **generated_question:** When did the warnings lead the I on the Irgun on carried the bombing and did the hotel staff on did evacuate?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** partial (The question includes details not present in the context and asks about specific actions (warnings, evacuation) that are not addressed in the provided sentences.)
- **path_coverage:** count=2 pass=True (covers 2 prior events, need >= 2 [PASS])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** No, they did not.
- **solver_eval_status:** ok
- **judge_answerable:** 0.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 0.33
- **quality_path_relevance:** 0.33
- **quality_difficulty_alignment:** 0.67
- **composite:** 0.333
- **item_id:** 10

### Hard #3

- **doc_id:** `3dcfd60153822a6a8f6a516f161fc506`
- **title:** Battle of Malacca (1641)
- **difficulty:** Hard
- **event_path:** began/Process_start -> launching/Military_operation -> took/Conquering -> agreed/Agree_or_refuse_to_act
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION
- **gold_answer_phrase:** `agreed not to seek territories or wage war with the Malay kingdoms`
- **gold_answer_sentence:** In line with the agreement with Johor in 1606, the Dutch took control of Malacca and agreed not to seek territories or wage war with the Malay kingdom
- **generated_question:** After the Dutch began their campaign to destroy Portuguese power and launched small incursions against the Portuguese, what did they agree to do in line with their agreement with Johor in 1606?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=2 pass=True (covers 2 prior events, need >= 2 [PASS])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Not seek territories.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 1.0
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.75
- **item_id:** 11

### Hard #4

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
- **path_coverage:** count=2 pass=True (covers 2 prior events, need >= 2 [PASS])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** False
- **final_filter_reason:** answer_consistency=no: extracted from text: asks=yes ans=no cons=no
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** False
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 37

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
- **path_coverage:** count=3 pass=True (covers 3 prior events, need >= 2 [PASS])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** was killed
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.717
- **item_id:** 40

### Hard #6

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **title:** Battle of Ciołków
- **difficulty:** Hard
- **event_path:** uprising/Change_of_leadership -> approached/Arriving -> refused/Agree_or_refuse_to_act -> ordered/Arranging
- **relations:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `ordered a charge of the Russians`
- **gold_answer_sentence:** However, Rogaliński refused to negotiate and ordered a charge of the Russians.
- **generated_question:** During the uprising, when the Russians approached the Polish quarters and the commander tried to negotiate, what did Rogaliński decide to do next?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=2 pass=True (covers 2 prior events, need >= 2 [PASS])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Rogaliński refused to negotiate.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 1.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.967
- **item_id:** 42

### Hard #7

- **doc_id:** `db50381e7d1dd4a41fb4ac60eaebe3a4`
- **title:** Battle of Orthez
- **difficulty:** Hard
- **event_path:** Moving/Motion -> surrounded/Surrounding -> pushed/Motion -> battle/Hostile_encounter
- **relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `the French marshal offered battle`
- **gold_answer_sentence:** Resuming their eastward drive, the remaining two Allied corps pushed Soult's army back to Orthez where the French marshal offered battle.
- **generated_question:** After the western-most Allied corps surrounded and isolated Bayonne, what event occurred as the remaining two Allied corps continued their eastward drive?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (weak trigger but valid phrase: 'the French marshal offered battle')
- **answer_consistency:** yes (brief)
- **path_coverage:** count=1 pass=False (covers 1 prior events, need >= 2 [FAIL])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** False
- **final_filter_reason:** path_coverage=covers 1 prior events, need >= 2 [FAIL]
- **solver_answer:** not_run
- **solver_eval_status:** not_run
- **judge_answerable:** True
- **judge_solver_correct:** ?
- **judge_support_covered:** ?
- **quality_fluency:** ?
- **quality_path_relevance:** ?
- **quality_difficulty_alignment:** ?
- **composite:** ?
- **item_id:** 63

### Hard #8

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **title:** Myyrmanni bombing
- **difficulty:** Hard
- **event_path:** crowded/Come_together -> took place/Process_start -> investigated/Criminal_investigation -> closed/Self_motion
- **relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION
- **gold_answer_phrase:** `closed in January 2003 without any indictments as Gerdt was the sole suspect`
- **gold_answer_sentence:** The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.
- **generated_question:** How was the Myyrmanni shopping mall when the bombing took place, and what was the subsequent investigation like?
- **generation_error:** False
- **retry_attempts:** 3
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=2 pass=True (covers 2 prior events, need >= 2 [PASS])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Crowded; no indictments.
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 1.0
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.75
- **item_id:** 66

### Hard #9

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **title:** Operation Deny Flight
- **difficulty:** Hard
- **event_path:** began/Process_start -> providing/Supply -> helped/Assistance -> adapted/Coming_to_be
- **relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE, TEMPORAL/CONTAINS
- **gold_answer_phrase:** `NATO had adapted to the post-Cold War era and could operate in environments other than a major force on force engagement on the plains of Central Europe`
- **gold_answer_sentence:** These engagements helped show that NATO had adapted to the post-Cold War era and could operate in environments other than a major force on force engag
- **generated_question:** After the operation began and started providing close air support, how did NATO demonstrate its ability in the post-Cold War era?
- **generation_error:** False
- **retry_attempts:** 1
- **grammar_pass:** True (pass)
- **answer_phrase_pass:** True (valid phrase)
- **weak_trigger_pass:** True (not a weak trigger)
- **answer_consistency:** yes (brief)
- **path_coverage:** count=2 pass=True (covers 2 prior events, need >= 2 [PASS])
- **shortcut_without_path:** no
- **needs_prior_events:** yes
- **hard_degraded:** False (not degraded: shortcut=no, needs_prior=yes)
- **final_filter_pass:** True
- **final_filter_reason:** all checks passed
- **solver_answer:** Operation Den Den Den
- **solver_eval_status:** ok
- **judge_answerable:** 1.0
- **judge_solver_correct:** 0.0
- **judge_support_covered:** 1.0
- **quality_fluency:** 1.0
- **quality_path_relevance:** 0.67
- **quality_difficulty_alignment:** 1.0
- **composite:** 0.717
- **item_id:** 68
