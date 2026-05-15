# Independent Difficulty Evaluation — Audit Sample

**Judge Model:** gpt-4o-mini
**Seed:** 42

This evaluation is independent of the solver and does not use solver answers or solver correctness.

## ICL-TargetQG

### [Easy -> Medium] doc_id=f46091471f38

- **Target difficulty:** Easy
- **Predicted difficulty:** Medium | **Steps:** 2
- **Path dependency:** strong | **Prior needed:** 2
- **Question:** What was the result of the operation carried out by the Irgun?
- **Answer:** carriedried out despite the operation being cancelled
- **Event path:** Disguised/Wearing -> operation/Military_operation
- **Relations:** TEMPORAL/BEFORE
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question requires connecting the operation's approval status and its execution, which involves understanding the cancellation of the operation and its consequences, thus needing two inference steps.
- **Path reason:** The question about the operation's result requires understanding both the disguise used by the Irgun and the context of the military operation.

### [Medium -> Easy] doc_id=a24058769038

- **Target difficulty:** Medium
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What happened to the Polish commander after the short hand-to-hand fight?
- **Answer:** lost his eye
- **Event path:** refused/Agree_or_refuse_to_act -> fight/Hostile_encounter -> lost/Earnings_and_losses
- **Relations:** TEMPORAL/BEFORE -> CAUSE/CAUSE
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The answer is directly stated in the context, specifically in sentence S7, making it easy to find.
- **Path reason:** The question can be answered solely by the final event sentence, which states the outcome for the Polish commander.

### [Easy -> Easy] doc_id=04a82d4eac37

- **Target difficulty:** Easy
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What event did Julio Cobos arrange for Raúl Alfonsín after his death?
- **Answer:** a state funeral at the Palace of
- **Event path:** died/Death -> funeral/Rite
- **Relations:** CAUSE/PRECONDITION
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question can be answered directly from sentence S3, which states that Julio Cobos arranged a state funeral for Alfonsín after his death.
- **Path reason:** The question can be answered by directly referencing the context sentence about the state funeral without needing to understand prior events.

### [Easy -> Easy] doc_id=dd2a791aa826

- **Target difficulty:** Easy
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What was used to gag her during the police raid?
- **Answer:** "Elastoplast" tape wrapped around her head
- **Event path:** raid/Attack -> wrapped/Filling
- **Relations:** TEMPORAL/CONTAINS
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question can be answered directly from S2, which states what was used to gag her.
- **Path reason:** The question can be answered solely by referring to the context sentence that describes the gagging, without needing to understand prior events.

### [Easy -> Easy] doc_id=dd2a791aa826

- **Target difficulty:** Easy
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What happened to the three police officers involved in the case in 1995?
- **Answer:** police officers involved stood trial for Gardner's
- **Event path:** placed/Placing -> involved/Cause_to_be_included
- **Relations:** TEMPORAL/BEFORE
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question can be answered directly from a single sentence about the trial of the police officers, making it easy and straightforward.
- **Path reason:** The question can be answered by directly referencing the sentence about the police officers without needing to understand prior events.

## PathQG-HardAware

### [Hard -> Medium] doc_id=3dcfd6015382

- **Target difficulty:** Hard
- **Predicted difficulty:** Medium | **Steps:** 2
- **Path dependency:** strong | **Prior needed:** 2
- **Question:** After the Dutch began their campaign to destroy Portuguese power and launched small incursions against the Portuguese, what did they agree to do in line with their agreement with Johor in 1606?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began/Process_start -> launching/Military_operation -> took/Conquering -> agreed/Agree_or_refuse_to_act
- **Relations:** TEMPORAL/BEFORE -> TEMPORAL/BEFORE -> CAUSE/PRECONDITION
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question requires connecting the Dutch campaign's initiation and their agreement with Johor, which involves two inference steps. The answer is not directly stated in a single sentence but can be inferred from the context.
- **Path reason:** The question specifically references the Dutch campaign and incursions, which are part of the prior events leading to the final agreement.

### [Medium -> Easy] doc_id=8af040b8bfa2

- **Target difficulty:** Medium
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What was the name given to the tropical cyclone that prompted the evacuation of 600,000 people in Bangladesh after it originated from an area of disturbed weather?
- **Answer:** also referred to as Tropical Storm
- **Event path:** Originating/Coming_to_be -> Tracking/Scrutiny -> storm/Catastrophe
- **Relations:** CAUSE/PRECONDITION -> TEMPORAL/BEFORE
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question can be answered directly from the information in the first sentence, which states the name of the cyclone and the evacuation it caused.
- **Path reason:** The question can be answered by directly referencing the context sentence about the cyclone without needing to understand prior events.

### [Medium -> Easy] doc_id=37d153abeafe

- **Target difficulty:** Medium
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What provided a morale-lifting consolation to the Chinese people following the Japanese invasion of Shanghai?
- **Answer:** people in the demoralizing aftermath of the Japanese invasion of Shanghai
- **Event path:** defense/Defending -> invasion/Attack -> aftermath/Catastrophe
- **Relations:** TEMPORAL/BEFORE -> CAUSE/CAUSE
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question can be answered directly from a single sentence that states the morale-lifting consolation provided to the Chinese people after the invasion.
- **Path reason:** The question can be answered solely by the context provided in the final event sentence.

### [Medium -> Easy] doc_id=9fcf7e509cc4

- **Target difficulty:** Medium
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What did these operations demonstrate about NATO's capabilities after expanding the mission and carrying out air strikes?
- **Answer:** These engagements helped show
- **Event path:** expanded/Expansion -> bombed/Attack -> helped/Assistance
- **Relations:** TEMPORAL/BEFORE -> TEMPORAL/BEFORE
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question can be answered directly from the context, specifically from the sentence that states 'These engagements helped show that NATO had adapted to the post-Cold War era.'
- **Path reason:** The question can be answered by the final event sentence alone, which states that the engagements helped show NATO's capabilities.

### [Medium -> Easy] doc_id=3dcfd6015382

- **Target difficulty:** Medium
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What was the result of the combined Dutch-Johor effort that followed the successful capture of Malacca from the Portuguese?
- **Answer:** removing their influence in the Malay archipelago
- **Event path:** capture/Conquering -> combined/Cause_to_amalgamate -> removing/Removing
- **Relations:** TEMPORAL/BEFORE -> CAUSE/CAUSE
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question directly asks for the result of the combined effort, which is explicitly stated in the context, requiring only one inference step to connect the effort to its outcome.
- **Path reason:** The question can be answered solely by the context provided in the final event sentence.

## SelfRefine

### [Medium -> Easy] doc_id=8af040b8bfa2

- **Target difficulty:** Medium
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What was the classification of the weather system that became Cyclone Forrest and prompted such a large evacuation?
- **Answer:** as Tropical Storm Forrest, was a powerful
- **Event path:** Originating/Coming_to_be -> Tracking/Scrutiny -> storm/Catastrophe
- **Relations:** CAUSE/PRECONDITION -> TEMPORAL/BEFORE
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question can be answered directly from the context in a single sentence, specifically from the information about Cyclone Forrest's classification.
- **Path reason:** The question can be answered by directly referencing the information in the context sentences without needing to understand the prior events.

### [Medium -> Easy] doc_id=275eb0bc9caa

- **Target difficulty:** Medium
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What was the reason Stone Cold Steve Austin needed to take time off after the event?
- **Answer:** time to recover from his injuries)
- **Event path:** delivered/Sending -> take/Creating -> recover/Recovering
- **Relations:** TEMPORAL/BEFORE -> TEMPORAL/CONTAINS
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question can be answered directly from the context, specifically from the sentence that states Austin needed time to recover from his injuries, which is clearly linked to the event.
- **Path reason:** The question can be answered solely based on the information provided in the context sentences without needing to reference prior events.

### [Medium -> Easy] doc_id=f83fc49c020e

- **Target difficulty:** Medium
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What offensive began on December 17, 2014, involving Kurdish forces and supported by US airstrikes?
- **Answer:** YPG forces started the December 2014 Sinjar
- **Event path:** reacted/Response -> airstrikes/Attack -> started/Process_start
- **Relations:** TEMPORAL/BEFORE -> TEMPORAL/BEFORE
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question directly asks for information that is explicitly stated in a single sentence, making it straightforward to answer.
- **Path reason:** The question can be answered solely by the context sentence that mentions the offensive without needing to reference prior events.

### [Easy -> Easy] doc_id=06f91ced00b4

- **Target difficulty:** Easy
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What did U.S. leaders do in response to Yamamoto's involvement in the attack on Pearl Harbor?
- **Answer:** leaders who blamed Yamamoto for the attack
- **Event path:** attack/Attack -> blamed/Judgment_communication
- **Relations:** CAUSE/PRECONDITION
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question can be answered directly from the information in the context, specifically from the sentence that states U.S. leaders blamed Yamamoto for the attack on Pearl Harbor.
- **Path reason:** The question can be answered by understanding the context provided in the final event without needing to reference prior events.

### [Easy -> Easy] doc_id=dd2a791aa826

- **Target difficulty:** Easy
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What type of tape was used to gag her during the police raid?
- **Answer:** "Elastoplast" tape wrapped around her head
- **Event path:** raid/Attack -> wrapped/Filling
- **Relations:** TEMPORAL/CONTAINS
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question can be answered directly from S2, which explicitly states the type of tape used.
- **Path reason:** The question can be answered solely by referring to the context sentence that describes the gagging incident.

## ZeroShot-TargetQG

### [Easy -> Medium] doc_id=f46091471f38

- **Target difficulty:** Easy
- **Predicted difficulty:** Medium | **Steps:** 2
- **Path dependency:** strong | **Prior needed:** 2
- **Question:** What was the name of the event that was part of the response to Operation Agatha and resulted in the deadliest attack on the British during the Mandate era?
- **Answer:** the operation was carried out
- **Event path:** Disguised/Wearing -> operation/Military_operation
- **Relations:** TEMPORAL/BEFORE
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question requires connecting the event of the operation being a response to Operation Agatha and identifying it as the deadliest attack, which involves synthesizing information from multiple sentences.
- **Path reason:** The question specifically references the operation that was a response to prior events, requiring knowledge of both the disguise and the military operation.

### [Medium -> Easy] doc_id=3dcfd6015382

- **Target difficulty:** Medium
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What action did the Dutch and their allies take to end Portuguese influence in the Malay archipelago?
- **Answer:** removing the last bastion of Portuguese influence in the Malay archipelago
- **Event path:** capture/Conquering -> combined/Cause_to_amalgamate -> removing/Removing
- **Relations:** TEMPORAL/BEFORE -> CAUSE/CAUSE
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question can be answered directly from the information in a single sentence, specifically from S7, which states the action taken by the Dutch and their allies.
- **Path reason:** The question can be answered by understanding the final event alone, as it directly asks about the action taken to end Portuguese influence.

### [Medium -> Easy] doc_id=04a82d4eac37

- **Target difficulty:** Medium
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What was the final action taken with Raúl Alfonsín's coffin after the military escort brought it to the La Recoleta Cemetery?
- **Answer:** Cemetery, and left him at the pantheon
- **Event path:** died/Death -> took/Bringing -> left/Placing
- **Relations:** CAUSE/PRECONDITION -> CAUSE/PRECONDITION
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question directly asks about the final action taken with the coffin, which is explicitly stated in the context, making it easy to answer.
- **Path reason:** The question can be answered solely by the final event sentence without needing context from prior events.

### [Easy -> Easy] doc_id=06f91ced00b4

- **Target difficulty:** Easy
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What action did U.S. leaders take against Yamamoto after the attack on Pearl Harbor?
- **Answer:** leaders who blamed Yamamoto for the attack
- **Event path:** attack/Attack -> blamed/Judgment_communication
- **Relations:** CAUSE/PRECONDITION
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question can be answered directly from the information in the context, specifically from the sentence that states U.S. leaders blamed Yamamoto for the attack on Pearl Harbor.
- **Path reason:** The question can be answered by understanding the final event alone, as it directly references the leaders' blame of Yamamoto.

### [Easy -> Easy] doc_id=28a13a10cb57

- **Target difficulty:** Easy
- **Predicted difficulty:** Easy | **Steps:** 1
- **Path dependency:** none | **Prior needed:** 0
- **Question:** What verb can be used to describe the process of the United States establishing a presence in the area that would become Minnesota during the 19th century?
- **Answer:** established a firm presence
- **Event path:** diminished/Cause_change_of_position_on_a_scale -> establish/Building
- **Relations:** TEMPORAL/BEFORE
- **Diff judge status:** ok
- **Path judge status:** ok
- **Diff reason:** The question directly asks for a verb related to the establishment of a presence, which is explicitly stated in S2.
- **Path reason:** The question can be answered by directly referencing the context sentence about the United States establishing a presence without needing prior events.
