# FairytaleQA Candidate Suitability Audit Report

Generated: 2026-05-13 16:26:48

## 1. Pool Counts Before/After by Difficulty

| Difficulty | Before | After | Removed | Retention |
|---|---:|---:|---:|---:|
| Easy | 659 | 646 | 13 | 98.0% |
| Medium | 1153 | 970 | 183 | 84.1% |
| Hard | 354 | 250 | 104 | 70.6% |

## 2. Rejection Reasons by Difficulty

### Easy

| Reason | Count |
|---|---:|
| ASA=partial (not yes) | 8 |
| ASA=no (not yes) | 5 |

### Medium

| Reason | Count |
|---|---:|
| no bridge sentences (Medium needs at least 1) | 115 |
| num_required_sentences=3 (>2) | 31 |
| num_required_sentences=5 (>2) | 12 |
| num_required_sentences=4 (>2) | 11 |
| num_required_sentences=6 (>2) | 5 |
| num_required_sentences=11 (>2) | 2 |
| num_required_sentences=18 (>2) | 2 |
| num_required_sentences=8 (>2) | 1 |
| num_required_sentences=12 (>2) | 1 |
| num_required_sentences=7 (>2) | 1 |
| num_required_sentences=17 (>2) | 1 |
| num_required_sentences=30 (>2) | 1 |

### Hard

| Reason | Count |
|---|---:|
| answer_type=emotion_label (scared .) | 11 |
| answer_type=emotion_label (sad .) | 10 |
| answer_type=emotion_label (happy .) | 7 |
| answer_type=emotion_label (confused .) | 5 |
| answer_type=emotion_label (upset .) | 5 |
| answer_type=emotion_label (grateful .) | 4 |
| necessity_type=answer_identification (weak) | 4 |
| answer_type=emotion_label (annoyed .) | 3 |
| answer_type=emotion_label (concerned .) | 3 |
| answer_type=emotion_label (excited .) | 3 |
| answer_type=emotion_label (worried .) | 3 |
| answer_type=short_label (three times .) | 2 |
| answer_type=emotion_label (unhappy .) | 2 |
| answer_type=short_label (poorly .) | 2 |
| answer_type=emotion_label (surprised .) | 2 |
| answer_type=emotion_label (afraid .) | 2 |
| answer_type=emotion_label (angry .) | 2 |
| answer_type=short_label (help her .) | 1 |
| answer_type=emotion_label (intrigued .) | 1 |
| answer_type=short_label (the mother .) | 1 |

## 3. Story Coverage Before/After

| Metric | Before | After |
|---|---:|---:|
| Unique stories | 228 | 227 |
| Story-matched eligible (>=1 per level) | 106 | 90 |

## 4. Easy Mismatch Audit

Easy candidates rejected: 13

| # | Story | Answer | ASA | num_req | Reject Reason |
|---|---|---|---|---|---|
| 1 | thomas-the-rhymer | thomas will become famous for his powers | no | 0 | ASA=no (not yes) |
| 2 | the-well-o-the-worlds-end | he was a wet , stick paddock . | no | 0 | ASA=no (not yes) |
| 3 | the-magic-bundle | marry her . | no | 0 | ASA=no (not yes) |
| 4 | the-little-spirit-or-boy-man | the boy stole their trout . | no | 0 | ASA=no (not yes) |
| 5 | the-adventures-of-gilla-na-chreck-a | tom defeated his men . | no | 0 | ASA=no (not yes) |
| 6 | the-boyhood-of-cuchulain | he wanted to join the military academy . | partial | 0 | ASA=partial (not yes) |
| 7 | the-boyhood-of-cuchulain | setanta will look for fergus mac roy . | partial | 0 | ASA=partial (not yes) |
| 8 | the-boyhood-of-cuchulain | the young nobles ' sport was interrupted | partial | 0 | ASA=partial (not yes) |
| 9 | the-boyhood-of-cuchulain | annoyed . | partial | 0 | ASA=partial (not yes) |
| 10 | the-boyhood-of-cuchulain | scared . | partial | 0 | ASA=partial (not yes) |
| 11 | the-boyhood-of-cuchulain | setanta had courage . | partial | 0 | ASA=partial (not yes) |
| 12 | the-boyhood-of-cuchulain | he admired the red branch . | partial | 0 | ASA=partial (not yes) |
| 13 | the-boyhood-of-cuchulain | setanta did not ring the gong before app | partial | 0 | ASA=partial (not yes) |

## 5. Hard Mismatch Audit

Hard candidates rejected: 104

### 5a. Hard: Short Emotion/State Answers

Count: 69

| # | Story | Answer | Necessity Type |
|---|---|---|---|
| 1 | youth-who-wanted-to-win-daughter-of | unhappy . | motivation_bridge |
| 2 | lame-dog | sad . | summary_synthesis |
| 3 | how-princess-pride-was-broken | annoyed . | motivation_bridge |
| 4 | master-girl | intrigued . | motivation_bridge |
| 5 | master-girl | confused . | summary_synthesis |
| 6 | thomas-the-rhymer | scared . | disambiguation |
| 7 | little-lasse | confused . | motivation_bridge |
| 8 | habetrot-the-spinstress | scared . | motivation_bridge |
| 9 | the-one-handed-girl | dissatisfied . | motivation_bridge |
| 10 | the-one-handed-girl | concerned . | summary_synthesis |
| 11 | the-one-handed-girl | sad . | motivation_bridge |
| 12 | the-one-handed-girl | surprised . | motivation_bridge |
| 13 | the-black-bull-of-norroway | upset . | summary_synthesis |
| 14 | the-black-bull-of-norroway | excited . | summary_synthesis |
| 15 | wunzh-the-father-of-indian-corn | happy . | motivation_bridge |
| 16 | kings-hares | upset . | motivation_bridge |
| 17 | the-three-crowns | furious . | motivation_bridge |
| 18 | sheem-the-forsaken-boy | surprised . | motivation_bridge |
| 19 | the-well-o-the-worlds-end | scared . | disambiguation |
| 20 | weendigoes-and-the-bone-dwarf | afraid . | disambiguation |

### 5b. Hard: All Rejection Reasons

| Reason | Count |
|---|---:|
| answer_type=emotion_label (scared .) | 11 |
| answer_type=emotion_label (sad .) | 10 |
| answer_type=emotion_label (happy .) | 7 |
| answer_type=emotion_label (confused .) | 5 |
| answer_type=emotion_label (upset .) | 5 |
| answer_type=emotion_label (grateful .) | 4 |
| necessity_type=answer_identification (weak) | 4 |
| answer_type=emotion_label (annoyed .) | 3 |
| answer_type=emotion_label (concerned .) | 3 |
| answer_type=emotion_label (excited .) | 3 |
| answer_type=emotion_label (worried .) | 3 |
| answer_type=short_label (three times .) | 2 |
| answer_type=emotion_label (unhappy .) | 2 |
| answer_type=short_label (poorly .) | 2 |
| answer_type=emotion_label (surprised .) | 2 |
| answer_type=emotion_label (afraid .) | 2 |
| answer_type=emotion_label (angry .) | 2 |
| answer_type=short_label (help her .) | 1 |
| answer_type=emotion_label (intrigued .) | 1 |
| answer_type=short_label (the mother .) | 1 |

### 5c. Hard: ASA != no (from original audit)

Count: 0

## 6. Final Story-Matched Suitable Pool

| Suitable stories (>=1 per level) | 90 |
| Selected (1 per level per story) | 270 (90E/90M/90H) |
| Selection target met (>=70) | YES |

## 7. Selected Examples (10 per level)

### Easy Examples

| # | Story | Answer | Answer Type | ASA | num_req | NT |
|---|---|---|---|---|---|---|
| 1 | Snow-man | a young girl and a young man . | explanatory | yes | 1 | background_context |
| 2 | a-legend-of-knockmany | he heard three whistles . | short_phrase | yes | 1 | background_context |
| 3 | a-lost-paradise | they wanted plenty of food to eat . | explanatory | yes | 1 | disambiguation |
| 4 | bokwewa-the-humpback | sad . | emotion_label | yes | 1 | disambiguation |
| 5 | brave-tin-soldier | they ran out of melted tin . | explanatory | yes | 1 | background_context |
| 6 | brother-sister | their stepmother was mean to them . | explanatory | yes | 1 | answer_identification |
| 7 | canonbie-dick-and-thomas-of-ercildo | the coins were not the gold that was use | explanatory | yes | 1 | background_context |
| 8 | comrade | the youth would have company . | short_phrase | yes | 1 | causal_bridge |
| 9 | cuchulain-of-muirthemne | it was night . | short_phrase | yes | 1 | background_context |
| 10 | dschang-liang | wanted dschang liang to fetch one of his | explanatory | yes | 1 | background_context |

### Medium Examples

| # | Story | Answer | Answer Type | ASA | num_req | NT |
|---|---|---|---|---|---|---|
| 1 | Snow-man | the yard - dog had been there longer tha | explanatory | no | 2 | causal_bridge |
| 2 | a-legend-of-knockmany | he wanted to give finn a considerable be | explanatory | no | 2 | causal_bridge |
| 3 | a-lost-paradise | they could not find work . | short_phrase | no | 2 | causal_bridge |
| 4 | bokwewa-the-humpback | kwasynd did not save her . | short_phrase | no | 2 | causal_bridge |
| 5 | brave-tin-soldier | the fish that swallowed him was caught a | explanatory | no | 2 | causal_bridge |
| 6 | brother-sister | to hide her lost eye . | short_phrase | no | 2 | motivation_bridge |
| 7 | canonbie-dick-and-thomas-of-ercildo | surprised . | emotion_label | no | 2 | motivation_bridge |
| 8 | comrade | help the youth get them back . | explanatory | no | 2 | causal_bridge |
| 9 | cuchulain-of-muirthemne | he wanted to get money out of conchubar  | explanatory | no | 2 | motivation_bridge |
| 10 | dschang-liang | his feigning illness loosed his soul fro | explanatory | no | 2 | causal_bridge |

### Hard Examples

| # | Story | Answer | Answer Type | ASA | num_req | NT |
|---|---|---|---|---|---|---|
| 1 | Snow-man | he had to . | short_phrase | no | 5 | causal_bridge |
| 2 | a-legend-of-knockmany | far rua 's teeth will fall out . | explanatory | no | 3 | summary_synthesis |
| 3 | a-lost-paradise | the wife will be curious what is inside  | explanatory | no | 3 | motivation_bridge |
| 4 | bokwewa-the-humpback | wanted kwasynd to succeed . | short_phrase | no | 6 | motivation_bridge |
| 5 | brave-tin-soldier | she fluttered into the stove and burned  | explanatory | no | 3 | summary_synthesis |
| 6 | brother-sister | because they would turn him into an anim | explanatory | no | 3 | summary_synthesis |
| 7 | canonbie-dick-and-thomas-of-ercildo | the stranger wanted to reveal his identi | explanatory | no | 3 | causal_bridge |
| 8 | comrade | he was not paying attention . | short_phrase | no | 3 | motivation_bridge |
| 9 | cuchulain-of-muirthemne | dechtire turned into a bird . | short_phrase | no | 4 | causal_bridge |
| 10 | dschang-liang | one of the traveling coaches was used as | explanatory | no | 3 | causal_bridge |
