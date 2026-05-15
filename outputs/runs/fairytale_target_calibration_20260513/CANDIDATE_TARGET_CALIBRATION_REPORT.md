# FairytaleQA Target-Label Calibration Report

Generated: 2026-05-14 00:21:25

Total candidates audited: 2166
Judge errors: 0/2166 (0.0%)

## 1. Agreement Matrix: evidence_difficulty x judge_predicted_difficulty

| Evidence \ Judge | Easy | Hard | Medium | Total |
|---|---:|---:|---:|---:|
| Easy | 250 | 23 | 386 | 659 |
| Medium | 140 | 77 | 936 | 1153 |
| Hard | 23 | 60 | 271 | 354 |

## 2. Calibrated Pool Size by Difficulty

| Difficulty | Calibrated | Total | Pct |
|---|---:|---:|---:|
| Easy | 35 | 659 | 5.3% |
| Medium | 936 | 1153 | 81.2% |
| Hard | 49 | 354 | 13.8% |
| **Total** | **1020** | **2166** | **47.1%** |

## 3. Rejection Reasons by Difficulty

### Easy

| Reason | Count |
|---|---:|
| judge_predicted=Medium (expected Easy) | 386 |
| answer=coreference_risk | 104 |
| answer=short_emotion (not self-contained for Easy) | 60 |
| answer=fragment | 35 |
| judge_predicted=Hard (expected Easy) | 23 |
| asa=no (expected yes) | 12 |
| asa=partial (expected yes) | 4 |

### Medium

| Reason | Count |
|---|---:|
| judge_predicted=Easy (expected Medium) | 140 |
| judge_predicted=Hard (expected Medium) | 77 |

### Hard

| Reason | Count |
|---|---:|
| judge_predicted=Medium (expected Hard) | 271 |
| judge_predicted=Easy (expected Hard) | 23 |
| answer=short_emotion (inappropriate for Hard) | 9 |
| asa=partial (expected no) | 2 |

## 4. Easy Mismatch Examples (evidence=Easy, not calibrated)

| # | Story | Question | Answer | Judge Pred | ASA | Reason |
|---|---|---|---|---|---|---|
| 1 | three-dogs | how will the king feel when he learns what happened ? | angry . | Easy | yes | answer=short_emotion (not self-contained for Easy) |
| 2 | three-dogs | how did the two princesses feel when they were reunited ? | glad . | Medium | partial | judge_predicted=Medium (expected Easy) |
| 3 | how-boots-befooled-king | what happened when boots yelled out "break pots, break pots" | the old woman started breaking | Medium | no | judge_predicted=Medium (expected Easy) |
| 4 | youth-who-wanted-to-win-d | what happened to the son's clothes when he skipped and dance | ripped . | Medium | partial | judge_predicted=Medium (expected Easy) |
| 5 | youth-who-wanted-to-win-d | how will the youth feel about the food provided by the rat ? | unsure . | Medium | partial | judge_predicted=Medium (expected Easy) |
| 6 | youth-who-wanted-to-win-d | how did the youth feel when the rat allowed him to go above  | excited . | Medium | partial | judge_predicted=Medium (expected Easy) |
| 7 | youth-who-wanted-to-win-d | why was the youth in no hurry to go down the hole again ? | he did not like being down the | Medium | partial | judge_predicted=Medium (expected Easy) |
| 8 | youth-who-wanted-to-win-d | what will the youth find when he returns home ? | piles of wool . | Medium | no | judge_predicted=Medium (expected Easy) |
| 9 | youth-who-wanted-to-win-d | what will the youth and his mother do with the wool ? | make clothes . | Medium | no | judge_predicted=Medium (expected Easy) |
| 10 | youth-who-wanted-to-win-d | what will happen now that the youth has returned again ? | the wedding . | Medium | partial | judge_predicted=Medium (expected Easy) |
| 11 | youth-who-wanted-to-win-d | how did the youth feel when he saw the beautiful castle ? | shocked . | Medium | no | judge_predicted=Medium (expected Easy) |
| 12 | rooster-handmill-swarm-of | how will the peasant feel when he discovers his rooster did  | angry . | Hard | no | judge_predicted=Hard (expected Easy) |
| 13 | rooster-handmill-swarm-of | why did the old woman steal his mill ? | she wanted the mill for hersel | Easy | yes | answer=coreference_risk |
| 14 | werewolf | how did the king feel when the queen fell ill and died ? | sad . | Medium | no | judge_predicted=Medium (expected Easy) |
| 15 | werewolf | why did the king's daughter believe the woman ? | she was a child . | Medium | partial | judge_predicted=Medium (expected Easy) |

## 5. Hard Mismatch Examples (evidence=Hard, not calibrated)

| # | Story | Question | Answer | Judge Pred | ASA | Reason |
|---|---|---|---|---|---|---|
| 1 | three-dogs | why did the two princes seize the youth by the throat and st | they were jealous of the youth | Medium | no | judge_predicted=Medium (expected Hard) |
| 2 | youth-who-wanted-to-win-d | how did the woman feel about her son singing and dancing ? | unhappy . | Medium | no | judge_predicted=Medium (expected Hard) |
| 3 | youth-who-wanted-to-win-d | why did the mother think the youth's idea was not a bad idea | she could not afford to care f | Medium | no | judge_predicted=Medium (expected Hard) |
| 4 | werewolf | what will the tiny old man do for the king's daughter ? | help her . | Medium | no | judge_predicted=Medium (expected Hard) |
| 5 | werewolf | how did the queen treat the princess after the king left ? | poorly . | Medium | partial | judge_predicted=Medium (expected Hard) |
| 6 | swan-maiden | why did the prince climb the swan maiden's hair ? | to get to the top of the fir - | Medium | no | judge_predicted=Medium (expected Hard) |
| 7 | lame-dog | why did the two princesses laugh and joke about the youngest | she did not care who she marri | Medium | no | judge_predicted=Medium (expected Hard) |
| 8 | lame-dog | who will the king's daughters marry ? | they will marry someone exactl | Medium | partial | judge_predicted=Medium (expected Hard) |
| 9 | lame-dog | how did the youngest daughter feel about her marriage ? | sad . | Medium | partial | judge_predicted=Medium (expected Hard) |
| 10 | how-princess-pride-was-br | how did the princess feel about giving the gooseherd twenty  | annoyed . | Hard | partial | asa=partial (expected no) |
| 11 | master-girl | what was different about the second and third kettle ? | the second kettle turned his h | Medium | no | judge_predicted=Medium (expected Hard) |
| 12 | master-girl | how did the king's son feel about the kettles ? | intrigued . | Medium | no | judge_predicted=Medium (expected Hard) |
| 13 | master-girl | why did the king's son pretend to not know of the master gir | he did not want the giant to k | Medium | no | judge_predicted=Medium (expected Hard) |
| 14 | master-girl | why did the master girl not slaughter the king's son like th | she did not want to kill him . | Medium | partial | judge_predicted=Medium (expected Hard) |
| 15 | master-girl | how did the giant feel when he woke up and could not find th | confused . | Hard | no | answer=short_emotion (inappropriate for Hard) |

## 6. Story-Matched Calibrated Pool

Stories with calibrated Easy+Medium+Hard: **4**

Target: >= 70 stories
Status: **FAIL (not enough stories)**

## 7. Bottleneck Analysis

| Difficulty | Total Candidates | Calibrated | Missing per story (avg) |
|---|---:|---:|---:|
| Easy | 659 | 35 | 2.7 |
| Medium | 1153 | 936 | 1.0 |
| Hard | 354 | 49 | 1.3 |

### Top Rejection Reasons (all difficulties)

| Reason | Count |
|---|---:|
| judge_predicted=Medium (expected Easy) | 386 |
| judge_predicted=Medium (expected Hard) | 271 |
| judge_predicted=Easy (expected Medium) | 140 |
| answer=coreference_risk | 104 |
| judge_predicted=Hard (expected Medium) | 77 |
| answer=short_emotion (not self-contained for Easy) | 60 |
| answer=fragment | 35 |
| judge_predicted=Hard (expected Easy) | 23 |
| judge_predicted=Easy (expected Hard) | 23 |
| asa=no (expected yes) | 12 |
