# DCQG 椤圭洰鐘舵€佷笌瀹為獙鍙拌处

**鏈€鍚庢洿鏂?*锛?026-05-01
**鐘舵€?*锛氭柟娉曟鏋跺凡鎴愬瀷锛屼富瀹為獙宸插湪鍥哄畾 300 鏉℃牱鏈笂瀹屾垚涓€杞粺涓€璇勪及銆傚凡淇鍥涗釜鏁版嵁閾捐矾闂锛屽凡娓呯悊鏃т唬鐮侊紙鍒犻櫎 8 涓枃浠讹級锛岄毦搴﹁瘎鍒嗘敼涓?hop-based锛屽叏閾捐矾 debug trace 宸查泦鎴愩€? 
**缁存姢瑙勫垯**锛氭瘡娆′慨鏀圭敓鎴愩€佽繃婊ゃ€佽瘎浼般€侀噰鏍锋垨 baseline 鍚庯紝蹇呴』鍚屾鏇存柊鏈枃浠朵腑鐨勨€滄渶鏂板疄楠屾暟鎹€濃€滃凡瑙ｅ喅闂鈥濃€滃緟瑙ｅ喅闂鈥濃€滄寚鏍囧畾涔変笌渚濇嵁鈥濆洓閮ㄥ垎銆傚悗缁畾浣嶉棶棰樺繀椤讳紭鍏堟煡鐪?trace log锛屼笉鍏佽鍙牴鎹眹鎬绘暟瀛楃寽娴嬪師鍥犮€?
---

## 1. 褰撳墠棰勮娴佺▼

鏈」鐩綋鍓嶆渶鍚堢悊鐨勬暣浣撴祦绋嬫槸锛?
```text
MAVEN-ERE valid documents
        |
        v
Document-level event graph construction
        |
        v
Rule-based path sampling
  - Easy: 1-hop event path
  - Medium: 2-hop event path
  - Hard: 3-hop event path
        |
        v
Stage 1 rule prefilter
  - weak final trigger filtering
  - answer phrase extraction / validation
  - relation composition check
  - support span / single-sentence risk check
        |
        v
Target-event-grounded question generation
  - context
  - target final event / answer phrase
  - event path
  - relation sequence
  - difficulty-specific prompt
        |
        v
Automatic quality filtering
  - grammar
  - asks target event
  - answer consistency
  - path coverage
  - hard degradation
        |
        v
Solver evaluation + LLM judge
        |
        v
Manual calibration sample
```

鍏抽敭瀹氫綅锛?
- 鎴戜滑涓嶆槸绾嚜鐢辩敓鎴愰棶棰橈紝鑰屾槸 **target-event-grounded QG**銆?- `trigger` 涓嶈兘鍐嶈褰撴垚鍞竴 gold answer锛涘畠鏇撮€傚悎浣滀负 final event 鐨勫畾浣嶉敋鐐广€?- 璁烘枃涓荤嚎寤鸿鍐欐垚锛?*浜嬩欢璺緞绾︽潫濡備綍甯姪鐢熸垚绗﹀悎鐩爣鎺ㄧ悊闅惧害鐨勯棶棰?*銆?- 闅惧害鏍囩鐢辫矾寰勮烦鏁板喅瀹氾紙Easy=1hop, Medium=2hop, Hard=3hop锛夛紝涓嶅啀浣跨敤鍥涚淮璇勫垎銆傛棫鐨?`difficulty_scorer.py` 宸插垹闄ゃ€?
---

## 2. 鏁版嵁涓庤矾寰勯噰鏍风姸鎬?
### 2.1 Event graph construction

鏉ユ簮鏂囦欢锛歚event_qg/outputs/graph_building_report.json`

| Item | Value |
|---|---:|
| Split | valid |
| Documents | 50 |
| Events | 1307 |
| Edges | 12064 |

Relation distribution:

| Relation subtype | Count |
|---|---:|
| TEMPORAL/BEFORE | 10379 |
| CAUSE/PRECONDITION | 664 |
| TEMPORAL/CONTAINS | 629 |
| CAUSE/CAUSE | 146 |
| SUBEVENT | 139 |
| TEMPORAL/SIMULTANEOUS | 54 |
| TEMPORAL/OVERLAP | 53 |

缁撹锛氬浘涓?temporal relation 鍗犳瘮鏋侀珮锛岃繖鏄?Hard 闂瀹规槗閫€鍖栦负鏃堕棿椤哄簭妫€绱㈢殑鏍规湰鍘熷洜涔嬩竴銆?
### 2.2 Path sampling

鏉ユ簮鏂囦欢锛歚event_qg/outputs/path_sampling_report.json`

| Difficulty | Count |
|---|---:|
| Easy | 1156 |
| Medium | 1458 |
| Hard | 1276 |
| Total | 3890 |

Relation group distribution:

| Relation group | Count |
|---|---:|
| TEMPORAL | 2600 |
| MIXED | 994 |
| CAUSE | 291 |
| SUBEVENT | 5 |

褰撳墠璺緞閲囨牱鏂规硶锛?
- Easy锛氭灇涓?1-hop directed edge銆?- Medium锛氬師鐩爣鏄灇涓?2-hop directed path锛屼絾褰撳墠瀹炵幇瀛樺湪鏂瑰悜鏋勯€犻闄╋紝闇€瑕佷慨澶嶄负涓ユ牸 `src -> mid -> tgt`銆?- Hard锛氫粠璧风偣 BFS 鎼滅储 3-hop directed path锛屽苟鍋氫竴瀹?hard 鏉′欢杩囨护銆?- 璺緞鑼冨洿锛氬崟鏂囨。鍐呴儴閲囨牱锛屼笉璺ㄦ枃妗ｃ€?
宸茶В鍐抽棶棰橈紙2026-05-01 淇锛夛細

- `path_sampler.py` 鐜板湪淇濈暀 MAVEN 鍘熷 `offset`锛宎nswer phrase 浣跨敤 clause-aware 鎶藉彇銆?- `path_sampler.py` Medium 璺緞宸蹭慨澶嶄负涓ユ牸 `src -> mid -> tgt` 鏈夊悜璺緞銆?- `difficulty_scorer.py` 宸插垹闄わ紝闅惧害鏍囩鏀逛负绾?hop-based锛圗asy=1hop, Medium=2hop, Hard=3hop锛夈€?- 鏃т唬鐮侊紙`compare.py`, `baselines.py`, `inspect_data.py`, `run_stage1.py`, `stage2_prototype.py`, `scripts/`锛夊凡娓呯悊銆?
### 2.3 Rule prefilter

鏉ユ簮鏂囦欢锛歚event_qg/outputs/path_prefilter_report.json`

| Difficulty | Total | Passed | Pass rate |
|---|---:|---:|---:|
| Easy | 1156 | 1064 | 92.04% |
| Medium | 1458 | 1334 | 91.50% |
| Hard | 1276 | 1191 | 93.34% |
| Overall | 3890 | 3589 | 92.26% |

閲嶈瑙傚療锛?
- Weak trigger distribution: none 3514, hard_blacklist 153, needs_phrase 223銆?- Temporal-only Hard: 434 / 1276 = 34.01%銆?- Single-sentence risk: high 324, medium 137, low 3429銆?- `answer_phrase_pass_rate = 1.0` 杩囦簬鐞嗘兂锛屽凡缁忕‘璁ゅ師鍥犱箣涓€鏄?fixed-window extractor 鍋忓锛屼細鎶婃埅鏂煭璇鍒や负 valid phrase銆?
### 2.4 Failure diagnosis: answer phrase and path direction

鏈疆浜哄伐杩借釜浜嗕粠 MAVEN 鍘熷鏁版嵁鍒?gpt-4o path judge 鐨勯摼璺紝瀹氫綅濡備笅锛?
| Stage | Status | Evidence |
|---|---|---|
| MAVEN raw document | 鍩烘湰姝ｅ父 | 鍘熷 event mention 鏈?`trigger_word`銆乣sent_id`銆乣offset`锛屽師鍙ュ畬鏁?|
| `graph_builder.py` | 鍩烘湰姝ｅ父 | `get_event_info()` 宸茶鍙?`offset` |
| `path_sampler.py` | 鏈夐棶棰?| 杈撳嚭 `events` 鏃朵涪鎺?`offset`锛汳edium path 鏂瑰悜鏋勯€犲瓨鍦ㄩ闄?|
| `answer_extraction.enrich_path_item()` | 已修复并独立成模块 | `extract_answer_phrase_local()` 使用 clause-aware expansion，并在 trace 中记录 `answer_phrase_status` |
| `path_prefilter.py` | 缁ф壙涓婅堪闂 | 璋冪敤 `enrich_path_item()` 鍚庢妸鎴柇 phrase 鏍囨垚 pass |
| `path_llm_judge.py` / gpt-4o | 鍒ゆ柇鍩烘湰鍚堢悊 | gpt-4o 姝ｇ‘鎸囧嚭 answer phrase 鎴柇鍜?single-sentence shortcut |

鍏稿瀷鎴柇渚嬪瓙锛?
| Trigger | Sentence | Current bad phrase | Better phrase |
|---|---|---|---|
| described | It has been described as one of the most horrific frontier massacres of the war. | has been described as one of the | described as one of the most horrific frontier massacres of the war |
| rise | ... leading to the rise of the ghazis and the conclusive Byzantine-Ottoman wars. | to the rise of the ghazis and | the rise of the ghazis |
| claimed | The U.S. pilots claimed to have shot down three twin-engined bombers and two fighters... | U.S. pilots claimed to have shot down | claimed to have shot down three twin-engined bombers and two fighters |
| drove | ... Expedition which drove the Iroquois out of western New York. | Expedition which drove the Iroquois out of | drove the Iroquois out of western New York |

闃舵鎬х粨璁猴細

- 涔嬪墠 weak results 涓嶈兘鐩存帴璇存槑 PathQG 鏂规硶鏃犳晥锛涗竴閮ㄥ垎澶辫触鏉ヨ嚜涓婃父 target-answer extraction 鍜?path direction 鍣０銆?- 鍚庣画蹇呴』鍏堜慨 `answer phrase extraction + path direction`锛屽啀閲嶆柊璺?path judge 鍜?QG銆?- 浠讳綍鏂扮粨璁洪兘蹇呴』鑳介€氳繃 full trace 杩芥函鍒?MAVEN 鍘熷彞銆乪vent offset銆乸ath direction銆乤nswer extraction 鍜?judge raw response銆?
---

## 3. 涓诲疄楠岃缃?
鍥哄畾鏍锋湰锛?
| Setting | Value |
|---|---|
| Sample file | `event_qg/outputs/sample_300_seed42.jsonl` |
| Total items | 300 |
| Difficulty distribution | Easy 100 / Medium 100 / Hard 100 |
| Unique documents | 49 |
| Generator | Qwen2.5-7B |
| Solver / Judge | Qwen2.5-32B |
| Known limitation | generator 鍜?judge/solver 灞炰簬鍚屼竴妯″瀷瀹舵棌 |

缁熶竴璇勪及锛?
1. Generate question銆?2. Grammar / basic format filter銆?3. LLM solver 鍥炵瓟闂銆?4. LLM judge 鍒ゆ柇 answerable銆乻olver_correct銆乻upport_covered銆?5. Quality judge 鍒ゆ柇 fluency銆乺elevance銆乨ifficulty alignment銆?
娉ㄦ剰锛氬綋鍓?composite 鏄唴閮ㄥ姞鏉冨垎鏁帮紝涓嶅缓璁綔涓鸿鏂囦富鎸囨爣銆?
```text
Composite = 0.25 * solver_correct
          + 0.20 * answerable
          + 0.15 * support_covered
          + 0.15 * fluency
          + 0.10 * relevance
          + 0.15 * diff_align
```

杩欎釜鏉冮噸涓嶆槸宸叉湁璁烘枃鏍囧噯鍏紡锛屽彧鑳戒綔涓鸿緟鍔╄瘖鏂垨鍐呴儴鎺掑簭鎸囨爣銆?
---

## 4. Baseline 璁捐

### 4.1 Main baselines

| Method | Input | 鏄惁鐪?target answer | 鏄惁鐪?event path | 鐩殑 |
|---|---|---:|---:|---|
| ZeroShotTargetQG | context + target answer + difficulty | Yes | No | 鐩爣浜嬩欢 QG 鐨勯浂鏍锋湰鍩虹嚎 |
| ICLTargetQG | context + target answer + difficulty + examples | Yes | No | CrossQG-style few-shot / ICL baseline |
| SelfRefine | ZeroShotTargetQG + critique + revise | Yes | No | 娴嬭瘯鑷垜淇鏄惁鎻愬崌璐ㄩ噺 |
| PathQG-HardAware | context + target event + event path + relation sequence + difficulty prompt | Yes | Yes | 鎴戜滑鐨勬柟娉?|

杩欓噷鐨?ICL 鏄皯鏍锋湰 in-context learning锛氱粰 2 涓悓闅惧害绀轰緥锛岃妯″瀷鎸夌ず渚嬫牸寮忕敓鎴愩€?
### 4.2 Ablations

| Method | Removed component | 鐩殑 |
|---|---|---|
| PathOnlyQG | context | 楠岃瘉娌℃湁涓婁笅鏂囨椂锛屼粎闈犺矾寰勬槸鍚﹁冻澶?|
| RelationTypeQG | specific event path | 楠岃瘉鍙粰鍏崇郴绫诲瀷銆佷笉鎸囧畾鍏蜂綋璺緞鏄惁瓒冲 |

杩欎袱涓洿閫傚悎鍐欏湪 ablation锛屼笉閫傚悎浣滀负澶栭儴涓?baseline銆?
---

## 5. 鏈€鏂颁富瀹為獙缁撴灉

鏉ユ簮鏂囦欢锛歚review-stage/RESULTS_SUMMARY.md`

| Method | N gen | N pass | Pass% | Answerable | Solver Correct | Composite |
|---|---:|---:|---:|---:|---:|---:|
| PathQG-HardAware | 300 | 186 | 62.0% | 0.747 | 0.274 | 0.693 |
| ZeroShotTargetQG | 300 | 127 | 42.3% | 0.921 | 0.325 | 0.728 |
| ICLTargetQG | 300 | 135 | 45.0% | 0.874 | 0.363 | 0.737 |
| SelfRefine | 300 | 143 | 47.7% | 0.923 | 0.308 | 0.729 |

瑙ｉ噴锛?
- PathQG-HardAware 褰撳墠鏈€楂樼殑鏄?pass rate锛岃€屼笉鏄?solver correctness銆?- ICLTargetQG 褰撳墠 solver_correct 鍜?composite 鏈€楂橈紝璇存槑寮?prompt + few-shot 瀵?target-aware QG 闈炲父鏈夋晥銆?- 鍥犳璁烘枃涓嶈兘鍐欌€滄垜浠叏闈紭浜?ICL baseline鈥濓紱鏇村悎鐞嗙殑璇存硶鏄€滀簨浠惰矾寰勭害鏉熸彁鍗囩敓鎴愰€氳繃鐜囥€佽矾寰勭粦瀹氬拰闅惧害鎺у埗锛屼絾閫氱敤闂璐ㄩ噺浠嶉渶鏀硅繘鈥濄€?
### 5.1 Solver correct by difficulty

| Method | Easy | Medium | Hard |
|---|---:|---:|---:|
| PathQG-HardAware | 0.260 | 0.371 | 0.203 |
| ZeroShotTargetQG | 0.375 | 0.375 | 0.211 |
| ICLTargetQG | 0.434 | 0.326 | 0.308 |
| SelfRefine | 0.277 | 0.380 | 0.261 |

褰撳墠闂锛?
- PathQG-HardAware 鍑虹幇 Medium > Easy锛岃繚鍙?Easy >= Medium >= Hard 鐨勭悊鎯抽毦搴﹁秼鍔裤€?- 杩欒鏄庡綋鍓嶉毦搴︽帶鍒惰繕娌℃湁瀹屽叏绋冲畾锛屼笉鑳界洿鎺ュ０绉扳€滀弗鏍煎崟璋冩帶鍒堕毦搴︹€濄€?
### 5.2 Fair metrics

| Method | Pass% | Conditional SolCor | Macro-Avg SolCor | E2E SolCor |
|---|---:|---:|---:|---:|
| PathQG-HardAware | 62.0% | 0.274 | 0.278 | 0.170 |
| ZeroShotTargetQG | 42.3% | 0.325 | 0.320 | 0.137 |
| ICLTargetQG | 45.0% | 0.363 | 0.356 | 0.163 |
| SelfRefine | 47.7% | 0.308 | 0.306 | 0.147 |

鍏朵腑锛?
```text
Conditional metric = mean(metric over passed/scored questions)
Macro metric       = (metric_Easy + metric_Medium + metric_Hard) / 3
E2E metric         = sum(metric over passed/scored questions) / 300
```

E2E 鎸囨爣瀵圭敓鎴愬け璐ユ牱鏈 0锛屽洜姝よ兘鍙嶆槧瀹屾暣 pipeline 鐨勪骇鍑鸿兘鍔涖€?
### 5.3 Ablation results

| Method | Component removed | N pass | Pass% | Answerable | Solver Correct | Composite |
|---|---|---:|---:|---:|---:|---:|
| PathQG-HardAware | none | 186 | 62.0% | 0.747 | 0.274 | 0.693 |
| RelationTypeQG | specific path | 134 | 44.7% | 0.791 | 0.179 | 0.652 |
| PathOnlyQG | context | 158 | 52.7% | 0.285 | 0.114 | 0.534 |

鍙敮鎾戠殑缁撹锛?
- 鍘绘帀 context 鍚?answerability 澶у箙涓嬮檷锛岃鏄庝笂涓嬫枃鏄彲鍥炵瓟鎬х殑蹇呰鏉′欢銆?- 鍘绘帀 specific event path 鍚?solver_correct 涓嬮檷锛岃鏄庡叿浣撹矾寰勭害鏉熸湁璐＄尞銆?- 浣?ablation 涓嶈兘璇佹槑褰撳墠闅惧害鎺у埗宸茬粡瀹屽叏鎴愬姛銆?
---

## 6. 鏈€鏂拌川閲?pilot 缁撴灉

鏉ユ簮鏂囦欢锛歚event_qg/outputs/quality_pilot_90_v3/filter_report.json`

| Metric | Value |
|---|---:|
| Total | 90 |
| Passed | 35 |
| Overall pass rate | 38.89% |
| Generation error | 4.44% |
| Grammar fail | 12.2% |
| asks_target_event | 91.38% |
| Answer consistency yes | 72.46% |
| Answer consistency yes + partial | 76.81% |
| Judge error | 26.58% |

Per-level pass rate:

| Difficulty | Passed / Total | Pass rate |
|---|---:|---:|
| Easy | 19 / 30 | 63.33% |
| Medium | 8 / 30 | 26.67% |
| Hard | 8 / 30 | 26.67% |

Path coverage:

| Difficulty | Avg coverage | Pass rate |
|---|---:|---:|
| Easy | 0.87 | 73.33% |
| Medium | 1.27 | 30.00% |
| Hard | 1.50 | 56.67% |

Hard degradation:

| Metric | Value |
|---|---:|
| Hard degraded count | 10 |
| Hard degraded ratio | 33.33% |

浜哄伐鎶芥煡 judge_error 鐨?20 鏉★細

| Manual label | Count |
|---|---:|
| yes | 12 |
| partial | 4 |
| no | 4 |

缁撹锛歫udge_error 涓湁澶ч噺鍏跺疄鍙敤鐨勯棶棰橈紝褰撳墠 judge JSON 瑙ｆ瀽/鎻愮ず涓嶇ǔ锛屾槸杩戞湡鏈€鍊煎緱淇殑闂涔嬩竴銆?
---

## 7. 宸茶В鍐崇殑闂

| Problem | Current status |
|---|---|
| baseline 涓?ours 鏍锋湰涓嶇粺涓€ | 宸茬粺涓€鍒?`sample_300_seed42.jsonl` |
| PathOnlyQG / RelationTypeQG 琚褰撲富 baseline | 宸茶皟鏁翠负 ablation |
| 缂哄皯 ICL baseline | 宸插姞鍏?ICLTargetQG |
| SelfRefine 缂哄け | 宸插姞鍏?target-aware SelfRefine |
| 鏃?answerability trigger matching 涓嶅叕骞?| 宸叉敼鎴愮粺涓€ LLM judge |
| PathQG-HardAware prompt 涓嶅宸ユ暣 | 宸叉敼涓?CrossQG-style structured prompt |
| Medium path binding 杩囦弗 | 宸蹭粠 2 events 鏀炬澗涓?1 prior event |
| 缂哄皯 fair metrics | 宸插姞鍏?macro / E2E / difficulty-control diagnostics |
| trigger 琚洿鎺ュ綋绛旀 | 宸插紑濮嬪紩鍏?answer phrase / final event semantic target |
| 缂哄皯閫愭 debug log | 宸叉湁 `quality_pilot_9_test/debug_traces` 绀轰緥锛岄渶瑕佹墿澶у埌涓绘祦绋?|
| 缂哄皯瑙勫垯棰勭瓫 | 宸插疄鐜?path prefilter 鍒濈増 |
| 缂哄皯 LLM path quality judge | 宸叉柊澧?`event_qg/src/path_llm_judge.py`锛屽苟宸茬敤 gpt-4o 璺?15 鏉?pilot 鍋氫汉宸ユ牳鏌?|
| 涓婃父澶辫触鏍瑰洜涓嶆竻 | 宸查€氳繃 trace 瀹氫綅鍒?fixed-window answer phrase 鎴柇鍜?Medium path 鏂瑰悜椋庨櫓 |

---

## 8. 寰呰В鍐崇殑闂

| Problem | Why it matters | Status |
|---|---|---|
| fixed-window answer phrase 鎴柇 | gold answer 鍣０浼氭薄鏌?QG銆乫ilter銆乻olver judge | **宸蹭慨澶?* 鈥?clause-aware extraction |
| Medium path 鏂瑰悜鏋勯€犻闄?| 鍙兘鐢熸垚闈炵湡瀹?directed path锛屽鑷撮毦搴﹀拰鍏崇郴涓嶅彲淇?| **宸蹭慨澶?* 鈥?涓ユ牸 `src -> mid -> tgt` |
| relation subtype 鍙嶅悜鏌ユ壘鎺╃洊閿欒 | 鍙嶅悜琛ヨ竟浼氳閿欒 path 鐪嬭捣鏉ュ悎娉?| **宸蹭慨澶?* 鈥?`difficulty_scorer.py` 宸插垹闄わ紝hop-based scoring |
| 鍏ㄩ摼璺?trace 涓嶅畬鏁?| 涓嶈兘蹇€熷畾浣嶈川閲忛棶棰樻潵鑷摢涓€鐜?| **宸蹭慨澶?* 鈥?`trace_utils.py` 鍏ㄩ摼璺?trace |
| Hard 闂浠嶅鏄撳崟鍙ュ彲绛?| 涓嶈兘璇佹槑澶氳烦闅惧害 | 寰呰В鍐?|
| Medium / Hard path coverage 涓嶅 | 闂娌℃湁鐪熸渚濊禆璺緞鍓嶇疆浜嬩欢 | 寰呰В鍐?|
| judge_error 杩囬珮 | 浣庝及鐪熷疄 pass rate | 寰呰В鍐?|
| answer phrase 瑙勫垯杩囧 | prefilter pass rate 杩囬珮涓斾細鎶婃埅鏂?phrase 鏍囨垚 valid | 淇鍚庢娊 50 鏉′汉宸ユ牎楠?answer phrase |
| LLM path judge 灏氭湭 90 鏉￠獙璇?| 15 鏉?gpt-4o pilot 鏄剧ず鏂瑰悜姝ｇ‘浣嗘牱鏈お灏?| 淇涓婃父鍚庣敤 gpt-4o 璺?Easy/Medium/Hard 鍚?30 鏉?|
| solver/judge 涓?generator 鍚屾ā鍨嬪鏃?| circular evaluation 椋庨櫓 | 鎹笉鍚屾ā鍨嬪鏃?judge 鎴栧姞鍏ヤ汉宸ユ爣娉?|
| Difficulty monotonicity 涓嶇ǔ瀹?| 涓诲紶闅惧害鎺у埗浼氳璐ㄧ枒 | 浠?human-rated difficulty + solver trend 鍙岄噸楠岃瘉 |
| 4D difficulty claim 涓嶆垚绔?| RD/ES/EA 瀵规渶缁堟爣绛捐础鐚笉瓒?| 鏀瑰啓涓?path-length primary + relation/evidence auxiliary |
| ICLTargetQG 璐ㄩ噺寮轰簬 ours | ours 涓嶈兘涓绘墦閫氱敤璐ㄩ噺鑳滃嚭 | 涓绘墦 path controllability銆乨ifficulty alignment銆乤blation contribution |

---

## 9. Primary Metrics 鏄惁鍚堢悊

### 9.1 Difficulty Consistency

瀹氫箟锛氱敓鎴愰棶棰樼殑浜虹被/LLM 鍒ゆ柇闅惧害鏄惁绛変簬鐩爣 Easy / Medium / Hard銆?
```text
DifficultyConsistency = (1 / N) * sum_i 1[predicted_difficulty_i = target_difficulty_i]
```

褰撳墠瀹炵幇杩戜技锛?
- 浣跨敤 quality judge 鐨?`quality_difficulty_alignment`銆?- 鎴栦娇鐢ㄤ汉宸ユ爣娉?difficulty 鍚庤绠椾竴鑷寸巼銆?
璁烘枃鏀拺锛?
- [CrossQG: Improving Difficulty-Controllable Question Generation through Consistency Enhancement](https://aclanthology.org/2025.findings-emnlp.151/) 鏄庣‘浠?target difficulty consistency 浣滀负 difficulty-controllable QG 鐨勬牳蹇冪洰鏍囥€?
鏄惁鍙綔涓轰富鎸囨爣锛?*鍙互**銆?
娉ㄦ剰锛?
- 鏈€濂戒娇鐢ㄤ笁鍒嗙被 accuracy / macro accuracy锛屼笉瑕佸彧鐢ㄨ嚜瀹氫箟 0-1 LLM 鍒嗐€?- 濡傛灉浣跨敤 LLM judge锛岄渶瑕佷汉宸ユ娊鏍烽獙璇?judge 涓庝汉绫绘爣娉ㄧ殑涓€鑷存€с€?
### 9.2 Inference-step Consistency

瀹氫箟锛氱敓鎴愰棶棰樺疄闄呴渶瑕佺殑 event reasoning steps 鏄惁涓庣洰鏍?hop level 涓€鑷淬€?
寤鸿瀹氫箟锛?
```text
target_steps(Easy)   = 1
target_steps(Medium) = 2
target_steps(Hard)   = 3

StepConsistency = (1 / N) * sum_i 1[estimated_steps_i = target_steps_i]
```

涔熷彲浠ユ姤鍛?relaxed 鐗堟湰锛?
```text
StepConsistency_relaxed = (1 / N) * sum_i 1[estimated_steps_i >= target_steps_i]
```

璁烘枃鏀拺锛?
- [HotpotQA](https://aclanthology.org/D18-1259/) 浣跨敤 supporting facts 鏉ユ敮鎾?explainable multi-hop QA銆?- [MuSiQue](https://aclanthology.org/2022.tacl-1.31/) 寮鸿皟鏋勯€犻渶瑕?connected multi-hop reasoning 鐨勯棶棰橈紝骞舵帶鍒?2-4 hop銆?
鏄惁鍙綔涓轰富鎸囨爣锛?*鍙互锛屼絾蹇呴』璇存槑鏄湰浠诲姟鐨?adapted metric**銆?
娉ㄦ剰锛?
- 杩欎釜鎸囨爣涓嶆槸 CrossQG 鐨勫師濮嬪叕寮忋€?- 鎴戜滑涓嶈兘鎶婂綋鍓?`path_coverage` 鐩存帴绛夊悓浜?required inference steps銆傛洿鍚堢悊鍋氭硶鏄汉宸?LLM 鍒ゆ柇 鈥滈渶瑕佸嚑姝ヤ簨浠舵帹鐞嗏€濓紝骞舵娊鏍蜂汉宸ユ牎鍑嗐€?
### 9.3 Solver Accuracy by Difficulty

瀹氫箟锛氱敤鍚屼竴 solver 鍥炵瓟鐢熸垚闂锛屾鏌?solver correct 鏄惁闅忕洰鏍囬毦搴﹀崌楂樿€屼笅闄嶃€?
鍩虹鍏紡锛?
```text
SolCor_l = (1 / N_l) * sum_{i in level l} 1[solver_answer_i is correct]
```

鐞嗘兂瓒嬪娍锛?
```text
SolCor_Easy >= SolCor_Medium >= SolCor_Hard
```

鍙姤鍛?gaps锛?
```text
E-M gap = SolCor_Easy - SolCor_Medium
M-H gap = SolCor_Medium - SolCor_Hard
E-H gap = SolCor_Easy - SolCor_Hard
```

褰撳墠鍐呴儴璇婃柇锛?
```text
DC Score = max(0, E-M gap) + max(0, M-H gap)
Violations = 1[SolCor_Easy < SolCor_Medium] + 1[SolCor_Medium < SolCor_Hard]
```

璁烘枃鏀拺锛?
- 澶氳烦 QA 鏁版嵁闆嗛€氬父閫氳繃妯″瀷鎬ц兘涓嬮檷浣撶幇闂鏇撮毦锛屼緥濡?HotpotQA 鍜?MuSiQue 閮藉己璋冩洿澶嶆潅鎺ㄧ悊瀵?QA 绯荤粺鏇村叿鎸戞垬鎬с€?
鏄惁鍙綔涓轰富鎸囨爣锛?*鍙互浣滀负 operational proxy锛屼絾涓嶈兘鍗曠嫭浣滀负鏈€缁堥毦搴﹁瘉鏄?*銆?
娉ㄦ剰锛?
- `DC Score` 鍜?`Violations` 鏄垜浠嚜宸辩殑璇婃柇鍏紡锛屼笉鏄凡鏈夎鏂囨爣鍑嗘寚鏍囥€?- 璁烘枃涓彲浠ユ姤鍛?`SolCor by difficulty` 鍜?monotonic trend锛屼絾涓嶈鎶?`DC Score` 鍖呰鎴愬凡鏈夋爣鍑嗘寚鏍囥€?
### 9.4 Question Quality

瀹氫箟锛氱敓鎴愰棶棰樻槸鍚﹁嚜鐒躲€佺浉鍏炽€佸彲鍥炵瓟銆佺瓟妗堜竴鑷淬€?
褰撳墠缁村害锛?
- Fluency
- Relevance
- Answerability
- Answer Consistency

璁烘枃鏀拺锛?
- [QGEval](https://aclanthology.org/2024.emnlp-main.658/) 灏?QG 璇勪环鎷嗘垚 fluency銆乧larity銆乧onciseness銆乺elevance銆乧onsistency銆乤nswerability銆乤nswer consistency 绛夌淮搴︺€?- [QuestEval](https://aclanthology.org/2021.emnlp-main.529/) 鏀寔鐢?QA-based evaluation 璇勪及 consistency銆乫luency銆乺elevance 绛夌淮搴︺€?
鏄惁鍙綔涓轰富鎸囨爣锛?*閫傚悎浣滀负璐ㄩ噺鎸囨爣锛屼絾寤鸿浣滀负 secondary metrics**銆?
寤鸿鍏紡锛?
```text
Fluency      = mean human_or_judge_score_fluency
Relevance    = mean human_or_judge_score_relevance
Answerability = (1 / N) * sum_i 1[question_i answerable from context]
AnswerConsistency = (1 / N) * sum_i 1[answer_i matches target final event meaning]
```

娉ㄦ剰锛?
- Fluency / Relevance 閫氬父鏄汉宸ユ垨 judge 鎵撳垎缁村害锛屼笉涓€瀹氭湁鍞竴鏍囧噯鍏紡銆?- Answer Consistency 蹇呴』浠?鈥渢rigger string match鈥?鏀逛负 鈥渇inal event semantic match鈥濄€?- 濡傛灉瑕佸拰 QGEval 瀵归綈锛屽缓璁噰鐢?1-5 Likert 鎴?binary yes/no锛屽苟鏄庣‘璇勫垎缁嗗垯銆?
---

## 10. 鎸囨爣浣跨敤寤鸿

涓诲疄楠屼腑寤鸿杩欐牱缁勭粐锛?
### Primary metrics

| Metric | Role | Publish status |
|---|---|---|
| Difficulty Consistency | 鐩爣闅惧害鏄惁涓€鑷?| 涓绘寚鏍?|
| Inference-step Consistency | 鏄惁鐪熺殑闇€瑕佺洰鏍?hop steps | 涓绘寚鏍?|
| Solver Accuracy by Difficulty | 闅惧害鏄惁浣撶幇涓?solver performance drop | 涓绘寚鏍囷紝浣嗕綔涓?proxy |

### Secondary metrics

| Metric | Role |
|---|---|
| Answerability | 闂鑳藉惁浠?context 鍥炵瓟 |
| Answer Consistency | 绛旀鏄惁瀵瑰簲 final event semantic target |
| Fluency | 璇█鑷劧鎬?|
| Relevance | 鏄惁涓?context / path 鐩稿叧 |
| Pass rate | 瀹屾暣 pipeline 鐢熸垚鍙敤闂鐨勬瘮渚?|

### 涓嶅缓璁綔涓轰富鎸囨爣

| Metric | Reason |
|---|---|
| Composite | 鏉冮噸鏄嚜瀹氫箟鐨勶紝瀹规槗琚川鐤?|
| DC Score | 鑷畾涔夎瘖鏂寚鏍囷紝涓嶆槸宸叉湁鏍囧噯 |
| Trigger exact match | trigger 鏈韩缁忓父涓嶅畬鏁存垨閿欒 |
| Path coverage lexical only | 涓嶈兘鍙潬浠ｈ〃璇箟鎺ㄧ悊姝ユ暟 |

---

## 11. 褰撳墠鍙互姹囨姤鐨勭粨璁?
鍙互姹囨姤锛?
1. 鎴戜滑宸茬粡鏋勫缓浜?document-level event graph锛屽苟浠庝腑閲囨牱 Easy / Medium / Hard 璺緞銆?2. 鎴戜滑鐨勬柟娉曚笉鏄嚜鐢辩敓鎴愶紝鑰屾槸鎶?event path 浣滀负缁撴瀯鍖栫害鏉熸敞鍏ラ棶棰樼敓鎴愩€?3. 鍦ㄥ浐瀹?300 鏉℃牱鏈笂锛孭athQG-HardAware 鐨?pass rate 鏈€楂橈細62.0%锛岄珮浜?ZeroShot銆両CL 鍜?SelfRefine銆?4. Ablation 鏄剧ず context 鍜?specific event path 閮芥湁璐＄尞銆?5. 褰撳墠 ICLTargetQG 鐨?solver_correct 鏈€楂橈紝璇存槑 ours 涓嶈兘涓绘墦閫氱敤璐ㄩ噺鑳滃嚭銆?6. 宸查€氳繃 trace-first 鎺掓煡瀹氫綅鍒颁笂娓歌川閲忛棶棰橈細fixed-window answer phrase 鎴柇銆丮edium path 鏂瑰悜鏋勯€犻闄┿€乸ath length 涓庣湡瀹炴帹鐞嗘鏁颁笉涓€鑷淬€?7. 褰撳墠鏈€鏍稿績鐨勫緟瑙ｅ喅闂鏄細淇 answer phrase extraction銆佷慨澶?directed path sampling銆佽ˉ鍏ㄥ叏閾捐矾 trace銆侀噸鏂拌窇 path judge 鍜?QG銆?
涓嶅缓璁眹鎶ユ垚锛?
1. 鈥滃洓缁撮毦搴﹁瘎鍒嗘湁鏁堝尯鍒嗛毦搴︺€傗€濆凡搴熷純锛屾敼涓?hop-based scoring銆?2. 鈥滄垜浠叏闈㈣秴杩?ICL baseline銆傗€濆綋鍓嶆暟鎹笉鏀寔銆?3. 鈥淗ard 闂绋冲畾闇€瑕佸璺虫帹鐞嗐€傗€濆綋鍓?hard degraded 浠嶆湁 33.33%銆?4. 鈥渢rigger 灏辨槸 gold answer銆傗€漷rigger 鍙兘浣滀负浜嬩欢閿氱偣銆?
---

## 12. 鑷姩鏇存柊鍗忚

姣忔瀹為獙鎴栦唬鐮佹敼鍔ㄥ悗锛屾寜浠ヤ笅椤哄簭鏇存柊鏈枃浠讹細

### 12.1 Trace-first debugging rule

鍚庣画浠讳綍璐ㄩ噺闂閮藉繀椤讳紭鍏堜緷鎹?trace log 瀹氫綅锛屼笉鍏佽鍙牴鎹眹鎬昏〃鐚滄祴鍘熷洜銆?
鎺掓煡椤哄簭鍥哄畾涓猴細

```text
MAVEN raw sentence/event/offset
 -> graph node and directed edge
 -> sampled path and relation direction
 -> difficulty score
 -> answer phrase extraction
 -> prefilter result
 -> LLM path judge prompt/raw/parsed
 -> QG prompt/raw/repair
 -> quality filter
 -> solver answer and judge raw response
```

濡傛灉鏌愪釜缁撹涓嶈兘杩芥函鍒板叿浣?trace item锛屼笉搴斿啓鍏ョ粨鏋滅粨璁烘垨璁烘枃鍙欒堪銆?
1. 濡傛灉鏀逛簡璺緞閲囨牱鎴?prefilter锛?   - 鏇存柊绗?2 鑺傘€?   - 鏉ユ簮锛歚graph_building_report.json`銆乣path_sampling_report.json`銆乣path_prefilter_report.json`銆?
2. 濡傛灉鏀逛簡 baseline 鎴栦富瀹為獙锛?   - 鏇存柊绗?4銆? 鑺傘€?   - 鏉ユ簮锛歚review-stage/RESULTS_SUMMARY.md` 鍜屽搴?evaluated jsonl銆?
3. 濡傛灉鏀逛簡 quality filter / judge锛?   - 鏇存柊绗?6銆? 鑺傘€?   - 鏉ユ簮锛歚quality_pilot_* / filter_report.json`銆乵anual review 鏂囦欢銆?
4. 濡傛灉鏀逛簡 LLM path judge锛?   - 鏇存柊绗?6銆? 鑺傘€?   - 鏉ユ簮锛歚path_judge_pilot_* / path_judge_report.json`銆乣path_judge_trace.jsonl`銆?
5. 濡傛灉鏀逛簡鎸囨爣瀹氫箟锛?   - 鏇存柊绗?9銆?0 鑺傘€?   - 蹇呴』妫€鏌ユ槸鍚︽湁宸插彂琛ㄨ鏂囨敮鎸併€?   - 濡傛灉鏄嚜瀹氫箟鍏紡锛屽彧鑳芥爣涓?diagnostic / internal銆?
6. 姣忔鏇存柊閮借鏀癸細
   - 椤堕儴鈥滄渶鍚庢洿鏂扳€濇棩鏈熴€?   - 鈥滃凡瑙ｅ喅鐨勯棶棰樷€濄€?   - 鈥滃緟瑙ｅ喅鐨勯棶棰樷€濄€?   - 鈥滃綋鍓嶅彲浠ユ眹鎶ョ殑缁撹鈥濄€?
寤鸿浠ュ悗瀵?Claude Code 浣跨敤杩欎釜鎸囦护锛?
```text
璇疯鍙?review-stage/PROJECT_STATUS.md銆乺eview-stage/RESULTS_SUMMARY.md 浠ュ強鏈€鏂拌緭鍑虹洰褰曚腑鐨?filter_report.json锛?鎶?PROJECT_STATUS.md 涓殑鏈€鏂板疄楠屾暟鎹€佸凡瑙ｅ喅闂銆佸緟瑙ｅ喅闂鍜屽彲姹囨姤缁撹鍚屾鏇存柊銆?涓嶈鏀瑰彉鎸囨爣瀹氫箟锛岄櫎闈炲悓鏃剁粰鍑哄凡鏈夎鏂囦緷鎹€?```

---

## 13. References

- Li and Zhang. 2025. [CrossQG: Improving Difficulty-Controllable Question Generation through Consistency Enhancement](https://aclanthology.org/2025.findings-emnlp.151/).
- Yang et al. 2018. [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://aclanthology.org/D18-1259/).
- Trivedi et al. 2022. [MuSiQue: Multihop Questions via Single-hop Question Composition](https://aclanthology.org/2022.tacl-1.31/).
- Fu et al. 2024. [QGEval: Benchmarking Multi-dimensional Evaluation for Question Generation](https://aclanthology.org/2024.emnlp-main.658/).
- Scialom et al. 2021. [QuestEval: Summarization Asks for Fact-based Evaluation](https://aclanthology.org/2021.emnlp-main.529/).

---

## 14. Failure Diagnosis

### 14.1 Medium Path Direction Bug
- **Root cause:** `path_sampler.py` lines 115-117 constructed V-shape paths `src <- mid -> tgt` instead of directed chains `src -> mid -> tgt`. Both `src` and `tgt` were sampled from `g.get_out_neighbors(mid)`, making both edges point away from `mid`.
- **Impact:** Medium paths were not real directed event chains. Relation subtypes and difficulty scores for these paths were unreliable. `difficulty_scorer.py` masked this by using bidirectional edge lookup.
- **Fix:** Changed iteration to follow outgoing edges: `src -> mid` via `g.get_out_neighbors(src)`, `mid -> tgt` via `g.get_out_neighbors(mid)`. Also added `offset` field to `events_detail`.
- **Validation:** Re-sampled 50 docs, verified all Medium paths form valid directed chains.

### 14.2 Answer Phrase Truncation
- **Root cause:** `compare_hardaware.py` `extract_answer_phrase_local()` used fixed window `[trigger-2, trigger+4]` words.
- **Impact:** Truncated phrases like "has been described as one of the" polluted gold answer, causing downstream QG, filter, and solver failures.
- **Fix:** Replaced with clause-aware expansion that stops at clause boundaries (commas, semicolons, conjunctions, sentence boundaries). Added `answer_phrase_status` field (`complete`/`partial`/`invalid`).
- **Validation:** Tested on known truncation examples, all produce clause-complete phrases.

### 14.3 API Error Misclassification
- **Root cause:** `path_llm_judge.py` produced synthetic `path_questionable: "no"` on API errors, indistinguishable from genuine rejections.
- **Impact:** Valid paths silently dropped due to transient API failures.
- **Fix:** Added `llm_path_judge_status` field (`ok`/`api_error`/`parse_error`). Paths with API errors are kept and marked for review instead of dropped.
- **Validation:** Tested with deliberately bad API URL, verified correct status tagging.

### 14.4 Bidirectional Edge Lookup Masking
- **Root cause:** `difficulty_scorer.py` `compute_relation_diversity_score()` and `get_path_relation_subtypes()` tried both edge directions `(src,tgt)` and `(tgt,src)`, masking the path direction bug.
- **Impact:** Relation subtypes and RD scores were computed correctly even for misdirected paths, hiding the Medium path direction bug.
- **Fix:** `difficulty_scorer.py` deleted entirely. Difficulty scoring changed to pure hop-based (Easy=1hop, Medium=2hop, Hard=3hop). Relation subtypes extracted by forward-only helper in `path_sampler.py`.
- **Validation:** Full pipeline smoke test passes with hop-based scoring.

### 14.5 Code Cleanup (2026-05-01)
- **Deleted files:** `difficulty_scorer.py`, `compare.py`, `inspect_data.py`, `run_stage1.py`, `stage2_prototype.py`, `scripts/reeval_32b_solver.py`, `scripts/sample_for_analysis.py`
- **Reason:** Dead code not in current pipeline dependency chain. `difficulty_scorer.py` replaced by hop-based logic in `path_sampler.py`. Other deleted files were standalone scripts superseded by current pipeline modules.
- **Important correction:** `baselines.py` is **not** dead code and must be kept. It contains baseline generation and unified evaluation.
- **Remaining files (13):** `graph_builder.py`, `path_sampler.py`, `path_prefilter.py`, `path_llm_judge.py`, `answer_extraction.py`, `compare_hardaware.py`, `quality_filter.py`, `quality_pilot.py`, `full_pipeline_smoke.py`, `trace_utils.py`, `evaluator.py`, `evaluator_v2.py`, `baselines.py`
- **Cleaned fields:** Removed `PL`, `RD`, `ES`, `EA`, `difficulty_score` from `path_sampler.py`, `trace_utils.py`, `full_pipeline_smoke.py`, `compare_hardaware.py`
- **Refactor:** Answer phrase extraction and final-event validity helpers moved from `compare_hardaware.py` to `answer_extraction.py`; path prefilter and pilots now import the shared module directly.

---

## 15. Trace Protocol

All pipeline stages now use `trace_utils.py` (`event_qg/src/trace_utils.py`) for consistent full-chain tracing.

### Trace record structure:
- `raw_source`: doc_id, event_count, relation_count from MAVEN raw data
- `graph_stage`: nodes, edges, isolated_nodes from graph_builder
- `path_sampling`: difficulty, hop_count, path_events (with offset), relation_subtypes, relation_distribution
- `answer_extraction`: trigger, answer_phrase, answer_sentence, answer_phrase_status, extraction_method
- `prefilter`: prefilter_pass, prefilter_reason, weak_trigger_type, relation_group, support_span, rule_single_sentence_risk
- `llm_path_judge`: llm_path_judge_status (ok/api_error/parse_error), path_questionable, recommended_difficulty, judge_raw_response
- `qg_generation`: generator_model, prompts, raw_responses, parsed_question, retry_attempts
- `quality_filter`: grammar_check, weak_trigger_check, answer_phrase_check, consistency_judge (with raw), path_coverage (with raw), asks_target_event, hard_degraded (with raw), filter_pass/fail
- `solver_eval`: solver_result, solver_confidence

### Debugging workflow:
1. Start from `full_trace.jsonl` for the specific item.
2. Check `raw_source` to verify MAVEN data integrity.
3. Check `graph_stage` for node/edge anomalies.
4. Check `path_sampling.path_events` for correct direction and offset.
5. Check `answer_extraction.answer_phrase_status` for truncation.
6. Check `prefilter` for rule-based filtering decisions.
7. Check `llm_path_judge.llm_path_judge_status` for API/parse errors.
8. Check `qg_generation` for prompt/response pairs and retry behavior.
9. Check `quality_filter` for all judge raw responses.
10. Check `solver_eval` for final solver outcome.

---

## 16. Full Pipeline Smoke Test

Added `event_qg/src/full_pipeline_smoke.py` for true small-scale end-to-end tracing.

Use this when validating the complete pipeline:

```powershell
python event_qg/src/full_pipeline_smoke.py `
  --limit 5 `
  --output_dir event_qg/outputs/full_pipeline_smoke_5
```

This differs from `quality_pilot.py`:

- `quality_pilot.py` runs path sampling/enrichment, question generation, and question filtering.
- `full_pipeline_smoke.py` runs prefiltered paths, LLM path judge, question generation, question filtering, solver, LLM judge, and full trace export.

Latest smoke result:

- Output: `event_qg/outputs/full_pipeline_smoke_5_4omini/`
- Items: 5
- Graph stage: present for 5/5
- Prefilter fields: present for 5/5
- LLM path judge fields: present for 5/5, `llm_path_judge_status=ok` for 5/5 with `gpt-4o-mini`.
- LLM path judge kept 4/5 paths; one Hard path was skipped due `hard_single_sentence_risk=high`.
- QG prompt/raw response: present for generated items; skipped item explicitly records `generation_status=not_run`.
- Quality filter: present for 5/5
- Solver eval: ran for final-filter-pass items only; `solver_ok=2/5`, other items marked `not_run` with reason.
- Current 5-item smoke is for trace validation only, not a performance estimate.

Important trace rule update:

If a stage is not executed, trace must explicitly record `status=not_run` or `api_error`; empty fields are not acceptable for full-chain debugging.

### 16.1 No-Skip Full-Chain Trace Validation

Latest no-skip smoke:

```powershell
python event_qg/src/full_pipeline_smoke.py `
  --limit 1 `
  --output_dir event_qg/outputs/full_pipeline_smoke_1_noskip_consistent
```

Result:

- `path_judge_status=ok`: 1/1
- `path_keep`: 1/1
- `generated`: 1/1
- `filter_pass`: 1/1
- `solver_ok`: 1/1

Trace finding:

- The full trace is now usable for locating cross-stage quality issues.
- A no-skip run exposed answer phrase drift: path judge/QG used one phrase, while `quality_filter.py` overwrote `gold_answer_phrase` with an LLM-extracted phrase.
- Fix: `full_pipeline_smoke.py` refreshes `answer_extraction` before path judge; `quality_filter.py` now keeps upstream `gold_answer_phrase` as the canonical target and stores LLM extraction only in `llm_answer_phrase*` diagnostic fields.
- Remaining observed issue in the example: solver reached `status=ok` but `judge_solver_correct=0.0`; use the trace fields `solver_answer`, `gold_answer_phrase`, and judge scores to inspect whether this is a solver failure, judge strictness issue, or answer phrase granularity issue.
