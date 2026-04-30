# DCQG 项目状态与实验台账

**最后更新**：2026-04-30  
**状态**：方法框架已成型，主实验已在固定 300 条样本上完成一轮统一评估。已修复四个数据链路问题：Medium path 方向 bug、answer phrase 截断、API error 误判、bidirectional edge lookup 掩盖。已加入全链路 debug trace (`trace_utils.py`)。  
**维护规则**：每次修改生成、过滤、评估、采样或 baseline 后，必须同步更新本文件中的“最新实验数据”“已解决问题”“待解决问题”“指标定义与依据”四部分。后续定位问题必须优先查看 trace log，不允许只根据汇总数字猜测原因。

---

## 1. 当前预设流程

本项目当前最合理的整体流程是：

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

关键定位：

- 我们不是纯自由生成问题，而是 **target-event-grounded QG**。
- `trigger` 不能再被当成唯一 gold answer；它更适合作为 final event 的定位锚点。
- 论文主线建议写成：**事件路径约束如何帮助生成符合目标推理难度的问题**。
- 当前四维难度分数 `D = PL + RD + ES + EA` 仍在代码中存在，但实验观察显示难度标签实际主要由路径长度决定。因此论文中应弱化“四维评分是核心贡献”的说法。

---

## 2. 数据与路径采样状态

### 2.1 Event graph construction

来源文件：`event_qg/outputs/graph_building_report.json`

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

结论：图中 temporal relation 占比极高，这是 Hard 问题容易退化为时间顺序检索的根本原因之一。

### 2.2 Path sampling

来源文件：`event_qg/outputs/path_sampling_report.json`

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

当前路径采样方法：

- Easy：枚举 1-hop directed edge。
- Medium：原目标是枚举 2-hop directed path，但当前实现存在方向构造风险，需要修复为严格 `src -> mid -> tgt`。
- Hard：从起点 BFS 搜索 3-hop directed path，并做一定 hard 条件过滤。
- 路径范围：单文档内部采样，不跨文档。

已定位风险：

- `path_sampler.py` 当前输出每个 event 时没有保留 MAVEN 原始 `offset`，导致后续 answer phrase 抽取只能靠 trigger 字符串和固定窗口。
- `path_sampler.py` 的 Medium 构造逻辑疑似把 `mid` 的 outgoing neighbor 反向放到 `mid` 前面，可能产生非真实 directed path。
- `difficulty_scorer.py` 的 relation subtype 查找会尝试反向边，可能掩盖 path 方向问题。

### 2.3 Rule prefilter

来源文件：`event_qg/outputs/path_prefilter_report.json`

| Difficulty | Total | Passed | Pass rate |
|---|---:|---:|---:|
| Easy | 1156 | 1064 | 92.04% |
| Medium | 1458 | 1334 | 91.50% |
| Hard | 1276 | 1191 | 93.34% |
| Overall | 3890 | 3589 | 92.26% |

重要观察：

- Weak trigger distribution: none 3514, hard_blacklist 153, needs_phrase 223。
- Temporal-only Hard: 434 / 1276 = 34.01%。
- Single-sentence risk: high 324, medium 137, low 3429。
- `answer_phrase_pass_rate = 1.0` 过于理想，已经确认原因之一是 fixed-window extractor 偏宽，会把截断短语误判为 valid phrase。

### 2.4 Failure diagnosis: answer phrase and path direction

本轮人工追踪了从 MAVEN 原始数据到 gpt-4o path judge 的链路，定位如下：

| Stage | Status | Evidence |
|---|---|---|
| MAVEN raw document | 基本正常 | 原始 event mention 有 `trigger_word`、`sent_id`、`offset`，原句完整 |
| `graph_builder.py` | 基本正常 | `get_event_info()` 已读取 `offset` |
| `path_sampler.py` | 有问题 | 输出 `events` 时丢掉 `offset`；Medium path 方向构造存在风险 |
| `compare_hardaware.enrich_path_item()` | 明确有问题 | `extract_answer_phrase_local()` 使用 trigger 前 2 词 + 后 4 词固定窗口 |
| `path_prefilter.py` | 继承上述问题 | 调用 `enrich_path_item()` 后把截断 phrase 标成 pass |
| `path_llm_judge.py` / gpt-4o | 判断基本合理 | gpt-4o 正确指出 answer phrase 截断和 single-sentence shortcut |

典型截断例子：

| Trigger | Sentence | Current bad phrase | Better phrase |
|---|---|---|---|
| described | It has been described as one of the most horrific frontier massacres of the war. | has been described as one of the | described as one of the most horrific frontier massacres of the war |
| rise | ... leading to the rise of the ghazis and the conclusive Byzantine-Ottoman wars. | to the rise of the ghazis and | the rise of the ghazis |
| claimed | The U.S. pilots claimed to have shot down three twin-engined bombers and two fighters... | U.S. pilots claimed to have shot down | claimed to have shot down three twin-engined bombers and two fighters |
| drove | ... Expedition which drove the Iroquois out of western New York. | Expedition which drove the Iroquois out of | drove the Iroquois out of western New York |

阶段性结论：

- 之前 weak results 不能直接说明 PathQG 方法无效；一部分失败来自上游 target-answer extraction 和 path direction 噪声。
- 后续必须先修 `answer phrase extraction + path direction`，再重新跑 path judge 和 QG。
- 任何新结论都必须能通过 full trace 追溯到 MAVEN 原句、event offset、path direction、answer extraction 和 judge raw response。

---

## 3. 主实验设置

固定样本：

| Setting | Value |
|---|---|
| Sample file | `event_qg/outputs/sample_300_seed42.jsonl` |
| Total items | 300 |
| Difficulty distribution | Easy 100 / Medium 100 / Hard 100 |
| Unique documents | 49 |
| Generator | Qwen2.5-7B |
| Solver / Judge | Qwen2.5-32B |
| Known limitation | generator 和 judge/solver 属于同一模型家族 |

统一评估：

1. Generate question。
2. Grammar / basic format filter。
3. LLM solver 回答问题。
4. LLM judge 判断 answerable、solver_correct、support_covered。
5. Quality judge 判断 fluency、relevance、difficulty alignment。

注意：当前 composite 是内部加权分数，不建议作为论文主指标。

```text
Composite = 0.25 * solver_correct
          + 0.20 * answerable
          + 0.15 * support_covered
          + 0.15 * fluency
          + 0.10 * relevance
          + 0.15 * diff_align
```

这个权重不是已有论文标准公式，只能作为辅助诊断或内部排序指标。

---

## 4. Baseline 设计

### 4.1 Main baselines

| Method | Input | 是否看 target answer | 是否看 event path | 目的 |
|---|---|---:|---:|---|
| ZeroShotTargetQG | context + target answer + difficulty | Yes | No | 目标事件 QG 的零样本基线 |
| ICLTargetQG | context + target answer + difficulty + examples | Yes | No | CrossQG-style few-shot / ICL baseline |
| SelfRefine | ZeroShotTargetQG + critique + revise | Yes | No | 测试自我修正是否提升质量 |
| PathQG-HardAware | context + target event + event path + relation sequence + difficulty prompt | Yes | Yes | 我们的方法 |

这里的 ICL 是少样本 in-context learning：给 2 个同难度示例，让模型按示例格式生成。

### 4.2 Ablations

| Method | Removed component | 目的 |
|---|---|---|
| PathOnlyQG | context | 验证没有上下文时，仅靠路径是否足够 |
| RelationTypeQG | specific event path | 验证只给关系类型、不指定具体路径是否足够 |

这两个更适合写在 ablation，不适合作为外部主 baseline。

---

## 5. 最新主实验结果

来源文件：`review-stage/RESULTS_SUMMARY.md`

| Method | N gen | N pass | Pass% | Answerable | Solver Correct | Composite |
|---|---:|---:|---:|---:|---:|---:|
| PathQG-HardAware | 300 | 186 | 62.0% | 0.747 | 0.274 | 0.693 |
| ZeroShotTargetQG | 300 | 127 | 42.3% | 0.921 | 0.325 | 0.728 |
| ICLTargetQG | 300 | 135 | 45.0% | 0.874 | 0.363 | 0.737 |
| SelfRefine | 300 | 143 | 47.7% | 0.923 | 0.308 | 0.729 |

解释：

- PathQG-HardAware 当前最高的是 pass rate，而不是 solver correctness。
- ICLTargetQG 当前 solver_correct 和 composite 最高，说明强 prompt + few-shot 对 target-aware QG 非常有效。
- 因此论文不能写“我们全面优于 ICL baseline”；更合理的说法是“事件路径约束提升生成通过率、路径绑定和难度控制，但通用问题质量仍需改进”。

### 5.1 Solver correct by difficulty

| Method | Easy | Medium | Hard |
|---|---:|---:|---:|
| PathQG-HardAware | 0.260 | 0.371 | 0.203 |
| ZeroShotTargetQG | 0.375 | 0.375 | 0.211 |
| ICLTargetQG | 0.434 | 0.326 | 0.308 |
| SelfRefine | 0.277 | 0.380 | 0.261 |

当前问题：

- PathQG-HardAware 出现 Medium > Easy，违反 Easy >= Medium >= Hard 的理想难度趋势。
- 这说明当前难度控制还没有完全稳定，不能直接声称“严格单调控制难度”。

### 5.2 Fair metrics

| Method | Pass% | Conditional SolCor | Macro-Avg SolCor | E2E SolCor |
|---|---:|---:|---:|---:|
| PathQG-HardAware | 62.0% | 0.274 | 0.278 | 0.170 |
| ZeroShotTargetQG | 42.3% | 0.325 | 0.320 | 0.137 |
| ICLTargetQG | 45.0% | 0.363 | 0.356 | 0.163 |
| SelfRefine | 47.7% | 0.308 | 0.306 | 0.147 |

其中：

```text
Conditional metric = mean(metric over passed/scored questions)
Macro metric       = (metric_Easy + metric_Medium + metric_Hard) / 3
E2E metric         = sum(metric over passed/scored questions) / 300
```

E2E 指标对生成失败样本记 0，因此能反映完整 pipeline 的产出能力。

### 5.3 Ablation results

| Method | Component removed | N pass | Pass% | Answerable | Solver Correct | Composite |
|---|---|---:|---:|---:|---:|---:|
| PathQG-HardAware | none | 186 | 62.0% | 0.747 | 0.274 | 0.693 |
| RelationTypeQG | specific path | 134 | 44.7% | 0.791 | 0.179 | 0.652 |
| PathOnlyQG | context | 158 | 52.7% | 0.285 | 0.114 | 0.534 |

可支撑的结论：

- 去掉 context 后 answerability 大幅下降，说明上下文是可回答性的必要条件。
- 去掉 specific event path 后 solver_correct 下降，说明具体路径约束有贡献。
- 但 ablation 不能证明当前难度控制已经完全成功。

---

## 6. 最新质量 pilot 结果

来源文件：`event_qg/outputs/quality_pilot_90_v3/filter_report.json`

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

人工抽查 judge_error 的 20 条：

| Manual label | Count |
|---|---:|
| yes | 12 |
| partial | 4 |
| no | 4 |

结论：judge_error 中有大量其实可用的问题，当前 judge JSON 解析/提示不稳，是近期最值得修的问题之一。

---

## 7. 已解决的问题

| Problem | Current status |
|---|---|
| baseline 与 ours 样本不统一 | 已统一到 `sample_300_seed42.jsonl` |
| PathOnlyQG / RelationTypeQG 被误当主 baseline | 已调整为 ablation |
| 缺少 ICL baseline | 已加入 ICLTargetQG |
| SelfRefine 缺失 | 已加入 target-aware SelfRefine |
| 旧 answerability trigger matching 不公平 | 已改成统一 LLM judge |
| PathQG-HardAware prompt 不够工整 | 已改为 CrossQG-style structured prompt |
| Medium path binding 过严 | 已从 2 events 放松为 1 prior event |
| 缺少 fair metrics | 已加入 macro / E2E / difficulty-control diagnostics |
| trigger 被直接当答案 | 已开始引入 answer phrase / final event semantic target |
| 缺少逐步 debug log | 已有 `quality_pilot_9_test/debug_traces` 示例，需要扩大到主流程 |
| 缺少规则预筛 | 已实现 path prefilter 初版 |
| 缺少 LLM path quality judge | 已新增 `event_qg/src/path_llm_judge.py`，并已用 gpt-4o 跑 15 条 pilot 做人工核查 |
| 上游失败根因不清 | 已通过 trace 定位到 fixed-window answer phrase 截断和 Medium path 方向风险 |

---

## 8. 待解决的问题

| Problem | Why it matters | Suggested next step |
|---|---|---|
| fixed-window answer phrase 截断 | gold answer 噪声会污染 QG、filter、solver judge | 用 offset/clause 或 LLM canonicalizer 重写 answer phrase extraction |
| Medium path 方向构造风险 | 可能生成非真实 directed path，导致难度和关系不可信 | 修复为严格 `src -> mid -> tgt`，并输出 direction_check |
| relation subtype 反向查找掩盖错误 | 反向补边会让错误 path 看起来合法 | `difficulty_scorer.py` 只按 path 方向查边，UNKNOWN 要进入 trace |
| 全链路 trace 不完整 | 不能快速定位质量问题来自哪一环 | 实现 raw -> graph -> path -> answer -> judge -> QG -> solver 全链路 trace |
| Hard 问题仍容易单句可答 | 不能证明多跳难度 | 增强 hard prompt，过滤 single-sentence answerable cases |
| Medium / Hard path coverage 不够 | 问题没有真正依赖路径前置事件 | 用 path coverage trace 找失败模板，改 prompt 和 repair |
| judge_error 过高 | 低估真实 pass rate | 简化 judge prompt，增加 fallback parser |
| answer phrase 规则过宽 | prefilter pass rate 过高且会把截断 phrase 标成 valid | 修复后抽 50 条人工校验 answer phrase |
| LLM path judge 尚未 90 条验证 | 15 条 gpt-4o pilot 显示方向正确但样本太少 | 修复上游后用 gpt-4o 跑 Easy/Medium/Hard 各 30 条 |
| solver/judge 与 generator 同模型家族 | circular evaluation 风险 | 换不同模型家族 judge 或加入人工标注 |
| Difficulty monotonicity 不稳定 | 主张难度控制会被质疑 | 以 human-rated difficulty + solver trend 双重验证 |
| 4D difficulty claim 不成立 | RD/ES/EA 对最终标签贡献不足 | 改写为 path-length primary + relation/evidence auxiliary |
| ICLTargetQG 质量强于 ours | ours 不能主打通用质量胜出 | 主打 path controllability、difficulty alignment、ablation contribution |

---

## 9. Primary Metrics 是否合理

### 9.1 Difficulty Consistency

定义：生成问题的人类/LLM 判断难度是否等于目标 Easy / Medium / Hard。

```text
DifficultyConsistency = (1 / N) * sum_i 1[predicted_difficulty_i = target_difficulty_i]
```

当前实现近似：

- 使用 quality judge 的 `quality_difficulty_alignment`。
- 或使用人工标注 difficulty 后计算一致率。

论文支撑：

- [CrossQG: Improving Difficulty-Controllable Question Generation through Consistency Enhancement](https://aclanthology.org/2025.findings-emnlp.151/) 明确以 target difficulty consistency 作为 difficulty-controllable QG 的核心目标。

是否可作为主指标：**可以**。

注意：

- 最好使用三分类 accuracy / macro accuracy，不要只用自定义 0-1 LLM 分。
- 如果使用 LLM judge，需要人工抽样验证 judge 与人类标注的一致性。

### 9.2 Inference-step Consistency

定义：生成问题实际需要的 event reasoning steps 是否与目标 hop level 一致。

建议定义：

```text
target_steps(Easy)   = 1
target_steps(Medium) = 2
target_steps(Hard)   = 3

StepConsistency = (1 / N) * sum_i 1[estimated_steps_i = target_steps_i]
```

也可以报告 relaxed 版本：

```text
StepConsistency_relaxed = (1 / N) * sum_i 1[estimated_steps_i >= target_steps_i]
```

论文支撑：

- [HotpotQA](https://aclanthology.org/D18-1259/) 使用 supporting facts 来支撑 explainable multi-hop QA。
- [MuSiQue](https://aclanthology.org/2022.tacl-1.31/) 强调构造需要 connected multi-hop reasoning 的问题，并控制 2-4 hop。

是否可作为主指标：**可以，但必须说明是本任务的 adapted metric**。

注意：

- 这个指标不是 CrossQG 的原始公式。
- 我们不能把当前 `path_coverage` 直接等同于 required inference steps。更合理做法是人工/LLM 判断 “需要几步事件推理”，并抽样人工校准。

### 9.3 Solver Accuracy by Difficulty

定义：用同一 solver 回答生成问题，检查 solver correct 是否随目标难度升高而下降。

基础公式：

```text
SolCor_l = (1 / N_l) * sum_{i in level l} 1[solver_answer_i is correct]
```

理想趋势：

```text
SolCor_Easy >= SolCor_Medium >= SolCor_Hard
```

可报告 gaps：

```text
E-M gap = SolCor_Easy - SolCor_Medium
M-H gap = SolCor_Medium - SolCor_Hard
E-H gap = SolCor_Easy - SolCor_Hard
```

当前内部诊断：

```text
DC Score = max(0, E-M gap) + max(0, M-H gap)
Violations = 1[SolCor_Easy < SolCor_Medium] + 1[SolCor_Medium < SolCor_Hard]
```

论文支撑：

- 多跳 QA 数据集通常通过模型性能下降体现问题更难，例如 HotpotQA 和 MuSiQue 都强调更复杂推理对 QA 系统更具挑战性。

是否可作为主指标：**可以作为 operational proxy，但不能单独作为最终难度证明**。

注意：

- `DC Score` 和 `Violations` 是我们自己的诊断公式，不是已有论文标准指标。
- 论文中可以报告 `SolCor by difficulty` 和 monotonic trend，但不要把 `DC Score` 包装成已有标准指标。

### 9.4 Question Quality

定义：生成问题是否自然、相关、可回答、答案一致。

当前维度：

- Fluency
- Relevance
- Answerability
- Answer Consistency

论文支撑：

- [QGEval](https://aclanthology.org/2024.emnlp-main.658/) 将 QG 评价拆成 fluency、clarity、conciseness、relevance、consistency、answerability、answer consistency 等维度。
- [QuestEval](https://aclanthology.org/2021.emnlp-main.529/) 支持用 QA-based evaluation 评估 consistency、fluency、relevance 等维度。

是否可作为主指标：**适合作为质量指标，但建议作为 secondary metrics**。

建议公式：

```text
Fluency      = mean human_or_judge_score_fluency
Relevance    = mean human_or_judge_score_relevance
Answerability = (1 / N) * sum_i 1[question_i answerable from context]
AnswerConsistency = (1 / N) * sum_i 1[answer_i matches target final event meaning]
```

注意：

- Fluency / Relevance 通常是人工或 judge 打分维度，不一定有唯一标准公式。
- Answer Consistency 必须从 “trigger string match” 改为 “final event semantic match”。
- 如果要和 QGEval 对齐，建议采用 1-5 Likert 或 binary yes/no，并明确评分细则。

---

## 10. 指标使用建议

主实验中建议这样组织：

### Primary metrics

| Metric | Role | Publish status |
|---|---|---|
| Difficulty Consistency | 目标难度是否一致 | 主指标 |
| Inference-step Consistency | 是否真的需要目标 hop steps | 主指标 |
| Solver Accuracy by Difficulty | 难度是否体现为 solver performance drop | 主指标，但作为 proxy |

### Secondary metrics

| Metric | Role |
|---|---|
| Answerability | 问题能否从 context 回答 |
| Answer Consistency | 答案是否对应 final event semantic target |
| Fluency | 语言自然性 |
| Relevance | 是否与 context / path 相关 |
| Pass rate | 完整 pipeline 生成可用问题的比例 |

### 不建议作为主指标

| Metric | Reason |
|---|---|
| Composite | 权重是自定义的，容易被质疑 |
| DC Score | 自定义诊断指标，不是已有标准 |
| Trigger exact match | trigger 本身经常不完整或错误 |
| Path coverage lexical only | 不能可靠代表语义推理步数 |

---

## 11. 当前可以汇报的结论

可以汇报：

1. 我们已经构建了 document-level event graph，并从中采样 Easy / Medium / Hard 路径。
2. 我们的方法不是自由生成，而是把 event path 作为结构化约束注入问题生成。
3. 在固定 300 条样本上，PathQG-HardAware 的 pass rate 最高：62.0%，高于 ZeroShot、ICL 和 SelfRefine。
4. Ablation 显示 context 和 specific event path 都有贡献。
5. 当前 ICLTargetQG 的 solver_correct 最高，说明 ours 不能主打通用质量胜出。
6. 已通过 trace-first 排查定位到上游质量问题：fixed-window answer phrase 截断、Medium path 方向构造风险、path length 与真实推理步数不一致。
7. 当前最核心的待解决问题是：修复 answer phrase extraction、修复 directed path sampling、补全全链路 trace、重新跑 path judge 和 QG。

不建议汇报成：

1. “四维难度评分有效区分难度。”目前证据不足。
2. “我们全面超过 ICL baseline。”当前数据不支持。
3. “Hard 问题稳定需要多跳推理。”当前 hard degraded 仍有 33.33%。
4. “trigger 就是 gold answer。”trigger 只能作为事件锚点。

---

## 12. 自动更新协议

每次实验或代码改动后，按以下顺序更新本文件：

### 12.1 Trace-first debugging rule

后续任何质量问题都必须优先依据 trace log 定位，不允许只根据汇总表猜测原因。

排查顺序固定为：

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

如果某个结论不能追溯到具体 trace item，不应写入结果结论或论文叙述。

1. 如果改了路径采样或 prefilter：
   - 更新第 2 节。
   - 来源：`graph_building_report.json`、`path_sampling_report.json`、`path_prefilter_report.json`。

2. 如果改了 baseline 或主实验：
   - 更新第 4、5 节。
   - 来源：`review-stage/RESULTS_SUMMARY.md` 和对应 evaluated jsonl。

3. 如果改了 quality filter / judge：
   - 更新第 6、8 节。
   - 来源：`quality_pilot_* / filter_report.json`、manual review 文件。

4. 如果改了 LLM path judge：
   - 更新第 6、8 节。
   - 来源：`path_judge_pilot_* / path_judge_report.json`、`path_judge_trace.jsonl`。

5. 如果改了指标定义：
   - 更新第 9、10 节。
   - 必须检查是否有已发表论文支持。
   - 如果是自定义公式，只能标为 diagnostic / internal。

6. 每次更新都要改：
   - 顶部“最后更新”日期。
   - “已解决的问题”。
   - “待解决的问题”。
   - “当前可以汇报的结论”。

建议以后对 Claude Code 使用这个指令：

```text
请读取 review-stage/PROJECT_STATUS.md、review-stage/RESULTS_SUMMARY.md 以及最新输出目录中的 filter_report.json，
把 PROJECT_STATUS.md 中的最新实验数据、已解决问题、待解决问题和可汇报结论同步更新。
不要改变指标定义，除非同时给出已有论文依据。
```

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
- **Fix:** Changed to forward-only lookup. Unknown edges marked as `"UNKNOWN"` and propagated to trace.
- **Validation:** After path_sampler fix, `"UNKNOWN"` count should be near zero.

---

## 15. Trace Protocol

All pipeline stages now use `trace_utils.py` (`event_qg/src/trace_utils.py`) for consistent full-chain tracing.

### Trace record structure:
- `raw_source`: doc_id, event_count, relation_count from MAVEN raw data
- `graph_stage`: nodes, edges, isolated_nodes from graph_builder
- `path_sampling`: difficulty, hop_count, path_events (with offset), relation_subtypes, difficulty_score
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
