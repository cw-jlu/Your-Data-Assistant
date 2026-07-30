# Harness `exp_128_logicrag_multistage` — 5-Stage Sub-Agent Pipeline 設計書

> exp_122 (= v5, LB 0.5789) から構造的に「がっつり変更」する案。
> LogicRAG / E-SQL / MAC-SQL / StructRAG / CoVe の合成。
> 想定 v7 候補、目標 local 0.84-0.89 → LB 0.63-0.69 (= 当時 top 0.6311 超え)。

---

## 0. なぜ変えるか — exp_122 (= 4-phase 単一 agent) の構造的限界

| 軸 | exp_122 の現状 | 問題 |
|---|---|---|
| Context | plan / explore / answer / verify が **同一文脈** | 蓄積でノイズ混入、長文脈で重要情報埋没 |
| Question Interpretation | 暗黙 (= 質問は raw のまま plan に渡る) | 50-task 手解きで判明: "lowest cost" = MIN(expense.cost) vs SUM(budget.spent) のような **意図解釈の罠** に弱い |
| Aggregation | 暗黙 (= explore 結果が raw で answer に流れる) | 不要な探索ログが answer 文脈を汚す、構造化された中間表現がない |
| Verify | 同一 agent | self-reinforcement bias、CoVe で実証された「独立 verify が bias を打ち消す」効果なし |
| 50-task patterns | knowledge.md 末尾 hint のみ | Stage 1 で entity 列挙させれば ID anchor 結合 / atom_id 数値ソート / 単価判定を早期検出可能 |

---

## 1. 5 段階パイプライン全体像

```
[Question + raw context]
        │
   ┌────▼─────────────────────────┐
   │ Stage 1: Question Interpreter│  Sub-Agent A (= context isolated)
   │   - リフレーズ (entity 明示化)│
   │   - 必要 col/table/id 列挙   │
   │   - sub-Q 分解 (1〜N)        │
   └────┬─────────────────────────┘
        │ Question Plan (= 固定 JSON)
   ┌────▼─────────────────────────┐
   │ Stage 2: Explorer            │  Sub-Agent B (= tool-using ReAct)
   │   - sub-Q ごとに schema/sample│
   │   - prose section 抽出       │
   │   - tools: read_table, sql,  │
   │     grep_doc, extract_struct │
   └────┬─────────────────────────┘
        │ Exploration Log (= raw)
   ┌────▼─────────────────────────┐
   │ Stage 3: Aggregator          │  Sub-Agent C (= structurer)
   │   - log を Structured Brief に│
   │   - 固定 schema JSON         │
   └────┬─────────────────────────┘
        │ Structured Brief (= 固定 JSON)
   ┌────▼─────────────────────────┐
   │ Stage 4: Answer Generator    │  Adaptive Vote (= 既存)
   │   - Brief のみ context       │
   │   - SQL or 自由形式 N candi  │
   └────┬─────────────────────────┘
        │ Candidate Answers (N=3)
   ┌────▼─────────────────────────┐
   │ Stage 5: Independent Verifier│  Sub-Agent D (= CoVe-style)
   │   - verify Q 群を独立生成    │
   │   - 各 verify Q を独立回答   │
   │   - mismatch → revise loop   │
   └────┬─────────────────────────┘
        ▼
[Final Answer + adaptive_vote]
```

**Sub-agent 分離の意義**: 各 stage で context を新規に組み立て、不要なログを次段に持ち越さない。MARS-SQL (77.84% dev) / MAC-SQL の主因。

---

## 2. Stage 1 — Question Interpreter

### 2.1 目的
質問を構造化して、後続段階が「何を探すか」「何を返すか」を曖昧さなく伝える。

### 2.2 Input
- `question` (str): 元質問
- `db_schema_preview` (str): 全 table の name + columns + dtype のみ (= 1000 token 以内)
- `prose_doc_toc` (str): doc/*.md の見出し階層のみ (= 200 token 以内)
- `knowledge_md` (str): knowledge.md 全文 (= 通常 1000-2000 token)

### 2.3 System Prompt (= 骨子)

```
You are the Question Interpreter for a data-analysis agent.
Your job: turn the user's question into a structured plan so the next stage can search efficiently.

OUTPUT — strictly valid JSON, no prose around it:
{
  "rephrased": "<one sentence, entity-explicit rephrasing>",
  "intent": "<one of: lookup | count | sum | avg | min/max | ratio | filter+list | compare | rank | other>",
  "result_shape": "<one of: scalar | single_row | multi_row | percentage>",
  "result_columns": ["<col1>", ...],
  "entities": [
    {"name": "<entity name>", "table": "<table>", "column": "<column>", "value_hint": "<exact value or pattern>"},
    ...
  ],
  "sub_questions": [
    {"id": "Q1", "text": "<atomic sub-question>", "depends_on": []},
    {"id": "Q2", "text": "...", "depends_on": ["Q1"]},
    ...
  ],
  "verification_targets": ["<what to verify in stage 5>", ...],
  "pitfall_hints": ["<which 50-task pattern applies, if any>"]
}

RULES:
- "rephrased": Make ALL implicit entities explicit. "lowest cost" → "MIN of expense.cost over rows linked to the target event"
- "entities": List every concrete value/column the question references, with the BIRD-evidence-style hint
- "sub_questions": 1-5 atomic steps. Each step is a single SELECT or fact lookup
- "pitfall_hints": Match against this list (one of):
  - "amount_vs_spent": "amount" = budgeted, "spent" = actual
  - "per_unit_price": "per unit" = Price/Amount, NOT Price alone
  - "case_insensitive_title": exact title string may have case mismatch
  - "atom_id_numeric_sort": atom_id like TR000_4 — sort by numeric suffix
  - "id_anchor_prose": same entity appears in multiple prose sections; join via ID
  - "directional_csv": connected.csv / hero_power.csv has both directions — DISTINCT bond_id
  - "subset_data": _1k.db or partial prose may be subset; answer from available rows
```

### 2.4 Output 例 (task_25: "Which event has the lowest cost?")

```json
{
  "rephrased": "Find the event_name(s) of the event linked to the row(s) with the minimum cost in the expense table",
  "intent": "min/max",
  "result_shape": "multi_row",
  "result_columns": ["event_name"],
  "entities": [
    {"name": "cost", "table": "expense", "column": "cost", "value_hint": "numeric, may have ties"},
    {"name": "event", "table": "event", "column": "event_name", "value_hint": "join via expense.link_to_budget → budget.link_to_event"}
  ],
  "sub_questions": [
    {"id": "Q1", "text": "What is the MIN(cost) across all expense rows?", "depends_on": []},
    {"id": "Q2", "text": "Which expense_ids have cost = MIN found in Q1?", "depends_on": ["Q1"]},
    {"id": "Q3", "text": "For those expense rows, what is the event_name via budget → event JOIN?", "depends_on": ["Q2"]}
  ],
  "verification_targets": [
    "Verify cost source is expense.cost, NOT budget.spent",
    "Verify ties in MIN are all included"
  ],
  "pitfall_hints": ["amount_vs_spent"]
}
```

### 2.5 Failure Modes
- JSON 不正 → fallback to "rephrased = question, sub_questions = [{id:Q1, text: question}]"
- entities 空 → 警告だけで続行 (= Stage 2 が自前で schema 探す)

---

## 3. Stage 2 — Explorer (ReAct + tools)

### 3.1 目的
Question Plan の sub_questions を順番に解いて、必要な事実を集める。

### 3.2 Input
- Question Plan (Stage 1 出力)
- DB 接続 / file system access (= tools)

### 3.3 System Prompt (= 骨子)

```
You are the Explorer. Solve each sub-question in the Question Plan using tools.

For each sub_question (Q1, Q2, ...):
1. Issue minimum tool calls to gather the fact
2. Record the answer as JSON: {"sub_question_id": "Q1", "answer": <value>, "evidence": "<SQL or doc excerpt>", "confidence": "high|medium|low"}
3. Move to next sub_question

Tools available:
- execute_sql(sql) — DuckDB unified data layer (= csv/json/sqlite as views)
- read_doc_section(file, heading) — fetch a prose section
- grep_doc(pattern, file) — find lines matching pattern
- extract_structured(doc_text, schema) — LLM sub-call to extract JSON from prose
- (... existing tools ...)

DO NOT generate the final SQL or answer here. ONLY gather facts.

OUTPUT — JSON array of fact records, one per sub_question.
```

### 3.4 Output 例 (task_25)

```json
[
  {"sub_question_id": "Q1", "answer": 6.0, "evidence": "SELECT MIN(cost) FROM expense WHERE approved='true'", "confidence": "high"},
  {"sub_question_id": "Q2", "answer": ["e1", "e2", "e3"], "evidence": "SELECT expense_id FROM expense WHERE cost=6.0", "confidence": "high"},
  {"sub_question_id": "Q3", "answer": ["November Speaker", "October Speaker", "September Speaker"], "evidence": "JOIN expense->budget->event", "confidence": "high"}
]
```

### 3.5 Failure Modes
- tool_call_failed → low confidence + 部分結果で続行
- sub_question 未解決 → "unresolved": true + 推測値

---

## 4. Stage 3 — Aggregator

### 4.1 目的
Stage 2 の生ログを「Answer に必要な情報だけ」に圧縮した固定スキーマ JSON にする。

### 4.2 Input
- Question Plan
- Exploration Log (Stage 2 出力)

### 4.3 System Prompt

```
You are the Aggregator. Convert the raw exploration log into a Structured Brief.

OUTPUT — strictly valid JSON:
{
  "rephrased_question": "<from Stage 1>",
  "expected_result_shape": "<scalar | single_row | multi_row | percentage>",
  "expected_columns": [...],
  "relevant_tables": {"<table>": ["<col1>", "<col2>", ...]},
  "key_values": {"<entity>": "<exact value>"},
  "sub_question_answers": [
    {"id": "Q1", "answer": <...>, "evidence_summary": "<1-2 sentence>"}
  ],
  "join_path": "<table1.col1 = table2.col1 → table2.col2 = table3.col2>",
  "filter_predicates": ["<column op value>", ...],
  "ordering": "<asc/desc on column, if any>",
  "limit_or_distinct": "<LIMIT N | DISTINCT | none>",
  "unresolved_assumptions": ["<list of remaining ambiguities>"],
  "verification_targets": [...]
}

RULES:
- Be ruthless: do NOT include exploration noise (= failed tool calls, unrelated lookups)
- "filter_predicates": Translate qualitative constraints to SQL-ready predicates
- "join_path": Explicit, in the format "tableA.colX = tableB.colX → tableB.colY = tableC.colY"
- "unresolved_assumptions": If Stage 2 had low-confidence answers, list them
```

### 4.4 Output 例 (task_25)

```json
{
  "rephrased_question": "Find event_name(s) for rows with MIN(expense.cost)",
  "expected_result_shape": "multi_row",
  "expected_columns": ["event_name"],
  "relevant_tables": {"expense": ["expense_id", "cost", "link_to_budget"], "budget": ["budget_id", "link_to_event"], "event": ["event_id", "event_name"]},
  "key_values": {"MIN_cost": 6.0},
  "sub_question_answers": [
    {"id": "Q1", "answer": 6.0, "evidence_summary": "MIN(cost) from expense"},
    {"id": "Q3", "answer": ["November Speaker", "October Speaker", "September Speaker"], "evidence_summary": "3-way tie via JOIN"}
  ],
  "join_path": "expense.link_to_budget = budget.budget_id → budget.link_to_event = event.event_id",
  "filter_predicates": ["expense.cost = (SELECT MIN(cost) FROM expense)"],
  "ordering": "asc on cost (= equivalent to MIN filter)",
  "limit_or_distinct": "DISTINCT event_name",
  "unresolved_assumptions": [],
  "verification_targets": ["confirm 3-way tie is correct (= MIN may not be unique)"]
}
```

---

## 5. Stage 4 — Answer Generator (= 既存 adaptive_vote 維持)

### 5.1 目的
Structured Brief **のみ** を入力に最終 SQL/answer を生成。N=3 attempts + adaptive_vote。

### 5.2 Input
- Structured Brief (= **唯一の文脈**、raw question / exploration log は含まない)

### 5.3 System Prompt

```
You are the Answer Generator. Generate the final SQL based on the Structured Brief.

CONSTRAINTS:
- ONLY use the Brief — no further exploration
- Output: a single SQL query or a deterministic answer
- Match the expected_result_shape and expected_columns exactly
- If unresolved_assumptions is non-empty, choose the most conservative interpretation

OUTPUT format: ```sql\n<SQL>\n```
```

### 5.4 Adaptive Vote (= 既存 exp_122 から継承)
- 3 attempt × T=0.6 で SQL 候補生成
- subset-pick > numeric-convergent majority > union の優先順
- 2dp normalization (= v5 で追加)
- prose-file SQL guard (= v5 で追加)

---

## 6. Stage 5 — Independent Verifier (CoVe-style)

### 6.1 目的
Stage 4 候補に対して、**Stage 4 と独立な context** で verify 質問群を生成・回答。bias を打ち消す。

### 6.2 Input
- 元 question (= raw)
- Structured Brief
- Candidate Answer (from Stage 4)

### 6.3 System Prompt

```
You are an Independent Verifier. Verify a candidate answer using fresh queries.

PROCESS:
1. Read the question and candidate answer
2. Generate 2-4 verification questions that, if all answered consistently, would confirm the candidate
3. Answer each verification question INDEPENDENTLY using tools (= NOT by re-deriving from the candidate)
4. Compare candidate vs verification answers
5. Output: pass | revise | reject

OUTPUT — JSON:
{
  "verification_questions": [
    {"q": "<verify Q>", "independent_answer": <...>, "matches_candidate": true|false}
  ],
  "verdict": "pass" | "revise" | "reject",
  "suggested_revision": "<if revise: brief description>"
}

VERDICT RULES:
- pass: All verification answers match candidate
- revise: Most match but minor discrepancy (e.g., row order, decimal precision)
- reject: Fundamental mismatch (= different value, wrong columns)

REVISE LOOP: If revise, Stage 4 re-runs once with the suggested_revision appended to the Brief.
MAX REVISE: 1 (= 2 total Answer attempts)
```

### 6.4 Output 例 (task_25)

```json
{
  "verification_questions": [
    {"q": "What is COUNT(DISTINCT event_name) where event has an expense with cost=6.0?", "independent_answer": 3, "matches_candidate": true},
    {"q": "What is MIN(cost) in expense table?", "independent_answer": 6.0, "matches_candidate": true}
  ],
  "verdict": "pass",
  "suggested_revision": null
}
```

---

## 7. 全 JSON スキーマまとめ

| Stage | 出力 schema | 厳格度 |
|---|---|---|
| 1 | Question Plan | strict (= validator で reject) |
| 2 | Exploration Log (array of facts) | lenient (= 部分許容) |
| 3 | Structured Brief | strict |
| 4 | SQL string | format check のみ |
| 5 | Verification Verdict | strict |

JSON validator は jsonschema lib + Pydantic で実装、不正なら 1 回 retry (= 同 prompt + "previous output failed validation: ..." prefix)。

---

## 8. Failure Mode Table

| 段階 | 起こりうる失敗 | 対処 |
|---|---|---|
| 1 | JSON 不正 | fallback to identity rephrasing |
| 1 | sub_questions = [] | Stage 2 が question raw で動く |
| 2 | tool_call timeout (per call) | low confidence + 続行 |
| 2 | 全 sub_Q 未解決 | Stage 3 が空 brief 生成、Stage 4 が直接 question から SQL |
| 3 | brief schema 違反 | retry 1 回 |
| 4 | SQL parse error | adaptive_vote の既存 fallback |
| 5 | verify_questions = [] | verdict = pass (= verify スキップ) |
| 5 | revise loop で再 reject | candidate を採用 (= no infinite loop) |

---

## 9. Smoke Test Plan

### 9.1 5 task で動作確認 (= POC stage)
- task_25 (= "amount_vs_spent" pitfall) — Stage 1 で pitfall_hint = "amount_vs_spent" 出力できるか
- task_180 (= "per_unit_price") — Stage 1 で Price/Amount の単価判定 hint 出るか
- task_344 (= "subset_data") — Stage 1 で subset 認識、Stage 5 で reject せず最善 answer 採用
- task_379 (= "atom_id_numeric_sort") — Stage 3 の ordering で数値ソート明記
- task_396 (= "id_anchor_prose") — Stage 2 で extract_structured 利用、Stage 3 で 3-section 結合

### 9.2 期待される動作
- 各 stage の JSON が schema 合致
- 5 task のうち少なくとも 3 task で final answer が gold 一致
- 1 run 完了時間 ≤ 75 min × 1.5 = 110 min (= sub-agent 増による遅延許容)

---

## 10. 実装ステップ (= prompt 確定後)

| ステップ | 成果物 | 所要 |
|---|---|---|
| 1. prompt 設計確定 (= 本ドキュメント) | docs/HARNESS_EXP_128.md | 完了 |
| 2. POC: 5 stage を `scripts/test_logicrag_poc.py` で個別検証 | スクリプト + 5-task 結果 | 1-2 day |
| 3. `src/experiments/exp_128_logicrag_multistage/` 本実装 | runner.py + 5 phase modules + config | 2-3 day |
| 4. smoke (= 50 task n=1) | run_001 evaluation.json | 2-3 hr |
| 5. n=3 variance | replication summary | 5-6 hr |
| 6. variance OK なら v7 候補 → 提出 | manifest + Docker | 1 day |

---

## 11. 期待効果 (= 根拠付き)

| 段階 | 根拠論文 | ΔEX 期待 |
|---|---|---|
| Stage 1 (rephrase + sub-Q) | E-SQL +5% EX (arXiv 2409.16751), MAC-SQL decomposer | +0.02〜0.04 |
| Stage 3 (Structured Brief) | StructRAG (arXiv 2410.08815), RAPTOR | +0.01〜0.03 |
| Stage 5 (Independent Verify) | CoVe (arXiv 2309.11495), CHESS NL unit test, MARS-SQL Validation | +0.01〜0.02 |
| 50-task pitfall hints (Stage 1) | 50 task 手解き report 直接対応 | +0.01〜0.02 |
| **合計** | | **+0.05〜0.11 local** |

→ local 0.85〜0.91 → LB 0.64〜0.71 = **top 0.6311 圏内〜上回り**。

---

## 12. Risk & 棄却した代替案

### Risk
- **Speed**: sub-agent × 4 → 1 task に LLM call 6-10 回、75 min/run → 3-5 hr/run の可能性 → workers=20 でカバー、step budget 圧縮で対応
- **Prompt 品質依存**: 5 stage 各々の prompt が品質を決める → POC で 5 task 反復改善
- **Inter-stage 情報落ち**: Stage 3 が重要事実を捨てる risk → "unresolved_assumptions" 明示でカバー
- **smoke 失敗時の debug**: どの stage が壊れたか切り分け → 各 stage 出力を artifact に dump

### 棄却した代替案
- **LogicRAG DAG 完全実装**: 推論時 DAG 構築は overkill (= DABench は 1-3 hop が大半、50-task 手解き確認済)
- **MARS-SQL の RL Validation**: RL training は競技範囲外
- **CHASE-SQL の 3 candidate multi-gen**: adaptive_vote が類似機能を既に提供
- **LogicRAG topological linearization**: sub_question の `depends_on` で簡易対応、formal topo sort は不要

---

## 13. exp_122 (= v5) と完全互換に保つ部分

- Model: qwen3.5-35b-a3b (= sampling パラメータ同じ)
- DuckDB unified data layer (= csv/json/sqlite views)
- column_auditor / filter_auditor (= 既存)
- adaptive_vote (= Stage 4 内で再利用)
- 2dp normalization + prose-file SQL guard (= v5 fix を継承)

---

## 参考文献

- LogicRAG (arXiv 2508.06105) — DAG-based decompose + topological reasoning
- E-SQL (arXiv 2409.16751) — Question enrichment, BIRD test 66.29%
- MAC-SQL (arXiv 2312.11242) — Selector + Decomposer + Refiner
- StructRAG (arXiv 2410.08815) — Router + Structurizer + Utilizer
- CoVe (arXiv 2309.11495) — Chain-of-Verification
- CHESS (arXiv 2405.16755) — IR + Schema Sel + Candidate + NL Unit Test
- MARS-SQL (arXiv 2511.01008) — ReAct + Validation, BIRD dev 77.84%
- Self-RAG (arXiv 2310.11511) — retrieve + critique tokens
- RAPTOR (arXiv 2401.18059) — recursive cluster + summary tree
