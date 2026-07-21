# RAG Final Evaluation Report

## 数据集说明
- Dataset: `datasets/evaluation/rag_eval_v1.jsonl`
- Samples: 24
- Tags: {'agent_route': 5, 'ambiguous': 2, 'fact': 8, 'global': 6, 'multi_hop': 2, 'relation': 7, 'unanswerable': 2}

## 运行环境说明
- Created at (UTC): 2026-07-21T08:46:49.988251+00:00
- Python: 3.9.13
- Platform: Windows-10-10.0.26200-SP0
- Runner summary: {"naive": {"runner_type": "mock_naive", "status": "ok", "is_mock": true}, "advanced": {"runner_type": "mock_advanced", "status": "ok", "is_mock": true}, "modular": {"runner_type": "mock_modular", "status": "ok", "is_mock": true}, "graph": {"runner_type": "mock_graph", "status": "ok", "is_mock": true}, "agentic": {"runner_type": "mock_agentic", "status": "ok", "is_mock": true}}

> **Note:** Mock runners validate metrics and report schema offline. They do not represent live LLM retrieval/generation quality.

## 五模式指标对比表

| Mode | Runner | Status | Hit@1 | Hit@3 | MRR | Latency(ms) | Tokens | Citations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive | mock_naive | ok | 1.0 | 1.0 | 1.0 | 0.06675 | 245.5 | 1.0 |
| advanced | mock_advanced | ok | 1.0 | 1.0 | 1.0 | 0.063667 | 245.5 | 1.0 |
| modular | mock_modular | ok | 1.0 | 1.0 | 1.0 | 0.059708 | 245.5 | 1.0 |
| graph | mock_graph | ok | 1.0 | 1.0 | 1.0 | 0.06575 | 245.5 | 1.0 |
| agentic | mock_agentic | ok | 1.0 | 1.0 | 1.0 | 0.066 | 245.5 | 1.0 |

## 按标签对比表

| Tag | Mode | Hit@1 | Count |
| --- | --- | ---: | ---: |
| agent_route | advanced | 1.0 | 5 |
| agent_route | agentic | 1.0 | 5 |
| agent_route | graph | 1.0 | 5 |
| agent_route | modular | 1.0 | 5 |
| agent_route | naive | 1.0 | 5 |
| ambiguous | advanced | 1.0 | 2 |
| ambiguous | agentic | 1.0 | 2 |
| ambiguous | graph | 1.0 | 2 |
| ambiguous | modular | 1.0 | 2 |
| ambiguous | naive | 1.0 | 2 |
| fact | advanced | 1.0 | 8 |
| fact | agentic | 1.0 | 8 |
| fact | graph | 1.0 | 8 |
| fact | modular | 1.0 | 8 |
| fact | naive | 1.0 | 8 |
| global | advanced | 1.0 | 6 |
| global | agentic | 1.0 | 6 |
| global | graph | 1.0 | 6 |
| global | modular | 1.0 | 6 |
| global | naive | 1.0 | 6 |
| multi_hop | advanced | 1.0 | 2 |
| multi_hop | agentic | 1.0 | 2 |
| multi_hop | graph | 1.0 | 2 |
| multi_hop | modular | 1.0 | 2 |
| multi_hop | naive | 1.0 | 2 |
| relation | advanced | 1.0 | 7 |
| relation | agentic | 1.0 | 7 |
| relation | graph | 1.0 | 7 |
| relation | modular | 1.0 | 7 |
| relation | naive | 1.0 | 7 |
| unanswerable | advanced | 1.0 | 2 |
| unanswerable | agentic | 1.0 | 2 |
| unanswerable | graph | 1.0 | 2 |
| unanswerable | modular | 1.0 | 2 |
| unanswerable | naive | 1.0 | 2 |

## Naive vs Advanced 结论
Advanced relative to Naive: ΔHit@1=0.0, ΔMRR=0.0, ΔLatency=-0.003083 ms, ΔTokens=0.0.

## Modular 模块结论
Modular relative to Naive: ΔHit@1=0.0, ΔMRR=0.0, ΔLatency=-0.007042 ms, ΔTokens=0.0.

## GraphRAG 适用问题结论
Graph mode is intended for relation/multi-hop style questions. relation Hit@1=1.0, multi_hop Hit@1=1.0, graph_path_presence_rate=1.0.

## Agentic 工具选择结论
Agentic mock run tool_selection_accuracy=1.0, trajectory_success_rate=1.0, agent_route Hit@1=1.0.

## 质量/耗时/Token 权衡
Best MRR in this run: naive (1.0). Lowest token average: naive (245.5). Highest latency: naive (0.06675 ms).

## 失败样例摘要
No failed samples in this run.

## 剩余风险
- One or more modes used mock runners; quality numbers are contract/smoke checks only, not production quality estimates.
- Agentic mode used a placeholder mock runner; tool metrics do not validate a real agent workflow.

## 回归检查
- Passed: True (13/13)

## 后续改进建议
- Replace mock runners with configured RAGService runs when API keys and KB data are available.
- Add LLM-as-judge only after deterministic offline metrics remain stable.
- Extend agentic evaluation once the production agent workflow is wired to the unified runner.
