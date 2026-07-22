#!/usr/bin/env python
"""
Ablation Evaluation Script - Acceptance Criterion 4

Runs a fixed evaluation set through Naive (no rewrite, no rerank) and
Advanced (rewrite + rerank) pipelines, then computes Hit@K and MRR.

Usage:
  cd E:\rag\langchain-rag
  $env:PYTHONPATH = "."
  python eval_ablation.py

Output: comparison table + JSON report saved to eval_results.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# --- Load .env ---
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

# --- Fixed evaluation dataset ---
# Each entry: question + list of keyword sets that MUST appear in a relevant chunk
# A hit is counted if any keyword set is fully present in the retrieved chunk text
EVAL_DATASET = [
    {
        "question": "归母净利润",
        "answer_keywords": ["归属于母公司", "净利润"],
    },
    {
        "question": "半年利润",
        "answer_keywords": ["归属于母公司", "净利润"],
    },
    {
        "question": "扣非利润",
        "answer_keywords": ["非经常性损益", "净利润"],
    },
    {
        "question": "营业收入",
        "answer_keywords": ["营业收入"],
    },
    {
        "question": "存货",
        "answer_keywords": ["存货"],
    },
    {
        "question": "经营活动产生的现金流量净额",
        "answer_keywords": ["经营活动", "现金流量", "净额"],
    },
    {
        "question": "总资产",
        "answer_keywords": ["总资产"],
    },
    {
        "question": "基本每股收益",
        "answer_keywords": ["基本每股收益"],
    },
    {
        "question": "加权平均净资产收益率",
        "answer_keywords": ["加权平均", "净资产收益率"],
    },
    {
        "question": "茅台酒收入",
        "answer_keywords": ["茅台酒", "收入"],
    },
]


def is_relevant(chunk_text: str, keyword_sets: list[list[str]]) -> bool:
    """Check if a chunk is relevant: all keywords in any keyword set must be present."""
    lower = chunk_text.lower()
    for kw_set in keyword_sets:
        if all(kw.lower() in lower for kw in kw_set):
            return True
    return False


def compute_hit_at_k(hits_relevance: list[bool], k: int) -> float:
    """Hit@K: 1.0 if any of the top-K hits is relevant."""
    return 1.0 if any(hits_relevance[:k]) else 0.0


def compute_mrr(hits_relevance: list[bool]) -> float:
    """MRR: 1/rank of first relevant hit."""
    for i, rel in enumerate(hits_relevance):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


async def run_evaluation():
    from src.api.real_rag_service import RealRAGService
    from src.api.api_models import RAGRequest, RAGMode

    service = RealRAGService()

    configs = {
        "Naive (no rewrite, no rerank)": {
            "rewrite_enabled": False,
            "rerank_enabled": False,
        },
        "Advanced (rewrite + rerank)": {
            "rewrite_enabled": True,
            "rerank_enabled": True,
        },
        "Rewrite only": {
            "rewrite_enabled": True,
            "rerank_enabled": False,
        },
        "Rerank only": {
            "rewrite_enabled": False,
            "rerank_enabled": True,
        },
    }

    results = {}
    K_VALUES = [1, 3, 5, 10]

    for config_name, opts in configs.items():
        print(f"\n{'='*60}")
        print(f"  Config: {config_name}")
        print(f"  Options: {opts}")
        print(f"{'='*60}")

        all_hit_at_k = {k: [] for k in K_VALUES}
        all_mrr = []
        total_time = 0
        config_details = []

        for i, item in enumerate(EVAL_DATASET):
            question = item["question"]
            keywords = item["answer_keywords"]

            request = RAGRequest(
                query=question,
                kb_id="default",
                mode=RAGMode.ADVANCED,
                options={**opts, "top_k": 10, "rerank_top_k": 5},
            )

            t0 = time.time()
            try:
                result = await service.query(request)
            except Exception as e:
                print(f"  [{i+1}/{len(EVAL_DATASET)}] ERROR: {e}")
                for k in K_VALUES:
                    all_hit_at_k[k].append(0.0)
                all_mrr.append(0.0)
                continue
            elapsed = time.time() - t0
            total_time += elapsed

            # Get pre-rerank hits from detail (these are the Top-10 retrieval results)
            detail = result.usage.get("detail", {})
            pre_hits = detail.get("pre_rerank_top_k", [])

            # Evaluate relevance for each retrieved chunk
            # We use the pre-rerank Top-10 for Hit@K evaluation
            hits_relevance = []
            for h in pre_hits:
                hits_relevance.append(is_relevant(h.get("text", ""), keywords))

            # Compute metrics
            hit_k = {k: compute_hit_at_k(hits_relevance, k) for k in K_VALUES}
            mrr = compute_mrr(hits_relevance)

            for k in K_VALUES:
                all_hit_at_k[k].append(hit_k[k])
            all_mrr.append(mrr)

            status = "HIT" if any(hits_relevance) else "MISS"
            print(f"  [{i+1}/{len(EVAL_DATASET)}] Q: {question:30s} | {status} | MRR={mrr:.3f} | {elapsed:.1f}s")

            config_details.append({
                "question": question,
                "hit_at_1": hit_k[1],
                "hit_at_3": hit_k[3],
                "hit_at_5": hit_k[5],
                "hit_at_10": hit_k[10],
                "mrr": mrr,
                "elapsed_s": round(elapsed, 2),
            })

        # Aggregate
        avg_hit = {k: sum(v) / len(v) if v else 0 for k, v in all_hit_at_k.items()}
        avg_mrr = sum(all_mrr) / len(all_mrr) if all_mrr else 0

        results[config_name] = {
            "avg_hit_at_1": round(avg_hit[1], 4),
            "avg_hit_at_3": round(avg_hit[3], 4),
            "avg_hit_at_5": round(avg_hit[5], 4),
            "avg_hit_at_10": round(avg_hit[10], 4),
            "avg_mrr": round(avg_mrr, 4),
            "total_time_s": round(total_time, 1),
            "details": config_details,
        }

        print(f"\n  Summary:")
        for k in K_VALUES:
            print(f"    Hit@{k:2d}: {avg_hit[k]:.4f}")
        print(f"    MRR:    {avg_mrr:.4f}")
        print(f"    Time:   {total_time:.1f}s")

    # --- Print comparison table ---
    print(f"\n\n{'='*70}")
    print("  COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"{'Config':<35} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR':>7}"
    print(header)
    print("-" * 70)
    for name, metrics in results.items():
        row = f"{name:<35} {metrics['avg_hit_at_1']:>7.4f} {metrics['avg_hit_at_3']:>7.4f} {metrics['avg_hit_at_5']:>7.4f} {metrics['avg_hit_at_10']:>7.4f} {metrics['avg_mrr']:>7.4f}"
        print(row)
    print("-" * 70)

    # --- Check acceptance criterion 4 ---
    naive_metrics = results.get("Naive (no rewrite, no rerank)", {})
    adv_metrics = results.get("Advanced (rewrite + rerank)", {})

    print("\n  Acceptance Criterion 4 Check:")
    better_count = 0
    for metric_name in ["avg_hit_at_1", "avg_hit_at_3", "avg_hit_at_5", "avg_hit_at_10", "avg_mrr"]:
        n_val = naive_metrics.get(metric_name, 0)
        a_val = adv_metrics.get(metric_name, 0)
        status = "PASS" if a_val > n_val else ("TIE" if a_val == n_val else "FAIL")
        if a_val > n_val:
            better_count += 1
        print(f"    {metric_name:>16}: Naive={n_val:.4f}  Advanced={a_val:.4f}  [{status}]")

    if better_count >= 1:
        print(f"\n  RESULT: PASS - Advanced is better in {better_count} metric(s)")
    else:
        print(f"\n  RESULT: FAIL - Advanced is not better in any metric")

    # --- Save JSON report ---
    report_path = Path(__file__).resolve().parent / "eval_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())