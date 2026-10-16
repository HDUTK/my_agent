# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于 原生 RAG 检索质量评测器 (100% 还原 Ragas Context Recall & Precision 逻辑)
彻底脱离官方 ragas 库与 LangChain 接口依赖，零报错运行。
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/10/14 16:04
version: V1.0
Target Python: 
"""

import warnings

# 🌟 过滤 jieba 引起的 pkg_resources 弃用警告
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import os
import re
import json
import requests
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 仅引入本地现成的检索器核心
from RAG.rag_retriever import HybridRerankRetriever
from config.agent_config import LLM_CONFIG

# =========================================================================
# 🌟 自动加载项目根目录的 .env 文件
# =========================================================================
# 无论在哪个目录下运行该脚本，都能精准找到项目根目录下的 .env
env_path = find_dotenv(usecwd=True)
if not env_path:
    # 备用方案：自动寻找当前脚本上级目录中的 .env
    env_path = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(dotenv_path=env_path)

# =========================================================================
# 🌟 裁判配置中心（支持 本地 Ollama 与 云端 API 一键切换）
# =========================================================================
# 模式选择："local" (本地 Ollama，默认) | "cloud" (云端大模型 API)
JUDGE_MODE = "cloud"
EVAL_TOP_K = 5  # 设置评测时评估的前 K 个切片数量（改为 5）

# 1. 本地 Ollama 配置 (需提前 ollama run deepseek-r1:8b/qwen3:8b 或其它模型)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

# 2. 云端 API 配置 (支持 DeepSeek、通义千问等 OpenAI 兼容接口)
CLOUD_MODEL = "qwen"  # 通义千问: qwen.
if CLOUD_MODEL == "qwen":
    API_KEY_NAME = "DASHSCOPE"
elif CLOUD_MODEL == "gpt":
    API_KEY_NAME = "OPENAI"
else:
    API_KEY_NAME = CLOUD_MODEL.upper()
CLOUD_API_KEY = os.getenv(API_KEY_NAME + "_API_KEY")
CLOUD_BASE_URL = LLM_CONFIG["model"]["remote"][CLOUD_MODEL]["base_url"]

# 并发线程数：本地 Ollama 建议 2；云端 API 建议 5~8
MAX_WORKERS = 2 if JUDGE_MODE == "local" else 5


def request_llm_judge(prompt: str) -> str:
    """底层 HTTP 请求函数"""
    if JUDGE_MODE == "local":
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            }
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
            return resp.json().get("response", "").strip() if resp.status_code == 200 else ""
        except Exception:
            return ""

    elif JUDGE_MODE == "cloud":
        try:
            headers = {"Authorization": f"Bearer {CLOUD_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": LLM_CONFIG["model"]["remote"][CLOUD_MODEL]["type"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            resp = requests.post(f"{CLOUD_BASE_URL.rstrip('/')}/chat/completions", headers=headers, json=payload,
                                 timeout=60)
            return resp.json()["choices"][0]["message"]["content"].strip() if resp.status_code == 200 else ""
        except Exception:
            return ""
    return ""


def evaluate_single_sample(item: dict) -> dict:
    """
    🌟 核心优化：Prompt Fusion（将 Recall 判定与 Top-K Precision 判定合二为一）
    仅消耗 1 次 LLM 调用即可同时解析出召回率与排序精确度。
    """
    q = item["question"]
    gt = item["ground_truth"]
    contexts = item["contexts"]

    if not contexts:
        return {"question": q, "ground_truth": gt, "context_recall": 0.0, "context_precision": 0.0}

    # 格式化上下文
    formatted_contexts = ""
    for idx, ctx in enumerate(contexts, 1):
        formatted_contexts += f"\n【片段 {idx}】：\n{ctx}\n"

    # 聚合打分 Prompt
    prompt = f"""你是一名严格的 RAG 检索评测专家。请基于【问题】和【标准答案】，评定检索到的【参考片段】。

    【问题】：{q}
    【标准答案】：{gt}
    【参考片段集合】：
    {formatted_contexts}
    
    请完成以下两项评判：
    1. 上下文召回率评估：判断所有参考片段合起来，是否覆盖了【标准答案】的全部事实要点？(完全覆盖给 1.0，部分覆盖给 0.5，完全缺失给 0.0)
    2. 片段相关性标注：依次判断【片段 1】到【片段 {len(contexts)}】各自是否对回答该问题提供了有价值的事实依据（有效标 1，无关噪声标 0）。
    
    请严格按照以下格式输出你的评判（严禁输出多余文字）：
    RECALL: <填 0.0 到 1.0 之间的一个浮点数>
    RELEVANCE: <用逗号隔开的 0 或 1，例如: 1, 1, 0, 0, 1>"""

    res = request_llm_judge(prompt)

    # 1. 提取 Recall
    recall = 0.5
    match_rec = re.search(r"RECALL:\s*([0-1](?:\.\d+)?)", res, re.IGNORECASE)
    if match_rec:
        recall = float(match_rec.group(1))

    # 2. 提取 Precision 并计算 Rank-weighting
    precision = 0.0
    match_rel = re.search(r"RELEVANCE:\s*([0-1,\s]+)", res, re.IGNORECASE)
    if match_rel:
        raw_text = match_rel.group(1)
        flags = [int(x) for x in re.findall(r"[01]", raw_text)]

        relevant_chunks_count = sum(flags)
        if flags and relevant_chunks_count > 0:
            p_at_k = []
            hits = 0
            for rank, flag in enumerate(flags, 1):
                if flag == 1:
                    hits += 1
                    p_at_k.append(hits / rank)

            # 分母改为实际相关的片段数 relevant_chunks_count，而非召回总数 len(flags)
            precision = sum(p_at_k) / relevant_chunks_count

    return {
        "question": q,
        "ground_truth": gt,
        "retrieved_count": len(contexts),
        "context_recall": recall,
        "context_precision": precision
    }

def merge_consecutive_chunks(docs: list) -> list:
    """
    对 Top-K 文档进行后处理：自动合并来自同一源文件且序号连续的相邻切片，修复上下文割裂
    """
    if not docs:
        return []

    merged = []
    skip_indices = set()

    for i in range(len(docs)):
        if i in skip_indices:
            continue

        curr_doc = docs[i]
        curr_source = curr_doc.metadata.get("source")
        curr_id = curr_doc.metadata.get("chunk_id")

        combined_content = curr_doc.page_content

        # 检查后续文档中是否有紧邻的切片
        for j in range(i + 1, len(docs)):
            if j in skip_indices:
                continue
            next_doc = docs[j]
            next_source = next_doc.metadata.get("source")
            next_id = next_doc.metadata.get("chunk_id")

            # 同一来源且 ID 连续（curr_id + 1 == next_id）
            if curr_source == next_source and isinstance(curr_id, int) and next_id == curr_id + 1:
                combined_content += "\n" + next_doc.page_content
                skip_indices.add(j)
                curr_id = next_id  # 允许连续向后串联多个

        curr_doc.page_content = combined_content
        merged.append(curr_doc)

    return merged


def run_fast_rag_evaluation():
    dataset_file = "eval_dataset.json"
    if not os.path.exists(dataset_file):
        print(f"❌ 找不到测试集: {dataset_file}")
        return

    with open(dataset_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print("\n" + "=" * 60)
    print(f"🚀 启动高速 RAG 检索评测管线 (裁判模式: {JUDGE_MODE} | 并发数: {MAX_WORKERS})")
    print(f"📋 共计 {len(test_data)} 个样本需要评测")
    print("=" * 60)

    # ----------------------------------------------------
    # 阶段一：极速本地检索召回 (GPU 跑 80 个样本只需几秒)
    # ----------------------------------------------------
    print("\n⚡ [阶段 1/2] 正在执行全库多路检索召回与重排...")
    retriever = HybridRerankRetriever()
    prepared_items = []

    total_samples = len(test_data)
    for idx, item in enumerate(test_data, 1):
        q = item["question"]
        gt = item["ground_truth"]

        # 🌟 实时打印当前检索进度，flush=True 确保控制台立即刷新
        print(f"\r  🔍 [检索中 {idx}/{total_samples}] 正在重排: {q[:25]}...", end="", flush=True)

        docs = retriever.search(q, score_threshold=0.0, top_k=EVAL_TOP_K)
        prepared_items.append({
            "id": idx,
            "question": q,
            "ground_truth": gt,
            "contexts": [d.page_content for d in docs]
        })

    print(f"✅ 阶段一完成！80 条样本切片提取完毕，开始进入裁判打分...")

    # ----------------------------------------------------
    # 阶段二：多线程并发裁判评审 (请求数削减 75%)
    # ----------------------------------------------------
    print(f"\n⚖️ [阶段 2/2] 正在启动 {MAX_WORKERS} 线程并发打分...")
    results = []
    completed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(evaluate_single_sample, item): item for item in prepared_items}

        for future in as_completed(future_map):
            res = future.result()
            results.append(res)
            completed_count += 1
            print(
                f"  -> 进度 [{completed_count}/{len(prepared_items)}] 完成 | Recall: {res['context_recall']:.2f} | Precision: {res['context_precision']:.2f}")

    # ----------------------------------------------------
    # 结果汇总与导出
    # ----------------------------------------------------
    df = pd.DataFrame(results)
    avg_recall = df["context_recall"].mean()
    avg_precision = df["context_precision"].mean()

    print("\n" + "🏆" * 25 + " 最终评测成果 " + "🏆" * 25)
    print(f"  - 样本总数                            : {len(df)}")
    print(f"  - 平均上下文召回率 (Context Recall)   : {avg_recall:.4f}")
    print(f"  - 平均上下文精确度 (Context Precision): {avg_precision:.4f}")
    print("=" * 65)

    report_path = "ragas_retrieval_report.csv"
    df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"📁 详细评测报表已导出至: {report_path}\n")


def inspect_zero_score_samples(target_ids: list[int] = [53, 63, 68], top_k: int = 5):
    """
    🌟 零分盲区深度诊断函数
    针对彻底漏检（Recall=0.0 / Precision=0.0）的样本，逐条打印其问题、标准答案、
    以及混合检索器（BGE-M3 + BM25 + Reranker）实际召回的 Top-K 切片原文与来源，
    方便直接排查是知识库未收录、切片被截断还是重排被噪声挤占。
    """
    # 兼容在根目录或 my_code 目录下运行的路径搜寻
    dataset_candidates = ["eval_dataset.json", "../eval_dataset.json", "my_code/eval_dataset.json"]
    dataset_file = next((p for p in dataset_candidates if os.path.exists(p)), None)

    if not dataset_file:
        print("❌ 未找到 eval_dataset.json，请确认测试集文件路径！")
        return

    with open(dataset_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print("\n" + "🔍" * 25 + " 零分盲区切片深度诊断 " + "🔍" * 25)
    print(f"📌 待排查样本列表: {target_ids} | 观测深度: Top-{top_k}")
    print("=" * 72)

    # 初始化混合检索器
    retriever = HybridRerankRetriever()

    for sample_id in target_ids:
        idx = sample_id - 1  # 题号转为 0-based 索引
        if idx < 0 or idx >= len(test_data):
            print(f"⚠️ 样本号 #{sample_id} 超出数据集范围 (总共 {len(test_data)} 条)，跳过。")
            continue

        item = test_data[idx]
        q = item["question"]
        gt = item["ground_truth"]

        print(f"\n❌ 【排查样本 #{sample_id}】")
        print(f"❓ 【提问 Question】：\n   {q}")
        print(f"🎯 【标准答案 Ground Truth】：\n   {gt}\n")

        # 重新执行全流程混合检索召回
        docs = retriever.search(q, score_threshold=0.0, top_k=top_k)

        print(f"📦 【实际打捞出的 Top-{len(docs)} 个切片原文】:")
        if not docs:
            print("   ⚠️ 警告：检索器没有返回任何切片（可能是关键词完全未匹配且相似度为 0）！")
        else:
            for rank, doc in enumerate(docs, 1):
                # 提取元数据来源与页码
                metadata = getattr(doc, "metadata", {}) or {}
                source_path = metadata.get("source", "未知来源")
                source_name = os.path.basename(str(source_path))
                page_info = f" (第 {metadata.get('page')} 页)" if "page" in metadata else ""

                print(f"  ------------------------------------------------------------")
                print(f"  ▶ [Rank {rank}] 来源文件: {source_name}{page_info}")

                # 打印切片内容（做缩进美化）
                content_preview = doc.page_content.strip()
                indented = "\n     ".join(content_preview.splitlines())
                print(f"     {indented}")

        print("=" * 72)


if __name__ == "__main__":
    # 1. 运行完整评测（如果只想看诊断，可以把这一行先注释掉）
    # run_fast_rag_evaluation()

    # 2. 🌟 诊断第 53、63、68 题的实际打捞情况
    inspect_zero_score_samples(target_ids=[15,24,41], top_k=EVAL_TOP_K)
