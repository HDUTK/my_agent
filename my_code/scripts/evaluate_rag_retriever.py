    #!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于 Ragas 检索评测脚本，入库完毕后可以使用该脚本进行评测
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/10/14 16:04
version: V1.0
Target Python: 
"""

import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# 导入现有的企业级检索管线与配置
from RAG.rag_retriever import HybridRerankRetriever
from config.agent_config import MODEL_REGISTRY
from config.system_config import DEVICE


def run_ragas_retrieval_evaluation():
    # 1. 加载评测数据集
    with open("eval_dataset.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 2. 启动你已写好的混合检索引擎（BGE-M3 + BM25 + BGE-Reranker）
    retriever = HybridRerankRetriever()

    questions = []
    ground_truths = []
    retrieved_contexts = []

    print("🚀 正在执行检索并收集 Context 片段...")
    for item in test_data:
        q = item["question"]
        gt = item["ground_truth"]

        # 调用你的混合重排检索器，获取 Top-3 的最终精选文档
        docs = retriever.search(q, score_threshold=0.0, top_k=3)
        # 提取文档正文列表
        contexts = [doc.page_content for doc in docs]

        questions.append(q)
        ground_truths.append(gt)
        retrieved_contexts.append(contexts)

    # 3. 构建 Ragas 所需的 HuggingFace Dataset 格式
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "contexts": retrieved_contexts,
        "ground_truth": ground_truths
    })

    # 4. 配置裁判 LLM 和 Embedding（用于 Ragas 内部计算相似度）
    # 裁判模型建议使用结构化遵从度高的模型（本地使用 qwen2.5:7b/14b 或云端 API）
    judge_llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    judge_embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_REGISTRY["bge_m3"]["model_name"],
        model_kwargs={'device': DEVICE}
    )

    print("⚖️ Ragas 正在对检索结果进行多维打分（Context Recall & Context Precision）...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings
    )

    # 5. 输出量化评估报表
    df_results = result.to_pandas()
    print("\n" + "=" * 60)
    print("📊 混合检索器 Ragas 评测成绩单：")
    print(f"平均上下文精确度 (Context Precision): {result['context_precision']:.4f}")
    print(f"平均上下文召回率 (Context Recall)   : {result['context_recall']:.4f}")
    print("=" * 60)

    # 导出到 CSV 逐条分析排查差评项
    df_results.to_csv("ragas_retrieval_report.csv", index=False, encoding="utf-8-sig")
    print("📁 详细评测记录已保存至: ragas_retrieval_report.csv")


if __name__ == "__main__":
    run_ragas_retrieval_evaluation()

