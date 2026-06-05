#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于纯本地混合检索器 (作为 MCP 的底层数据工具)
只负责找资料，不负责生成回答！
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/06/04 15:32
version: V1.0
Target Python: 
"""

from RAG.rag_retriever import HybridRerankRetriever


class LocalKnowledgeExpert:
    def __init__(self):
        print("\n" + "=" * 50)
        print(" 正在后台启动【本地知识检索器】(仅检索无生成)...")
        # 仅唤醒双路检索与重排模型，不加载任何大语言模型！
        self.retriever = HybridRerankRetriever()
        print("【本地知识检索器】就绪，等待 MCP 索要数据！")
        print("=" * 50 + "\n")

    def retrieve_docs(self, user_query: str, threshold: float = 0.3) -> str:
        """
        核心对外接口：只负责捞出及格的 Top 3 原文，拼成字符串返回。
        threshold 参数：低于此分数的文档将被直接判定为不相关并丢弃。
        """
        best_docs = self.retriever.search(user_query, score_threshold=threshold, top_k=3)

        if not best_docs:
            return "⚠️ [系统提示]：本地数据库中未检索到任何与此问题相关的记录。请使用你的知识进行回答，并向用户明确说明：本地知识库无相关记录。"

        # 把捞出来的原文拼在一起
        result_text = "【以下是本地数据库检索到的参考资料】\n"
        for i, doc in enumerate(best_docs):
            result_text += f"\n--- 资料片段 {i + 1} (来源: {doc.metadata.get('source', '未知')}) ---\n"
            result_text += doc.page_content + "\n"

        return result_text

