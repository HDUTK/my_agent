#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于agent内部相关配置
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/06/05 16:19
version: V1.0
Target Python: 
"""

# ==========================================
# LLM相关
# ==========================================
LLM_CONFIG = {
    "model": {
        "gemini": {
            "model_name": "gemini",
            # gemini-2.0-flash/gemini-2.5-flash/gemini-2.5-pro/gemini-3-flash-preview/
            # gemini-3-pro-preview/gemini-3.1-pro-preview/gemini-3.5-flash
            "type": "gemini-2.5-flash"
        },
        "gpt": {
            "model_name": "gpt",
            "type": "gpt-4o-mini"
        },
        "qwen": {
            "model_name": "qwen",
            # qwen-plus（已用82%）/qwen3.7-plus
            "type": "qwen3.7-plus"
        },
        "zhipu": {
            "model_name": "zhipu",
            "type": "glm-4-flash"
        },
        "hunyuan": {
            "model_name": "hunyuan",
            "type": "hunyuan-lite"
        },
        "spark": {
            "model_name": "spark",
            "type": "4.0Ultra"
        }
    },
    "input_max_tokens": 3000,  # 限制大模型最多生成 3000 个 Token (防废话，防破产)
    "history_max_messages": 6,  # 每次提问前，先对历史记忆进行安全截断,保留最近 max_messages 条信息
}

# ==========================================
# RAG相关
# ==========================================
# RAG 检索参数：返回相关度最大的K个结果
RETRIEVER_TOP_K = 3
# RAG 检索参数：返回相关度必须要大于这个threshold（0-1）
RERANK_THRESHOLD = 0.30
# PDF文档解析时超时限制 (目前超时设为 600 秒)
PDF_PARSER_TIMEOUT = 600
# 划分chunk时的每个chunk的最大字数 以及 重叠字数（防止上下文断裂）
CHUNK_CONFIG = {
    "markdown_or_text": {
        "CHUNK_SIZE": 600,
        "CHUNK_OVERLAP": 100,
        "SEPARATOR": ["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    },
    "spreadsheet": {
        "CHUNK_SIZE": 300,
        "CHUNK_OVERLAP": 50,
        "SEPARATOR": ["\n\n", "\n"]
    },
    "json": {
        "CHUNK_SIZE": 500,
        "CHUNK_OVERLAP": 50,
        "SEPARATOR": ["\n\n", "\n", "},", "],", "}", "]"]
    },
}

# ==========================================
# 集中管理所有本地模型的网络与路径策略
# ==========================================
MODEL_REGISTRY = {
    "bge_m3": {
        "model_name": "BAAI/bge-m3",
        "model_name_simple": "bge_m3",
        "device": "cuda",
        "offline": True,  # 🌟 是否开启终极离线模式
        "use_mirror": True,  # 🌟 是否使用国内镜像源（若 offline=True，此项自动失效）
        "local_path": "D:/Python31210/HuggingFace_Models",
        "search_number": 15  # 从向量数据库的海量数据里先捞出 x 条最神似的
    },
    "BM25_search_number": 10,  # 关键词初筛时获取最形似的前 y 条结果
    "bge_reranker": {
        "model_name": "BAAI/bge-reranker-v2-m3",
        "model_name_simple": "bge_reranker",
        "device": "cuda",
        "offline": True,  # 🌟 比如重排模型想允许联网检查更新或首次下载
        "use_mirror": True,  # 🌟 联网时使用国内镜像（若 offline=True，此项自动失效）
        "local_path": "D:/Python31210/HuggingFace_Models"
    },
    "marker_pdf": {
        "model_name": "marker-pdf",  # 这里的名字仅作占位说明，底层命令还是 marker_single
        "model_name_simple": "marker_pdf",
        "device": "cuda",
        "offline": True,  # 开启终极离线，防止 marker 偷偷连网下载
        "use_mirror": False,
        "local_path": "D:/Python31210/HuggingFace_Models"
    }
}

# ==========================================
# 大模型 Token 消耗与输出限制
# ==========================================
# 规划师 (Planner) 的最大输出 Token：只负责输出结构化 JSON，不需要太多
PLANNER_MAX_TOKENS = 800
# 执行者 (Executor) 的最大输出 Token：负责输出长文本/总结，需要给足空间
EXECUTOR_MAX_TOKENS = 3000
