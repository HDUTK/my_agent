#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于MCP服务
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/18 20:17
version: V1.0
Target Python:  3.14 -> 3.12
"""

import logging
from mcp.server.fastmcp import FastMCP
import inspect
from core_tools import tools

from RAG.local_RAG import LocalKnowledgeExpert
from utils.logger_print import sys_logger
from config.agent_config import RERANK_THRESHOLD, RETRIEVER_TOP_K

# 🌟 屏蔽底层库的 INFO 级别刷屏日志，只放行 WARNING 和 ERROR
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

# 1. 创建标准化 Server 实例
mcp = FastMCP("my_server")

# ==========================================
# 🌟 在全局实例化大模型专家！
# 极其重要：必须写在全局，保证 6GB 的模型只在 MCP 服务启动时加载 1 次，
# 绝对不能写在下面的 tool 函数里，否则每次对话都会重新加载导致显存爆炸。
# ==========================================
expert_instance = LocalKnowledgeExpert()


# ==========================================
# 🌟 注册“本地专家”大模型调度工具
# ==========================================
@mcp.tool()
def consult_local_knowledge_expert(query: str) -> str:
    """
    当用户询问有关于本地信息等深度专业知识时，可以调用此工具。
    该工具会在本地向量数据库中进行高级混合检索，并由本地数据库给出尽量相关的几段内容。

    ⚠️【给大模型你的指令】：
    拿到此工具返回的资料后，你必须遵守以下纪律回答用户：
    1. 如果有觉得工具返回的内容有相关性高的内容，优先基于工具返回的资料进行回答。
    2. 若资料中未提及，可以使用你的专业知识补充，但必须在补充部分明确标注：“（注：此部分内容由大模型自身知识库补充）”。
    3. 绝对禁止捏造工具返回值里没有的数据。
    4. 若工具返回了“⚠️ [系统提示]：本地数据库中未检索到...”，你必须遵守该提示，用自身知识作答，并向用户坦白本地无记录。

    参数:
        query (str): 用户提出的具体问题，例如“9号窟的湿度异常怎么处理？”或者“论文的结论是什么？”等等问题

    返回:
        str: 基于本地知识库给出的解答。
    """
    # 💡 在 MCP 的 stdio 模式下，尽量使用 logging.warning 输出日志，
    # 直接使用 print 有概率会污染底层的 JSON 通信管道导致通信失败。
    sys_logger.info(f" [MCP Tool 触发] 正在向后台老专家提问: {query}")

    # 拿到的是纯原始文本！
    raw_docs_text = expert_instance.retrieve_docs(query, threshold=RERANK_THRESHOLD,
                                                  top_k=RETRIEVER_TOP_K)

    sys_logger.info(" [MCP Tool 完毕] 本地知识库已给出详细解答。")
    return raw_docs_text


# 2. 自动扫描并注册 tools.py 中的所有公开函数
for name, func in inspect.getmembers(tools, inspect.isfunction):
    if not name.startswith("_"):
        mcp.add_tool(func)
        sys_logger.info(f"✅ 工具已成功挂载: {name}")

if __name__ == "__main__":
    sys_logger.info("🚀 FastMCP Server 启动，正通过 stdio 监听通信通道...")
    # 3. 启动标准 stdio 服务
    mcp.run()
