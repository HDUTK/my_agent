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
import inspect
import warnings
from mcp.server.mcpserver import MCPServer
from core_tools import tools

from RAG.local_RAG import LocalKnowledgeExpert
from utils.logger_print import sys_logger
from config.agent_config import RERANK_THRESHOLD, RETRIEVER_TOP_K
from config.system_config import RAG_TOOL_DOCSTRING

# 静音 jieba 词典加载的 DEBUG / INFO 刷屏日志
import jieba
jieba.setLogLevel(logging.WARNING)
# 🌟 屏蔽底层库的 INFO 级别刷屏日志，只放行 WARNING 和 ERROR
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
# 消除 pkg_resources 弃用警告
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# 1. 创建标准化 Server 实例
# mcp = FastMCP("my_server")
mcp = MCPServer("MyServer")

# ==========================================
# 🌟 在全局实例化大模型专家！
# 极其重要：必须写在全局，保证 6GB 的模型只在 MCP 服务启动时加载 1 次，
# 绝对不能写在下面的 tool 函数里，否则每次对话都会重新加载导致显存爆炸。
# ==========================================
expert_instance = LocalKnowledgeExpert()


# ==========================================
# 🌟 注册“本地专家”大模型调度工具
# ==========================================
def consult_local_knowledge_expert(query: str) -> str:
    # 💡 在 MCP 的 stdio 模式下，尽量使用 logging.warning 输出日志，
    # 直接使用 print 有概率会污染底层的 JSON 通信管道导致通信失败。
    sys_logger.info(f" [MCP Tool 触发] 正在向后台老专家提问: {query}")

    # 拿到的是纯原始文本！
    raw_docs_text = expert_instance.retrieve_docs(query, threshold=RERANK_THRESHOLD,
                                                  top_k=RETRIEVER_TOP_K)

    sys_logger.info(" [MCP Tool 完毕] 本地知识库已给出详细解答。")
    return raw_docs_text

# 🌟 动态注入从配置文件中读取的说明书
consult_local_knowledge_expert.__doc__ = RAG_TOOL_DOCSTRING
# 🌟 显式注册进 MCP Server
mcp.add_tool(consult_local_knowledge_expert)
sys_logger.info("✅ 本地知识库检索工具已动态挂载 (带配置说明书)")

# 2. 自动扫描并注册 tools.py 中的所有公开函数
for name, func in inspect.getmembers(tools, inspect.isfunction):
    if not name.startswith("_"):
        mcp.add_tool(func)
        sys_logger.info(f"✅ 工具已成功挂载: {name}")

if __name__ == "__main__":
    sys_logger.info("🚀 FastMCP Server 启动，正通过 stdio 监听通信通道...")
    # 3. 启动标准 stdio 服务
    mcp.run()
