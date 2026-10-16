#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于系统相关配置
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/06/05 15:46
version: V1.0
Target Python: 
"""

from pathlib import Path
import torch

# 调用本地的大模型
USE_LOCAL_MODEL = False  # True 代表使用本地 Ollama，False 代表使用远程云端 API

# 本地 Ollama 模型配置
# 你的可用库存: 'deepseek-r1', 'gemma3', 'gemma4', 'qwen2.5-coder', 'qwen3'
LOCAL_MODEL_NAME = "qwen3" # 当前选中的本地模型

# 调用远程云端的大模型
# Gemini/GPT/Qwen/Zhipu/Hunyuan（超时）/Spark（没有key）
REMOTE_MODEL_NAME: str = "Qwen"
# 使用的System Prompts
# default_chat/a63_sensor/code_review/complex_task
my_scenario_name: str = "complex_task"

# 自动定位当前项目的根目录 (AI_Agent_Project)
# __file__ 是当前文件，parent 是 config 文件夹，parent.parent 就是项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent  # D:\PythonProject\AI_Agent\my_code
# Prompts的地址
PROMPTS_DIR = str(BASE_DIR / "AdvancedPrompts")

# 向量数据库配置 (RAG)：自动将路径拼接为根目录下的 db_storage/chroma_db
DB_PERSIST_PATH = r"E:/Vector_Database_for_Agent/db_storage/chroma_db"
# 集合的名字（相当于关系型数据库的表名）
COLLECTION_NAME = "tk_knowledge"

# agent需要的模型下载与缓存路径
HF_MODELS_PATH = "D:/Python31210/HuggingFace_Models"

# 日志输出目录
LOG_CONFIG = {
    "path" : str(BASE_DIR / "logs"),
    "rotation" : 5, # 每当日志文件达到 x MB 时，自动新建一个文件
    "retention": 7,  # 历史日志保留 x 天，防止撑爆硬盘
    # 日志格式
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
}

# cpu/cuda
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================================================================
# 🌟 RAG 作为工具的 DOCSTRING：从同目录下的 rag_tool_docstring.md 动态读取
# =========================================================================
RAG_DOC_FILE = Path(__file__).resolve().parent / "rag_tool_docstring.md"

def load_rag_tool_docstring() -> str:
    """从外部 Markdown 文件动态加载 RAG 说明书，带异常与缺失防御"""
    if not RAG_DOC_FILE.exists():
        # 防呆兜底：如果外部 md 文件丢失，返回基础说明，防止启动抛异常崩溃
        return "【本地核心专业知识库检索工具】用于检索云冈石窟项目相关内容、博士资料和博士学术论文等本地资料。"

    try:
        return RAG_DOC_FILE.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"⚠️ [配置警告] 读取 {RAG_DOC_FILE.name} 失败: {e}，将启用简短兜底说明。")
        return "【本地核心专业知识库检索工具】用于检索本地专业知识库资料。"

# 对外暴露的常量：保持原变量名不变，mcp_server.py 等其他模块无需做任何修改
RAG_TOOL_DOCSTRING = load_rag_tool_docstring()
