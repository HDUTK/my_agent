#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于存放所有与业务逻辑无关的通用工具函数，如读 Prompt、修剪记忆
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/31 16:40
version: V1.0
Target Python:  3.14 -> 3.12
"""

import os
import warnings

from pydantic_ai.messages import ModelRequest, ModelResponse, ToolReturnPart
from config.system_config import PROMPTS_DIR

from config.agent_config import MODEL_REGISTRY
from utils.logger_print import sys_logger


# ==========================================
# Advanced Prompts
# ==========================================
def load_prompt(scenario_name: str) -> str:
    """根据场景名称，从本地文件中读取 Prompt"""
    file_path = PROMPTS_DIR + f"/{scenario_name}.md"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            sys_logger.info(f"[Prompt 加载] 成功读取系统提示词模板: {scenario_name}.md")
            return content
    except FileNotFoundError:
        sys_logger.warning(f"[Prompt 缺失] 找不到文件 {file_path}，已强制降级使用默认兜底提示词！")
        return "你是一个有用的人工智能助手。"  # 找不到文件时的默认兜底


# ==========================================
# 安全截断历史记忆 (滑动窗口)
# ==========================================
def trim_history(history: list, max_messages: int = 6) -> list:
    """
    像外科手术一样精准截断历史记录。
    不仅限制长度，还能避开“切断工具调用链”的致命雷区。
    """
    if len(history) <= max_messages:
        return history

    # 1. 粗略切取最后 max_messages 条
    trimmed = history[-max_messages:]

    # 2. 精密排雷：确保切下来的第一条消息是纯净的、由用户发起的问题
    while trimmed:
        first_msg = trimmed[0]
        is_invalid_start = False

        # 雷区 A：如果开头第一条是模型生成的回复（ModelResponse），逻辑断裂
        if isinstance(first_msg, ModelResponse):
            is_invalid_start = True

        # 雷区 B：如果开头第一条包含了工具执行的返回结果（ToolReturnPart），说明大模型请求工具的那句话被切没了，逻辑断裂
        elif isinstance(first_msg, ModelRequest):
            if any(isinstance(part, ToolReturnPart) for part in first_msg.parts):
                is_invalid_start = True

        # 如果踩雷，就把这条残缺的记忆扔掉，往后找下一条，直到找到安全的起点
        if is_invalid_start:
            trimmed.pop(0)
        else:
            break

    return trimmed


# ==========================================
# 动态环境变注入函数
# ==========================================
def apply_model_environment(model_key: str):
    """
    根据传入的模型 key (如 'bge_m3')，动态注入对应的 HuggingFace 环境变量。
    必须在对应的 RAG 组件 import 底层大模型库之前调用！
    """
    # 🌟 强制终端使用 UTF-8 编码
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # 🌟 消除 Flash Attention 警告(rag_retriever.py和vector_builder.py)
    # warnings.filterwarnings("ignore", message=".*1Torch was not compiled with flash attention.*")
    # 消除警告
    warnings.filterwarnings("ignore")

    config = MODEL_REGISTRY.get(model_key)
    if not config:
        sys_logger.error(f" 未在 MODEL_REGISTRY 中找到模型配置: {model_key}")
        raise ValueError(f" 未在 MODEL_REGISTRY 中找到模型配置: {model_key}")

    # 配置当前模型专属的本地缓存路径
    os.environ["HF_HOME"] = config["local_path"]

    # 基础安全清理：强行拔掉 Python 的代理管子，防止网络卡死
    os.environ['HTTP_PROXY'] = ""
    os.environ['HTTPS_PROXY'] = ""
    # 🌟 彻底关闭 ChromaDB 的后台匿名数据收集线程（防止卡死）
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    # 🌟 关闭 Tokenizer 导致的多线程死锁
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if config["offline"]:
        # 开启终极离线模式
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # 离线时清除镜像源设置
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
    else:
        # 允许联网模式
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        if config["use_mirror"]:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        else:
            if "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]

    sys_logger.info(
        f"⚙ [环境配置] 已成功为模型 [{model_key}] 注入网络与路径环境变（离线={config['offline']}, 镜像={config['use_mirror']})")
