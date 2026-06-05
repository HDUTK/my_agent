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

from pydantic_ai.messages import ModelRequest, ModelResponse, ToolReturnPart


# ==========================================
# Advanced Prompts
# ==========================================
def load_prompt(scenario_name: str) -> str:
    """根据场景名称，从本地文件中读取 Prompt"""
    file_path = f"AdvancedPrompts/{scenario_name}.md"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "你是一个有用的人工智能助手。" # 找不到文件时的默认兜底


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

