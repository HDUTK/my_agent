#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于日志
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/06/05 14:44
version: V1.0
Target Python: 
"""

from loguru import logger
import os


def setup_logger():
    # 确保日志文件夹存在
    os.makedirs("logs", exist_ok=True)

    # 1. 移除 Loguru 默认往控制台(屏幕)输出的处理器
    # 这一步最关键！这保证了日志不会在屏幕上和 LLM 的对话抢地盘
    logger.remove()

    # 2. 添加一个输出到文件的处理器
    logger.add(
        sink="logs/agent_system.log",  # 日志保存路径
        rotation="5 MB",  # 每当日志文件达到 5MB 时，自动新建一个文件
        retention="7 days",  # 历史日志保留 7 天，防止撑爆硬盘
        encoding="utf-8",
        level="INFO",  # 记录 INFO 及以上级别的日志
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    # 3. 可以单独把 ERROR 级别的日志拆分到一个文件，方便查 Bug
    logger.add(
        sink="logs/agent_error.log",
        rotation="5 MB",
        encoding="utf-8",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    return logger

# 实例化并暴露给其他文件使用
sys_logger = setup_logger()


def print_and_log(message: str, level: str = "info", **kwargs):
    """
    把消息同时打在屏幕上并写入日志。
    如果 level='error'，Loguru 会自动将它同时写入 system.log 和 error.log。
    """
    print(message, **kwargs)
    if level == "info":
        sys_logger.info(message)
    elif level == "warning":
        sys_logger.warning(message)
    elif level == "error":
        sys_logger.error(message)

