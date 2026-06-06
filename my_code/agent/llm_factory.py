#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于模型锻造厂。专门负责加载各大平台的 API 密钥并返回对应的模型实例
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/31 16:40
version: V1.0
Target Python:  3.14 -> 3.12
"""

from config.agent_config import LLM_CONFIG
from utils.logger_print import sys_logger

import os
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel


# ==========================================
# 核心配置库
# ==========================================
def get_llm_model(platform: str):
    platform = platform.lower()
    sys_logger.info(f"🏭 [模型锻造厂] 开始装载大模型引擎，指定平台: {platform}")

    # ----------------------------------------
    # 1. Google Gemini
    # ----------------------------------------
    if platform == "gemini":
        model_name = LLM_CONFIG["model"]["gemini"]["type"]
        sys_logger.info(f"✅ [模型锻造厂] 成功装载 Google Gemini 模型: {model_name}")
        return GoogleModel(model_name)

    # ----------------------------------------
    # 2. OpenAI (ChatGPT)
    # ----------------------------------------
    elif platform == "gpt":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            sys_logger.error("❌ 找不到 OPENAI_API_KEY，无法启动 GPT 模型。")
            raise ValueError("❌ 找不到 OPENAI_API_KEY")

        os.environ["OPENAI_API_KEY"] = api_key

        model_name = LLM_CONFIG["model"]["gpt"]["type"]
        sys_logger.info(f"✅ [模型锻造厂] 成功装载 OpenAI 模型: {model_name}")
        return OpenAIChatModel(model_name)

    # ----------------------------------------
    # 3. 阿里 - 通义千问 (Qwen)
    # ----------------------------------------
    elif platform == "qwen":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            sys_logger.error("❌ 找不到 DASHSCOPE_API_KEY，无法启动通义千问模型。")
            raise ValueError("❌ 找不到 DASHSCOPE_API_KEY")

        # 🌟 终极黑科技：直接修改底层系统变量，欺骗 OpenAI SDK
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # 现在只需要传一个纯净的名字，括号里什么都不用加！
        model_name = LLM_CONFIG["model"]["qwen"]["type"]
        sys_logger.info(f"✅ [模型锻造厂] 成功装载 阿里通义千问模型: {model_name}")
        return OpenAIChatModel(model_name)

    # ----------------------------------------
    # 4. 智谱 - GLM
    # ----------------------------------------
    elif platform == "zhipu":
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            sys_logger.error("❌ 找不到 ZHIPU_API_KEY，无法启动智谱 GLM 模型。")
            raise ValueError("❌ 找不到 ZHIPU_API_KEY")

        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4/"

        model_name = LLM_CONFIG["model"]["zhipu"]["type"]
        sys_logger.info(f"✅ [模型锻造厂] 成功装载 智谱 GLM 模型: {model_name}")
        return OpenAIChatModel(model_name)

    # ----------------------------------------
    # 5. 腾讯 - 混元 (Hunyuan)
    # ----------------------------------------
    elif platform == "hunyuan":
        api_key = os.getenv("HUNYUAN_API_KEY")
        if not api_key:
            sys_logger.error("❌ 找不到 HUNYUAN_API_KEY，无法启动腾讯混元模型。")
            raise ValueError("❌ 找不到 HUNYUAN_API_KEY")

        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.hunyuan.cloud.tencent.com/v1"

        model_name = LLM_CONFIG["model"]["hunyuan"]["type"]
        sys_logger.info(f"✅ [模型锻造厂] 成功装载 腾讯混元模型: {model_name}")
        return OpenAIChatModel(model_name)

    # ----------------------------------------
    # 6. 讯飞 - 星火 (Spark)
    # ----------------------------------------
    elif platform == "spark":
        api_key = os.getenv("SPARK_API_KEY")
        if not api_key:
            sys_logger.error("❌ 找不到 SPARK_API_KEY，无法启动讯飞星火模型。")
            raise ValueError("❌ 找不到 SPARK_API_KEY")

        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = "https://spark-api-open.xf-yun.com/v1"

        model_name = LLM_CONFIG["model"]["spark"]["type"]
        sys_logger.info(f"✅ [模型锻造厂] 成功装载 讯飞星火模型: {model_name}")
        return OpenAIChatModel(model_name)

    # 如果输入的平台名字不在这 6 个里面，抛出异常

    sys_logger.error(f"❌ 严重错误: 暂不支持的模型平台 '{platform}'，请检查 config 配置。")
    raise ValueError(f"❌ 暂不支持的模型平台: {platform}")
