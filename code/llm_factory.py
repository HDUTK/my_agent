#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于模型锻造厂。专门负责加载各大平台的 API 密钥并返回对应的模型实例
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/31 16:40
version: V1.0
Target Python: 
"""

import os
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel


# ==========================================
# 核心配置库
# ==========================================
def get_llm_model(platform: str):
    platform = platform.lower()

    # ----------------------------------------
    # 1. Google Gemini
    # ----------------------------------------
    if platform == "gemini":
        # gemini-2.0-flash/gemini-2.5-flash/gemini-2.5-pro/gemini-3-flash-preview/
        # gemini-3-pro-preview/gemini-3.1-pro-preview/gemini-3.5-flash
        return GoogleModel("gemini-2.5-flash")

    # ----------------------------------------
    # 2. OpenAI (ChatGPT)
    # ----------------------------------------
    elif platform == "gpt":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("❌ 找不到 OPENAI_API_KEY")

        os.environ["OPENAI_API_KEY"] = api_key
        return OpenAIChatModel("gpt-4o-mini")

    # ----------------------------------------
    # 3. 阿里 - 通义千问 (Qwen)
    # ----------------------------------------
    elif platform == "qwen":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key: raise ValueError("❌ 找不到 DASHSCOPE_API_KEY")

        # 🌟 终极黑科技：直接修改底层系统变量，欺骗 OpenAI SDK
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # 现在只需要传一个纯净的名字，括号里什么都不用加！
        return OpenAIChatModel("qwen-plus")

    # ----------------------------------------
    # 4. 智谱 - GLM
    # ----------------------------------------
    elif platform == "zhipu":
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key: raise ValueError("❌ 找不到 ZHIPU_API_KEY")

        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4/"
        return OpenAIChatModel("glm-4-flash")

    # ----------------------------------------
    # 5. 腾讯 - 混元 (Hunyuan)
    # ----------------------------------------
    elif platform == "hunyuan":
        api_key = os.getenv("HUNYUAN_API_KEY")
        if not api_key: raise ValueError("❌ 找不到 HUNYUAN_API_KEY")

        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.hunyuan.cloud.tencent.com/v1"
        return OpenAIChatModel("hunyuan-lite")

    # ----------------------------------------
    # 6. 讯飞 - 星火 (Spark)
    # ----------------------------------------
    elif platform == "spark":
        api_key = os.getenv("SPARK_API_KEY")
        if not api_key: raise ValueError("❌ 找不到 SPARK_API_KEY")

        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = "https://spark-api-open.xf-yun.com/v1"
        return OpenAIChatModel("4.0Ultra")

    # 如果输入的平台名字不在这 6 个里面，抛出异常

    raise ValueError(f"❌ 暂不支持的模型平台: {platform}")
