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
from config.system_config import USE_LOCAL_MODEL, LOCAL_MODEL_NAME
from utils.logger_print import sys_logger

import os
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider


# ==========================================
# 核心配置库
# ==========================================
def get_llm_model(platform: str):
    if USE_LOCAL_MODEL:
        api_key = "ollama"  # Ollama 本地不需要真实 key，随便填个字符串占位即可
        os.environ["OLLAMA_API_KEY"] = api_key
        try:
            local_model_config = LLM_CONFIG["model"]["local"][platform.lower()]
            OLLAMA_BASE_URL = local_model_config["base_url"]
            os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
        except KeyError:
            sys_logger.error(f"❌ 严重错误: 本地暂不支持的模型平台 '{platform}'，请检查 config 配置。")
            raise ValueError(f"❌ 本地暂不支持的模型平台: {platform}")

        no_proxy_hosts = ["localhost", "127.0.0.1", "::1"]
        current_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        merged_no_proxy = ",".join(dict.fromkeys(
            [item.strip() for item in current_no_proxy.split(",") if item.strip()] + no_proxy_hosts
        ))
        os.environ["NO_PROXY"] = merged_no_proxy
        os.environ["no_proxy"] = merged_no_proxy

        sys_logger.info(f"🔌 [模型装配] 检测到本地开关已打开，正在连接本地 Ollama 模型: {LOCAL_MODEL_NAME}")

        model_name = local_model_config["ollama_name"]
        sys_logger.info(f"✅ [模型锻造厂] 成功装载 本地模型: {model_name}")

        return OllamaModel(model_name,
                           provider=OllamaProvider(base_url=OLLAMA_BASE_URL))

    else:
        platform = platform.lower()
        sys_logger.info(f"🏭 [模型锻造厂] 开始装载远程云端大模型引擎，指定平台: {platform}")

        # ----------------------------------------
        # 1. Google Gemini
        # ----------------------------------------
        if platform == "gemini":
            model_name = LLM_CONFIG["model"]["remote"]["gemini"]["type"]
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

            model_name = LLM_CONFIG["model"]["remote"]["gpt"]["type"]
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

            # 直接修改底层系统变量，欺骗 OpenAI SDK
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_BASE_URL"] = LLM_CONFIG["model"]["remote"]["qwen"]["base_url"]

            # 现在只需要传一个纯净的名字，括号里什么都不用加！
            model_name = LLM_CONFIG["model"]["remote"]["qwen"]["type"]
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
            os.environ["OPENAI_BASE_URL"] = LLM_CONFIG["model"]["remote"]["zhipu"]["base_url"]

            model_name = LLM_CONFIG["model"]["remote"]["zhipu"]["type"]
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
            os.environ["OPENAI_BASE_URL"] = LLM_CONFIG["model"]["remote"]["hunyuan"]["base_url"]

            model_name = LLM_CONFIG["model"]["remote"]["hunyuan"]["type"]
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
            os.environ["OPENAI_BASE_URL"] = LLM_CONFIG["model"]["remote"]["spark"]["base_url"]

            model_name = LLM_CONFIG["model"]["remote"]["spark"]["type"]
            sys_logger.info(f"✅ [模型锻造厂] 成功装载 讯飞星火模型: {model_name}")
            return OpenAIChatModel(model_name)

        # 如果输入的平台名字不在这 6 个里面，抛出异常

        sys_logger.error(f"❌ 严重错误: 远程暂不支持的模型平台 '{platform}'，请检查 config 配置。")
        raise ValueError(f"❌ 远程暂不支持的模型平台: {platform}")
