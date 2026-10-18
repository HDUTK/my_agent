#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于agent调度中心。只负责启动网络连接、调用装配车间，以及执行对话循环。
以下是示例输入：
你好
我本地test文件夹内有一个2文件夹，2文件夹内有一个Manuscript.docx，请查看这个文件，然后大概总结一下这篇论文内容
我本地test文件夹内有一个2文件夹，2文件夹内有一个Manuscript.docx，帮我把Manuscript.docx里面的 Abstract, Introduction, Research aim, Methodology, Results, Discussion, Conclusion 这 7 个部分进行总结提取
当前目录下有一个文件夹叫test2，里面有一个文件是A63.csv，记录了A63传感器从2023/06/07/00:00到2024/04/06/23:55的空气温度、空气湿度、壁面温度，我想要将里面的2023/10/10/00:00的空气温度、空气湿度、壁面温度提取出来并告诉我环境是否有风险（空气温度超过 20度 或空气湿度超过 60%，判定又风险）
列出test/1文件夹下所有的文件名，然后告诉我每一个文件用的什么语言编写的代码
请针对这个文件夹下的文件内容，把文件名后缀名的txt进行修改（根据代码的语言），如果你遇到什么困难可以先回复我有什么困难（例如遇到权限问题等）

author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/12 15:47
version: V1.0
Target Python: 3.14 -> 3.12
"""

import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits
from typing import Any, Optional

# 导入mcp客户端组件
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# 统一记忆门面总管（一行代码涵盖门禁、裁剪、排雷、摘要与事实落盘）
from utils.memory_manager import UnifiedMemoryManager

# 导入其他的模块
from agent.agent_builder import build_agent_with_mcp
from utils.logger_print import print_and_log
# 配置
from config.system_config import (USE_LOCAL_MODEL, LOCAL_MODEL_NAME, REMOTE_MODEL_NAME,
                                  my_scenario_name, MEMORY_CONFIG)
from config.agent_config import LLM_CONFIG

# langsmith监控
from langsmith import traceable

# Phoenix监控
import asyncio
from opentelemetry import trace
from utils.logger_print import sys_logger

# 加载所有环境变量配置
load_dotenv()

# ==========================================
# 🌟 Phoenix 动态开关装配
# ==========================================
# 读取 Phoenix 开关（不区分大小写）
ENABLE_PHOENIX = os.getenv("ENABLE_PHOENIX", "false").lower() in ("true", "1", "yes")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "TK_Personal_Agent_Phoenix")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
if ENABLE_PHOENIX:
    print("🔭 [可观测性] Phoenix 监控已启用，正在上报至 6006 端口")
    try:
        from phoenix.otel import register
        from openinference.instrumentation.openai import OpenAIInstrumentor

        tracer_provider = register(
            project_name=PHOENIX_PROJECT_NAME,
            endpoint=PHOENIX_ENDPOINT
        )
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        sys_logger.info("🔭 [可观测性] Phoenix 监控已启用，正在上报至 6006 端口")
    except Exception as e:
        sys_logger.warning(f"⚠️ [可观测性] Phoenix 初始化失败，降级为关闭状态: {e}")
else:
    print("🔒 [可观测性] Phoenix 监控处于关闭状态（No-op 模式）")
    sys_logger.info("🔒 [可观测性] Phoenix 监控处于关闭状态（No-op 模式）")

# 全局 Tracer（当未初始化 provider 时，自动返回 No-op 空操作追踪器）
tracer = trace.get_tracer(__name__)


# 🌟 将“单次回合”的逻辑单独抽离，并打上摄像头
@traceable(name="Agent_单回合对话_Turn")  # langSmith
async def process_single_turn(agent: Agent, user_input: str, history: list) -> list:
    """处理单次对话回合，并返回更新后的记忆"""
    # 🌟 核心修改：用 OTel 创建单回合根节点 (Root Span)，将该回合的所有操作聚合为一棵树
    with tracer.start_as_current_span("Agent_单回合对话_Turn") as root_span:
        # 将用户输入写入 Span 属性
        root_span.set_attribute("input.value", user_input)

        print_and_log("🤖 思考中...\n回答: ", "info", end="", flush=True)

        try:
            # 🌟 把 run() 换成 run_stream()，用 async with 打开“水龙头”
            # 🌟 核心控制：传入 model_settings，限制 max_tokens
            async with agent.run_stream(
                    user_input,
                    message_history=history,
                    model_settings=ModelSettings(max_tokens=LLM_CONFIG["input_max_tokens"]),
                    usage_limits=UsageLimits(request_limit=8, tool_calls_limit=5)
                    # 限制大模型最多生成 input_max_tokens 个 Token (防废话，防破产)
            ) as resp:

                content_chunks = []

                async for text_chunk in resp.stream_text(delta=True):
                    print(text_chunk, end="", flush=True)
                    content_chunks.append(text_chunk)
                print()  # 打印换行收尾

                # 🌟 必须等文字全部流完，再结算并打印聊天 Token 消耗
                full_reply = "".join(content_chunks)
                sys_logger.info(f"[对话记录] 用户输入: {user_input}\n[Agent完整回复]: \n{full_reply}")

                usage = resp.usage
                token_msg = f"📊 [Token结算 - 聊天界面] 输入: {usage.input_tokens} | 输出: {usage.output_tokens} | 累计: {usage.total_tokens}"
                print_and_log(f"\n{token_msg}", "info")

                # 🌟 在水流完之后，更新聊天记录
                return resp.all_messages()

        except Exception as e:
            # 报错写入日志，但屏幕上也要给用户一个温柔的提示
            sys_logger.error(f"大模型处理或工具调用时遇到错误：\n{e}")
            sys_logger.error("-" * 50)

            print(f"\n🚨 抱歉，大模型处理或工具调用时遇到错误，后台已自动捕获并记录。")
            print(" 程序仍在运行，您可以尝试重新提问，或等待网络恢复。")
            return history


# ==========================================
# 用户交互界面
# ==========================================
async def run_chat_loop(agent: Agent, memory_mgr: UnifiedMemoryManager):
    """
    控制台交互主引擎：
    在每轮交互中，协同调度【正则门禁】、【顺风车摘要】与【主 Agent 对话】
    """
    print_and_log("\n" + "═" * 50, "info")

    if USE_LOCAL_MODEL:
        print_and_log(f"🧠 [驱动模型]: {LLM_CONFIG['model']['local'][LOCAL_MODEL_NAME.lower()]['model_name']}", "info")
    else:
        print_and_log(f"🧠 [驱动模型]: {LLM_CONFIG['model']['remote'][REMOTE_MODEL_NAME.lower()]['type']}", "info")

    # 如果场景变量名不叫 scenario_name，请换成实际传递给 agent_builder 的那个变量
    print_and_log(f"🎭 [当前场景 (System Prompt)]: {my_scenario_name}", "info")
    # 显示长期事实档案状态
    if memory_mgr.fact_store and memory_mgr.fact_store.facts:
        print_and_log(f"📚 [事实档案]: 已成功加载本地长期记忆库 ({memory_mgr.fact_store.file_path.name})", "info")
    print_and_log("═" * 55, "info")

    print_and_log("\n🚀 Agent 已就绪，开始聊天模式。随时输入 'exit' 退出。", "info")

    history = []

    while True:
        user_input = await asyncio.to_thread(input, "\n输入：")
        cleaned_input = user_input.strip()

        if cleaned_input.strip().lower() == 'exit':
            print_and_log("👋 Agent 已退出。", "info")
            sys_logger.info("-" * 50)
            sys_logger.info("\n" * 5)
            break

        if not cleaned_input.strip():
            continue

        # 🌟【核心简化点】：1 行代码解决所有门禁、抽取、排雷、剪枝与滚动摘要
        clean_history = await memory_mgr.prepare_turn_context(cleaned_input, history)

        # -----------------------------------------------------------------
        # 🌟 主对话推理: 大模型只接收预算内的干净短期历史
        # （而长期事实和滚动前情提要，已由 agent_builder 中的动态钩子自动注入到了 System Prompt）
        # -----------------------------------------------------------------
        history = await process_single_turn(agent, cleaned_input, clean_history)

        print("-" * 50)


# ==========================================
# Main 函数
# ==========================================
async def main(local_remote_flag: bool, model_name: str, scenario_name: str):
    sys_logger.info(f"系统启动中... [当前指定模型: {model_name}]")

    # 第一步：初始化统一记忆总管 (内含轻量抽取模型、长期事实档案与滚动摘要引擎)
    memory_mgr = UnifiedMemoryManager(model_name=model_name)

    # 第二步：定义后厨位置——启动连接（对接 MCP Server）
    server_parameters = StdioServerParameters(command="python", args=["mcp_server.py"])

    # 第三步：接通管道并建立会话
    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            sys_logger.info("✅ 成功连接到 MCP Server！")

            # 第四步：把 session 交给装配厂，换回一个组装好的 Agent
            agent = await build_agent_with_mcp(
                session=session,
                model_name=model_name,
                scenario_name=scenario_name,
                memory_manager=memory_mgr
            )

            # 第四步：把 Agent 放进聊天引擎
            await run_chat_loop(agent, memory_mgr)


if __name__ == "__main__":
    active_model = LOCAL_MODEL_NAME if USE_LOCAL_MODEL else REMOTE_MODEL_NAME
    asyncio.run(
        main(
            local_remote_flag=USE_LOCAL_MODEL,
            model_name=active_model,
            scenario_name=my_scenario_name
        )
    )
