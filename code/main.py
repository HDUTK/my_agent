#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于agent编写
以下是示例输入：
你好
当前目录下有一个文件夹叫test2，里面有一个文件是A63.csv，记录了A63传感器从2023/06/07/00:00到2024/04/06/23:55的空气温度、空气湿度、壁面温度，我想要将里面的2023/10/10/00:00的空气温度、空气湿度、壁面温度提取出来并告诉我环境是否有风险（空气温度超过 20度 或空气湿度超过 60%，判定又风险）
列出test文件夹下所有的文件名，然后告诉我每一个文件用的什么语言编写的代码
请针对这个文件夹下的文件内容，把文件名后缀名的txt进行修改（根据代码的语言），如果你遇到什么困难可以先回复我有什么困难（例如遇到权限问题等）

author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/12 15:47
version: V1.0
Target Python: 3.14
"""

import os
import json
import asyncio
from typing import Dict, Any, Union
from dotenv import load_dotenv

from openai import AsyncOpenAI
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolReturnPart

from pydantic_ai.exceptions import ModelRetry

# 导入mcp客户端组件
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession


# 加载所有环境变量配置
load_dotenv()


# ==========================================
# 🌟 进阶模块：Advanced Prompts
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
# 模块一：核心配置库
# ==========================================
def get_llm_model(platform: str):
    platform = platform.lower()

    # ----------------------------------------
    # 1. Google Gemini
    # ----------------------------------------
    if platform == "gemini":
        # gemini-2.0-flash/gemini-2.5-flash/gemini-2.5-pro/gemini-3-flash-preview/
        # gemini-3-pro-preview/gemini-3.1-pro-preview/gemini-3.5-flash
        return GoogleModel("gemini-2.0-flash")

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

# ==========================================
# 模块二：工具制造厂 (上一步讲过的闭包工厂)
# ==========================================
def make_mcp_tool(session: ClientSession):
    """闭包工厂：接收一个 session，返回一个绑定好的工具调用网关"""
    async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
        print(f"\n[🔌 远程调用] 正在请求 Server 执行: {tool_name}")
        print(f"   传入参数: {arguments}")
        try:
            result = await session.call_tool(tool_name, arguments)
            text_result = [content.text for content in result.content if content.type == "text"]
            final_result = "\n".join(text_result)

            # 2. 打印工具的返回结果（如果超过 300 字就截断显示，防止刷屏）
            preview = final_result if len(final_result) < 300 else final_result[:300] + "\n... (内容太长，已省略后续输出)"
            print(f"[📥 Server 返回]\n{preview}\n")

            return final_result
        except Exception as e:
            error_msg = str(e).lower()

            # 🌟 诊断 1：如果是通讯/网络/超时问题 (可以根据实际的报错关键字调整)
            if "timeout" in error_msg or "connection" in error_msg or "network" in error_msg or "mcp" in error_msg:
                print(f"⚠️ [网络波动] 检测到通讯异常: {e}")
                print(f"⏳ 正在扣减重试额度并请求大模型重试...")
                # 抛出 ModelRetry
                # Pydantic-AI 会拦截这个异常，自动消耗一次 retries 额度，并让模型重新发起调用
                raise ModelRetry(f"由于网络通讯异常导致工具调用失败 ({e})。请重新尝试调用此工具。")

            # 🌟 诊断 2：如果是代码逻辑/GBK编码等致命死错
            else:
                print(f"❌ [致命错误] {e}")
                # 核心拦截：直接返回普通字符串作为结果
                # 大模型看到这句话后，就知道工具废了，不会再执着重试，而是直接回复用户
                return f"🚨 致命系统错误: {str(e)}。请立即放弃重试此工具，并直接向用户汇报系统故障！"
    return call_mcp_tool


# ==========================================
# 模块三：Agent 装配流水线
# ==========================================
async def build_agent_with_mcp(session: ClientSession, model_name:str,
                               scenario_name: str = "default_chat") -> Agent:
    """
    Agent 装配厂：负责找 Server 要工具列表，写说明书，最后把大脑和工具组装成 Agent
    """
    # 1. 获取工具列表
    tools_response = await session.list_tools()
    mcp_tools = tools_response.tools
    print(f"📦 发现 {len(mcp_tools)} 个远程工具: {[t.name for t in mcp_tools]}")

    # 2. 编写工具说明书 (系统提示词)
    tools_instruction = "你现在连接到了一个本地工具库，可以使用以下工具：\n\n"
    for tool in mcp_tools:
        tools_instruction += f"- 名称: {tool.name}\n"
        tools_instruction += f"  描述: {tool.description}\n"
        tools_instruction += f"  参数格式: {json.dumps(tool.inputSchema, ensure_ascii=False)}\n\n"
    tools_instruction += "请调用 `call_mcp_tool` 函数来使用上述工具。"

    # 🌟 进阶模块：系统提示词 (System Prompt) 策略
    advanced_prompt = load_prompt(scenario_name)

    # 将人设模板与动态工具列表拼接，形成最终的超级大脑设定
    final_system_prompt = advanced_prompt + tools_instruction

    # 3. 生产对讲机
    my_custom_tool = make_mcp_tool(session)

    # 4. 组装并返回 Agent
    return Agent(
        get_llm_model(model_name),
        system_prompt=final_system_prompt,
        tools=[my_custom_tool],
        output_type=str,  # 锁定输出结构
        retries=3  # 允许工具调用失败后最多重试 3 次，额度用完直接停止
    )


# ==========================================
# 🌟 进阶模块：安全截断历史记忆 (滑动窗口)
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
# 模块四：用户交互界面
# ==========================================
async def run_chat_loop(agent: Agent):
    """控制台聊天引擎：只负责跟用户互动"""
    print("\n🚀 Agent 已就绪，开始聊天模式。随时输入 'exit' 退出。")
    history = []

    while True:
        user_input = await asyncio.to_thread(input, "\n输入：")

        if user_input.strip().lower() == 'exit':
            print("👋 Agent 已退出。")
            break

        if not user_input.strip():
            continue

        # 🌟 每次提问前，先对历史记忆进行安全截断（保留最近 6 条有价值的信息）
        history = trim_history(history, max_messages=6)

        # print(f"🧹 (当前记忆长度: {len(history)} 条)")
        print("🤖 思考中...\n回答: ", end="", flush=True)

        try:
            # 🌟 把 run() 换成 run_stream()，用 async with 打开“水龙头”
            # 🌟 核心控制：传入 model_settings，限制 max_tokens
            async with agent.run_stream(
                    user_input,
                    message_history=history,
                    model_settings={'max_tokens': 800}  # 限制大模型最多生成 800 个 Token (防废话，防破产)
            ) as resp:

                async for text_chunk in resp.stream_text(delta=True):
                    print(text_chunk, end="", flush=True)
                print()  # 打印换行收尾

                # 🌟 在水流完之后，更新聊天记录
                history = resp.all_messages()

            print("-" * 50)
        except Exception as e:
            print(f"\n❌ 抱歉，大模型处理或工具调用时遇到错误：\n{e}")
            print("💡 程序仍在运行，您可以尝试重新提问，或等待网络恢复。")
            print("-" * 50)


# ==========================================
# 最终的 Main 函数
# Gemini/GPT/Qwen/Zhipu/Hunyuan（超时）/Spark（没有key）
# default_chat/a63_sensor/code_review
# ==========================================
async def main(model_name: str = "Qwen", scenario_name: str = "default_chat"):
    print(f"🔄 系统启动中... [当前指定模型: {model_name}]")

    # 第一步：定义后厨位置
    server_parameters = StdioServerParameters(command="python", args=["mcp_server.py"])

    # 第二步：接通管道并建立会话
    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ 成功连接到 MCP Server！")

            # 第三步：把 session 交给装配厂，换回一个组装好的 Agent
            agent = await build_agent_with_mcp(session, model_name, scenario_name)

            # 第四步：把 Agent 放进聊天引擎
            await run_chat_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())
