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
from typing import Dict, Any
from dotenv import load_dotenv

from openai import AsyncOpenAI
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai import Agent

# 导入mcp客户端组件
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# 导入模具
from schemas import SensorRecord


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
        try:
            result = await session.call_tool(tool_name, arguments)
            text_result = [content.text for content in result.content if content.type == "text"]
            return "\n".join(text_result)
        except Exception as e:
            return f"工具调用失败: {str(e)}"
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
        output_type=SensorRecord  # 锁定输出结构
    )

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

        print("🤖 思考中...")

        try:
            resp = await agent.run(user_input, message_history=history)
            history = list(resp.all_messages())

            print("\n回答:")
            # 检查返回的数据有没有 model_dump 方法 (是不是 Pydantic 结构化对象)
            if hasattr(resp.output, 'model_dump'):
                data_dict = resp.output.model_dump()
                for key, value in data_dict.items():
                    # 打印结构化字段
                    print(f"🔸 {key}: {value}")
            else:
                # 兜底：如果只是普通文本聊天，它没有 model_dump 方法，就会走到这里
                # 此时 resp.data 就是一个纯字符串，直接打印即可！
                print(resp.output)
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
