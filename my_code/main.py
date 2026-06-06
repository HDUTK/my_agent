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

import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

# 导入mcp客户端组件
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# 导入其他的模块
from utils.core_utils import trim_history
from agent.agent_builder import build_agent_with_mcp
from utils.logger_print import sys_logger, print_and_log
# 配置
from config.system_config import my_model_name, my_scenario_name
from config.agent_config import LLM_CONFIG

# 加载所有环境变量配置
load_dotenv()


# ==========================================
# 用户交互界面
# ==========================================
async def run_chat_loop(agent: Agent):
    """控制台聊天引擎：只负责跟用户互动"""
    print_and_log("\n🚀 Agent 已就绪，开始聊天模式。随时输入 'exit' 退出。", "info")
    history = []

    while True:
        user_input = await asyncio.to_thread(input, "\n输入：")

        if user_input.strip().lower() == 'exit':
            print_and_log("👋 Agent 已退出。", "info")
            sys_logger.info("-" * 50)
            sys_logger.info("-" * 50)
            sys_logger.info("-" * 50)
            break

        if not user_input.strip():
            continue

        # 🌟 每次提问前，先对历史记忆进行安全截断（保留最近 max_messages 条有价值的信息）
        history = trim_history(history, max_messages=LLM_CONFIG["history_max_messages"])

        # print(f"🧹 (当前记忆长度: {len(history)} 条)")
        print_and_log("🤖 思考中...\n回答: ", "info", end="", flush=True)

        try:
            # 🌟 把 run() 换成 run_stream()，用 async with 打开“水龙头”
            # 🌟 核心控制：传入 model_settings，限制 max_tokens
            async with agent.run_stream(
                    user_input,
                    message_history=history,
                    model_settings=ModelSettings(max_tokens=LLM_CONFIG["input_max_tokens"])
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
                history = resp.all_messages()

            print("-" * 50)
        except Exception as e:
            # 报错写入日志，但屏幕上也要给用户一个温柔的提示
            sys_logger.error(f"大模型处理或工具调用时遇到错误：\n{e}")
            sys_logger.error("-" * 50)

            print(f"\n🚨 抱歉，大模型处理或工具调用时遇到错误，后台已自动捕获并记录。")
            print(" 程序仍在运行，您可以尝试重新提问，或等待网络恢复。")
            print("-" * 50)


# ==========================================
# Main 函数
# ==========================================
async def main(model_name: str, scenario_name: str):
    sys_logger.info(f"系统启动中... [当前指定模型: {model_name}]")

    # 第一步：定义后厨位置——启动连接（对接 MCP Server）
    server_parameters = StdioServerParameters(command="python", args=["mcp_server.py"])

    # 第二步：接通管道并建立会话
    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            sys_logger.info("✅ 成功连接到 MCP Server！")

            # 第三步：把 session 交给装配厂，换回一个组装好的 Agent
            agent = await build_agent_with_mcp(session, model_name, scenario_name)

            # 第四步：把 Agent 放进聊天引擎
            await run_chat_loop(agent)


if __name__ == "__main__":
    asyncio.run(main(model_name=my_model_name, scenario_name=my_scenario_name))
