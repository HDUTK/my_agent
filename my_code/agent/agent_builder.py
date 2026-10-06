#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于Agent 装配车间。负责把模型、MCP 工具、本地高级引擎拼装成最终的 Agent
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/31 16:40
version: V1.0
Target Python:  3.14 -> 3.12
"""

import os
import json
import docx
from typing import Dict, Any

from pydantic_ai import Agent, Tool
from pydantic_ai.exceptions import ModelRetry
from mcp.client.session import ClientSession

# from agent.agent_engine import UniversalPlanExecuteEngine
from agent.agent_engine_2 import UniversalPlanExecuteEngine

from agent.llm_factory import get_llm_model
from utils.core_utils import load_prompt
from RAG.document_parser import read_any_file
from utils.logger_print import sys_logger


# ==========================================
# 工具制造厂 (闭包工厂)
# ==========================================
def make_mcp_tool(session: ClientSession):
    """闭包工厂：接收一个 session，返回一个绑定好的工具调用网关"""
    async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
        sys_logger.info(f"[远程调用] 正在请求 Server 执行工具: {tool_name} | 传入参数: {arguments}")

        try:
            result = await session.call_tool(tool_name, arguments)
            text_result = [content.text for content in result.content if content.type == "text"]
            final_result = "\n".join(text_result)

            # 2. 打印工具的返回结果（如果超过 300 字就截断显示，防止刷屏）
            preview = final_result if len(final_result) < 300 else final_result[:300] + "\n... (内容太长，已省略后续输出)"
            sys_logger.info(f"[Server 返回]\n{preview}")

            return final_result
        except Exception as e:
            error_msg = str(e).lower()

            # 🌟 诊断 1：拦截 JSON 400 报错，强迫模型重新生成！
            if "invalid_parameter" in error_msg or "json" in error_msg or "400" in error_msg:
                sys_logger.warning(f"[格式修复] 检测到大模型生成的 JSON 存在语法错误，已勒令其消耗额度重写...")
                raise ModelRetry(
                    "系统检测到刚才发送的工具参数存在 JSON 格式错误（可能包含了未转义的引号或换行）。请严格去除所有回车和双引号，重新调用本工具！")

            # 🌟 诊断 2：如果是通讯/网络/超时问题 (可以根据实际的报错关键字调整)
            if "timeout" in error_msg or "connection" in error_msg or "network" in error_msg or "mcp" in error_msg:
                sys_logger.warning(f"[网络波动] 检测到通讯异常: {e}。正在扣减重试额度并请求大模型重试...")
                # 抛出 ModelRetry
                # Pydantic-AI 会拦截这个异常，自动消耗一次 retries 额度，并让模型重新发起调用
                raise ModelRetry(f"由于网络通讯异常导致工具调用失败 ({e})。请重新尝试调用此工具。")

            # 🌟 诊断 3：如果是代码逻辑/GBK编码等致命死错
            else:
                sys_logger.error(f"[致命错误] 工具执行崩溃: {e}")
                # 核心拦截：直接返回普通字符串作为结果
                # 大模型看到这句话后，就知道工具废了，不会再执着重试，而是直接回复用户
                return f" 致命系统错误: {str(e)}。请立即放弃重试此工具，并直接向用户汇报系统故障！"
    return call_mcp_tool


# ==========================================
# 🌟 面向对象的万能引擎工具类
# ==========================================
class UniversalEngineTool:
    """
    将重度任务引擎封装为类，实现配置与逻辑的解耦。
    """
    def __init__(self, model_name: str):
        # 初始化时将系统级的参数（模型名称）冻结在实例内部
        self.model_name = model_name

    async def run_engine(
            self,
            goal: str,
            source_files: list[str] = None,
            save_to: str = "output_result.json",
            predefined_steps: list[str] = None
    ) -> str:
        """
        【万能重度任务专用引擎】当你遇到需要深度分析长文本、多步拆解、竞品调研、或者生成复杂报告等长线任务时，必须调用此工具！
        参数:
        - goal: 必须详细描述你要外包团队完成的终极目标。
        - source_files: 引擎需要读取的本地文件路径列表（如 ["test2/Manuscript.docx"]）。如果没有，传入空列表 []。
        - save_to: 最终结果保存的 JSON 文件路径，必须以 .json 结尾。
        - predefined_steps: 如果明确知道分几步，以列表传入（如 ["Abstract", "Introduction"]）。如果想让底层自己规划，传 null。

        ⚠️ 极其重要 (致命红线)：
        一旦你判断用户的任务需要调用此引擎，请【立即、马上】发起 Tool Call！
        绝对禁止向用户回复“好的”、“请稍等”、“我正在为您提取”等任何过渡性寒暄废话！只要你开口说普通文本，系统就会崩溃！
        """
        sys_logger.info(f"[主控中枢] 收到万能调度请求，正在组装上下文并移交 Universal 引擎...")

        # 1. 泛化多文件读取引擎
        context_text = ""
        if source_files:
            for fpath in source_files:
                file_content = read_any_file(fpath)
                context_text += f"\n--- 文件 [{fpath}] 内容 ---\n{file_content}\n"

        # 2. 启动外包流水线引擎 (复用 main.py 里的模型配置)
        engine_model = get_llm_model(self.model_name)
        engine = UniversalPlanExecuteEngine(model_instance=engine_model)

        # 3. 阻塞等待流水线跑完
        results = await engine.run(goal=goal, context=context_text, predefined_steps=predefined_steps)

        # 4. 泛化自动落盘 (大模型传什么路径，就存什么路径)
        try:
            # 如果大模型传了带有子目录的路径，确保目录存在
            os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
            with open(save_to, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            sys_logger.info(f"[引擎落盘] 复杂流水线结果已持久化到: {save_to}")
        except Exception as e:
            msg = f"[引擎落盘失败] 无法保存到 {save_to}: {e}"
            sys_logger.error(msg)
            return msg

        return f" 复杂流水线已在后台成功执行！共完成 {len(results)} 个步骤，详细结果已持久化到 {save_to}。请向用户汇报成功摘要。"


# ==========================================
# Agent 装配流水线
# ==========================================
async def build_agent_with_mcp(session: ClientSession, model_name:str,
                               scenario_name: str = "default_chat") -> Agent[Any, str]:
    """
    Agent 装配厂：负责找 Server 要工具列表，写说明书，最后把大脑和工具组装成 Agent
    """
    # 1. 获取工具列表
    tools_response = await session.list_tools()
    mcp_tools = tools_response.tools

    # 🌟 系统启动时的工具扫描日志
    tool_names = [t.name for t in mcp_tools]
    sys_logger.info(f"📦 [Agent组装] 成功从 Server 发现 {len(mcp_tools)} 个远程工具: {tool_names}")

    # 2. 编写工具说明书 (系统提示词)
    tools_instruction = (
        "你现在连接到了一个本地工具库，可以使用以下工具。\n\n"
        "【工具调用边界 - 必须严格遵守】\n"
        "1. 用户只是打招呼、闲聊、问候、致谢，或没有明确要求读取/查询/修改本地资源时，绝对不要调用任何工具，直接用自然语言简短回复。\n"
        "2. 只有当用户明确提出需要查看文件、列目录、查询 CSV、读取 Word、检索本地知识库、保存结果、修改文件等任务时，才允许调用工具。\n"
        "3. 不要根据示例、系统提示词、历史任务或自己的猜测编造文件夹/文件名。用户没有明确给路径时，先询问用户提供路径。\n"
        "4. 如果工具返回文件或文件夹不存在，最多再尝试一次合理的路径修正；仍失败时必须停止工具调用，并向用户说明需要确认路径。\n\n"
    )
    for tool in mcp_tools:
        tools_instruction += f"- 名称: {tool.name}\n"
        tools_instruction += f"  描述: {tool.description}\n"
        tools_instruction += f"  参数格式: {json.dumps(tool.inputSchema, ensure_ascii=False)}\n\n"
    tools_instruction += "请调用 `call_mcp_tool` 函数来使用上述工具。"

    # 🌟 系统提示词 (System Prompt) 策略
    advanced_prompt = load_prompt(scenario_name)

    # 将人设模板与动态工具列表拼接，形成最终的超级大脑设定
    final_system_prompt = advanced_prompt + tools_instruction

    # 3. 生产对讲机
    my_custom_tool = make_mcp_tool(session)

    # 实例化 OOP 版本的万能引擎，并将其方法提取为 Tool
    engine_instance = UniversalEngineTool(model_name)
    heavy_engine_tool = Tool(engine_instance.run_engine)

    sys_logger.info(f"🚀 [Agent组装完毕] 引擎挂载成功，即将进入对话循环。")

    # 4. 组装并返回 Agent
    return Agent(
        get_llm_model(model_name),
        system_prompt=final_system_prompt,
        tools=[my_custom_tool, heavy_engine_tool],
        output_type=str,  # 锁定输出结构
        retries=3  # 允许工具调用失败后最多重试 3 次，额度用完直接停止
    )
