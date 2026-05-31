#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于动态规划（Planner）、无状态执行（Executor）和本地便签本（State）
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/30 20:49
version: V1.0
Target Python: 
"""

from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models import Model


# ==========================================
# 1. 泛化状态与输出模型
# ==========================================
@dataclass
class EngineState:
    goal: str
    context: str
    scratchpad: dict[str, str]  # 记录步骤和结论的万能便签本


class TaskPlan(BaseModel):
    steps: list[str] = Field(description="为完成目标拆解出的按顺序执行的具体步骤清单")


# ==========================================
# 2. 泛化引擎核心类
# ==========================================
class UniversalPlanExecuteEngine:
    def __init__(self, model_instance: Model):
        """初始化引擎，装载规划师与执行者"""
        self.model = model_instance

        # 规划师：专职拆解任务，输出结构化 JSON 列表
        self.planner = Agent(
            model=self.model,  # 🌟 传入模型
            system_prompt="你是一个顶级的项目拆解专家...",
            output_type=TaskPlan
        )

        # 执行者：无状态的干活机器，专注处理单一任务
        self.executor = Agent(
            model=self.model,  # 🌟 传入模型
            output_type=str
        )

    async def run(
            self,
            goal: str,
            context: str = "",
            predefined_steps: Optional[list[str]] = None,
            max_tokens: int = 3000
    ) -> dict[str, str]:
        """
        启动万能流水线
        :param goal: 核心目标说明
        :param context: 背景资料（论文文本、需求文档等）
        :param predefined_steps: 如果您已经知道要干嘛（比如提炼7个章节），直接传进来；如果传 None，引擎会让 AI 自己规划。
        :param max_tokens: 最大token数量
        :return: 包含所有步骤详细执行结果的字典
        """
        print(f"\n⚙️ [引擎启动] 终极目标: {goal}")
        state = EngineState(goal=goal, context=context, scratchpad={})

        # -----------------------------------
        # 阶段一：任务拆解 (Plan)
        # -----------------------------------
        if predefined_steps:
            steps = predefined_steps
            print("📝 检测到预设任务清单，跳过 AI 规划，直接采用预设步骤。")
        else:
            print("🧠 正在呼叫规划师进行任务拆解...")
            plan_resp = await self.planner.run(
                f"目标：{goal}\n\n背景上下文：\n{context[:2000]}...",  # 规划时不必传全文
                model_settings=ModelSettings(max_tokens=800)
            )

            # 🌟 结算规划师的 Token 消耗
            usage = plan_resp.usage()
            print(
                f"📊 [Token结算 - 规划师] 输入: {usage.input_tokens} | 输出: {usage.output_tokens} | 累计: {usage.total_tokens}")

            steps = plan_resp.output.steps

        print(f"📋 执行清单已确认，共 {len(steps)} 步：")
        for i, s in enumerate(steps):
            print(f"   [{i + 1}] {s}")

        # -----------------------------------
        # 阶段二：分布式执行 (Execute)
        # -----------------------------------
        detailed_results = {}

        for i, current_task in enumerate(steps):
            print(f"\n👷 [执行节点 {i + 1}/{len(steps)}] 正在处理: {current_task}...")

            # 动态组装上下文：让大模型永远拥有大局观，但没有历史包袱
            scratchpad_view = "\n".join([f"- {k}: {v}" for k, v in state.scratchpad.items()]) or "目前是第一步，暂无历史进度。"

            dynamic_prompt = (
                f"你的全局终极目标是：{state.goal}\n\n"
                f"【全局便签本 / 当前进度摘要】\n{scratchpad_view}\n\n"
                f"【背景资料】\n{state.context}\n\n"
                f"⚠️ 系统指令：请基于资料，立刻执行当前任务：【{current_task}】。\n"
                f"直接输出执行结果的纯文本大白话，绝对禁止使用 JSON 或任何排版代码块。"
            )

            # 🌟 流式输出
            async with self.executor.run_stream(
                    dynamic_prompt,
                    model_settings=ModelSettings(max_tokens=max_tokens)
            ) as resp:

                # 开启水龙头，在屏幕上实时流式输出当前节点的内容
                content_chunks = []
                async for text_chunk in resp.stream_text(delta=True):
                    print(text_chunk, end="", flush=True)
                    content_chunks.append(text_chunk)
                print()  # 💡 当前节点内容流完后，打印一个换行收尾

            # 🌟 结算当前执行节点的 Token 消耗
            usage = resp.usage
            print(
                f"📊 [Token结算 - 执行节点 {i + 1}] 输入: {usage.input_tokens} | 输出: {usage.output_tokens} | 累计: {usage.total_tokens}")

            # 完整结果落袋为安
            content = "".join(content_chunks)
            detailed_results[current_task] = content

            # 记忆物理切除：只保留前 50 个字存入便签本供下一步参考
            snippet = content[:50].replace("\n", "") + "..."
            state.scratchpad[current_task] = f"[✅ 完成] 摘要: {snippet}"

        print("\n🎉 [引擎完工] 所有分布式思维链节点已执行完毕！")
        return detailed_results
