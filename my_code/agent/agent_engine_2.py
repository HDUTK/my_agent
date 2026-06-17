#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于动态规划（Planner）、无状态执行（Executor）和本地便签本（State）
【升级说明】已使用 LangGraph 状态机重构，彻底移除了死板的 for 循环。
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/06/17 16:13
version: V1.0
Target Python: 
"""

from typing import Optional, TypedDict
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models import Model

# 🌟 引入 LangGraph 的三大件
from langgraph.graph import StateGraph, START, END

from utils.logger_print import sys_logger, print_and_log
from config.agent_config import PLANNER_MAX_TOKENS, EXECUTOR_MAX_TOKENS


# ==========================================
# 1. 泛化状态 (State) - 流水线上的包裹
# ==========================================
class EngineState(TypedDict):
    """
    用 TypedDict 定义的全局便签本。
    这就是在各个节点之间传来传去的“包裹”，所有节点只能修改里面的数据。
    """
    goal: str
    context: str
    steps: list[str]              # 规划师拆解出来的所有步骤
    current_step_index: int       # 当前正在执行第几个步骤的索引 (相当于游标)
    scratchpad: dict[str, str]    # 记录每个步骤结论的便签本
    detailed_results: dict[str, str] # 最终返回的详细执行结果
    max_tokens: int               # 动态传入的 Token 限制


class TaskPlan(BaseModel):
    steps: list[str] = Field(description="为完成目标拆解出的按顺序执行的具体步骤清单")


# ==========================================
# 2. 泛化引擎核心类 (Graph 构造车间)
# ==========================================
class UniversalPlanExecuteEngine:
    def __init__(self, model_instance: Model):
        """初始化引擎，装载规划师与执行者，并拼接状态机流水线"""
        self.model = model_instance

        # --- 准备“工人” (大模型 Agent) ---
        self.planner = Agent(
            model=self.model,
            system_prompt="你是一个顶级的项目拆解专家...",
            output_type=TaskPlan
        )
        self.executor = Agent(
            model=self.model,
            output_type=str
        )

        # 🌟 构建 LangGraph 状态机流水线
        workflow = StateGraph(EngineState)

        # 1. 添加节点 (也就是车间的工位)
        workflow.add_node("planner_node", self.planner_node)
        workflow.add_node("executor_node", self.executor_node)

        # 2. 添加常规边 (定死的铁轨)
        workflow.add_edge(START, "planner_node")             # 启动后，必定先去规划师节点
        workflow.add_edge("planner_node", "executor_node")   # 规划完，必定交给执行者节点

        # 3. 🌟 添加条件边 (智能分拣道岔，替代原有的 for 循环)
        workflow.add_conditional_edges(
            "executor_node",           # 从哪个节点出来开始分流？
            self.should_continue,      # 负责裁决的条件函数
            {
                "continue": "executor_node", # 如果函数返回 "continue"，就流回执行节点（闭环循环）
                "end": END                   # 如果函数返回 "end"，就流入大结局
            }
        )

        # 编译为可执行应用 (拉下电闸)
        self.app = workflow.compile()


    # -----------------------------------
    # 节点函数区 (Node)
    # 规则：接收 state，只负责返回你想更新的 state 字段
    # -----------------------------------
    async def planner_node(self, state: EngineState):
        """节点 A：规划师。负责生成 steps 列表"""
        goal = state["goal"]
        context = state["context"]
        steps = state["steps"]

        if steps:
            sys_logger.info("检测到预设任务清单，跳过 AI 规划，直接采用预设步骤。")
            # 这里返回 {"steps": steps} 意味着更新包裹里的 steps 字段
            return {"steps": steps}

        sys_logger.info("正在呼叫规划师进行任务拆解...")
        plan_resp = await self.planner.run(
            f"目标：{goal}\n\n背景上下文：\n{context[:2000]}...",
            model_settings=ModelSettings(max_tokens=PLANNER_MAX_TOKENS)
        )

        usage = plan_resp.usage()
        print_and_log(f" [Token结算 - 规划师] 输入: {usage.input_tokens} | 输出: {usage.output_tokens}", "info")

        new_steps = plan_resp.output.steps
        sys_logger.info(f"执行清单已确认，共 {len(new_steps)} 步：")
        for i, s in enumerate(new_steps):
            sys_logger.info(f"   [{i + 1}] {s}")

        return {"steps": new_steps}

    async def executor_node(self, state: EngineState):
        """节点 B：单步执行者。它每次只处理 1 个步骤！"""
        idx = state["current_step_index"]
        steps = state["steps"]
        current_task = steps[idx]
        max_tokens = state["max_tokens"]

        print_and_log(f"\n⚙️ [执行节点 {idx + 1}/{len(steps)}] 正在处理: {current_task}...", "info")

        scratchpad_view = "\n".join([f"- {k}: {v}" for k, v in state["scratchpad"].items()]) or "目前是第一步，暂无历史进度。"

        dynamic_prompt = (
            f"你的全局终极目标是：{state['goal']}\n\n"
            f"【全局便签本 / 当前进度摘要】\n{scratchpad_view}\n\n"
            f"【背景资料】\n{state['context']}\n\n"
            f"⚠️ 系统指令：请基于资料，立刻执行当前任务：【{current_task}】。\n"
            f"直接输出执行结果的纯文本大白话，绝对禁止使用 JSON 或任何排版代码块。"
        )

        content_chunks = []
        async with self.executor.run_stream(
                dynamic_prompt,
                model_settings=ModelSettings(max_tokens=max_tokens)
        ) as resp:
            async for text_chunk in resp.stream_text(delta=True):
                print(text_chunk, end="", flush=True)
                content_chunks.append(text_chunk)
            print()

        usage = resp.usage()
        print_and_log(f" [Token结算 - 执行节点 {idx + 1}] 输出: {usage.output_tokens}", "info")

        content = "".join(content_chunks)
        sys_logger.info(f"[执行节点 {idx + 1} 完整回复]\n{content}")

        # 🌟 获取旧的数据字典，然后进行拷贝修改
        new_detailed_results = state["detailed_results"].copy()
        new_detailed_results[current_task] = content

        new_scratchpad = state["scratchpad"].copy()
        snippet = content[:100].replace("\n", "") + "..."
        new_scratchpad[current_task] = f"[✅ 完成] 摘要: {snippet}"

        # 核心：执行完后，把游标 (index) 加 1，并更新字典
        return {
            "current_step_index": idx + 1,
            "scratchpad": new_scratchpad,
            "detailed_results": new_detailed_results
        }


    # -----------------------------------
    # 边函数区 (Edges)
    # -----------------------------------
    def should_continue(self, state: EngineState) -> str:
        """分拣道岔：判断是循环回 executor 还是结束"""
        # 如果游标还没走到列表尽头，说明还有任务没做完
        if state["current_step_index"] < len(state["steps"]):
            return "continue"
        else:
            return "end"


    # -----------------------------------
    # 对外暴露的启动接口 (不变！)
    # -----------------------------------
    async def run(
            self,
            goal: str,
            context: str = "",
            predefined_steps: Optional[list[str]] = None,
            max_tokens: int = EXECUTOR_MAX_TOKENS
    ) -> dict[str, str]:
        """外部调用依然是旧的配方，内部已经是全新的流水线。"""
        sys_logger.info(f"\n⚙ [引擎启动] 终极目标: {goal}")

        # 1. 准备初始包裹 (包裹里的数据必须跟 TypedDict 严丝合缝)
        initial_state: EngineState = {
            "goal": goal,
            "context": context,
            "steps": predefined_steps or [],
            "current_step_index": 0,
            "scratchpad": {},
            "detailed_results": {},
            "max_tokens": max_tokens
        }

        # 2. 🌟 一键启动整个图！ainvoke 会把状态机跑到 END 为止，并返回最终状态
        final_state = await self.app.ainvoke(initial_state)

        sys_logger.info("[引擎完工] 所有分布式思维链节点已执行完毕！")

        # 3. 从最终的包裹里掏出我们想要的结果返回给外界
        return final_state["detailed_results"]

