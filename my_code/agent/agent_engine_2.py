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
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models import Model

# 🌟 引入 LangGraph 的三大件
from langgraph.graph import StateGraph, START, END

from utils.logger_print import sys_logger, print_and_log
from config.agent_config import PLANNER_MAX_TOKENS, EXECUTOR_MAX_TOKENS


# ==========================================
# 1. 泛化状态 (State) - 流水线上的包裹（TypedDict 版）
# ==========================================
# class EngineState(TypedDict):
#     """
#     用 TypedDict 定义的全局便签本。
#     这就是在各个节点之间传来传去的“包裹”，所有节点只能修改里面的数据。
#     """
#     goal: str
#     context: str
#     steps: list[str]              # 规划师拆解出来的所有步骤
#     current_step_index: int       # 当前正在执行第几个步骤的索引 (相当于游标)
#     scratchpad: dict[str, str]    # 记录每个步骤结论的便签本
#     detailed_results: dict[str, str] # 最终返回的详细执行结果
#     max_tokens: int               # 动态传入的 Token 限制


# ==========================================
# 1. 泛化状态 (State) - 流水线上的包裹 (Dataclass 版)
# ==========================================
# @dataclass
# class EngineState:
#     goal: str
#     context: str
#     max_tokens: int
#
#     # 使用 field 赋予默认值，这样初始化时就不用手动塞空列表和空字典了
#     steps: list[str] = field(default_factory=list)
#     current_step_index: int = 0
#     scratchpad: dict[str, str] = field(default_factory=dict)
#     detailed_results: dict[str, str] = field(default_factory=dict)


# ==========================================
# 1. 泛化状态 (State) - 流水线上的包裹 (Pydantic 版)
# ==========================================
class EngineState(BaseModel):
    goal: str = Field(description="核心终极目标")
    context: str = Field(default="", description="背景上下文文本")
    max_tokens: int

    # Pydantic 同样支持默认值工厂
    steps: list[str] = Field(default_factory=list)
    current_step_index: int = Field(default=0)
    scratchpad: dict[str, str] = Field(default_factory=dict)
    detailed_results: dict[str, str] = Field(default_factory=dict)

    # 🌟 🌟 🌟 【质检反馈字段】
    feedback: str = Field(default="", description="质检员的报错反馈。如果有值，说明被打回重做")


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
        # 🌟 🌟 🌟 【将新员工“质检员”安排上工位】
        workflow.add_node("reviewer_node", self.reviewer_node)

        # 2. 添加常规边 (定死的铁轨)
        workflow.add_edge(START, "planner_node")             # 启动后，必定先去规划师节点
        workflow.add_edge("planner_node", "executor_node")   # 规划完，必定交给执行者节点
        workflow.add_edge("executor_node", "reviewer_node")  # 执行者干完活，必须把结果交给质检员

        # 3. 🌟 添加条件边 (智能分拣道岔，替代原有的 for 循环)
        workflow.add_conditional_edges(
            "reviewer_node",    # 现在由质检员节点出来分流！
            self.should_continue,      # 负责裁决的条件函数
            {
                "retry": "executor_node",  # 驳回：打回给执行者重写
                "continue": "executor_node",  # 通过：继续做下一个任务
                "end": END  # 通过：全部做完，杀青
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
        # TypedDict
        # goal = state["goal"]
        # context = state["context"]
        # steps = state["steps"]

        # Dataclass/Pydantic
        goal = state.goal
        context = state.context
        steps = state.steps

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
        # TypedDict
        # idx = state["current_step_index"]
        # steps = state["steps"]
        # current_task = steps[idx]
        # max_tokens = state["max_tokens"]

        # Dataclass/Pydantic
        idx = state.current_step_index
        steps = state.steps
        current_task = steps[idx]
        max_tokens = state.max_tokens

        # 🌟 🌟 🌟 【如果有报错，将报错信息注入提示词，按头让大模型认错】
        feedback_msg = f"\n❌ [质检驳回信息]：{state.feedback}\n请立刻根据上述报错修改你的输出！\n" if state.feedback else ""
        # 打印日志（区分是初次执行还是重做）
        status_tag = "🔄 打回重做" if state.feedback else "⚙️ 正在处理"
        print_and_log(f"\n⚙️ [{status_tag} {idx + 1}/{len(steps)}] 目标: {current_task}...", "info")

        # TypedDict
        # scratchpad_view = "\n".join([f"- {k}: {v}" for k, v in state["scratchpad"].items()]) or "目前是第一步，暂无历史进度。"
        # Dataclass/Pydantic
        scratchpad_view = "\n".join([f"- {k}: {v}" for k, v in state.scratchpad.items()]) or "目前是第一步，暂无历史进度。"

        dynamic_prompt = (
            # TypedDict
            # f"你的全局终极目标是：{state['goal']}\n\n"
            # Dataclass/Pydantic
            f"你的全局终极目标是：{state.goal}\n\n"
            f"【全局便签本 / 当前进度摘要】\n{scratchpad_view}\n\n"
            # TypedDict
            # f"【背景资料】\n{state['context']}\n\n"
            # Dataclass/Pydantic
            f"【背景资料】\n{state.context}\n\n"
            f"⚠️ 系统指令：请基于资料，立刻执行当前任务：【{current_task}】。\n"
            f"直接输出执行结果的纯文本大白话，绝对禁止使用 JSON 或任何排版代码块。"
            f"{feedback_msg}"  # 🌟 把驳回信息拼接在最后，加重权重
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
        # TypedDict
        # new_detailed_results = state["detailed_results"].copy()
        # Dataclass/Pydantic
        new_detailed_results = state.detailed_results.copy()
        new_detailed_results[current_task] = content

        # TypedDict
        # new_scratchpad = state["scratchpad"].copy()
        # Dataclass/Pydantic
        new_scratchpad = state.scratchpad.copy()
        snippet = content[:100].replace("\n", "") + "..."
        new_scratchpad[current_task] = f"[✅ 暂存] 摘要: {snippet}"

        # 执行节点不再有资格推动进度 (游标 +1被移除了)】
        # 它只负责干活，能不能进入下一步，得看下一步的质检员点头！
        return {
            "scratchpad": new_scratchpad,
            "detailed_results": new_detailed_results
        }

    # 🌟 🌟 🌟 【质检员节点 (级别 A 纯代码规则审查)】
    @staticmethod
    async def reviewer_node(state: EngineState):
        """节点 C：质检员。使用 Python 规则检查执行者的产出"""
        idx = state.current_step_index
        current_task = state.steps[idx]

        # 把执行者刚刚存在字典里的最新结果拿出来检查
        latest_content = state.detailed_results.get(current_task, "")

        print_and_log(f"\n🔍 [审查节点] 正在对任务【{current_task}】的产出进行合规检查...", "info")

        # 规则 1：查字数（防敷衍）
        if len(latest_content.strip()) < 10:
            msg = "输出内容过短（少于10个字符），请重新思考并提供详细的分析过程！"
            sys_logger.warning(f"⚠️ [质检未通过] {msg}")
            return {"feedback": msg}  # 记录报错信息

        # 规则 2：查格式（因为我们在 prompt 里严禁了它输出 JSON）
        if "```json" in latest_content or ("{" in latest_content and "}" in latest_content):
            msg = "你生成的内容中包含了 JSON 格式字符。系统已明确要求必须使用纯文本！请去除所有代码块重写！"
            sys_logger.warning(f"⚠️ [质检未通过] {msg}")
            return {"feedback": msg}  # 记录报错信息

        # ✅ 一旦所有规则通过：清除历史报错，并把游标往前推！
        print_and_log("✅ [质检通过] 内容合规！准备归档并推进进度。", "info")
        return {
            "feedback": "",
            "current_step_index": idx + 1  # 🌟 只有质检员签字了，进度才能往前走！
        }


    # -----------------------------------
    # 边函数区 (Edges)
    # -----------------------------------
    @staticmethod
    def should_continue(state: EngineState) -> str:
        """分拣道岔：判断是循环还是结束"""
        # 如果游标还没走到列表尽头，说明还有任务没做完
        if state.feedback:
            return "retry"  # 包裹里夹带着报错单，立刻打回执行者重写！
        # TypedDict
        # if state["current_step_index"] < len(state["steps"]):
        # Dataclass/Pydantic
        if state.current_step_index < len(state.steps):
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

        # 1. 准备初始包裹 (包裹里的数据必须跟 TypedDict 严丝合缝) TypedDict
        # initial_state: EngineState = {
        #     "goal": goal,
        #     "context": context,
        #     "steps": predefined_steps or [],
        #     "current_step_index": 0,
        #     "scratchpad": {},
        #     "detailed_results": {},
        #     "max_tokens": max_tokens
        # }
        # Dataclass/Pydantic
        initial_state = EngineState(
            goal=goal,
            context=context,
            steps=predefined_steps or [],
            max_tokens=max_tokens
        )  # 其他的 index 和 dict 都会自动使用我们定义的默认值！

        # 2. 🌟 一键启动整个图！ainvoke 会把状态机跑到 END 为止，并返回最终状态
        final_state = await self.app.ainvoke(initial_state)

        sys_logger.info("[引擎完工] 所有分布式思维链节点已执行完毕！")

        # 3. 从最终的包裹里掏出我们想要的结果返回给外界
        if isinstance(final_state, dict):
            return final_state["detailed_results"]
        else:
            return final_state.detailed_results

