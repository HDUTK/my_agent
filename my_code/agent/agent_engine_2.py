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
    # current_step_index: int = Field(default=0)
    # 🌟 🌟 🌟 【废除 current_step_index，引入任务队列】
    # 里面存的是步骤的索引数字，比如 [0, 1, 2]。为空代表全部做完。
    pending_tasks: list[int] = Field(default_factory=list)

    scratchpad: dict[str, str] = Field(default_factory=dict)
    detailed_results: dict[str, str] = Field(default_factory=dict)

    # 🌟 🌟 🌟 【质检反馈字段】
    feedback: str = Field(default="", description="质检员的报错反馈。如果有值，说明被打回重做")

    # 🌟 🌟 🌟 【人类审批与 Critic 意见字段】
    user_approval: str = Field(default="", description="人类用户的审批结果：Y(满意) 或 N(不满意)")
    critic_feedback: str = Field(default="", description="Critic 专家节点或者人类给出的具体润色修改意见")

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
        # 🌟 🌟 🌟 【装配车间新员工 —— Critic 挑刺润色专家】
        self.critic = Agent(
            model=self.model,
            system_prompt="你是一个眼光毒辣、追求完美的学术与技术报告评审专家。你的任务是审阅现有的报告，结合用户的修改意见，指出报告中逻辑不通、数据模糊或流于表面等缺点，并给出一份详尽的重写指导方案。",
            output_type=str
        )

        # 🌟 构建 LangGraph 状态机流水线
        workflow = StateGraph(EngineState)

        # 1. 添加节点 (也就是车间的工位)
        workflow.add_node("planner_node", self.planner_node)
        workflow.add_node("executor_node", self.executor_node)
        # 🌟 🌟 🌟 【将新员工“质检员”安排上工位】
        workflow.add_node("reviewer_node", self.reviewer_node)
        # 🌟 🌟 🌟 【将人类审批工位与 Critic 工位挂载到图上】
        workflow.add_node("human_approval_node", self.human_approval_node)
        workflow.add_node("critic_node", self.critic_node)

        # 2. 添加常规边 (定死的铁轨)
        workflow.add_edge(START, "planner_node")             # 启动后，必定先去规划师节点
        workflow.add_edge("planner_node", "executor_node")   # 规划完，必定交给执行者节点
        workflow.add_edge("executor_node", "reviewer_node")  # 执行者干完活，必须把结果交给质检员

        # 3. 🌟 添加条件边 (智能分拣道岔，替代原有的 for 循环)
        # 🌟 🌟 🌟 【自动化质检员出来的分流道岔】
        workflow.add_conditional_edges(
            "reviewer_node",
            self.should_continue_after_review,
            {
                "retry_level_a": "executor_node",  # 级别 A 拦截：格式或字数不对，直接原地重试
                "continue": "executor_node",  # 自动化步骤未完：继续做下一个任务
                "go_to_human": "human_approval_node"  # 自动化步骤全完：正式进入级别 C 人工断点审查
            }
        )

        # 🌟 🌟 🌟 【铺设人类审批节点流出后的条件道岔】
        workflow.add_conditional_edges(
            "human_approval_node",
            self.should_continue_after_human,
            {
                "pass_and_end": END,  # 用户敲 Y：大结局，直接输出保存
                "fail_to_critic": "critic_node"  # 用户敲 N：打入冷宫，交给 Critic 节点批判润色
            }
        )

        # 🌟 🌟 🌟 【铺设 Critic 专家批判完后的铁轨】
        # Critic 节点出具完修改方案后，铁轨死死地指向 executor_node，强迫系统带着反思重新整改！
        workflow.add_edge("critic_node", "executor_node")

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
            new_steps = steps
        else:
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

        # 如果有 3 步，队列就是 [0, 1, 2]
        initial_queue = list(range(len(new_steps)))

        return {"steps": new_steps,
                "pending_tasks": initial_queue
                }

    async def executor_node(self, state: EngineState):
        """节点 B：单步执行者。它每次只处理 1 个步骤！"""
        # 🌟 防御性校验
        if not state.pending_tasks:
            return {}

        # TypedDict
        # idx = state["current_step_index"]
        # steps = state["steps"]
        # current_task = steps[idx]
        # max_tokens = state["max_tokens"]

        # Dataclass/Pydantic
        # idx = state.current_step_index
        # 🌟 🌟 🌟 【核心修改 3：永远从队列最前面拿任务，干完为止】
        idx = state.pending_tasks[0]
        steps = state.steps
        current_task = steps[idx]
        max_tokens = state.max_tokens

        # 🌟 🌟 🌟 【如果有报错，将报错信息注入提示词，按头让大模型认错】
        feedback_msg = f"\n❌ [质检驳回信息]：{state.feedback}\n请立刻根据上述报错修改你的输出！\n" if state.feedback else ""

        # 🌟 🌟 🌟 【动态注入高级 Critic 专家和人类的联合整改意见】
        if state.critic_feedback:
            feedback_msg += (
                f"\n⚠️ ⚠️ ⚠️ [专家评审组与用户的联合整改意见]：\n"
                f"{state.critic_feedback}\n"
                f"请彻底吸取上述教训，深度重构并润色当前章节，拒绝敷衍！\n"
            )

        # 打印日志（区分是初次执行还是重做）
        status_tag = "🔄 专家整改重写" if state.critic_feedback else ("❌ 自动化重试" if state.feedback else "⚙️ 正在处理")
        print_and_log(f"\n[{status_tag} {idx + 1}/{len(steps)}] 目标: {current_task}...", "info")

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
            f"直接输出执行结果的纯文本大白话，绝对禁止使用 JSON 或任何排版代码块（除非我要求你这么做）。"
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
        if not state.pending_tasks: return {}

        # idx = state.current_step_index
        # 检查正在执行的这个任务（依然是队列头）
        idx = state.pending_tasks[0]
        current_task = state.steps[idx]

        # 把执行者刚刚存在字典里的最新结果拿出来检查
        latest_content = state.detailed_results.get(current_task, "")

        print_and_log(f"\n🔍 [审查节点] 正在对任务【{current_task}】的产出进行合规检查...", "info")

        # 规则 1：查字数（防敷衍）
        if len(latest_content.strip()) < 10:
            msg = "输出内容过短（少于10个字符），请重新思考并提供详细的分析过程！"
            sys_logger.warning(f"⚠️ [质检未通过] {msg}")
            return {"feedback": msg}  # 记录报错信息

        # 🌟 🌟 🌟 动态判断：用户/规划师是不是本来就想要 JSON？
        is_json_expected = "json" in state.goal.lower() or "json" in current_task.lower()
        # 规则 2：查格式（只有在不期望输出 JSON 的时候，才执行拦截！）
        if not is_json_expected:
            if "```json" in latest_content or ("{" in latest_content and "}" in latest_content):
                msg = "系统检测到非预期的 JSON 格式。本环节必须使用纯文本描述！请去除所有代码块重写！"
                sys_logger.warning(f"⚠️ [质检未通过] {msg}")
                return {"feedback": msg}

        # ✅ 一旦所有规则通过：清除历史报错，并把游标往前推！
        print_and_log("✅ [质检通过] 内容合规！准备归档并推进进度。", "info")

        # 🌟 🌟 🌟 【质检通过！把这个任务从待办队列中“弹”出去！】
        # 比如原本是 [0, 2]，现在变成了 [2]
        new_pending = state.pending_tasks[1:]

        return {
            "feedback": "",
            # "current_step_index": idx + 1  # 🌟 只有质检员签字了，进度才能往前走！
            "pending_tasks": new_pending  # 更新队列！
        }

    # 🌟 🌟 🌟 【级别 C —— 控制台人工审批断点节点】
    @staticmethod
    async def human_approval_node(state: EngineState):
        """人类审查节点。在此挂起流传，把最终合成的完整结果呈报给人类看"""
        print("\n" + "═" * 40 + " 📊 最终报告呈报中心 (人类断点审查) " + "═" * 40)

        # 拼装目前所有的章节内容展现给用户看
        for title, content in state.detailed_results.items():
            print(f"\n【章节：{title}】")
            print("-" * 50)
            print(content)
            print("-" * 50)

        print("\n" + "═" * 110)

        # 💡 利用标准 Python input() 强制卡死传送带，等待主人落子！
        while True:
            user_choice = input(
                "👉 以上为系统为您生成的全量报告。您是否满意？(输入 Y 批准输出 | 输入 N 驳回并送去专家组整改): ").strip().upper()
            if user_choice in ['Y', 'N']:
                break
            print("❌ 输入非法！请严格输入 Y 或 N。")

        # 如果不满意，顺便收集一下主人的“御旨”
        user_opinion = ""

        redo_queue = []
        # 🌟 🌟 🌟 【如果人类不满意，询问具体重做哪些步骤！】
        if user_choice == 'N':
            # 💡 贴心设计：打印一个精简版的步骤目录，防止用户看完长文后忘记序号
            print("\n" + "═" * 20 + " 📑 报告步骤清单（供您点选） " + "═" * 20)
            for i, title in enumerate(state.steps):
                print(f"  [{i + 1}] {title}")
            print("═" * 63 + "\n")

            while True:
                redo_input = input(
                    "🎯 请输入需要重做的步骤序号 (例如输入 '2' 重做第二步，'1,3' 重做第一和第三步。填 '0' 代表全部重做): ").strip()
                if not redo_input:
                    continue

                if redo_input == '0':
                    # 全部重做，队列满载
                    redo_queue = list(range(len(state.steps)))
                    break
                else:
                    try:
                        # 解析逗号分隔的输入 (用户输入 1, 3 -> 我们转成程序认识的 0, 2)
                        redo_queue = [int(x.strip()) - 1 for x in redo_input.split(',')]
                        # 防呆校验
                        if all(0 <= x < len(state.steps) for x in redo_queue):
                            break
                        else:
                            print(f"❌ 序号越界！请输入 1 到 {len(state.steps)} 之间的数字。")
                    except ValueError:
                        print("❌ 格式错误！请使用纯数字和英文逗号（如 1,3）。")

            # 第二问：选定步骤后，再针对性地询问修改意见
            user_opinion = input("\n📝 请输入您的具体修改意见（比如：‘第三章数据太少，重新分析’）：").strip()

            print_and_log(f"🔄 系统已记录重做队列，准备回炉重造步骤: {[x + 1 for x in redo_queue]}", "warning")

        # if user_choice == 'N':
        #     user_opinion = input("📝 请输入您的具体修改意见（比如：‘第三章写得太肤浅，多加点传感器异常案例’）：").strip()

        # 把人类的选择和意见打包存入包裹，送去下一个道岔
        return {
            "user_approval": user_choice,
            "critic_feedback": user_opinion,
            "pending_tasks": redo_queue  # 🌟 将新的重做清单塞回任务队列！
        }

    # 🌟 🌟 🌟 【级别 B —— Critic 智能反思节点】
    async def critic_node(self, state: EngineState):
        """Critic 专家节点。当人类说 N 时触发，用高强度 Prompt 压榨 LLM 生成挑刺方案"""
        print_and_log("\n🛑 [评审会商中...] 正在召集 AI 专家组联合诊断报告缺陷...", "info")

        # 🌟 🌟 🌟 【Critic 只需要看用户圈出来要修改的那些章节，而不是一顿乱喷】
        redo_titles = [state.steps[i] for i in state.pending_tasks]

        # 把当前的报告和人类的意见揉在一起
        current_report_dump = "\n".join([f"## {k}\n{v}" for k, v in state.detailed_results.items()])

        critic_prompt = (
            f"【当前生成的报告初稿如下】\n{current_report_dump}\n\n"
            f"【最终用户（老板）的无情驳回意见如下】\n{state.critic_feedback}\n\n"
            f"请站在极度严苛的视角，指出这份初稿为什么不能让用户满意？"
            f"请给出一份高水平的、一针见血的‘整改方案清单’，告诉 Executor 接下来该如何重写。"
        )

        # 呼叫 Critic 智能体
        critic_resp = await self.critic.run(critic_prompt)

        expert_opinion = critic_resp.output
        print_and_log(f"\n🧠 [专家组评审报告出炉]：\n{expert_opinion}\n", "info")

        # 💡 将游标重置为减 1，让它退回到最后一个任务进行“原地重写整改”，而不是从头洗牌
        # 如果你依然想从头洗牌，可以将这里改为 0
        rollback_index = max(0, state.current_step_index - 1)

        return {
            "critic_feedback": f"【人类意见】：{state.critic_feedback}\n【专家组方案】：{expert_opinion}",
            # "current_step_index": rollback_index  # 时光倒流到倒数最后一步，精准整改
        }


    # -----------------------------------
    # 边函数区 (Edges)
    # -----------------------------------
    # 🌟 🌟 🌟 【分拆并重构条件道岔逻辑】
    @staticmethod
    def should_continue_after_review(state: EngineState) -> str:
        """自动化质检员后面的分流器"""
        if state.feedback:
            return "retry_level_a"  # 级别 A 没过：格式错了，原地重做

        # 🌟 🌟 🌟 【核心修改 6：道岔只看队列里还有没有任务】
        if len(state.pending_tasks) > 0:
            return "continue"
        else:
            return "go_to_human"

        # # 如果自动化流程还没跑完所有步骤，继续推着游标往前跑
        # if state.current_step_index < len(state.steps):
        #     # 此时游标已经在 reviewer_node 内部完成递增了，直接进行边界判断
        #     return "continue"
        # else:
        #     # 自动化全部干完，终于有资格面圣了，移交人类工位
        #     return "go_to_human"

    @staticmethod
    def should_continue_after_human(state: EngineState) -> str:
        """人类审批节点后面的分流器"""
        if state.user_approval == 'Y':
            print_and_log("\n🎉 [大喜] 报告获得人类最终批准！正在准备落盘输出...", "info")
            return "pass_and_end"
        else:
            print_and_log("\n⚠️ [打回] 报告遭到人类驳回！正式移交专家组处理...", "warning")
            return "fail_to_critic"

    # -----------------------------------
    # 对外暴露的启动接口
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

