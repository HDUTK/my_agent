#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于工业级 Agent 全层级记忆管理系统
核心架构:
  - 第 1 层: Token 敏感短期裁剪 + 工具输出剪枝瘦身 + API 协议原子排雷
  - 第 2 层: 滚动会话摘要 (Summary Memory)
  - 第 3 层: 长期事实档案 (Fact Store)
    * 策略 1: 正则门禁（Gatekeeper）0 开销拦截显式设定，实时原子覆盖
    * 策略 2: 顺风车合并提取（Piggybacking），在淘汰消息做滚动摘要时顺手打捞潜藏事实
    * 策略 3: 本地结构化 JSON 存储，支持 Notepad++ 随时手工审查与修改
  - 🌟 统一门面: UnifiedMemoryManager 将上述所有生命周期封装为单一接口，彻底净化主调度中心
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/10/17 19:02
version: V1.0
Target Python: 
"""

import os
import re
import json
import copy
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart
)

from agent.llm_factory import get_llm_model
from config.system_config import MEMORY_CONFIG
from utils.logger_print import sys_logger

# =========================================================================
# 1. Token 计量组件 (基于 tiktoken，支持未安装时平滑降级)
# =========================================================================
try:
    import tiktoken

    # 使用与主流大模型 (GPT-4 / Qwen / Claude 兼容端点) 对齐的 cl100k_base 编码
    _ENCODER = tiktoken.get_encoding("cl100k_base")

    def count_string_tokens(text: str) -> int:
        if not text:
            return 0
        return len(_ENCODER.encode(str(text), disallowed_special=()))
except ImportError:
    # 优雅降级：若未安装 tiktoken，按中英文混合经验公式估算 (1 token ≈ 1.2 字符)
    def count_string_tokens(text: str) -> int:
        if not text:
            return 0
        return int(len(str(text)) * 0.8) + 1


def estimate_message_tokens(message: Any) -> int:
    """估算单条 Pydantic-AI 消息所占用的 Token 数量"""
    total_text = []

    # 1. 获取消息内包含的所有部件 (parts)
    parts = getattr(message, "parts", [])
    for part in parts:
        # 如果是用户提问、大模型回复或工具返回值，提取它的文本内容
        if hasattr(part, "content") and part.content:
            total_text.append(str(part.content))
        # 如果是大模型发起的工具调用指令，提取它的参数字典 (args)
        if hasattr(part, "args") and part.args:
            total_text.append(str(part.args))

    # 2. 把拆出来的所有文本拼在一起，送去计算 Token
    full_payload = "".join(total_text)
    # 基础消息开销预留 4 个 Token,
    # 主流大模型在底层传输消息时，每条消息前后都会包裹几位固定的协议元数据标记（ChatML 格式）
    # 这些协议标记（角色名、起始符、结束符、换行符）本身固定占用 3 ~ 4 个 Token
    return count_string_tokens(full_payload) + 4


# =========================================================================
# 2. 工具输出剪枝与安全历史修剪 (第 1 层短期记忆)
# =========================================================================
def prune_older_tool_returns(history: List[ModelMessage],
                             keep_last_n_tools: Optional[int] = None,
                             char_threshold: Optional[int] = None) -> List[ModelMessage]:
    """
    对历史中的工具返回结果进行剪枝瘦身：
    只保留最近 N 轮工具调用的完整返回值；更早轮次的庞大文本替换为轻量占位符。
    """
    if not history:
        return history

    keep_n = keep_last_n_tools if keep_last_n_tools is not None else MEMORY_CONFIG["keep_last_n_tools"]
    threshold = char_threshold if char_threshold is not None else MEMORY_CONFIG["tool_prune_char_threshold"]

    # 1. 深度拷贝，避免污染外部运行时的原始消息对象
    cloned_history = copy.deepcopy(history)

    # 2. 定位所有包含 ToolReturnPart 的 ModelRequest 索引
    tool_req_indices = []
    for idx, msg in enumerate(cloned_history):
        if isinstance(msg, ModelRequest):
            if any(isinstance(part, ToolReturnPart) for part in msg.parts):
                tool_req_indices.append(idx)

    # 如果工具调用次数未超过保留轮数，无需折叠
    if len(tool_req_indices) <= keep_n:
        return cloned_history

    # 需要被折叠瘦身的请求索引集合
    indices_to_compact = set(tool_req_indices[:-keep_n])

    # 3. 执行剪枝与内容浓缩(执行占位替换)
    for idx in indices_to_compact:
        msg = cloned_history[idx]
        new_parts = []
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                raw_content = str(part.content)
                if len(raw_content) > threshold:
                    snippet = raw_content[:40].replace("\n", " ").strip()
                    compact_desc = (
                        f"[历史工具调用: {part.tool_name} 结果已由大模型总结完毕 | "
                        f"前缀摘要: {snippet}... | 原始数据已折叠(省去 {len(raw_content)} 字符)]"
                    )

                    if hasattr(part, "model_copy"):
                        compact_part = part.model_copy(update={"content": compact_desc})
                    elif hasattr(part, "copy"):
                        compact_part = part.copy(update={"content": compact_desc})
                    else:
                        part.content = compact_desc
                        compact_part = part

                    new_parts.append(compact_part)
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)

        msg.parts = new_parts

    return cloned_history


# # =========================================================================
# # 3. 独立安全修剪函数 (原 core_utils 中的 trim_history 升级版)
# # =========================================================================
# def trim_history(
#         history: List[ModelMessage],
#         token_budget: Optional[int] = None,
#         max_messages: Optional[int] = None,
#         prune_tools: Optional[bool] = None
# ) -> List[ModelMessage]:
#     """
#     轻量单步历史裁剪器：
#     1. 工具剪枝 -> 2. Token 预算打捞 -> 3. API 协议安全排雷
#     """
#     if not history:
#         return []
#
#     budget = token_budget if token_budget is not None else MEMORY_CONFIG["token_budget"]
#     msg_limit = max_messages if max_messages is not None else MEMORY_CONFIG["max_messages"]
#     do_prune = prune_tools if prune_tools is not None else MEMORY_CONFIG["enable_tool_pruning"]
#
#     # 阶段 1：工具剪枝
#     processed_history = prune_older_tool_returns(history) if do_prune else history
#
#     # 阶段 2：基于 Token 预算倒序截取
#     selected: List[ModelMessage] = []
#     current_tokens = 0
#
#     for msg in reversed(processed_history):
#         t_cost = estimate_message_tokens(msg)
#         if (current_tokens + t_cost > budget) or (len(selected) >= msg_limit):
#             break
#         selected.append(msg)
#         current_tokens += t_cost
#
#     # 恢复正序时间流
#     selected.reverse()
#
#     # 阶段 3：原子调用排雷（首条必须是合法用户请求）
#     while selected:
#         first_msg = selected[0]
#         is_invalid = False
#
#         # 雷区 A：首条不能是模型生成的回复
#         if isinstance(first_msg, ModelResponse):
#             is_invalid = True
#
#         # 雷区 B：首条不能是包含了孤立工具结果的请求
#         elif isinstance(first_msg, ModelRequest):
#             if any(isinstance(p, ToolReturnPart) for p in first_msg.parts):
#                 is_invalid = True
#
#         if is_invalid:
#             selected.pop(0)
#         else:
#             break
#
#     # 极端防御：如果排雷后列表为空，保留最后一条有效消息作为保底
#     return selected if selected else (history[-1:] if history else [])


# =========================================================================
# 淘汰消息转译器 (供摘要大模型阅读)
# =========================================================================
def serialize_evicted_messages(messages: List[ModelMessage]) -> str:
    """将滑出短期工作记忆的复杂对象序列化为干净的纯文本对话纪要"""
    dialogue_lines = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    dialogue_lines.append(f"用户: {part.content}")
                elif isinstance(part, ToolReturnPart):
                    preview = str(part.content)[:60].replace("\n", " ").strip()
                    dialogue_lines.append(f"[工具 {part.tool_name} 返回: {preview}...]")

        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart) and part.content:
                    dialogue_lines.append(f"AI助手: {part.content.strip()}")
                elif isinstance(part, ToolCallPart):
                    dialogue_lines.append(f"[AI助手决定调用工具: {part.tool_name}]")

    return "\n".join(dialogue_lines)


# =========================================================================
# 3. 长期事实档案库 (FactStore) - 策略 3 纯文本透明落盘
# =========================================================================
class FactStore:
    """
    长期事实档案管理器：
    - 本地 JSON 结构化存储，便于 Notepad++ 随时手工核对修改
    - 采用语义 Key-Value 唯一键原则：相同 Key 自动物理覆盖，绝无历史矛盾
    - 提供 ADD (新增), UPDATE (覆盖修改), DELETE (作废清除) 三大操作
    """

    def __init__(self, file_path: Optional[str] = None):
        # 1. 优先使用外部传入路径，若未传则读取 system_config 配置中的 fact_store_path
        target_path = file_path or MEMORY_CONFIG.get("fact_store_path")

        # 2. 极端缺失兜底：如果配置项不存在，默认指向同级 config/user_facts.json
        if not target_path:
            from config.system_config import CONFIG_DIR
            target_path = str(CONFIG_DIR / "user_facts.json")

        self.file_path = Path(target_path)
        sys_logger.info(f"📂 [事实档案] 已绑定持久化文件路径: {self.file_path.resolve()}")

        # 3. 从本地加载现有事实
        self.facts: Dict[str, Dict[str, str]] = self._load_facts()

    def _load_facts(self) -> Dict[str, Dict[str, str]]:
        """从本地磁盘读取 JSON 档案，若不存在则返回三类标准空白骨架"""
        if not self.file_path.exists():
            return {
                "user_profile": {},         # 用户科研身份、个人背景
                "project_constraints": {},  # 监测窟号、模型名称、点位规范等刚性约束
                "user_preferences": {}      # 开发框架、回复风格等偏好
            }
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            sys_logger.error(f"❌ [事实档案] 读取 {self.file_path.name} 失败: {e}，启用空档案。")
            return {"user_profile": {}, "project_constraints": {}, "user_preferences": {}}

    def _save_facts(self) -> bool:
        """将内存中的最新事实原子写入硬盘 (如果 config 目录未创建则自动创建)"""
        try:
            # 确保 my_code/config 目录存在
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                # ensure_ascii=False 确保中文字符可读，indent=4 便于人类阅读排版
                json.dump(self.facts, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            sys_logger.error(f"❌ [事实档案] 落盘失败: {e}")
            return False

    def apply_fact_changes(self, change_items: List[Dict[str, Any]]) -> None:
        """
        根据大模型提取出的变动指令列表，执行原子增、删、改
        每个 item 格式:
        {"category": "user_profile", "key": "phd_topic", "value": "...", "action": "ADD/UPDATE/DELETE"}
        """
        if not change_items:
            return

        modified = False
        for item in change_items:
            category = item.get("category", "project_constraints")
            key = item.get("key", "").strip()
            value = item.get("value", "").strip()
            action = item.get("action", "ADD").upper()

            if not key:
                continue

            if category not in self.facts:
                self.facts[category] = {}

            # ----------------------------------------------------
            # 动作 1: 删除陈旧事实
            # ----------------------------------------------------
            if action == "DELETE":
                if key in self.facts[category]:
                    removed_val = self.facts[category].pop(key)
                    sys_logger.info(f"🗑️ [事实删除] [{category}] -> {key} 已移除 (原值: {removed_val})")
                    modified = True

            # ----------------------------------------------------
            # 动作 2: 新增或覆盖已有事实 (ADD / UPDATE)
            # ----------------------------------------------------
            else:
                is_override = key in self.facts[category]
                old_val = self.facts[category].get(key, "")

                # 键值赋值，完成天然覆盖
                self.facts[category][key] = value

                if is_override:
                    sys_logger.info(f"🔄 [事实覆盖] [{category}] -> {key} 已修改:\n   旧: {old_val}\n   新: {value}")
                else:
                    sys_logger.info(f"✨ [事实新增] [{category}] -> {key}: {value}")
                modified = True

        if modified:
            self._save_facts()

    def get_formatted_facts(self) -> str:
        """格式化输出为 System Prompt 可读的背景事实块"""
        if not any(self.facts.values()):
            return ""

        lines = ["📌【长期记忆事实档案 (Persistent Facts)】"]
        titles = {
            "user_profile": "👤 用户科研与个人背景",
            "project_constraints": "🏛️ 项目业务约束与术语规范",
            "user_preferences": "⚙️ 交互偏好与开发环境"
        }

        for cat, items in self.facts.items():
            if not items:
                continue
            lines.append(f"\n{titles.get(cat, cat)}:")
            for k, v in items.items():
                lines.append(f"- **{k}**: {v}")

        return "\n".join(lines)


# =========================================================================
# 4. 策略 1 核心组件：0 开销正则门禁与即时提取器
# =========================================================================
class FastFactGatekeeper:
    """
    【策略 1 门禁】
    使用纯 Python 正则表达式进行毫秒级意图拦截。
    耗时 < 0.1ms，0 Token 开销，过滤 99% 的普通问答，只有明确表达设定时才放行。
    """

    def __init__(self):
        patterns = MEMORY_CONFIG.get("fact_trigger_keywords", [])
        self.regex = re.compile("|".join(patterns)) if patterns else None

    def should_trigger(self, user_text: str) -> bool:
        """检查用户提问是否命中事实变更意图"""
        if not self.regex or not user_text or len(user_text.strip()) < 3:
            return False
        return bool(self.regex.search(user_text))


# =========================================================================
# 5. 滚动摘要与顺风车提炼引擎 (RollingSummaryEngine)
# =========================================================================
class RollingSummaryEngine:
    """负责 Token 切分判定、协议排雷以及摘要与事实顺风车合并提取"""
    def __init__(self, summarizer_model: Any, token_budget: Optional[int] = None):
        self.summarizer_model = summarizer_model
        self.token_budget = token_budget if token_budget is not None else MEMORY_CONFIG["token_budget"]
        self.current_summary: str = ""

        self.summary_agent = Agent(
            model=self.summarizer_model,
            system_prompt="你是一个长程对话核心纪要提炼专家，负责将旧对话要点增量合并入全局背景摘要，并挖掘需要持久化的关键业务事实。",
            output_type=str
        )

    def extract_evicted_and_surviving(
        self,
        history: List[ModelMessage]
    ) -> Tuple[List[ModelMessage], List[ModelMessage]]:
        """
        核心记忆切分逻辑：
        步骤 1: 工具结果剪枝
        步骤 2: 基于 Token 预算回溯打捞
        步骤 3: 严格 API 协议排雷，杜绝 400 孤立调用
        """
        if not history:
            return [], []

        # 步骤 1: 工具输出剪枝
        if MEMORY_CONFIG["enable_tool_pruning"]:
            history = prune_older_tool_returns(history)

        surviving: List[ModelMessage] = []
        accumulated_tokens = 0

        # 步骤 2: 基于 Token 与条数预算倒序打捞
        for msg in reversed(history):
            t_cost = estimate_message_tokens(msg)
            if (accumulated_tokens + t_cost > self.token_budget) or (len(surviving) >= MEMORY_CONFIG["max_messages"]):
                break
            surviving.append(msg)
            accumulated_tokens += t_cost

        surviving.reverse()
        cutoff_index = len(history) - len(surviving)
        evicted = history[:cutoff_index]

        # 步骤 3: 协议排雷，确保起点因果闭环
        while surviving:
            first_msg = surviving[0]
            is_invalid = False
            if isinstance(first_msg, ModelResponse):
                is_invalid = True
            elif isinstance(first_msg, ModelRequest):
                if any(isinstance(p, ToolReturnPart) for p in first_msg.parts):
                    is_invalid = True

            if is_invalid:
                evicted.append(surviving.pop(0))
            else:
                break

        return surviving, evicted

    async def update_summary_and_harvest_facts(
        self,
        evicted_messages: List[ModelMessage],
        fact_store: Optional[FactStore] = None
    ) -> None:
        """单次调用大模型，合并更新摘要并顺带沉淀事实"""
        if not evicted_messages:
            return

        evicted_text = serialize_evicted_messages(evicted_messages)
        if len(evicted_text.strip()) < MEMORY_CONFIG["summary_trigger_min_chars"]:
            return

        sys_logger.info(f"🧠 [记忆管家] {len(evicted_messages)} 条旧对话滑出窗口，启动顺风车增量提炼...")

        prompt = f"""请阅读下方【即将移出的旧对话】，并结合【现存的历史摘要】，完成两项任务：
                【现存的历史摘要】：
                {self.current_summary if self.current_summary else "（暂无历史摘要，当前为初始对话归档）"}
                【即将移出的旧对话】：
                {evicted_text}
                ==================================================
                【任务指令】请严格输出符合以下 JSON 结构的纯文本，严禁包含任何多余解释：
                {{
                  "updated_summary": "在此处输出提炼后的最新全局对话摘要（严格控制在 {MEMORY_CONFIG['summary_max_words']} 字以内，剔除客套与工具细节，保留核心结论）",
                  "extracted_facts": [
                    {{
                      "category": "user_profile 或 project_constraints 或 user_preferences",
                      "key": "语义唯一英文Key",
                      "value": "长期有效的事实描述",
                      "action": "ADD 或 UPDATE 或 DELETE"
                    }}
                  ]
                }}
                若旧对话中未包含需要长期保存的学术/业务固定事实，则 extracted_facts 填空列表 []。
                ==================================================
                """
        try:
            resp = await self.summary_agent.run(prompt)
            raw_text = resp.output.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").removeprefix("json").strip()

            result_data = json.loads(raw_text)

            new_summary = result_data.get("updated_summary", "").strip()
            if new_summary:
                self.current_summary = new_summary
                sys_logger.info(f"✅ [滚动摘要已更新] (长 {len(self.current_summary)} 字):\n{self.current_summary}")

            facts_to_save = result_data.get("extracted_facts", [])
            if facts_to_save and fact_store:
                sys_logger.info(f"🌾 [顺风车捕获] 成功打捞出 {len(facts_to_save)} 条长期事实！")
                fact_store.apply_fact_changes(facts_to_save)
        except Exception as e:
            sys_logger.error(f"❌ [顺风车提炼异常] {e}")


# =========================================================================
# 🌟 6. 统一记忆外观总管 (UnifiedMemoryManager) - 彻底净化 main.py
# =========================================================================
class UnifiedMemoryManager:
    """
    全层级记忆调度外观中枢：
    把长期事实、正则门禁、滚动摘要、Token 裁剪与协议排雷全量收敛至内部。
    外部调用者（main.py）每轮对话只需调用 `prepare_turn_context` 单一接口。
    """

    def __init__(self, model_name: str):
        # 1. 实例化专用后台轻量抽取模型
        self.summary_llm = get_llm_model(model_name)

        # 2. 初始化事实档案 (第 3 层)
        self.fact_store = FactStore() if MEMORY_CONFIG.get("enable_fact_store", True) else None

        # 3. 初始化毫秒级正则门禁 (策略 1)
        self.gatekeeper = FastFactGatekeeper()

        # 4. 初始化滚动摘要与顺风车引擎 (第 2 层 + 策略 2)
        self.summary_engine = RollingSummaryEngine(summarizer_model=self.summary_llm)

    async def prepare_turn_context(self, user_input: str, history: List[ModelMessage]) -> List[ModelMessage]:
        """
        【主调度接口】：在单轮对话推理前执行完整的记忆维护闭环
        1. 策略 1: 0 Token 门禁拦截显式修改并即时落盘；
        2. 工具剪枝 -> Token 预算打捞 -> 协议安全排雷；
        3. 策略 2: 顺风车增量提炼滑出窗口的旧记忆；
        4. 返回符合协议标准且经过预算控制的短期上下文列表。
        """
        # --- 步骤 A: 策略 1 门禁即时提取 (0 Token 开销拦截) ---
        if self.fact_store and self.gatekeeper.should_trigger(user_input):
            try:
                await self._extract_instant_facts(user_input)
            except Exception as e:
                sys_logger.warning(f"⚠️ [记忆总管] 策略 1 即时提取跳过: {e}")

        # --- 步骤 B: 核心三步走（工具剪枝 + Token 预算打捞 + 协议排雷） ---
        surviving, evicted = self.summary_engine.extract_evicted_and_surviving(history)

        # --- 步骤 C: 策略 2 顺风车增量提炼淘汰对话 ---
        if evicted:
            try:
                await self.summary_engine.update_summary_and_harvest_facts(evicted, fact_store=self.fact_store)
            except Exception as e:
                sys_logger.warning(f"⚠️ [记忆总管] 策略 2 顺风车提炼跳过: {e}")

        # --- 步骤 D: 返回干净的短期历史 ---
        return surviving

    async def _extract_instant_facts(self, user_text: str) -> None:
        """策略 1 执行逻辑：唤醒模型提取单句设定"""
        sys_logger.info(f"⚡ [门禁命中] 捕获到用户显式意图，启动即时事实抽取...")
        extractor_agent = Agent(
            model=self.summary_llm,
            system_prompt="你是一个专业的信息抽取引擎。请从用户输入中提取需要持久化的设定或修改，输出 JSON 数组格式。",
            output_type=str
        )
        prompt = f"""请分析用户此句输入：
                \"{user_text}\"
                【提取要求】：
                1. 若包含长期有效的个人身份、研究对象、模型算法命名、硬件测点约束或工具偏好，输出 JSON 数组：
                   [ {{"category": "user_profile|project_constraints|user_preferences", "key": "唯一英文Key", "value": "描述", "action": "ADD|UPDATE|DELETE"}} ]
                2. 若无长期事实，直接输出空数组 []。
                3. 严禁任何说明，只输出合法 JSON 文本！
                """
        resp = await extractor_agent.run(prompt)
        raw_text = resp.output.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").removeprefix("json").strip()

        changes = json.loads(raw_text)
        if isinstance(changes, list) and changes and self.fact_store:
            self.fact_store.apply_fact_changes(changes)

    def get_system_prompt_injection(self) -> str:
        """供 Agent 装配厂一键获取注入内容（长期事实 + 会话摘要）"""
        blocks = []
        if self.fact_store:
            facts_str = self.fact_store.get_formatted_facts()
            if facts_str:
                blocks.append(facts_str)

        if self.summary_engine.current_summary:
            blocks.append(
                f"📌【前情提要 / 历史关键脉络摘要】：\n{self.summary_engine.current_summary}"
            )

        if blocks:
            sep = "=" * 55
            return f"\n\n{sep}\n" + "\n\n".join(blocks) + f"\n{sep}\n"
        return ""
