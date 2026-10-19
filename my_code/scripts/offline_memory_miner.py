#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于离线情境经验提炼与向量化引擎 (offline_memory_miner.py)
执行模式: 离线批处理 (Offline Batch Processing)
运行逻辑:
  1. 扫描 logs/chat_history/pending 目录下待处理的 session_*.json 文件；
  2. 纯代码硬规则过滤：剔除小于 3 轮或无工具调用的琐碎日常闲聊 (0 Token 开销)；
  3. 大模型二合一审计与蒸馏：判断会话是否形成闭环科研结论，提取结构化经验卡片；
  4. 将经验卡片存入 ChromaDB 的 agent_episodic_memory 独立集合；
  5. 更新文件状态为 processed / skipped，并移入 archived 归档目录。
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/10/19 20:59
version: V1.0
Target Python: 
"""

import os
import re
import json
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

import chromadb
from pydantic_ai import Agent

from config.system_config import (
    PENDING_SESSIONS_DIR,
    ARCHIVED_SESSIONS_DIR,
    DB_PERSIST_PATH,
    EPISODIC_COLLECTION_NAME,
    REMOTE_MODEL_NAME
)
from agent.llm_factory import get_llm_model
from utils.logger_print import sys_logger


# =========================================================================
# 1. 纯代码硬门禁过滤规则 (0 Token 开销)
# =========================================================================
def is_session_worth_analyzing(session_data: Dict[str, Any]) -> bool:
    """
    通过元数据做极速初筛，过滤 80% 的日常琐碎问答：
    1. 总轮次 >= 3 轮；
    2. 至少包含 1 次实质性工具调用；
    3. 命中石窟、传感器等科研业务关键词。
    """
    stats = session_data.get("meta", {}).get("stats", {})
    total_turns = stats.get("total_turns", 0)
    tool_calls = stats.get("tool_call_count", 0)

    # 规则 1 & 2: 轮次不够或无工具调用，直接跳过
    if total_turns < 2 or tool_calls == 0:
        return False

    # 规则 3: 关键词快速命中扫描
    text_content = json.dumps(session_data.get("turns", []), ensure_ascii=False)
    domain_pattern = re.compile(r"(传感器|测点|A63|9号窟|10号窟|结露|温湿度|壁温|露点|阈值|排查|方案|实验|算法)")
    if not domain_pattern.search(text_content):
        return False

    return True


# =========================================================================
# 2. 格式化会话为大模型易读的动作链摘要
# =========================================================================
def format_session_for_llm(session_data: Dict[str, Any]) -> str:
    """提取 user_input -> tool_calls(args + summary) -> agent_reply 的纯净链条"""
    lines = []
    lines.append(f"【会话 ID】: {session_data.get('session_id')}")
    lines.append(
        f"【起止时间】: {session_data.get('meta', {}).get('start_time')} ~ {session_data.get('meta', {}).get('end_time')}\n")

    for t in session_data.get("turns", []):
        lines.append(f"--- 第 {t.get('turn_id')} 轮 ---")
        lines.append(f"👤 用户: {t.get('user_input')}")

        tools = t.get("tool_calls", [])
        if tools:
            for tool in tools:
                lines.append(f"🔧 工具执行: {tool.get('tool_name')} | 参数: {tool.get('arguments')}")
                lines.append(f"   ↳ 返回结果切片: {tool.get('result_summary')}")

        reply_preview = t.get("agent_reply", "").strip()
        lines.append(f"🤖 助手结论: {reply_preview}\n")

    return "\n".join(lines)


# =========================================================================
# 3. 大模型二合一审计与卡片提炼
# =========================================================================
async def distill_episode_card(formatted_text: str, llm_model: Any) -> Optional[Dict[str, Any]]:
    """调用大模型执行价值判定与标准经验卡片抽取"""
    miner_agent = Agent(
        model=llm_model,
        system_prompt="你是一个严谨的文物保护科研知识工程专家，负责从 Agent 的长线历史操作记录中提炼出具有复盘价值的闭环经验卡片。",
        output_type=str
    )

    prompt = f"""请审查以下历史会话流，完成【价值判定与经验卡片蒸馏】：

{formatted_text}

==================================================
【判定准则】：
1. 具备长期价值的条件：有明确排查目标、经历工具/实验验证、最终确立了清晰的参数、预警阈值、事故原因或有效代码方案。
2. 坚决排除的场景：中途放弃未完成的任务、纯语法报错、无明确数字或业务结论的测试。

【输出规范】：
- 若不具备长期价值，严格输出 JSON: {{"is_valuable": false}}
- 若具备长期价值，严格输出 JSON:
{{
  "is_valuable": true,
  "episode_card": {{
    "title": "任务简述标题 (如: 9号窟前室A63测点冷凝结露风险排查与阈值测定)",
    "target_caves": ["涉及的窟号，如 9号窟"],
    "sensors": ["涉及的测点代号，如 A63"],
    "problem_and_goal": "核心排查目的或遇到的异常现象",
    "method_and_tools": "调用的工具、核验的文件以及分析方法",
    "final_conclusion": "最终确立的刚性参数、预警阈值、物理定性或解决方案",
    "embedding_summary": "一段用于未来向量检索的密集文本摘要 (150字以内，包含核心实体、动作和最终确立的数值参数)"
  }}
}}
只输出纯 JSON，严禁输出任何多余说明与 markdown 包裹符！
==================================================
"""
    try:
        resp = await miner_agent.run(prompt)
        raw_text = resp.output.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").removeprefix("json").strip()

        result = json.loads(raw_text)
        if result.get("is_valuable"):
            return result.get("episode_card")
        return None
    except Exception as e:
        sys_logger.error(f"❌ [卡片蒸馏异常] {e}")
        return None


# =========================================================================
# 4. ChromaDB 向量入库
# =========================================================================
def save_card_to_chroma(card: Dict[str, Any], session_id: str, chroma_dir: str, collection_name: str):
    """将经验卡片存入独立的 ChromaDB Collection"""
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name=collection_name)

    doc_text = card.get("embedding_summary", "")
    metadata = {
        "session_id": session_id,
        "title": card.get("title", ""),
        "target_caves": ",".join(card.get("target_caves", [])),
        "sensors": ",".join(card.get("sensors", [])),
        "problem_and_goal": card.get("problem_and_goal", "")[:200],
        "final_conclusion": card.get("final_conclusion", "")[:200]
    }

    # 使用 session_id 作为唯一 ID，具备天然的幂等覆盖能力
    collection.upsert(
        ids=[session_id],
        documents=[doc_text],
        metadatas=[metadata]
    )
    print(f"  ✨ [向量入库] 经验卡片已存入 ChromaDB -> [{collection_name}]: {card.get('title')}")


# =========================================================================
# 5. 批处理主流程
# =========================================================================
async def run_batch_mining():
    print("\n" + "═" * 60)
    print("🚀 [离线记忆提炼引擎启动] 正在扫描待处理会话日志...")
    print("═" * 60)

    if not PENDING_SESSIONS_DIR.exists():
        print("📂 未发现 pending 目录，暂无待提炼会话。")
        return

    session_files = list(PENDING_SESSIONS_DIR.glob("session_*.json"))
    if not session_files:
        print("📂 暂无待处理的会话文件。")
        return

    print(f"📦 共发现 {len(session_files)} 个待处理会话日志，准备提炼...\n")

    ARCHIVED_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    llm = get_llm_model(REMOTE_MODEL_NAME)

    processed_count = 0
    distilled_count = 0

    for file_path in session_files:
        print(f"📄 正在处理: {file_path.name}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 步骤 1: 纯代码前置过滤
            if not is_session_worth_analyzing(data):
                print("  ⏩ [硬规则跳过] 属于简单日常问答或无工具调用，无需提炼。")
                data["meta"]["status"] = "skipped"
            else:
                # 步骤 2: 格式化并交由大模型审计提炼
                formatted_history = format_session_for_llm(data)
                card = await distill_episode_card(formatted_history, llm)

                if card:
                    # 步骤 3: 存入 ChromaDB
                    save_card_to_chroma(
                        card=card,
                        session_id=data.get("session_id"),
                        chroma_dir=DB_PERSIST_PATH,
                        collection_name=EPISODIC_COLLECTION_NAME
                    )
                    data["meta"]["status"] = "processed"
                    data["distilled_card"] = card
                    distilled_count += 1
                else:
                    print("  🛑 [审计驳回] 会话未得出闭环科研结论，放弃入库。")
                    data["meta"]["status"] = "rejected"

            # 步骤 4: 更新文件状态并移动到归档目录
            dest_path = ARCHIVED_SESSIONS_DIR / file_path.name
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 删除原 pending 文件
            file_path.unlink()
            processed_count += 1

        except Exception as e:
            print(f"  ❌ 处理文件 {file_path.name} 失败: {e}")

    print("\n" + "═" * 60)
    print(f"🏁 [提炼完成] 共归档 {processed_count} 个会话，成功沉淀 {distilled_count} 张高价值经验卡片！")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_batch_mining())