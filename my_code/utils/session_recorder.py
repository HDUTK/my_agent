#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于交互会话轻量化记录器 (SessionRecorder)
设计目标:
  - 零在线算力负担，仅利用纯 Python 解析单轮交互消息；
  - 完整保留工具入参 (arguments) 与业务实体，工具返回值严格截断 (300字以内)；
  - 会话退出时，以标准 JSON 格式持久化到 logs/chat_history/pending 目录。
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/10/19 20:45
version: V1.0
Target Python: 
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart
)

from config.system_config import PENDING_SESSIONS_DIR
from utils.logger_print import sys_logger


class SessionRecorder:
    """会话全生命周期记录器"""

    def __init__(self, model_name: str, scenario_name: str, is_local: bool):
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now().isoformat()
        self.model_info = {
            "driver_type": "local" if is_local else "remote",
            "model_name": model_name,
            "scenario": scenario_name
        }
        self.turns: List[Dict[str, Any]] = []
        self.total_tool_calls = 0
        self.total_tokens_consumed = 0

    def record_turn(
            self,
            turn_id: int,
            user_input: str,
            agent_reply: str,
            new_messages: List[ModelMessage],
            usage: Any
    ) -> None:
        """
        记录单次回合。
        从本轮产生的新消息中精准提取工具调用参数与截断后的返回值。
        """
        tool_records = []

        # 1. 临时存储本轮的模型工具发起与工具返回
        calls_map: Dict[str, Dict[str, Any]] = {}
        returns_map: Dict[str, str] = {}

        for msg in new_messages:
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        calls_map[part.tool_call_id] = {
                            "tool_name": part.tool_name,
                            "args": part.args if isinstance(part.args, dict) else str(part.args)
                        }
            elif isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        # 核心防爆机制：工具返回值严格截断到 300 字符以内
                        raw_ret = str(part.content)
                        if len(raw_ret) > 300:
                            summary_ret = raw_ret[:300].replace("\n",
                                                                " ").strip() + f"... [数据截断，共 {len(raw_ret)} 字符]"
                        else:
                            summary_ret = raw_ret.replace("\n", " ").strip()

                        tool_call_id = getattr(part, "tool_call_id", part.tool_name)
                        returns_map[tool_call_id] = summary_ret

        # 2. 组装工具调用记录
        for call_id, call_info in calls_map.items():
            ret_content = returns_map.get(call_id, returns_map.get(call_info["tool_name"], "[已执行无文本回显]"))
            tool_records.append({
                "tool_name": call_info["tool_name"],
                "arguments": call_info["args"],
                "result_summary": ret_content
            })

        self.total_tool_calls += len(tool_records)
        turn_tokens = getattr(usage, "total_tokens", 0)
        self.total_tokens_consumed += turn_tokens

        # 3. 追加单轮记录
        self.turns.append({
            "turn_id": turn_id,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "tool_calls": tool_records,
            "agent_reply": agent_reply,
            "tokens": {
                "input": getattr(usage, "input_tokens", 0),
                "output": getattr(usage, "output_tokens", 0),
                "total": turn_tokens
            }
        })

    def finalize_and_save(self) -> Optional[Path]:
        """
        退出时将整场会话持久化落盘。
        若整场会话完全没有任何有效轮次，则不产生垃圾文件。
        """
        if not self.turns:
            return None

        PENDING_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = PENDING_SESSIONS_DIR / f"{self.session_id}.json"

        payload = {
            "session_id": self.session_id,
            "meta": {
                "start_time": self.start_time,
                "end_time": datetime.now().isoformat(),
                "status": "pending",  # 供离线脚本扫描的状态游标
                "model_info": self.model_info,
                "stats": {
                    "total_turns": len(self.turns),
                    "tool_call_count": self.total_tool_calls,
                    "total_tokens_consumed": self.total_tokens_consumed
                }
            },
            "turns": self.turns
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            sys_logger.info(f"💾 [会话落盘] 本场对话已安全持久化至: {file_path.name}")
            return file_path
        except Exception as e:
            sys_logger.error(f"❌ [会话落盘失败] {e}")
            return None
