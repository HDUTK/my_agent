#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于存放所有与业务逻辑无关的通用工具函数，如读 Prompt、修剪记忆
同时作为本地边界安全网关核心：
1. 文件/目录物理访问权限沙箱控制 (白名单/黑名单机制)
2. 敏感数据双向可逆脱敏 (污点标记与按需解密)
3. 密级分流与安全墓碑拦截 (L0/L2/L4/L6)
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/31 16:40
version: V1.0
Target Python:  3.14 -> 3.12
"""

import os
import fnmatch
import warnings
import functools
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

from config.system_config import PROMPTS_DIR
from config.agent_config import MODEL_REGISTRY
from utils.logger_print import sys_logger

# 导入系统安全与权限配置
try:
    from config.system_config import (
        DATA_SECURITY_CONFIG,
        ENABLE_LOCAL_SECURITY_PROXY,
        ENABLE_FILE_ACCESS_CONTROL,
        FILE_ACCESS_CONFIG,
        BASE_DIR
    )
except ImportError:
    ENABLE_LOCAL_SECURITY_PROXY = True
    ENABLE_FILE_ACCESS_CONTROL = True
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_SECURITY_CONFIG = {
        "enable_proxy": True,
        "default_level": "L6",
        "rules_file": str(BASE_DIR / "config" / "masking_rules.md"),
        "tombstone_template": "【⚠️ 系统安全拦截声明】：文件 [{source}] 密级为 [{security_level}]，已阻断！",
        "compliance_prompt_header": "【数据安全说明】：已执行合规脱敏。"
    }
    FILE_ACCESS_CONFIG = {
        "enable_access_control": True,
        "rules_file": str(BASE_DIR / "config" / "access_control.md")
    }


# =========================================================================
# 🌟 本地边界安全网关中心 (Local Security Proxy Engine)
# =========================================================================
class LocalSecurityProxy:
    """
    【本地边界安全网关单例类】
    统一管理两大安全中枢：
    1. 路径访问控制中枢 (Gate 0)：根据 access_control.md 执行物理路径白名单与黑名单安全校验；
    2. 数据合规脱敏中枢 (Gate 1~4)：可逆脱敏、参数解密、切面清洗与污点追踪按需还原。
    """

    def __init__(self):
        # 内部维护的单轮脱敏映射表 {"【测点_A63】": "A63"}
        self._mask_map: Dict[str, str] = {}
        # 🌟 核心标记：本轮对话是否发生过真实的加密/脱敏动作
        self.has_masked_in_turn: bool = False

        # 缓存解析后的白名单与黑名单规则
        self._whitelist_rules: List[str] = []
        self._blacklist_rules: List[str] = []
        self._rules_loaded: bool = False

    @property
    def is_enabled(self) -> bool:
        """检查全局开关状态"""
        return DATA_SECURITY_CONFIG.get("enable_proxy", True)

    @property
    def is_access_control_enabled(self) -> bool:
        """检查文件访问权限控制开关状态"""
        return FILE_ACCESS_CONFIG.get("enable_access_control", True)

    def reset_turn_state(self) -> None:
        """每轮对话开始前必须调用：重置脏标记与映射表"""
        self._mask_map.clear()
        self.has_masked_in_turn = False

    # -------------------------------------------------------------
    # 🌟 权限规则解析：从 config/access_control.md 加载规则
    # -------------------------------------------------------------
    def load_access_control_rules(self) -> Tuple[List[str], List[str]]:
        """从 access_control.md 中动态解析白名单和黑名单路径规则"""
        if self._rules_loaded:
            return self._whitelist_rules, self._blacklist_rules

        rules_path = Path(FILE_ACCESS_CONFIG.get("rules_file", ""))
        whitelist, blacklist = [], []

        if not rules_path.exists():
            sys_logger.warning(f"⚠️ [权限网关] 未找到权限规则文件: {rules_path}，降级为默认拒绝所有访问。")
            return [], []

        current_section = None
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        # 识别区块标题
                        if "白名单" in line or "Whitelist" in line:
                            current_section = "white"
                        elif "黑名单" in line or "Blacklist" in line:
                            current_section = "black"
                        continue

                    if line.startswith("-") or line.startswith("*"):
                        rule = line.lstrip("-*").strip()
                        if rule:
                            if current_section == "white":
                                whitelist.append(rule)
                            elif current_section == "black":
                                blacklist.append(rule)

            self._whitelist_rules = whitelist
            self._blacklist_rules = blacklist
            self._rules_loaded = True
            sys_logger.info(
                f"🔒 [权限网关] 成功加载访问控制规则: 白名单 {len(whitelist)} 条 | 黑名单 {len(blacklist)} 条")
            return whitelist, blacklist
        except Exception as e:
            sys_logger.error(f"❌ [权限网关] 解析 access_control.md 失败: {e}")
            return [], []

    # -------------------------------------------------------------
    # 🌟 路径权限硬校验 (Gate 0)：白名单 + 黑名单判定
    # -------------------------------------------------------------
    def verify_path_permission(self, target_path_str: str) -> Tuple[bool, str]:
        """
        校验目标路径访问权限：
        1. 物理规范化路径（消除 ../ 等路径穿越逃逸攻击）；
        2. 黑名单校验（黑名单绝对优先，命中即拒绝）；
        3. 白名单校验（必须属于白名单目录或文件，未命中即默认拒绝）。

        返回: (是否允许访问: bool, 错误拒绝说明: str)
        """
        if not self.is_access_control_enabled or not target_path_str:
            return True, ""

        # 1. 规范化为物理绝对路径 (解决相对路径与 ../ 目录遍历)
        try:
            target_path = Path(target_path_str)
            if not target_path.is_absolute():
                target_path = (BASE_DIR / target_path).resolve()
            else:
                target_path = target_path.resolve()
        except Exception as e:
            return False, f"【安全权限拦截】：路径格式非法或无法解析: '{target_path_str}' ({e})"

        whitelist_rules, blacklist_rules = self.load_access_control_rules()
        target_str_posix = target_path.as_posix()
        target_name = target_path.name

        # 2. 检查黑名单 (Blacklist - 绝对优先)
        for rule in blacklist_rules:
            # A. 通配符模式匹配 (如 *.py, **/*.py, *secret*)
            if any(char in rule for char in ["*", "?", "["]):
                if fnmatch.fnmatch(target_name, rule) or fnmatch.fnmatch(target_str_posix, rule):
                    err_msg = f"【安全权限拦截】：拒绝访问！文件/目录 '{target_path_str}' 命中系统黑名单保护规则 ('{rule}')。"
                    sys_logger.warning(f"🛑 [权限拦截] {err_msg}")
                    return False, err_msg
            else:
                # B. 明确路径或目录匹配
                rule_path = Path(rule)
                if not rule_path.is_absolute():
                    rule_path = (BASE_DIR / rule_path).resolve()
                else:
                    rule_path = rule_path.resolve()

                # 若目标是黑名单目录本身或其子文件/子目录
                if target_path == rule_path or target_path.is_relative_to(rule_path):
                    err_msg = f"【安全权限拦截】：拒绝访问！路径 '{target_path_str}' 位于受黑名单保护的区域 ('{rule}') 内。"
                    sys_logger.warning(f"🛑 [权限拦截] {err_msg}")
                    return False, err_msg

        # 3. 检查白名单 (Whitelist - 默认拒绝原则)
        matched_whitelist = False
        for rule in whitelist_rules:
            rule_path = Path(rule)
            if not rule_path.is_absolute():
                rule_path = (BASE_DIR / rule_path).resolve()
            else:
                rule_path = rule_path.resolve()

            # 目标属于白名单目录及其子路径，或是白名单中的具体文件
            if target_path == rule_path or target_path.is_relative_to(rule_path):
                matched_whitelist = True
                break

        if not matched_whitelist:
            err_msg = (
                f"【安全权限拦截】：拒绝访问！目标路径 '{target_path_str}' 未在系统允许的白名单工作区范围内。\n"
                f"允许的工作区目录包括: {whitelist_rules}。请检查路径或向用户申请访问权限。"
            )
            sys_logger.warning(f"🛑 [权限拦截] {err_msg}")
            return False, err_msg

        # 校验合法通过
        return True, ""

    def load_masking_rules(self) -> Dict[str, str]:
        """从 config/masking_rules.md 中动态解析脱敏实体词表"""
        rules_path = Path(DATA_SECURITY_CONFIG.get("rules_file", ""))
        if not rules_path.exists():
            return {}

        mapping = {}
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("-") and "->" in line:
                        parts = line.lstrip("-").strip().split("->")
                        if len(parts) == 2:
                            raw_entity = parts[0].strip()
                            masked_token = parts[1].strip()
                            if raw_entity and masked_token:
                                mapping[raw_entity] = masked_token
            return mapping
        except Exception as e:
            sys_logger.error(f"❌ [安全网关] 解析规则文件失败: {e}")
            return {}

    def detect_security_level(self, file_path: str) -> str:
        """根据文件所在目录判定密级 (L0/L2/L4/L6)"""
        if not file_path:
            return DATA_SECURITY_CONFIG.get("default_level", "L6")

        normalized = "/" + str(file_path).replace("\\", "/").strip("/") + "/"
        if "/l0/" in normalized.lower():
            return "L0"
        elif "/l2/" in normalized.lower():
            return "L2"
        elif "/l4/" in normalized.lower():
            return "L4"
        elif "/l6/" in normalized.lower():
            return "L6"
        return DATA_SECURITY_CONFIG.get("default_level", "L6")

    # -------------------------------------------------------------
    # 🌟 关卡 1: 用户输入脱敏 (发给大模型前)
    # -------------------------------------------------------------
    def mask_user_input(self, user_text: str) -> str:
        """
        扫描用户提问：若命中实体，执行脱敏并将 has_masked_in_turn 置为 True
        """
        if not self.is_enabled or not user_text:
            return user_text

        rules = self.load_masking_rules()
        if not rules:
            return user_text

        masked_text = user_text
        did_mask = False

        for entity, placeholder in sorted(rules.items(), key=lambda x: len(x[0]), reverse=True):
            if entity in masked_text:
                masked_text = masked_text.replace(entity, placeholder)
                self._mask_map[placeholder] = entity
                did_mask = True

        # 🌟 只要发生过一次替换，标记为已加密
        if did_mask:
            self.has_masked_in_turn = True

        return masked_text

    # -------------------------------------------------------------
    # 🌟 关卡 2: 工具入参反向还原 (大模型发起 Tool Call 本地执行前)
    # -------------------------------------------------------------
    def demask_tool_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        若当前对话存在脱敏 (has_masked_in_turn == True)，将参数中的代号还原为本地真实文件名；
        若无脱敏，直接 O(1) 原样返回，跳过解析！
        """
        if not self.is_enabled or not arguments:
            return arguments

        # 🌟 快捷短路：若本轮根本没脱敏过，直接跳过
        if not self.has_masked_in_turn or not self._mask_map:
            return arguments

        restored_args = {}
        for k, v in arguments.items():
            if isinstance(v, str):
                restored_val = v
                for placeholder, original in sorted(self._mask_map.items(), key=lambda x: len(x[0]), reverse=True):
                    if placeholder in restored_val:
                        restored_val = restored_val.replace(placeholder, original)
                restored_args[k] = restored_val
            else:
                restored_args[k] = v

        return restored_args

    # -------------------------------------------------------------
    # 🌟 关卡 3: 工具返回与 RAG Chunk 切面处理 (根据 L 标签决定是否加密)
    # -------------------------------------------------------------
    def process_tool_payload(self, payload: Dict[str, Any]) -> str:
        """
        核心根据 L 标签做分流：
        - L4: 原样直通，不触发 flag；
        - L0/L6: 物理抹除正文，返回墓碑；
        - L2: 执行脱敏，并激活 has_masked_in_turn = True！
        """
        content = payload.get("content", "")
        level = payload.get("security_level", "L6")
        file_path = payload.get("file_path", "未知文件")
        file_name = Path(file_path).name if file_path else "未知文件"

        # 若总开关关闭，直接透传
        if not self.is_enabled:
            return content

        # 1. 公开级别 (L4)：直接全文放行，不影响 flag
        if level == "L4":
            return f"[密级: L4 - 公开]\n{content}"

        # 2. 受控敏感级别 (L2)：本地可逆脱敏，并设置 flag = True
        elif level == "L2":
            # 扫描并脱敏其中的实体
            masked_body = self.mask_user_input(content)
            return f"[密级: L2 - 已执行可逆脱敏]\n{masked_body}"

        # 3. 机密与未定级 (L0 / L6)：物理阻断，返回墓碑
        else:
            return DATA_SECURITY_CONFIG.get("tombstone_template", "").format(
                source=file_name,
                security_level=level
            )

    # -------------------------------------------------------------
    # 🌟 关卡 4: 最终回复当场反向还原 (大模型推理完毕流出前)
    # -------------------------------------------------------------
    def demask_response(self, text: str) -> str:
        """
        🌟 核心短路优化：
        只有当 self.has_masked_in_turn == True 时才执行解密遍历；
        若本轮未发生过脱敏，直接 O(1) 返回原始文本！
        """
        if not self.is_enabled or not text:
            return text

        # 🌟 关键判断：若发出去的内容从未经过脱敏，直接跳过，零开销！
        if not self.has_masked_in_turn or not self._mask_map:
            return text

        restored_text = text
        for placeholder, original in sorted(self._mask_map.items(), key=lambda x: len(x[0]), reverse=True):
            if placeholder in restored_text:
                restored_text = restored_text.replace(placeholder, original)

        return restored_text


# 🌟 全局单例网关对象
security_proxy = LocalSecurityProxy()


def security_guard(func):
    """
    🌟 【双向切面安全装饰器】
    1. 【执行前 (Gate 0 路径门禁)】：
       提取目标路径，若目标超出白名单或命中黑名单，物理阻断工具调用，并当场返回拦截警告！
    2. 【执行后 (Gate 3 数据清洗)】：
       接收工具返回的标准字典载荷，执行 L0 阻断、L2 脱敏或 L4 放行。
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. 动态提取路径参数 (兼容 file_path, folder_path, path 及第一个位置参数)
        target_path = kwargs.get("file_path") or kwargs.get("folder_path") or kwargs.get("path")
        if not target_path and args:
            if isinstance(args[0], str):
                target_path = args[0]

        # 2. 🌟 前置路径权限校验 (Gate 0)
        if target_path and isinstance(target_path, str):
            is_allowed, deny_reason = security_proxy.verify_path_permission(target_path)
            if not is_allowed:
                # 权限被拒绝：直接返回阻断提示字符串，根本不调用底层读写逻辑
                sys_logger.warning(f"🛑 [权限拦截] 工具 [{func.__name__}] 拒绝访问路径: {target_path}")
                return deny_reason

        # 3. 校验通过，放行执行原工具逻辑
        result = func(*args, **kwargs)

        # 4. 🌟 后置 DLP 分级清洗 (Gate 3)
        if isinstance(result, dict) and "security_level" in result:
            return security_proxy.process_tool_payload(result)

        return result

    return wrapper


def detect_security_level(file_path: str) -> str:
    """向外暴露的密级探测接口"""
    return security_proxy.detect_security_level(file_path)


# ==========================================
# Advanced Prompts
# ==========================================
def load_prompt(scenario_name: str) -> str:
    """根据场景名称，从本地文件中读取 Prompt"""
    file_path = PROMPTS_DIR + f"/{scenario_name}.md"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            sys_logger.info(f"[Prompt 加载] 成功读取系统提示词模板: {scenario_name}.md")
            return content
    except FileNotFoundError:
        sys_logger.warning(f"[Prompt 缺失] 找不到文件 {file_path}，已强制降级使用默认兜底提示词！")
        return "你是一个有用的人工智能助手。"  # 找不到文件时的默认兜底


# ==========================================
# 动态环境变注入函数
# ==========================================
def apply_model_environment(model_key: str):
    """
    根据传入的模型 key (如 'bge_m3')，动态注入对应的 HuggingFace 环境变量。
    必须在对应的 RAG 组件 import 底层大模型库之前调用！
    """
    # 🌟 强制终端使用 UTF-8 编码
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # 🌟 消除 Flash Attention 警告(rag_retriever.py和vector_builder.py)
    # warnings.filterwarnings("ignore", message=".*1Torch was not compiled with flash attention.*")
    # 消除警告
    warnings.filterwarnings("ignore")

    config = MODEL_REGISTRY.get(model_key)
    if not config:
        sys_logger.error(f" 未在 MODEL_REGISTRY 中找到模型配置: {model_key}")
        raise ValueError(f" 未在 MODEL_REGISTRY 中找到模型配置: {model_key}")

    # 配置当前模型专属的本地缓存路径
    os.environ["HF_HOME"] = config["local_path"]

    # 基础安全清理：强行拔掉 Python 的代理管子，防止网络卡死
    os.environ['HTTP_PROXY'] = ""
    os.environ['HTTPS_PROXY'] = ""
    # 🌟 彻底关闭 ChromaDB 的后台匿名数据收集线程（防止卡死）
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    # 🌟 关闭 Tokenizer 导致的多线程死锁
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if config["offline"]:
        # 开启终极离线模式
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # 离线时清除镜像源设置
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
    else:
        # 允许联网模式
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        if config["use_mirror"]:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        else:
            if "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]

    sys_logger.info(
        f"⚙ [环境配置] 已成功为模型 [{model_key}] 注入网络与路径环境变（离线={config['offline']}, 镜像={config['use_mirror']})")
