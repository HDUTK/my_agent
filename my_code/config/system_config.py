#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于系统相关配置
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/06/05 15:46
version: V1.0
Target Python: 
"""

from pathlib import Path
import torch

# =========================================================================
# 🌟 调用的大模型选择相关
# =========================================================================
# 调用本地的大模型
USE_LOCAL_MODEL = False  # True 代表使用本地 Ollama，False 代表使用远程云端 API

# 本地 Ollama 模型配置
# 你的可用库存: 'deepseek-r1', 'gemma3', 'gemma4', 'qwen2.5-coder', 'qwen3'
LOCAL_MODEL_NAME = "qwen3" # 当前选中的本地模型

# 调用远程云端的大模型
# Gemini/GPT/Qwen/Zhipu/Hunyuan（超时）/Spark（没有key）
REMOTE_MODEL_NAME: str = "Qwen"
# 使用的System Prompts
# default_chat/a63_sensor/code_review/complex_task
my_scenario_name: str = "complex_task"
# =========================================================================


# =========================================================================
# 🌟 当前根目录/目录地址相关
# =========================================================================
# 自动定位当前项目的根目录 (AI_Agent_Project)
# __file__ 是当前文件，parent 是 config 文件夹，parent.parent 就是项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent  # D:\PythonProject\AI_Agent\my_code
# Prompts的地址
PROMPTS_DIR = str(BASE_DIR / "AdvancedPrompts")
# config的地址
CONFIG_DIR = str(BASE_DIR / "config")
# =========================================================================


# =========================================================================
# 🌟 向量数据库相关
# =========================================================================
# 向量数据库配置 (RAG)：自动将路径拼接为根目录下的 db_storage/chroma_db
DB_PERSIST_PATH = r"E:/Vector_Database_for_Agent/db_storage/chroma_db"
# 集合的名字（相当于关系型数据库的表名）
COLLECTION_NAME = "tk_knowledge"
# =========================================================================


# =========================================================================
# 🌟 会话历史日志与离线情境经验 (Episodic Memory) 配置
# =========================================================================
# =========================================================================
# 🌟 第 4 层：离线情境经验与会话记录配置 (Episodic Memory Config)
# =========================================================================
EPISODIC_MEMORY_CONFIG = {
    # ----------------------------------------------------
    # 1. 在线会话记录器参数 (Session Recorder)
    # ----------------------------------------------------
    "enable_session_recording": True,               # 是否开启会话历史记录落盘
    "recorder_tool_summary_chars": 300,             # 记录到 JSON 的工具返回值最大截断字符数
    "recorder_min_turns_to_save": 1,                # 最少有效轮次，避免退出时产生空文件

    # ----------------------------------------------------
    # 2. 本地存储路径与独立向量库 (Paths & ChromaDB)
    # ----------------------------------------------------
    "session_log_dir": str(BASE_DIR / "logs" / "chat_history"),
    "pending_dir": str(BASE_DIR / "logs" / "chat_history" / "pending"),
    "archived_dir": str(BASE_DIR / "logs" / "chat_history" / "archived"),
    "episodic_collection_name": "agent_episodic_memory", # 独立经验卡片集合
    "chroma_db_dir": str(BASE_DIR / "db_storage" / "chroma_db"),

    # ----------------------------------------------------
    # 3. 离线提炼前置硬门禁参数 (0 Token 过滤规则)
    # ----------------------------------------------------
    "miner_min_turns": 2,                           # 触发大模型提炼的最低交互轮次
    "miner_require_tool_call": True,                # 是否要求会话中必须存在工具调用
    "miner_min_total_chars": 200,                   # 会话总字符数下限
    "miner_domain_keywords": [                      # 业务关键词命中列表 (命中其一才提炼)
        "传感器", "测点", "A63", "9号窟", "10号窟", "结露",
        "温湿度", "壁温", "露点", "阈值", "排查", "方案", "实验", "算法"
    ],

    # ----------------------------------------------------
    # 4. 离线大模型审计参数 (LLM Distillation)
    # ----------------------------------------------------
    "miner_model_name": REMOTE_MODEL_NAME,          # 离线提炼使用的模型 (可单独指定轻量模型)
    "card_embedding_summary_max_words": 150,         # 卡片向量检索浓缩摘要字数限制

    # ----------------------------------------------------
    # 5. 热路径 (Hot Path) 常驻目录配置
    # ----------------------------------------------------
    "hot_path_enabled": True,                       # 是否启用 System Prompt 常驻热目录
    "hot_path_cards_limit": 3,                      # 热目录常驻的最近完成任务数量 (默认 3 条)
    "hot_path_title_max_chars": 40,                 # 热目录中标题截断长度

    # ----------------------------------------------------
    # 6. 冷路径 (Cold Path) 向量检索与两级穿透配置
    # ----------------------------------------------------
    "cold_search_top_k": 3,                         # 工具检索默认返回的候选卡片数量
    "cold_confidence_warning": (                    # 注入大模型上下文的防幻觉警示语
        "⚠️【冷库粗筛防幻觉提示】：以上卡片为数学余弦相似度粗筛结果，置信度【待定】！"
        "可能存在字面接近但实际无关的情况。请严格核验其实际关联度；若确认无关，严禁强行作为参数依据！"
    )
}
# 会话日志物理存放根目录
SESSION_LOG_DIR = BASE_DIR / "logs" / "chat_history"
PENDING_SESSIONS_DIR = SESSION_LOG_DIR / "pending"    # 待提炼的原始会话 JSON
ARCHIVED_SESSIONS_DIR = SESSION_LOG_DIR / "archived"  # 已提炼归档的会话 JSON
# 情境经验卡片专用的向量库 Collection 名称 (物理隔离于行业文献库 tk_knowledge)
EPISODIC_COLLECTION_NAME = "agent_episodic_memory"
# 本地向量库根目录
CHROMA_DB_DIR = DB_PERSIST_PATH


# =========================================================================
# 🌟 日志log相关
# =========================================================================
# 日志输出目录
LOG_CONFIG = {
    "path" : str(BASE_DIR / "logs"),
    "rotation" : 5, # 每当日志文件达到 x MB 时，自动新建一个文件
    "retention": 7,  # 历史日志保留 x 天，防止撑爆硬盘
    # 日志格式
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
}
# =========================================================================


# =========================================================================
# 🌟 全局记忆管理配置 (Memory Configuration)
# =========================================================================
MEMORY_CONFIG = {
    # 【第 1 层：短期工作记忆与剪枝】
    "token_budget": 3000,               # 短期工作记忆的最大 Token 预算上限
    "max_messages": 10,                 # 短期工作记忆容纳的最大消息条数硬上限
    "keep_last_n_tools": 2,             # 历史中保留完整原始输出的最近工具调用轮数
    "enable_tool_pruning": True,        # 是否开启历史工具返回值自动折叠剪枝
    "tool_prune_char_threshold": 200,   # 工具返回值超过此字符长度时，才执行折叠压缩

    # 【第 2 层：滚动会话摘要】
    "summary_trigger_min_chars": 20,    # 淘汰消息的有效字符下限（低于此字数不唤醒模型，防无意义消耗）
    "summary_max_words": 350,           # 全局滚动摘要建议保留的最大字数 (中文字符)

    # 【第 3 层：长期外部事实档案 Fact Store (策略 1 + 2 + 3)】
    "enable_fact_store": True,          # 是否启用长期事实持久化功能
    "fact_store_path": str(Path(CONFIG_DIR) / "user_facts.json"),  # 本地持久化路径

    # 策略 1 专用：触发即时事实提取的正则关键词列表（0 Token 快速拦截门禁）
    "fact_trigger_keywords": [
        # 显式记忆指令
        r"记一下", r"记住", r"别忘了", r"提醒你", r"记录一下",
        # 身份与学术课题定义
        r"我是", r"我的课题", r"研究对象", r"主要研究", r"博士阶段",
        # 习惯与环境偏好
        r"偏好", r"习惯用", r"以后统一", r"默认使用", r"技术栈",
        # 纠偏与修改动作
        r"改成", r"修改为", r"调整为", r"不是.*而是", r"不再.*而是"
    ]
}
# =========================================================================


# =========================================================================
# 🌟 RAG相关
# =========================================================================
# RAG 作为工具的 DOCSTRING：从同目录下的 rag_tool_docstring.md 动态读取
RAG_DOC_FILE = Path(__file__).resolve().parent / "rag_tool_docstring.md"

def load_rag_tool_docstring() -> str:
    """从外部 Markdown 文件动态加载 RAG 说明书，带异常与缺失防御"""
    if not RAG_DOC_FILE.exists():
        # 防呆兜底：如果外部 md 文件丢失，返回基础说明，防止启动抛异常崩溃
        return "【本地核心专业知识库检索工具】用于检索云冈石窟项目相关内容、博士资料和博士学术论文等本地资料。"

    try:
        return RAG_DOC_FILE.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"⚠️ [配置警告] 读取 {RAG_DOC_FILE.name} 失败: {e}，将启用简短兜底说明。")
        return "【本地核心专业知识库检索工具】用于检索本地专业知识库资料。"

# 对外暴露的常量：保持原变量名不变，mcp_server.py 等其他模块无需做任何修改
RAG_TOOL_DOCSTRING = load_rag_tool_docstring()
# =========================================================================


# =========================================================================
# 🌟 文件与目录访问权限控制 (File Access Control / Sandbox)
# =========================================================================
# 文件访问权限总开关：
# True  -> 开启沙箱权限管控：严格检查白名单与黑名单，超出或命中黑名单一律报错拒绝
# False -> 调试/全放行模式：关闭路径检查，允许访问系统任意可读写路径
ENABLE_FILE_ACCESS_CONTROL: bool = True

FILE_ACCESS_CONFIG = {
    "enable_access_control": ENABLE_FILE_ACCESS_CONTROL,
    # 权限规则配置文件路径
    "rules_file": str(BASE_DIR / "config" / "access_control.md")
}


# =========================================================================
# 🌟 【统一安全配置】本地安全边界网关配置 (Local Security Proxy Config)
# =========================================================================
# 一键总开关：
# True  -> 生产模式：启用带有“污点标记”的智能边界网关 (有脱敏才解密，无脱敏零开销)
# False -> 调试模式：彻底关闭所有网关拦截，100% 原始数据直接透传
ENABLE_LOCAL_SECURITY_PROXY: bool = True

DATA_SECURITY_CONFIG = {
    "enable_proxy": ENABLE_LOCAL_SECURITY_PROXY,
    "default_level": "L6",  # 无法判定密级时的零信任兜底级别 (视为 L0 处理)
    "rules_file": str(BASE_DIR / "config" / "masking_rules.md"),

    # 安全墓碑模板 (L0 / L6 阻断时发给大模型的提示)
    "tombstone_template": (
        "【⚠️ 系统安全拦截声明 - 涉密内容未披露】：\n"
        "此处引用的来源文件 [{source}] 属于受保护的高密级资产 ({security_level})。\n"
        "根据文物保护数据合规安全红线，该部分正文已被本地边界安全网关物理阻断，未向云端披露！\n"
        "请注意：由于此关键数据未提供，严禁臆测核心推导细节；若用户追问，请明确告知此数据受本地密级保护。"
    ),

    # 注入远程大模型的合规上下文声明头部
    "compliance_prompt_header": "【数据安全合规说明】：当前上下文已由本地边界安全网关执行合规清洗与可逆脱敏。"
}


# =========================================================================
# 🌟 CPU/GPU
# =========================================================================
# cpu/cuda
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# =========================================================================


# =========================================================================
# 🌟 agent下载与缓存路径
# =========================================================================
# agent需要的模型下载与缓存路径
HF_MODELS_PATH = "D:/Python31210/HuggingFace_Models"
# =========================================================================
