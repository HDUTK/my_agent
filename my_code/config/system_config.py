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

# 调用的模型
# Gemini/GPT/Qwen/Zhipu/Hunyuan（超时）/Spark（没有key）
my_model_name: str = "Qwen"
# 使用的System Prompts
# default_chat/a63_sensor/code_review/paper_review
my_scenario_name: str = "paper_review"

# 自动定位当前项目的根目录 (AI_Agent_Project)
# __file__ 是当前文件，parent 是 config 文件夹，parent.parent 就是项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent  # D:\PythonProject\AI_Agent\my_code
# Prompts的地址
PROMPTS_DIR = str(BASE_DIR / "AdvancedPrompts")

# 向量数据库配置 (RAG)：自动将路径拼接为根目录下的 db_storage/chroma_db
DB_PERSIST_PATH = r"E:/Vector_Database_for_Agent/db_storage/chroma_db"
# 集合的名字（相当于关系型数据库的表名）
COLLECTION_NAME = "tk_knowledge"

# agent需要的模型下载与缓存路径
HF_MODELS_PATH = "D:/Python31210/HuggingFace_Models"

# 日志输出目录
LOG_CONFIG = {
    "path" : str(BASE_DIR / "logs"),
    "rotation" : 5, # 每当日志文件达到 x MB 时，自动新建一个文件
    "retention": 7,  # 历史日志保留 x 天，防止撑爆硬盘
    # 日志格式
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
}

# cpu/cuda
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
