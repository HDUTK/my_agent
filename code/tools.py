#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于agent能够使用的工具等
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/12 16:06
version: V1.0
Target Python: 3.14
"""

import os
import platform
import json
import subprocess
import psutil
import pandas as pd


def get_host_info() -> str:
    """
    获取主机信息
    """
    info: dict[str, str] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "memory_gb": str(round(psutil.virtual_memory().total / (1024 ** 3), 2))
    }

    cpu_count = psutil.cpu_count(logical=True)
    if cpu_count is None:
        info["cpu_count"] = "-1"
    else:
        info["cpu_count"] = str(cpu_count)

    info["cpu_model"] = platform.processor()

    return json.dumps(info, indent=4)


def read_file(file_path: str) -> str:
    """读取指定路径的文件内容，返回字符串"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 '{file_path}' 不存在"
    except Exception as e:
        return f"读取文件时出错：{e}"


def get_file_names(folder_path: str) -> list[str]:
    """返回指定文件夹内所有文件的文件名（不包括子文件夹内的文件）"""
    try:
        # 列出文件夹内所有条目
        entries = os.listdir(folder_path)
        # 过滤出文件（保留文件名，不含路径）
        files = [f for f in entries if os.path.isfile(os.path.join(folder_path, f))]
        return files
    except FileNotFoundError:
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return []
    except PermissionError:
        print(f"错误：没有权限访问文件夹 '{folder_path}'")
        return []
    except Exception as e:
        print(f"发生未知错误：{e}")
        return []


def rename_file(file_path: str, new_name: str) -> bool:
    """
    将文件重命名为新名字（不包含路径，仅文件名或完整路径均可）
    参数：
        file_path: 原文件的完整路径
        new_name:  新文件名（例如 "newfile.txt"），可以包含扩展名
    """
    try:
        if not os.path.isfile(file_path):
            print(f"错误：文件 '{file_path}' 不存在")
            return False

        dir_name = os.path.dirname(file_path)
        new_path = os.path.join(dir_name, new_name)

        os.rename(file_path, new_path)
        print(f"重命名成功：{file_path} -> {new_path}")
        return True
    except Exception as e:
        print(f"重命名失败：{e}")
        return False


def query_csv(file_path: str, query_string: str) -> str:
    """
    专门用于查询大型 CSV 文件的数据过滤工具。
    【大模型注意】
    1. 必须用 Pandas 的 query 语法编写 query_string。
    2. 请严格使用 CSV 的真实列名（通常是英文，如 time, air_temperature）。
    """
    print(f"[工具执行] 正在查询 CSV: {file_path} | 条件: {query_string}")

    if not os.path.exists(file_path):
        return f"❌ 错误: 找不到文件 {file_path}"

    try:
        # 1. 读取 CSV (消除警告)
        df = pd.read_csv(filepath_or_buffer=file_path)

        if not isinstance(df, pd.DataFrame):
            return "❌ 错误: 读取的结果不是有效的 DataFrame"

        # 2. 执行大模型写好的查询语句
        result_df = df.query(query_string)

        # 🌟 核心升级 1：查不到数据时的“格式纠偏”
        if result_df.empty:
            cols = df.columns.tolist()
            sample_data = df.iloc[0].to_dict() if not df.empty else "无数据"
            return (f"⚠️ 查询执行成功，但结果为空！\n"
                    f"这通常是因为你的 query_string (特别是时间格式) 与 CSV 的实际文本不匹配。\n"
                    f"请看一眼 CSV 真实的列名和第一条数据的格式，然后修改你的条件重试！\n"
                    f"-> 真实列名: {cols}\n"
                    f"-> 真实数据示例: {sample_data}")

        # 3. 正常返回 JSON 字符串
        json_result = str(result_df.to_json(orient="records", force_ascii=False))
        return json_result

    except Exception as e:
        # 🌟 核心升级 2：语法报错时的“列名纠偏”
        cols = df.columns.tolist() if 'df' in locals() else "未知"
        return (f"❌ 查询语法报错！错误信息: {str(e)}\n"
                f"这通常是因为你使用了不存在的列名（比如用了中文'时间'，但实际是'time'）。\n"
                f"请使用真实的列名重新查询: {cols}")


# ==========================================
# 🌟 新增数据提交工具
# ==========================================
def submit_sensor_record(
        target_time: str,
        air_temperature: float,
        air_humidity: float,
        wall_temperature: float,
        has_risk: bool,
        action_advice: str
) -> str:
    """
    当用户要求提取石窟传感器数据并评估风险时，必须调用此工具提交最终的结构化诊断报告。

    参数:
    - target_time: 提取的目标时间字符串
    - air_temperature: 空气温度值 (float)
    - air_humidity: 空气湿度值 (float)
    - wall_temperature: 壁面温度值 (float)
    - has_risk: 是否有风险 (bool)
    - action_advice: 系统就绪状态或针对风险给出的具体处置建议
    """
    print(f"\n[报告接收中心] 成功捕获到大模型提交的结构化数据：")
    print(f"   - 时间: {target_time}")
    print(f"   - 空气温湿度: {air_temperature} ℃ | {air_humidity} %")
    print(f"   - 壁面温度: {wall_temperature} ℃")
    print(f"   - 风险判定: {'有风险' if has_risk else '安全'}")
    print(f"   - 处置建议: {action_advice}\n")

    # 这里可以扩展业务逻辑，比如写入数据库、保存到本地 JSON 文件等
    # 目前直接返回一个确认信息给大模型
    return "成功：结构化诊断报告已安全提交至系统后台。"


# 使用示例
if __name__ == "__main__":
    folder = "./test"  # 替换为你的文件夹路径
    names = get_file_names(folder)
    # rename_file("C:/temp/old.txt", "new.txt")
    print(names)
    print(get_host_info())
