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
    大模型注意：请用 Pandas 的 query 语法编写 query_string。

    参数:
    - file_path: csv文件的相对或绝对路径 (例如 'test2/A63.csv')
    - query_string: Pandas 过滤条件 (例如 "时间 == '2023/10/10 00:00'")
    """
    print(f"[工具执行] 正在查询 CSV: {file_path} | 条件: {query_string}")

    if not os.path.exists(file_path):
        return f"❌ 错误: 找不到文件 {file_path}"

    try:
        # 1. 明确参数名 filepath_or_buffer，消除第一个绿线
        df = pd.read_csv(file_path)

        # 2. 类型收窄：明确告诉 PyCharm 这绝对是个 DataFrame，消除第二个绿线
        if not isinstance(df, pd.DataFrame):
            return "❌ 错误: 读取的结果不是有效的 DataFrame"

        result_df = df.query(query_string)

        if result_df.empty:
            return f"⚠️ 查询成功，但在文件 {file_path} 中没有找到符合 '{query_string}' 的数据。"

        # 3. 明确强转为 str
        json_result = str(result_df.to_json(orient="records", force_ascii=False))
        return json_result

    except Exception as e:
        return f"❌ 查询失败，可能是 query_string 语法错误或列名不存在。Python报错: {str(e)}"


# 使用示例
if __name__ == "__main__":
    folder = "./test"  # 替换为你的文件夹路径
    names = get_file_names(folder)
    # rename_file("C:/temp/old.txt", "new.txt")
    print(names)
    print(get_host_info())
