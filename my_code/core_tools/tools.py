#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于agent能够使用的工具等
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/12 16:06
version: V1.0
Target Python:  3.14 -> 3.12
"""

import os
import platform
import json
import psutil
import pandas as pd
import docx

from utils.logger_print import sys_logger, print_and_log


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
            msg = f.read()
            sys_logger.info(f"[工具执行] 成功读取文件: {file_path} (长度: {len(msg)})")
            return msg
    except FileNotFoundError:
        msg = f"错误：文件 '{file_path}' 不存在"
        sys_logger.error(msg)
        return msg
    except Exception as e:
        msg = f"读取文件时出错：{e}"
        sys_logger.error(msg)
        return msg


def get_file_names(folder_path: str) -> str:
    """返回指定文件夹内所有文件的文件名（不包括子文件夹内的文件）"""
    try:
        # 列出文件夹内所有条目
        entries = os.listdir(folder_path)
        # 过滤出文件（保留文件名，不含路径）
        files = [f for f in entries if os.path.isfile(os.path.join(folder_path, f))]
        sys_logger.info(f"[工具执行] 成功获取文件夹 '{folder_path}' 的文件列表，共 {len(files)} 个文件。")
        return json.dumps(files, ensure_ascii=False)
    except FileNotFoundError:
        msg = f"错误：文件夹 '{folder_path}' 不存在。请停止猜测路径，并向用户确认正确路径。"
        sys_logger.error(msg)
        return msg
    except PermissionError:
        msg = f"错误：没有权限访问文件夹 '{folder_path}'。请停止重试，并向用户说明权限问题。"
        sys_logger.error(msg)
        return msg
    except Exception as e:
        msg = f"获取文件列表发生未知错误：{e}。请停止重试，并向用户说明问题。"
        sys_logger.error(msg)
        return msg


def rename_file(file_path: str, new_name: str) -> bool:
    """
    将文件重命名为新名字（不包含路径，仅文件名或完整路径均可）
    参数：
        file_path: 原文件的完整路径
        new_name:  新文件名（例如 "newfile.txt"），可以包含扩展名
    """
    try:
        if not os.path.isfile(file_path):
            sys_logger.warning(f"错误：文件 '{file_path}' 不存在")
            return False

        dir_name = os.path.dirname(file_path)
        new_path = os.path.join(dir_name, new_name)

        os.rename(file_path, new_path)
        sys_logger.info(f"[工具执行] 重命名成功：{file_path} -> {new_path}")
        return True
    except Exception as e:
        sys_logger.error(f"重命名失败：{e}")
        return False


def query_csv(file_path: str, query_string: str) -> str:
    """
    专门用于查询大型 CSV 文件的数据过滤工具。
    【大模型注意】
    1. 必须用 Pandas 的 query 语法编写 query_string。
    2. 请严格使用 CSV 的真实列名（通常是英文，如 time, air_temperature）。
    """
    sys_logger.info(f"[工具执行] 正在查询 CSV: {file_path} | 条件: {query_string}")

    if not os.path.exists(file_path):
        msg = f"错误: 找不到文件 {file_path}"
        sys_logger.error(msg)
        return msg
    if os.path.isdir(file_path):
        msg = f"错误: {file_path} 是文件夹，不是 CSV 文件。请停止把文件夹当作 CSV 查询，并向用户确认具体文件路径。"
        sys_logger.error(msg)
        return msg

    try:
        # 1. 读取 CSV (消除警告)
        df = pd.read_csv(filepath_or_buffer=file_path)

        if not isinstance(df, pd.DataFrame):
            msg = "错误: 读取的结果不是有效的 DataFrame"
            sys_logger.error(msg)
            return msg

        # 2. 执行大模型写好的查询语句
        result_df = df.query(query_string)

        # 🌟 查不到数据时的“格式纠偏”
        if result_df.empty:
            cols = df.columns.tolist()
            sample_data = df.iloc[0].to_dict() if not df.empty else "无数据"

            sys_logger.warning(
                f"⚠️ 查询执行成功，但结果为空！\n这通常是因为你的 query_string (特别是时间格式) 与 CSV 的实际文本不匹配。\n请看一眼 CSV 真实的列名和第一条数据的格式，然后修改你的条件重试！\n-> 真实列名: {cols}\n-> 真实数据示例: {sample_data}")

            return (f"⚠️ 查询执行成功，但结果为空！\n"
                    f"这通常是因为你的 query_string (特别是时间格式) 与 CSV 的实际文本不匹配。\n"
                    f"请看一眼 CSV 真实的列名和第一条数据的格式，然后修改你的条件重试！\n"
                    f"-> 真实列名: {cols}\n"
                    f"-> 真实数据示例: {sample_data}")

        # 3. 正常返回 JSON 字符串
        json_result = str(result_df.to_json(orient="records", force_ascii=False))
        sys_logger.info(f"[工具执行] CSV查询成功，返回了 {len(result_df)} 条数据。")
        return json_result

    except Exception as e:
        # 🌟 核心升级 2：语法报错时的“列名纠偏”
        cols = df.columns.tolist() if 'df' in locals() else "未知"
        sys_logger.error(f"查询语法报错！错误信息: {str(e)}\n")
        return (f"查询语法报错！错误信息: {str(e)}\n这通常是因为你使用了不存在的列名（比如用了中文'时间'，但实际是'time'）。\n请使用真实的列名重新查询: {cols}"
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
    log_msg = (
        f"[报告接收中心] 成功捕获到大模型提交的结构化数据：\n"
        f"   - 时间: {target_time}\n"
        f"   - 空气温湿度: {air_temperature} ℃ | {air_humidity} %\n"
        f"   - 壁面温度: {wall_temperature} ℃\n"
        f"   - 风险判定: {'有风险' if has_risk else '安全'}\n"
        f"   - 处置建议: {action_advice}"
    )
    # 一次性静默打入日志
    sys_logger.info(log_msg)

    # 这里可以扩展业务逻辑，比如写入数据库、保存到本地 JSON 文件等
    # 目前直接返回一个确认信息给大模型
    return "成功：结构化诊断报告已安全提交至系统后台。"


# ==========================================
# 🌟 学术论文 Word 文本提取器
# ==========================================
def read_docx_file(file_path: str) -> str:
    """
    读取本地 .docx 格式的论文手稿文件，并将其中的段落和表格文字完整提取为纯文本。

    参数:
    - file_path: 本地 Word 文件的路径（例如：'test2/Manuscript.docx'）
    """
    try:
        doc = docx.Document(file_path)
        full_text = []

        # 1. 提取所有段落文字
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())

        # 2. 提取所有表格中的文字（学术论文中数据表很重要）
        for table in doc.tables:
            for row in table.rows:
                # 把每一行的单元格文字用 | 拼起来
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    full_text.append(" [表格数据] " + " | ".join(row_text))

        if not full_text:
            sys_logger.warning(f"⚠️ 读取了 {file_path}，但未发现有效内容。")
            return "⚠️ 读取成功，但未在文档中发现有效的文本内容。"

        # 把所有文本用换行符连起来，变成大模型能读懂的长文章
        result_txt = "\n".join(full_text)
        sys_logger.info(f"[工具执行] 成功读取 DOCX 文件: {file_path} (解析出 {len(result_txt)} 字符)")
        return result_txt

    except Exception as e:
        sys_logger.error(f"读取 .docx 文件失败，原因: {str(e)}")
        return f"读取 .docx 文件失败，原因: {str(e)}"


# ==========================================
# 🌟 学术论文结构化 JSON 报告提交中心/蚂蚁搬家式论文提交工具
# ==========================================
def submit_paper_section(section_name: str, content: str) -> str:
    """
    分批次提交论文的结构化总结。为了系统的稳定性，你必须调用此工具 7 次，每次仅提交一个章节。
    参数 section_name 必须严格是以下之一:
    Abstract, Introduction, Research aim, Methodology, Results, Discussion, Conclusion。
    """
    file_path = "test2/paper_summary_result.json"

    # 1. 尝试读取已有的 JSON 文件，如果没有则创建一个空字典
    summary_data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
        except json.JSONDecodeError:
            pass  # 如果文件损坏或为空，直接用空字典

    # 2. 将大模型刚刚传过来的新章节，追加进字典
    summary_data[f"{section_name}总结"] = content

    # 3. 重新写回本地硬盘
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=4)

        sys_logger.info(f"[报告接收中心] 已安全接收并持久化章节: {section_name}")
        return f" {section_name} 章节已保存。请继续调用此工具提交下一个章节，直到 7 个章节全部提交完毕！"
    except Exception as e:
        sys_logger.error(f"写入 {section_name} 时发生本地错误: {str(e)}")
        return f"写入 {section_name} 时发生本地错误: {str(e)}"


# ==========================================
# 🌟 Agentic Scratchpad (智能体便签本)
# ==========================================
def update_task_notes(current_action: str, checklist_status: str, next_step: str) -> str:
    """
    通用任务状态记录本。当你执行复杂的多步任务时，必须用此工具来记录进度、更新检查清单，并规划下一步。

    参数:
    - current_action: 你刚才完成了什么动作？
    - checklist_status: 当前完整的任务清单及各项的完成状态（如：[x] 提取摘要, [ ] 提取引言）。
    - next_step: 你的下一步行动计划是什么？
    """
    checklist_view = checklist_status.replace(']', '] ').replace('\n', ' | ')
    log_msg = (
        f"[Agent 内部思考/便签本更新] \n"
        f"   动作: {current_action}\n"
        f"   进度: {checklist_view}\n"
        f"   规划: {next_step}"
    )
    sys_logger.info(log_msg)

    return "笔记已更新。请严格根据 next_step 继续执行下一个工具调用，在所有任务打钩前，绝对保持静默！"


# 使用示例
if __name__ == "__main__":
    print_and_log("\n" + "=" * 50, "info")
    print_and_log("🚀 启动工具箱本地测试...", "info")

    folder = "./test/1"  # 替换为你的文件夹路径
    names = get_file_names(folder)
    print_and_log(f"获取到的文件名列表: {names}", "info")

    # rename_file("C:/temp/old.txt", "new.txt")

    host_info = get_host_info()
    print_and_log(f"主机信息: \n{host_info}", "info")

    print_and_log("=" * 50 + "\n", "info")
