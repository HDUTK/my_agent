#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于agent能够使用的工具等
功能列表:
1. 获取宿主机硬件与系统信息 (get_host_info)
2. 本地文件纯文本读取 (read_file)
3. 文件夹下文件名提取 (get_file_names)
4. 本地文件重命名 (rename_file)
5. Pandas CSV 结构化数据过滤查询 (query_csv)
6. 石窟传感器结构化诊断报告提交 (submit_sensor_record)
7. 学术论文手稿 Word (.docx) 文本解析 (read_docx_file)
8. 论文 7 大章节结构化总结提交 (submit_paper_section)
9. 复杂任务便签本状态更新 (update_task_notes)
10. 冷路径向量经验检索 (search_past_experience): 热目录落空时按需检索过往实验与阈值参数
11. 经验卡片 ID 详情穿透 (get_experience_detail_by_id): 根据唯一 ID 二级拉取全量实验步骤与代码
12. 统一安全出口网关 (secure_file_output_gateway): 针对所有文件类型的统一清洗、可逆脱敏与物理阻断中枢
13. 本地 PDF 文本提取器 (read_pdf_file): 自动接入统一安全管道清洗
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

import chromadb
from typing import Optional

from utils.logger_print import sys_logger, print_and_log

# 集中读取 system_config.py 中的向量库与情境记忆配置
from config.system_config import (
    CHROMA_DB_DIR,
    EPISODIC_COLLECTION_NAME,
    EPISODIC_MEMORY_CONFIG
)

# 引入密级判定与切面安全装饰器
from utils.core_utils import detect_security_level, security_guard, security_proxy


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

@security_guard
def read_file(file_path: str) -> dict:
    """读取指定路径的文件内容（前置校验白名单/黑名单，后置执行DLP脱敏清洗）"""
    level = detect_security_level(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            msg = f.read()
            sys_logger.info(f"[工具执行] 成功读取文件: {file_path} (长度: {len(msg)})")
            # 🌟 返回标准载荷字典，由 @security_guard 切面拦截清洗
            return {"content": msg, "security_level": level, "file_path": file_path}
    except FileNotFoundError:
        msg = f"错误：文件 '{file_path}' 不存在"
        sys_logger.error(msg)
        return {"content": msg, "security_level": "L4", "file_path": file_path}
    except Exception as e:
        msg = f"读取文件时出错：{e}"
        sys_logger.error(msg)
        return {"content": msg, "security_level": "L4", "file_path": file_path}

@security_guard
def get_file_names(folder_path: str) -> str:
    """返回指定文件夹内所有文件的文件名（不包括子文件夹内的文件）（受白名单/黑名单安全网关保护，严禁遍历外部路径）"""
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

@security_guard
def rename_file(file_path: str, new_name: str) -> bool:
    """
    将文件重命名为新名字（不包含路径，仅文件名或完整路径均可）（受白名单/黑名单安全网关保护）
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

        # 针对目标重命名路径进行黑名单二次安全校验
        is_safe, deny_msg = security_proxy.verify_path_permission(new_path)
        if not is_safe:
            return deny_msg
        
        os.rename(file_path, new_path)
        sys_logger.info(f"[工具执行] 重命名成功：{file_path} -> {new_path}")
        return True
    except Exception as e:
        sys_logger.error(f"重命名失败：{e}")
        return False


@security_guard
def query_csv(file_path: str, query_string: str) -> dict:
    """
    专门用于查询大型 CSV 文件的数据过滤工具。
    【大模型注意】
    1. 必须用 Pandas 的 query 语法编写 query_string。
    2. 请严格使用 CSV 的真实列名（通常是英文，如 time, air_temperature）。
    """
    level = detect_security_level(file_path)
    sys_logger.info(f"[工具执行] 正在查询 CSV: {file_path} | 条件: {query_string}")

    if not os.path.exists(file_path):
        return {"content": f"错误: 找不到文件 {file_path}", "security_level": "L4", "file_path": file_path}
    if os.path.isdir(file_path):
        return {"content": f"错误: {file_path} 是文件夹，不是 CSV 文件。", "security_level": "L4", "file_path": file_path}

    try:
        df = pd.read_csv(filepath_or_buffer=file_path)
        if not isinstance(df, pd.DataFrame):
            return {"content": "错误: 读取的结果不是有效的 DataFrame", "security_level": "L4", "file_path": file_path}

        result_df = df.query(query_string)
        if result_df.empty:
            cols = df.columns.tolist()
            return {"content": f"⚠️ 查询结果为空！请核对列名: {cols}", "security_level": "L4", "file_path": file_path}

        json_result = str(result_df.to_json(orient="records", force_ascii=False))
        sys_logger.info(f"[工具执行] CSV查询成功，返回了 {len(result_df)} 条数据。")

        # 🌟 返回标准载荷字典，由 @security_guard 切面拦截清洗
        return {"content": json_result, "security_level": level, "file_path": file_path}

    except Exception as e:
        cols = df.columns.tolist() if 'df' in locals() else "未知"
        err_msg = f"查询语法报错: {str(e)}。当前列名: {cols}"
        return {"content": err_msg, "security_level": "L4", "file_path": file_path}


# ==========================================
# 🌟 数据提交工具
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
@security_guard
def read_docx_file(file_path: str) -> dict:
    """
    读取本地 .docx 格式的论文手稿文件，并将其中的段落和表格文字完整提取为纯文本。

    参数:
    - file_path: 本地 Word 文件的路径（例如：'test2/Manuscript.docx'）
    """
    level = detect_security_level(file_path)
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
            return {"content": "⚠️ 未在文档中发现有效的文本内容。", "security_level": "L4", "file_path": file_path}

        # 把所有文本用换行符连起来，变成大模型能读懂的长文章
        result_txt = "\n".join(full_text)
        sys_logger.info(f"[工具执行] 成功读取 DOCX 文件: {file_path} (解析出 {len(result_txt)} 字符)")
        return {"content": result_txt, "security_level": level, "file_path": file_path}

    except Exception as e:
        sys_logger.error(f"读取 .docx 文件失败，原因: {str(e)}")
        return {"content": f"读取 .docx 文件失败: {str(e)}", "security_level": "L4", "file_path": file_path}


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


# =========================================================================
# 🌟 第 4 层：情境经验冷库检索与详情穿透工具集 (Episodic Memory Tools)
# =========================================================================

def _get_episodic_collection():
    """
    内部辅助函数：安全获取经验卡片专属 Collection 实例。
    - 路径读取自 system_config.py 中的 CHROMA_DB_DIR
    - 集合名读取自 system_config.py 中的 EPISODIC_COLLECTION_NAME
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(name=EPISODIC_COLLECTION_NAME)


def search_past_experience(
        query: str,
        target_cave: Optional[str] = None,
        sensor_id: Optional[str] = None
) -> str:
    """
    【专有功能】: 仅用于从持久化经验冷库中检索过往完成的石窟微气候排查、故障分析、传感器阈值测定等科研历史任务。

    【调用触发铁律】:
    1. 当用户询问过去某次实验的具体结论、历史定案参数（如 A63 阈值、结露排查），且当前 System Prompt 中的【热目录】未列出该任务时，必须调用本工具！
    2. 支持通过 target_cave (如 "9号窟") 或 sensor_id (如 "A63") 缩小检索范围。

    【严禁调用场景】:
    1. 用户询问通用编程代码、通用数学计算、阅读本地文件时，严禁调用；
    2. 查阅石窟国家文保行业规范、专著文献时，应调用 search_knowledge_base，严禁调用此工具；
    3. 若【热目录】已明确给出了对应任务的结论，严禁调用本搜索工具！
    """
    try:
        col = _get_episodic_collection()

        # 库内完全为空时优雅直接返回
        if col.count() == 0:
            return "【冷库检索结果】：当前历史经验库尚无任何已归档的实验任务卡片。"

        top_k = EPISODIC_MEMORY_CONFIG.get("cold_search_top_k", 3)

        # 1. 动态构造元数据前置过滤条件 (缩小搜索范围，降低干扰)
        where_conditions = {}
        if sensor_id:
            where_conditions["sensors"] = {"$contains": sensor_id.strip()}
        if target_cave:
            where_conditions["target_caves"] = {"$contains": target_cave.strip()}

        where_clause = where_conditions if where_conditions else None

        # 2. 执行向量余弦相似度粗筛
        results = col.query(
            query_texts=[query],
            where=where_clause,
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]

        if not docs:
            return "【冷库检索结果】：未在历史经验库中检索到与该问题匹配的卡片。请确认测点代号或关键词是否准确。"

        # 3. 组装带有置信度分级警示与详情提取提示的文本块
        lines = [
            "==================================================",
            f"🟡【向量冷库语义检索候选卡片 (共召回 {len(docs)} 条)】："
        ]

        for idx, (doc_text, meta, item_id) in enumerate(zip(docs, metas, ids), 1):
            title = meta.get("title", "未命名任务") if meta else "未命名任务"
            caves = meta.get("target_caves", "-") if meta else "-"
            sensors = meta.get("sensors", "-") if meta else "-"
            conclusion = meta.get("final_conclusion", "无明确结论记录") if meta else "无明确结论记录"

            lines.append(f"\n--- [候选 #{idx} | ID: {item_id}] ---")
            lines.append(f"• 任务主题: {title}")
            lines.append(f"• 关联实体: 窟号: {caves} | 测点: {sensors}")
            lines.append(f"• 核心结论摘要: {conclusion}")
            lines.append(f"• 详细脉络切片: {doc_text}")
            lines.append(
                f"👉 (若需提取该任务的全量排查过程与代码，可调用 get_experience_detail_by_id(episode_id='{item_id}'))")

        # 4. 追加认知谦逊约束警示语 (来自 system_config.py)
        warning_text = EPISODIC_MEMORY_CONFIG.get("cold_confidence_warning", "")
        lines.append("\n" + warning_text)
        lines.append("==================================================")

        sys_logger.info(f"🔍 [冷路径召回] 成功按语义在冷库中检索到 {len(docs)} 条历史卡片。")
        raw_result_text = "\n".join(lines)
        # -------------------------------------------------------------
        # 冷库卡片发往大模型前统一脱敏
        # 即使卡片在 ChromaDB 中记录的是真实的 A63、9号窟，交还大模型时自动变成代号
        # -------------------------------------------------------------
        return security_proxy.mask_user_input(raw_result_text)

    except Exception as e:
        sys_logger.error(f"❌ [冷库检索异常] {e}")
        return f"检索历史经验冷库时出现内部错误: {e}"


def get_experience_detail_by_id(episode_id: str) -> str:
    """
    【专有功能】: 根据经验卡片的唯一 ID (如 session_20261019_153022) 穿透提取全量排查技术细节与刚性参数。

    【何时调用】: 当热目录或冷库搜索结果中的简要结论不足以解答细节，需追溯当时具体的排查步骤、对比数据或代码段时调用。
    """
    try:
        col = _get_episodic_collection()
        res = col.get(ids=[episode_id.strip()], include=["documents", "metadatas"])

        docs = res.get("documents", [])
        metas = res.get("metadatas", [])

        if not docs:
            return f"⚠️ 未在经验库中找到 ID 为 [{episode_id}] 的完整卡片记录。"

        meta = metas[0] if metas else {}
        doc = docs[0]

        detail_lines = [
            "==================================================",
            f"📋【历史任务全量经验档案 (ID: {episode_id})】",
            f"• 任务标题: {meta.get('title', '-')}",
            f"• 涉及洞窟: {meta.get('target_caves', '-')}",
            f"• 涉及测点: {meta.get('sensors', '-')}",
            f"• 核心问题与排查目标: {meta.get('problem_and_goal', '-')}",
            f"• 最终定案结论与参数: {meta.get('final_conclusion', '-')}",
            f"• 完整密集总结:\n  {doc}",
            "=================================================="
        ]
        raw_detail_text = "\n".join(detail_lines)

        # -------------------------------------------------------------
        # 🌟 穿透详情发往大模型前同样强制脱敏
        # -------------------------------------------------------------
        return security_proxy.mask_user_input(raw_detail_text)

    except Exception as e:
        sys_logger.error(f"❌ [穿透详情异常] {e}")
        return f"读取经验卡片详情失败: {e}"


# 使用示例
if __name__ == "__main__":
    print_and_log("\n" + "=" * 50, "info")
    print_and_log("🚀 启动工具箱本地测试...", "info")

    folder = "D:/PythonProject/AI_Agent/my_code/test/1"  # 替换文件夹路径
    names = get_file_names(folder)
    print_and_log(f"获取到的文件名列表: {names}", "info")

    # rename_file("C:/temp/old.txt", "new.txt")

    host_info = get_host_info()
    print_and_log(f"主机信息: \n{host_info}", "info")

    print_and_log("=" * 50 + "\n", "info")
