#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于知识库批量入库执行脚本
执行此脚本，会自动遍历指定文件夹下的所有支持文件，并依次完成解析、切分与入库。
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/06/08 16:38
version: V1.0
Target Python: 
"""


import os
from utils.logger_print import print_and_log

# 引入三大核心组件
from RAG.document_parser import read_any_file
from RAG.chunk import intelligent_chunker
from RAG.vector_builder import save_chunks_to_chroma, clear_vector_database


# ==========================================
# 🌟 配置入库目标文件夹
# ==========================================
TARGET_FOLDER = r"D:/PythonProject/AI_Agent/knowledge_source"  # 替换为实际存放文件的文件夹路径

# 🌟 是否在入库前清空历史库？(True: 彻底重置 | False: 追加更新)
CLEAR_OLD_DB = False


def batch_build_knowledge_base(folder_path: str):
    print_and_log("\n" + "🚀" * 20, "info")
    print_and_log(f" 启动批量入库管线！目标文件夹: {folder_path}", "info")
    print_and_log("🚀" * 20 + "\n", "info")

    if not os.path.exists(folder_path):
        print_and_log(f"❌ 找不到文件夹: {folder_path}，请检查路径！", "error")
        return

    # 🌟 在开始处理文件之前，根据开关决定是否清空数据库
    if CLEAR_OLD_DB:
        clear_vector_database()

    # 获取文件夹下的所有文件, 使用 os.walk 递归扫描所有子文件夹
    all_file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 拼出文件的完整绝对路径并收集起来
            all_file_paths.append(os.path.join(root, file))

    if not all_file_paths:
        print_and_log(f"⚠️ 文件夹 {folder_path} 是空的，没有需要入库的文件。", "warning")
        return

    print_and_log(f"📦 共发现 {len(all_file_paths)} 个待处理文件。开始执行流水线...\n", "info")

    success_count = 0
    failed_count = 0
    total_chunks_saved = 0

    # 开始循环处理每一个文件
    # 开始循环处理每一个文件
    for idx, file_path in enumerate(all_file_paths):
        # 从完整路径中提取出单纯的文件名，用来打日志
        file_name = os.path.basename(file_path)
        print_and_log(f"🔄 [{idx + 1}/{len(all_file_paths)}] 正在处理: {file_name} ...", "info")

        try:
            # 第一阶段：解析提取文本
            raw_text = read_any_file(file_path)
            if raw_text.startswith("⚠️") or raw_text.startswith("["):
                print_and_log(f"  ❌ 解析跳过: {file_name} ({raw_text})", "error")
                failed_count += 1
                continue

            # 第二阶段：智能切分
            chunks = intelligent_chunker(raw_text, file_name)
            if not chunks:
                print_and_log(f"  ⚠️ 切分跳过: {file_name} 未生成任何有效内容。", "warning")
                failed_count += 1
                continue

            # 第三阶段：向量化入库
            save_chunks_to_chroma(chunks)

            success_count += 1
            total_chunks_saved += len(chunks)
            print_and_log(f"  ✅ {file_name} 处理完成！(生成 {len(chunks)} 个碎片)\n", "info")

        except Exception as e:
            print_and_log(f"  💥 处理 {file_name} 时发生致命崩溃: {str(e)}", "error")
            failed_count += 1
            continue

    # 汇总报告
    print_and_log("\n" + "=" * 50, "info")
    print_and_log("🎉 批量入库任务执行完毕！", "info")
    print_and_log(f"📊 统计：", "info")
    print_and_log(f"   - 成功处理文件: {success_count} 个", "info")
    print_and_log(f"   - 失败/跳过文件: {failed_count} 个", "info")
    print_and_log(f"   - 累计入库碎片 (Chunks): {total_chunks_saved} 个", "info")
    print_and_log("=" * 50 + "\n", "info")


if __name__ == "__main__":
    batch_build_knowledge_base(TARGET_FOLDER)

