#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于ChromaDB 知识库运维管理面板
功能：
1. 查看知识库全局统计（总 Chunk 数、包含哪些文件、每个文件多少个切片）
2. 抽查/预览指定文件的文本切片内容
3. 精准按文件名删除（某篇文档废弃或内容更新时使用）
4. 物理级重置清空数据库
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/10/14 21:13
version: V1.0
Target Python: 
"""

import os
from collections import Counter
from utils.logger_print import print_and_log
from RAG.vector_builder import get_chroma_db, clear_vector_database


def inspect_database():
    """1. 查验知识库资产分布"""
    vector_db = get_chroma_db()
    # 仅提取元数据与 ID
    data = vector_db.get(include=["metadatas"])
    ids = data.get("ids", [])
    metadatas = data.get("metadatas", [])

    total_chunks = len(ids)
    print("\n" + "=" * 50)
    print(f"📊 知识库全局统计概览：")
    print(f"   - 总存储切片数 (Chunks): {total_chunks} 个")

    if total_chunks == 0:
        print("   - 目前库内为空，暂无数据。")
        print("=" * 50 + "\n")
        return

    # 统计每个文件贡献了多少切片
    sources = [m.get("source", "未知来源") for m in metadatas if m]
    source_counts = Counter(sources)

    print(f"   - 包含独立文件数: {len(source_counts)} 个")
    print("-" * 50)
    print(f"{'序号':<4} | {'切片数 (Chunks)':<14} | {'文件名'}")
    print("-" * 50)
    for idx, (source, count) in enumerate(source_counts.items(), 1):
        print(f"{idx:<4} | {count:<14} | {source}")
    print("=" * 50 + "\n")


def preview_file_chunks(source_filename: str, preview_count: int = 2):
    """2. 抽查某文件的切片内容"""
    vector_db = get_chroma_db()
    results = vector_db.get(
        where={"source": source_filename},
        include=["documents", "metadatas"]
    )

    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    if not docs:
        print(f"⚠️ 未在库中检索到文件名为 [{source_filename}] 的切片！")
        return

    print(f"\n🔍 文件 [{source_filename}] 共包含 {len(docs)} 个切片。展示前 {min(preview_count, len(docs))} 个：\n")
    for i in range(min(preview_count, len(docs))):
        print(f"--- [切片 #{i + 1}] ---")
        print(f"🏷️ 元数据: {metas[i]}")
        print(f"📝 正文预览:\n{docs[i][:300]}...")
        print("-" * 40)


def delete_file_from_db(source_filename: str):
    """3. 精准按文件名删除该文件的所有切片"""
    vector_db = get_chroma_db()
    # 先查出该文件对应的所有唯一 ID
    results = vector_db.get(
        where={"source": source_filename},
        include=["metadatas"]
    )
    ids_to_delete = results.get("ids", [])

    if not ids_to_delete:
        print(f"⚠️ 找不到名为 [{source_filename}] 的记录，无需删除。")
        return

    confirm = input(f"🚨 警告: 确认从数据库中删除 [{source_filename}] 的全部 {len(ids_to_delete)} 个切片吗？(y/n): ")
    if confirm.lower() == 'y':
        # 通过 ID 列表执行精准删除
        vector_db.delete(ids=ids_to_delete)
        print(f"✅ 成功移除 [{source_filename}] 及其所有向量切片！")
    else:
        print("已取消删除操作。")


def main_menu():
    while True:
        print("\n" + "🛠️" * 10 + " ChromaDB 运维管理工具 " + "🛠️" * 10)
        print("1. 查看知识库文件与切片统计")
        print("2. 抽查指定文件的切片内容")
        print("3. 按文件名精准删除数据")
        print("4. 彻底清空整个知识库")
        print("0. 退出")
        choice = input("请选择操作 (0-4): ").strip()

        if choice == "1":
            inspect_database()
        elif choice == "2":
            file_name = input("请输入要抽查的文件名 (如: 云冈项目任务书.pdf): ").strip()
            preview_file_chunks(file_name)
        elif choice == "3":
            file_name = input("请输入要删除的文件名 (如: 旧版方案.docx): ").strip()
            delete_file_from_db(file_name)
        elif choice == "4":
            confirm = input("⚠️ 此操作不可逆！确认物理清空数据库所有内容？(yes/no): ").strip()
            if confirm == "yes":
                clear_vector_database()
        elif choice == "0":
            print("退出管理工具。")
            break
        else:
            print("输入有误，请重新选择。")


if __name__ == "__main__":
    main_menu()
