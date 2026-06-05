#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于RAG 第二阶段：文档智能切分 (Chunking)
支持基于文件类型的路由分发策略，自动注入 Metadata。
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/06/02 14:53
version: V1.0
Target Python: 3.12
"""


import os

from config.agent_config import CHUNK_CONFIG
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_markdown_or_text(text: str, source_file: str) -> list[Document]:
    """
    【路线 A】针对 PDF(Marker输出), MD, TXT, DOCX
    策略：结构化 Markdown 标题切分 + 递归字符切分保底
    """
    # 1. 先按标题切分，保留逻辑骨架并自动生成元数据
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # 如果纯 TXT 没有标题，这里会原样返回一整个大块，不影响后续处理
    md_splits = md_splitter.split_text(text)

    # 2. 递归字符保底切分 (防止某个标题下的内容依然超长)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CONFIG["markdown_or_text"]["CHUNK_SIZE"],  # 每张纸条最大字数
        chunk_overlap=CHUNK_CONFIG["markdown_or_text"]["CHUNK_OVERLAP"],  # 重叠字数，防止上下文断裂
        separators=CHUNK_CONFIG["markdown_or_text"]["SEPARATOR"]
    )

    chunks = text_splitter.split_documents(md_splits)

    # 3. 补充基础 Metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = source_file
        chunk.metadata["chunk_index"] = i
        chunk.metadata["data_type"] = "natural_language"

    return chunks


def chunk_spreadsheet(text: str, source_file: str) -> list[Document]:
    """
    【路线 B】针对 CSV, XLSX (完美升级版)
    策略：先提取 Sheet 层级的元数据，再按行切分避免上下文丢失！
    """
    # 1. 🌟 拦截提取 Sheet 名称！
    # 利用在 Parser 中埋下的伏笔，把 "### 表格子页 (Sheet): XXX" 提取为 Metadata
    headers_to_split_on = [
        ("###", "Sheet_Name"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # 这步会将多 Sheet 长字符串，优雅地变成按 Sheet 划分的 Document 列表
    # 并且自动给每个块打上类似 {'Sheet_Name': '表格子页 (Sheet): 2025年温湿度'} 的标签！
    sheet_splits = md_splitter.split_text(text)

    # 2. 对每个 Sheet 里面的数据按行进行保底切分
    table_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CONFIG["spreadsheet"]["CHUNK_SIZE"],  # 表格数据密度大，大概包含 3-5 行数据
        chunk_overlap=CHUNK_CONFIG["spreadsheet"]["CHUNK_OVERLAP"],
        separators=CHUNK_CONFIG["spreadsheet"]["SEPARATOR"]  # 严禁按句号逗号切表格，保持表格行的完整
    )

    # 将包含 Sheet Metadata 的块传给递归切分器
    # 🌟 神奇之处：切出来的所有小行，都会自动继承父级 Sheet 的 Metadata！
    chunks = table_splitter.split_documents(sheet_splits)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = source_file
        chunk.metadata["chunk_index"] = i
        chunk.metadata["data_type"] = "spreadsheet_table"

    return chunks


def chunk_json(text: str, source_file: str) -> list[Document]:
    """
    【路线 C】针对 JSON
    策略：严格按照 JSON 括号层级进行递归切分
    """
    json_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CONFIG["json"]["CHUNK_SIZE"],
        chunk_overlap=CHUNK_CONFIG["json"]["CHUNK_OVERLAP"],
        separators=CHUNK_CONFIG["json"]["SEPARATOR"]  # 针对 JSON 结构的特殊分割符
    )

    raw_doc = [Document(page_content=text)]
    chunks = json_splitter.split_documents(raw_doc)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = source_file
        chunk.metadata["chunk_index"] = i
        chunk.metadata["data_type"] = "json_structured"

    return chunks


def intelligent_chunker(text_content: str, file_path: str) -> list[Document]:
    """
    🌟 路由中枢：根据原始文件扩展名，将文本分发给最合适的切分器
    """
    if not text_content or text_content.startswith("⚠️") or text_content.startswith("["):
        print(f"⚠️ 跳过切分：文件 {file_path} 内容为空或解析报错。")
        return []

    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)

    print(f"✂ 开始智能切分 [{file_name}] ...")

    if ext in ['.pdf', '.md', '.txt', '.docx']:
        return chunk_markdown_or_text(text_content, file_name)

    elif ext in ['.csv', '.xlsx', '.xls']:
        return chunk_spreadsheet(text_content, file_name)

    elif ext == '.json':
        return chunk_json(text_content, file_name)

    else:
        # 未知格式，采用标准文本切分兜底
        return chunk_markdown_or_text(text_content, file_name)


# 本地单元测试 (仅当直接运行此文件时执行)
# ==========================================
if __name__ == "__main__":
    # 🌟 从解析器文件中引入统一读取接口
    from RAG.document_parser import read_any_file
    print("🚀 启动 Chunker 模块本地真实文件遍历测试...\n")

    # 真实测试文件路径
    base_dir = r"../test/2/parser_test"

    # 6 个真实测试文件
    test_files = [
        # "d.txt",
        # "default_chat.md",
        # "paper_summary.json",
        # "杭电自动化13级新生住宿.xlsx",
        # "BYDX.docx",
        "博士在读成绩单.pdf",
        # "动漫游.pdf",
    ]

    for file_names in test_files:
        file_path = os.path.join(base_dir, file_names)
        print("\n" + "=" * 70)
        print(f"🎯 正在测试目标文件: {file_names}")
        print("=" * 70)

        # 1. 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"❌ 找不到文件: {file_path}\n请检查路径是否正确。")
            continue

        # 2. 呼叫 Parser 提取纯文本
        print("⏳ 正在解析提取纯文本...")
        raw_text = read_any_file(file_path)

        if raw_text.startswith("⚠️") or raw_text.startswith("["):
            print(f"❌ 解析异常终止:\n{raw_text}")
            continue

        # 3. 呼叫 Chunker 进行智能切分与打标签
        print("⏳ 正在进行智能切分 (Chunking)...")
        # 假设当前文件中的核心路由函数名为 intelligent_chunker
        chunks = intelligent_chunker(raw_text, file_names)

        if not chunks:
            print("⚠️ 警告：切分后没有获得任何块 (Chunk)。")
            continue

        print(f"✅ 完美！该文档被切分成了 {len(chunks)} 个 Chunk。")

        # 4. 质检抽查 (打印前 2 个 Chunk，限制字数防刷屏)
        print("-" * 40 + " 抽查预览 " + "-" * 40)
        display_count = min(2, len(chunks))

        for i in range(display_count):
            chunk = chunks[i]
            print(f"【📄 第 {i + 1} 块 Chunk】")
            print(f"🏷️ 元数据 (Metadata): {chunk.metadata}")

            content_preview = chunk.page_content
            if len(content_preview) > 200:
                content_preview = content_preview[:200] + "\n... (内容过长，已折叠展示) ..."

            print(f"📝 文本内容:\n{content_preview}\n")
            print("-" * 70)

    print("\n🎉 所有本地真实文件 Chunker 单元测试运行完毕！")
