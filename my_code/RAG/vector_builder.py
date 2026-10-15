#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于RAG 第三阶段：向量化 (Embedding) 与 ChromaDB 入库
加载中文最强开源模型 BGE
将数据存入指定的绝对路径。
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/06/02 16:59
version: V1.0
Target Python: 3.12
"""

import os
import gc
import hashlib
import shutil

from config.system_config import DB_PERSIST_PATH, COLLECTION_NAME, DEVICE
from config.agent_config import MODEL_REGISTRY
from utils.core_utils import apply_model_environment
from utils.logger_print import print_and_log


# 读取bge_m3模型的配置
apply_model_environment(MODEL_REGISTRY['bge_m3']['model_name_simple'])

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ----------------------------------------------------
# 🌟 模块级全局单例句柄：Embedding 模型与 Chroma 数据库连接
# ----------------------------------------------------
_CACHED_EMBEDDINGS = None
_CACHED_CHROMA_DB = None


def clear_vector_database():
    """
    🌟 物理级清空本地向量数据库
    """
    global _CACHED_CHROMA_DB
    print_and_log("\n" + "⚠️" * 20, "warning")
    print_and_log(" 收到清空指令，正在准备销毁历史向量数据库...", "warning")

    try:
        # 1. 尝试使用 Chroma 原生方式删除并重置集合（优雅且无锁冲突）
        vector_db = get_chroma_db()
        vector_db.delete_collection()
        print_and_log(f"🗑️ 已成功通过 Chroma 原生指令清除集合: [{COLLECTION_NAME}]", "info")

        # 重置内存单例，促使下次自动重新生成新 Collection
        _CACHED_CHROMA_DB = None
        gc.collect()
    except Exception as e:
        print_and_log(f"⚠️ 原生重置异常，尝试物理级销毁: {e}", "warning")

        # 2. 保底方案：显式停止底层系统线程并释放句柄
        if _CACHED_CHROMA_DB is not None:
            try:
                # 显式关闭 Chroma 底层持久化系统与 SQLite 线程
                _CACHED_CHROMA_DB._client._system.stop()
            except Exception:
                pass
            _CACHED_CHROMA_DB = None
            gc.collect()

        if os.path.exists(DB_PERSIST_PATH):
            try:
                shutil.rmtree(DB_PERSIST_PATH)
                print_and_log(f"🗑️ 已成功彻底删除历史数据库文件夹: {DB_PERSIST_PATH}", "info")
            except Exception as err:
                print_and_log(
                    f"❌ 物理删除失败！请检查是否有其他终端（如 main.py 或 PyCharm Console）正在运行: {err}",
                    "error"
                )

    print_and_log("⚠️" * 20 + "\n", "warning")

    return


def generate_chunk_id(chunk: Document) -> str:
    """
    🌟 利用 MD5 算法为每个纸条生成独一无二的“数字指纹”。
    只要文件的来源 (source) 和 文本内容 (page_content) 不变，生成的 ID 就绝对不变！
    """
    # 提取来源文件名，如果没有就用 unknown
    source = chunk.metadata.get('source', 'unknown')
    # 提取这段纸条的具体文字内容
    content = chunk.page_content

    # 将来源和内容拼接成一个超级字符串
    unique_string = f"{source}::{content}"

    # 计算 MD5 哈希值并返回 32 位字符串
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()


def get_bge_embeddings():
    """
    🌟 加载智源最新一代 BGE-M3 多语言/长文本向量模型 (纯本地运行)
    首次运行会自动去镜像源下载约 2.2GB 的模型权重，之后永久本地秒加载。
    """
    global _CACHED_EMBEDDINGS
    if _CACHED_EMBEDDINGS is not None:
        return _CACHED_EMBEDDINGS

    print_and_log(" 正在加载 " + MODEL_REGISTRY['bge_m3']['model_name'] + "  顶级向量模型引擎...", "info")

    # 🌟 模型名称 bge-m3
    model_name = MODEL_REGISTRY['bge_m3']['model_name']

    # device
    model_kwargs = {'device': DEVICE}

    # normalize_embeddings=True 依然极其重要！确保余弦相似度的准确计算
    encode_kwargs = {'normalize_embeddings': True}

    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print_and_log(MODEL_REGISTRY['bge_m3']['model_name'] + " 模型加载完毕！", "info")
    return _CACHED_EMBEDDINGS


def get_chroma_db(embeddings=None) -> Chroma:
    """
    🌟 获取 ChromaDB 客户端单例
    保持 SQLite 引擎与 HNSW 索引句柄常驻，避免每个文件处理时反复重复挂载与 I/O 校验。
    """
    global _CACHED_CHROMA_DB
    if _CACHED_CHROMA_DB is not None:
        return _CACHED_CHROMA_DB

    if embeddings is None:
        embeddings = get_bge_embeddings()

    _CACHED_CHROMA_DB = Chroma(
        persist_directory=DB_PERSIST_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return _CACHED_CHROMA_DB


def save_chunks_to_chroma(chunks: list[Document], embeddings=None):
    """
    将切分好的纸条 (Chunks) 连同它们的标签 (Metadata) 一起变成坐标，存入指定路径的 ChromaDB
    支持接收外部预热好的 embeddings 实例，避免重复创建连接。
    """
    if not chunks:
        print_and_log("⚠️ 传入的 Chunks 为空，无法入库。", "warning")
        return None

    print_and_log(f"📦 准备将 {len(chunks)} 个 Chunk 进行向量化并存入 ChromaDB...", "info")
    print_and_log(f"📁 目标存储路径: {DB_PERSIST_PATH}", "info")

    # 🌟 批内清洗与基于 MD5 的精准去重
    unique_chunks = []
    unique_chunk_ids = []
    seen_ids = set()
    filtered_empty_count = 0

    # # 🌟 批量为所有纸条生成固定的数字指纹 ID
    # chunk_ids = [generate_chunk_id(chunk) for chunk in chunks]
    # print_and_log(f"🔑 已生成固定唯一哈希 ID，启动防重复入库 (Upsert 机制)...", "info")
    #
    # # 1. 获取常驻的 Chroma 客户端
    # vector_db = get_chroma_db(embeddings=embeddings)
    #
    # # 2. 核心入库指令！
    # # Chroma.from_documents 会做两件事：
    # # ① 遍历所有的 chunk，让 BGE 模型把文字算成向量。
    # # ② 把文字、标签、向量打包存入 persist_directory 指定的本地文件夹中。
    # # 调用 add_documents 增量追加，不重建客户端
    # vector_db.add_documents(documents=chunks, ids=chunk_ids)
    #
    # print_and_log("🎉 恭喜！所有数据已成功向量化并持久化入库！", "info")
    # return vector_db

    for chunk in chunks:
        # 1. 过滤纯空白或换行符构成的无意义脏切片
        if not chunk.page_content or not chunk.page_content.strip():
            filtered_empty_count += 1
            continue

        chunk_id = generate_chunk_id(chunk)

        # 2. 拦截同一文档内内容完全相同的重复切片（如重复的页眉、页脚、表头）
        if chunk_id in seen_ids:
            continue

        seen_ids.add(chunk_id)
        unique_chunks.append(chunk)
        unique_chunk_ids.append(chunk_id)

    # 打印去重与清洗结果
    duplicate_count = len(chunks) - len(unique_chunks) - filtered_empty_count
    if duplicate_count > 0 or filtered_empty_count > 0:
        print_and_log(
            f"🧹 [数据清洗] 自动剔除同文档内重复切片: {duplicate_count} 个，空切片: {filtered_empty_count} 个",
            "info"
        )

    if not unique_chunks:
        print_and_log("⚠️ 经过清洗后无有效切片可入库。", "warning")
        return None

    print_and_log(f"🔑 实际入库有效独立切片数: {len(unique_chunks)} 个，正在写入向量库...", "info")

    # 3. 获取常驻客户端并安全追加
    vector_db = get_chroma_db(embeddings=embeddings)
    vector_db.add_documents(documents=unique_chunks, ids=unique_chunk_ids)

    print_and_log("🎉 当前批次数据已成功向量化并追加持久化入库！", "info")
    return vector_db


def query_chroma_db(query_text: str, top_k: int = 3):
    """
    测试检索功能：输入问题，寻找最相似的纸条
    ⭐️ 单元测试时使用
    """
    print_and_log(f"\n🔍 正在检索问题: '{query_text}'", "info")

    # 复用已初始化的客户端单例，不再重复读盘建立连接
    vector_db = get_chroma_db()

    # 执行余弦相似度检索，找回最相似的前 top_k 个结果
    results = vector_db.similarity_search_with_score(query_text, k=top_k)

    return results


if __name__ == "__main__":
    from RAG.document_parser import read_any_file
    from RAG.chunk import intelligent_chunker

    # 🎯 找一个本地真实的测试文件（挑一个内容丰富的，比如 JSON 或 Excel）
    test_file_path = r"D:/PythonProject/AI_Agent/my_code/test/2/Manuscript.docx"

    print_and_log("\n" + "=" * 60)
    print_and_log("🚀 RAG 全链路启动：解析 -> 切分 -> 向量入库")
    print_and_log("=" * 60)

    # 1. 解析
    raw_text = read_any_file(test_file_path)
    # 2. 切分
    chunks = intelligent_chunker(raw_text, os.path.basename(test_file_path))

    # 3. 入库
    if chunks:
        save_chunks_to_chroma(chunks)

        # 4. 立刻测试一下检索！
        # 随便搜个测试问题
        search_results = query_chroma_db("论文的结论是什么？", top_k=2)

        print("\n🏆 检索结果展示：")
        for i, (doc, score) in enumerate(search_results):
            print_and_log(f"\n【Top {i + 1}】(相似度分数: {score:.4f})")
            print_and_log(f"🏷️ 标签: {doc.metadata}")
            print_and_log(f"📝 内容:\n{doc.page_content}")
