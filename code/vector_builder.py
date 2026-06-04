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
import warnings
import hashlib


# 🌟 消除 Flash Attention 警告
warnings.filterwarnings("ignore", message=".*1Torch was not compiled with flash attention.*")

# 🌟 强制终端使用 UTF-8 编码
os.environ["PYTHONIOENCODING"] = "utf-8"

# 🌟 指定大模型位置
os.environ["HF_HOME"] = "D:/Python31210/HuggingFace_Models"

# 🌟 强行拔掉 Python 的代理管子，防止关闭 VPN 后端口卡死
os.environ['HTTP_PROXY'] = ""
os.environ['HTTPS_PROXY'] = ""

# 🌟 彻底关闭 ChromaDB 的后台匿名数据收集线程（防止卡死）
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# 🌟 关闭 Tokenizer 导致的多线程死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🌟 强行接管下载通道，指向国内 HF 镜像源，防止 BGE 模型下载卡死！
# (需要检查更新模型时打开，同时注释掉之后下面两行代码)
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 🌟 开启终极离线模式：严禁程序偷偷连网检查更新，纯本地物理读取！
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ==========================================
# 🌟 指定的数据库存放位置！
# 建议在项目里建一个专门的文件夹，比如 db_storage
# ==========================================
DB_PERSIST_PATH = r"E:\Vector_Database_for_Agent\db_storage\chroma_db"
COLLECTION_NAME = "tk_knowledge"  # 集合的名字（相当于关系型数据库的表名）


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
    print("⏳ 正在加载 BGE-M3 顶级向量模型引擎...")

    # 🌟 模型名称 bge-m3
    model_name = "BAAI/bge-m3"

    # device: 如果电脑有英伟达显卡，强烈建议改为 'cuda' 会有百倍加速
    # 如果没有显卡或用的是轻薄本，保持 'cpu' 即可
    model_kwargs = {'device': 'cuda'}

    # normalize_embeddings=True 依然极其重要！确保余弦相似度的准确计算
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("✅ BGE-M3 模型加载完毕！")
    return embeddings


def save_chunks_to_chroma(chunks: list[Document]):
    """
    将切分好的纸条 (Chunks) 连同它们的标签 (Metadata) 一起变成坐标，存入指定路径的 ChromaDB
    """
    if not chunks:
        print("⚠️ 传入的 Chunks 为空，无法入库。")
        return None

    print(f"📦 准备将 {len(chunks)} 个 Chunk 进行向量化并存入 ChromaDB...")
    print(f"📁 目标存储路径: {DB_PERSIST_PATH}")

    # 🌟 批量为所有纸条生成固定的数字指纹 ID
    chunk_ids = [generate_chunk_id(chunk) for chunk in chunks]
    print(f"🔑 已生成固定唯一哈希 ID，启动防重复入库 (Upsert 机制)...")

    # 1. 唤醒模型
    embeddings = get_bge_embeddings()

    # 2. 核心入库指令！
    # Chroma.from_documents 会做两件事：
    # ① 遍历所有的 chunk，让 BGE 模型把文字算成向量。
    # ② 把文字、标签、向量打包存入 persist_directory 指定的本地文件夹中。
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=chunk_ids,
        persist_directory=DB_PERSIST_PATH,
        collection_name=COLLECTION_NAME
    )

    print("🎉 恭喜！所有数据已成功向量化并持久化入库！")
    return vector_db


def query_chroma_db(query_text: str, top_k: int = 3):
    """
    测试检索功能：输入问题，寻找最相似的纸条
    ⭐️ 单元测试时使用
    """
    print(f"\n🔍 正在检索问题: '{query_text}'")

    # 唤醒模型（把问题也变成坐标）
    embeddings = get_bge_embeddings()

    # 连接到我们之前建好的本地数据库
    vector_db = Chroma(
        persist_directory=DB_PERSIST_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # 执行余弦相似度检索，找回最相似的前 top_k 个结果
    results = vector_db.similarity_search_with_score(query_text, k=top_k)

    return results


if __name__ == "__main__":
    from document_parser import read_any_file
    from chunk import intelligent_chunker

    # 🎯 找一个本地真实的测试文件（挑一个内容丰富的，比如 JSON 或 Excel）
    test_file_path = r"D:\PythonProject\AI_Agent\code\test2\parser_test\Manuscript.docx"

    print("\n" + "=" * 60)
    print("🚀 RAG 全链路启动：解析 -> 切分 -> 向量入库")
    print("=" * 60)

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
            print(f"\n【Top {i + 1}】(相似度分数: {score:.4f})")
            print(f"🏷️ 标签: {doc.metadata}")
            print(f"📝 内容:\n{doc.page_content}")
