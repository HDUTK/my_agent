#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于RAG 第四阶段：多路混合召回 (Chroma + BM25) 与 BGE-Reranker 精准重排管线
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/06/03 21:13
version: V1.0
Target Python: 
"""

from config.system_config import DB_PERSIST_PATH, COLLECTION_NAME, DEVICE
from config.agent_config import MODEL_REGISTRY
from utils.core_utils import apply_model_environment
from utils.logger_print import sys_logger, print_and_log

# 连续为本文件需要的两个模型注入配置
apply_model_environment(MODEL_REGISTRY['bge_m3']['model_name_simple'])
apply_model_environment(MODEL_REGISTRY['bge_reranker']['model_name_simple'])

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class HybridRerankRetriever:
    """
    🌟 工业级混合检索与重排核心控制器
    """

    def __init__(self):
        sys_logger.info("\n" + "=" * 50)
        sys_logger.info(" 正在初始化企业级混合检索与重排引擎...")

        # 1. 初始化左路：加载 BGE-M3 向量模型
        sys_logger.info(" 正在加载左路：" + MODEL_REGISTRY['bge_m3']['model_name_simple'] + " 向量模型 (GPU 加速)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_REGISTRY["bge_m3"]["model_name"],
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 2. 连接本地 ChromaDB 数据库
        self.vector_db = Chroma(
            persist_directory=DB_PERSIST_PATH,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )

        # 3. 初始化右路：从 ChromaDB 中提取所有文本，构建本地 BM25 关键词索引
        sys_logger.info(" 正在加载右路：提取全库文本并构建 BM25 关键词索引...")
        # 原生读取库里所有的原始纸条
        all_content = self.vector_db.get()
        self.all_documents = []

        # 将读取出来的散装数据重新组装成 LangChain 的 Document 对象
        for idx in range(len(all_content['ids'])):
            doc = Document(
                page_content=all_content['documents'][idx],
                metadata=all_content['metadatas'][idx] if all_content['metadatas'] else {}
            )
            self.all_documents.append(doc)

        sys_logger.info(f" 基础数据库加载成功！当前知识库共包含 【 {len(self.all_documents)} 】 个文本片段。")

        # 极其关键：对文本进行基础切分（按字/词切分），以便 BM25 计算频率
        # 关于中文字符，最简单的做法就是直接变成一个个单独的字列表
        tokenized_corpus = [list(doc.page_content) for doc in self.all_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 4. 初始化终极总监：加载 BGE-Reranker-v2-m3 重排模型
        sys_logger.info(" 正在加载终极重排大模型: " + MODEL_REGISTRY['bge_reranker']['model_name_simple'] + " ...")
        # 使用 CrossEncoder 架构直接加载重排器，并强制推向 CUDA 显卡加速
        self.reranker = CrossEncoder(MODEL_REGISTRY['bge_reranker']['model_name'], device="cuda")

        sys_logger.info(" 混合检索与重排引擎全部就绪！你可以开始精准大海捞针了！")
        sys_logger.info("=" * 50 + "\n")

    def search(self, query: str, score_threshold: float = 0.0, top_k: int = 2) -> list[Document]:
        """
        核心搜寻函数：经历【向量找 15个】 + 【关键词找 10个】 -> 【合并去重】 -> 【重排】 -> 【截取前k个】
        """
        sys_logger.info(f" 收到用户深度提问: '{query}'")

        # ------------ 【第一步：左路向量初筛】 ------------
        # 从海量数据里先捞出 x 条最神似的
        dense_results = self.vector_db.similarity_search(query, k=MODEL_REGISTRY["bge_m3"]["search_number"])
        sys_logger.info(f"  -> [左路向量] 成功初筛出 {len(dense_results)} 条语义相关文档。")

        # ------------ 【第二步：右路关键词初筛】 ------------
        # 将问题也切成字的列表，丢给 BM25 算法去算分
        tokenized_query = list(query)
        # 获取最形似的前 y 条结果
        sparse_results = self.bm25.get_top_n(tokenized_query, self.all_documents, n=MODEL_REGISTRY["BM25_search_number"])
        sys_logger.info(f"  -> [右路关键词] 成功初筛出 {len(sparse_results)} 条字面精准文档。")

        # ------------ 【第三步：合并与绝对去重】 ------------
        all_candidates = dense_results + sparse_results

        # 利用字典（以文本内容为 Key）进行绝对去重
        unique_candidates_dict = {}
        for doc in all_candidates:
            unique_candidates_dict[doc.page_content] = doc

        candidates = list(unique_candidates_dict.values())
        sys_logger.info(f"  -> [合并去重] 双路会师完成，剔除重复项后，共有 {len(candidates)} 篇文档进入重排总决赛。")

        if not candidates:
            return []

        # ------------ 【第四步：技术总监 BGE-Reranker 终极重排】 ------------
        sys_logger.info(f" 启动 " + MODEL_REGISTRY["bge_reranker"]["model_name_simple"] + " 神经网络进行交叉对比打分...")

        # 组装重排模型需要的标准格式：[[问题, 文档1文本], [问题, 文档2文本], ...]
        pairs = [[query, doc.page_content] for doc in candidates]

        # 显卡全速运转，直接预测出所有候选人的真实得分（返回一个包含浮点数的列表）
        scores = self.reranker.predict(pairs)

        # 将分数绑定到 Document 的元数据（metadata）中，方便后面查看
        for idx, score in enumerate(scores):
            candidates[idx].metadata["rerank_score"] = float(score)

        # 根据重排分数，从大到小（降序）重新排列整支队伍
        candidates.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

        # ------------ 【第五步：根据分数和数量截取最精华的 Top-K】 ------------
        final_results = []
        for doc in candidates:
            # 过滤掉低于分数阈值的垃圾水军（如果设了阈值的话）
            if doc.metadata["rerank_score"] >= score_threshold:
                final_results.append(doc)
            if len(final_results) == top_k:
                break

        sys_logger.info(f" 重排总决赛结束！已为您精准筛选出最顶尖的 {len(final_results)} 个黄金片段。")
        return final_results


# ==========================================
# 🧪 自动化测试沙盒
# ==========================================
if __name__ == "__main__":
    # 1. 初始化超级检索器
    retriever = HybridRerankRetriever()

    # 2. 扔一个带有高精度数字和方位的学术/工程典型问题
    test_query = "What is the conclusion of the paper?"

    # 3. 执行搜索，我们要最精准的前 3 名
    best_docs = retriever.search(test_query, top_k=5)

    # 4. 华丽展示结果
    print_and_log("\n" + "🏆" * 20 + " 终极精准检索结果展示 " + "🏆" * 20, "info")
    for i, doc in enumerate(best_docs):
        print_and_log(f"\n 【Top {i + 1}】 (Rerank 神经网络严审得分: {doc.metadata['rerank_score']:.4f})", "info")
        print_and_log(f"🏷 标签来源: {doc.metadata.get('source', '未知')}", "info")
        print_and_log(f" 纯净内容:\n{doc.page_content}", "info")
    print_and_log("\n" + "=====" * 13, "info")

