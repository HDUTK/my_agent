#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于ChromaDB 知识库运维管理面板
功能：
1. 查看知识库全局统计（总 Chunk 数、包含哪些文件、每个文件多少个切片）
2. 抽查/预览指定文件的文本切片内容
3. 精准按文件名删除（某篇文档废弃或内容更新时使用）
4. 物理级重置清空数据库
🌟 升级功能 (V2.1 多集合多粒度安全运维版):
5. 动态感知库内所有 Collection (如 tk_knowledge, agent_episodic_memory 等)，支持随时交互式切换工作集合；
6. 异构资产自适应分析：自动区分“文献切片型集合”与“情境任务经验卡片型集合”，自动切换对应的统计与预览面板；
7. 三级安全清空机制彻底解耦：
   - 级别 A: 清空当前集合的数据 (仅清空切片，保留集合骨架)；
   - 级别 B: 彻底注销当前集合 (连同集合本身从库中完全移除 Drop Collection)；
   - 级别 C: 全库物理格式化 (彻底清除所有集合与全部底层物理索引)。
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/10/14 21:13
version: V1.0
Target Python: 
"""

from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional

# 导入 ChromaDB 原生客户端用于多 Collection 动态管理与底层反射
import chromadb

# 保留原有向量构建模块的导入，维持底层逻辑与配置契约向后兼容
from RAG.vector_builder import get_chroma_db, clear_vector_database

# 尝试从系统集中配置中导入本地向量数据库根路径，若导入失败则使用工程默认路径兜底
try:
    from config.system_config import CHROMA_DB_DIR
except ImportError:
    # 路径兜底：指向项目根目录下的 db_storage/chroma_db
    CHROMA_DB_DIR = str(Path(__file__).resolve().parent.parent / "db_storage" / "chroma_db")


# =========================================================================
# 🌟 底层集合探测与客户端管理辅助函数
# =========================================================================
def get_chroma_client() -> chromadb.PersistentClient:
    """
    获取 ChromaDB 本地持久化客户端实例。
    - 确保目标持久化文件夹物理存在；
    - 具备跨集合列举与底层管理的权限。
    """
    db_path = Path(CHROMA_DB_DIR)
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def list_available_collections(client: chromadb.PersistentClient) -> List[str]:
    """
    动态扫描并获取当前 ChromaDB 中已存在的所有 Collection 名称列表。
    - 不写死任何集合名，未来新增的 Collection 自动在此可见；
    - 返回按字母排序后的名称列表。
    """
    raw_collections = client.list_collections()
    names = []
    for col in raw_collections:
        # 兼容不同版本 ChromaDB 对 Collection 对象的返回类型差异 (对象或字符串)
        if hasattr(col, "name"):
            names.append(col.name)
        elif isinstance(col, str):
            names.append(col)
    return sorted(names)


def get_target_collection(client: chromadb.PersistentClient, collection_name: str):
    """
    根据集合名称获取或自动创建 Collection 实例。
    - 当指定的集合不存在时，自动安全创建，防止抛出 CollectionNotFoundError。
    """
    return client.get_or_create_collection(name=collection_name)


# =========================================================================
# 1. 查验知识库资产分布 (保留原注释，支持指定 Collection 与异构数据)
# =========================================================================
def inspect_database(collection_name: str = "tk_knowledge"):
    """1. 查验知识库资产分布"""
    # 获取底层通用客户端与当前指定操作的 Collection
    client = get_chroma_client()
    vector_db = get_target_collection(client, collection_name)

    # 仅提取元数据与 ID
    data = vector_db.get(include=["metadatas"])
    ids = data.get("ids", [])
    metadatas = data.get("metadatas", [])

    total_chunks = len(ids)
    print("\n" + "=" * 55)
    print(f"📊 集合全局统计概览 [当前正在查看: {collection_name}]：")
    print(f"   - 总存储切片数 (Chunks) / 记录数: {total_chunks} 个")

    if total_chunks == 0:
        print("   - 目前库内为空，暂无数据。")
        print("=" * 55 + "\n")
        return

    # 探测集合的数据结构特征：检查第一条记录是否包含 "source" 字段
    sample_meta = metadatas[0] if metadatas else {}

    # -------------------------------------------------------------
    # 🌟 分支 A：文献知识库模式 (如 tk_knowledge，包含 source 字段)
    # -------------------------------------------------------------
    if "source" in sample_meta:
        # 统计每个文件贡献了多少切片
        sources = [m.get("source", "未知来源") for m in metadatas if m]
        source_counts = Counter(sources)

        print(f"   - 包含独立文件数: {len(source_counts)} 个")
        print("-" * 55)
        print(f"{'序号':<4} | {'切片数 (Chunks)':<14} | {'文件名'}")
        print("-" * 55)
        for idx, (source, count) in enumerate(source_counts.items(), 1):
            print(f"{idx:<4} | {count:<14} | {source}")

    # -------------------------------------------------------------
    # 🌟 分支 B：情境经验卡片模式 (如 agent_episodic_memory，包含 title / session_id)
    # -------------------------------------------------------------
    elif "title" in sample_meta or "session_id" in sample_meta:
        print(f"   - 集合属性: 🧠 科研任务情境经验卡片库 (Episodic Memory)")
        print(f"   - 包含独立经验记录: {len(ids)} 条")
        print("-" * 55)
        print(f"{'序号':<4} | {'任务主题 / 实验标题':<30} | {'涉及实体 (窟号/测点)'}")
        print("-" * 55)
        for idx, meta in enumerate(metadatas, 1):
            if not meta:
                continue
            title = meta.get("title", meta.get("session_id", "未命名会话"))[:28]
            caves = meta.get("target_caves", "-")
            sensors = meta.get("sensors", "-")
            entity_tag = f"窟:{caves} | 点:{sensors}"
            print(f"{idx:<4} | {title:<30} | {entity_tag}")

    # -------------------------------------------------------------
    # 🌟 分支 C：通用向量集合模式
    # -------------------------------------------------------------
    else:
        print(f"   - 集合属性: 🏷️ 通用键值向量数据表")
        print(f"   - 采样前 5 条 ID: {ids[:5]}")

    print("=" * 55 + "\n")


# =========================================================================
# 2. 抽查/预览指定文件的文本切片内容 (保留原注释，支持智能降级与模糊搜索)
# =========================================================================
def preview_file_chunks(source_filename: str, preview_count: int = 2, collection_name: str = "tk_knowledge"):
    """2. 抽查某文件的切片内容"""
    client = get_chroma_client()
    vector_db = get_target_collection(client, collection_name)

    # 首先探测该集合的元数据结构特征
    sample_data = vector_db.get(limit=1, include=["metadatas"])
    sample_meta = sample_data.get("metadatas", [{}])[0]

    # --- 针对文献切片库（包含 source 属性）：按文件名严格或模糊匹配 ---
    if "source" in sample_meta or not sample_meta:
        results = vector_db.get(
            where={"source": source_filename},
            include=["documents", "metadatas"]
        )

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        # 🌟 增强容错：如果用户输入的只是部分文件名，执行模糊打捞
        if not docs:
            all_data = vector_db.get(include=["documents", "metadatas"])
            matched_docs = []
            matched_metas = []
            for d, m in zip(all_data.get("documents", []), all_data.get("metadatas", [])):
                if m and source_filename.lower() in m.get("source", "").lower():
                    matched_docs.append(d)
                    matched_metas.append(m)
            docs = matched_docs
            metas = matched_metas

        if not docs:
            print(f"⚠️ 未在当前集合 [{collection_name}] 中检索到文件名为 [{source_filename}] 的切片！")
            return

        print(f"\n🔍 文件 [{source_filename}] 共包含 {len(docs)} 个切片。展示前 {min(preview_count, len(docs))} 个：\n")
        for i in range(min(preview_count, len(docs))):
            print(f"--- [切片 #{i + 1}] ---")
            print(f"🏷️ 元数据: {metas[i]}")
            print(f"📝 正文预览:\n{docs[i][:300]}...")
            print("-" * 40)

    # --- 针对情境经验卡片库：按任务标题、session_id 或语义关键词匹配 ---
    else:
        # 优先使用余弦语义相似度进行小样本召回预览
        query_res = vector_db.query(
            query_texts=[source_filename],
            n_results=preview_count,
            include=["documents", "metadatas"]
        )
        docs = query_res.get("documents", [[]])[0]
        metas = query_res.get("metadatas", [[]])[0]

        if not docs:
            print(f"⚠️ 在经验库 [{collection_name}] 中未找到与 [{source_filename}] 相关的任务卡片！")
            return

        print(f"\n🔍 命中经验卡片 {len(docs)} 张，展示如下：\n")
        for i in range(len(docs)):
            print(f"--- [经验卡片 #{i + 1}] ---")
            print(f"🏷️ 元数据 (Metadata):")
            for k, v in (metas[i] or {}).items():
                print(f"   • {k}: {v}")
            print(f"📝 卡片密集摘要/正文:\n{docs[i]}")
            print("-" * 40)


# =========================================================================
# 3. 精准按文件名删除（某篇文档废弃或内容更新时使用）(保留原注释)
# =========================================================================
def delete_file_from_db(source_filename: str, collection_name: str = "tk_knowledge"):
    """3. 精准按文件名删除该文件的所有切片"""
    client = get_chroma_client()
    vector_db = get_target_collection(client, collection_name)

    sample_data = vector_db.get(limit=1, include=["metadatas"])
    sample_meta = sample_data.get("metadatas", [{}])[0]

    # --- 模式 A: 针对文献库，按 source 元数据字段匹配删除 ---
    if "source" in sample_meta:
        # 先查出该文件对应的所有唯一 ID
        results = vector_db.get(
            where={"source": source_filename},
            include=["metadatas"]
        )
        ids_to_delete = results.get("ids", [])

        if not ids_to_delete:
            print(f"⚠️ 在当前集合 [{collection_name}] 中找不到名为 [{source_filename}] 的记录，无需删除。")
            return

        confirm = input(f"🚨 警告: 确认从集合 [{collection_name}] 中删除 [{source_filename}] 的全部 {len(ids_to_delete)} 个切片吗？(y/n): ")
        if confirm.lower() == 'y':
            # 通过 ID 列表执行精准删除
            vector_db.delete(ids=ids_to_delete)
            print(f"✅ 成功从 [{collection_name}] 移除 [{source_filename}] 及其所有向量切片！")
        else:
            print("已取消删除操作。")

    # --- 模式 B: 针对经验库或其他集合，按 ID 或 session_id 删除 ---
    else:
        res_by_id = vector_db.get(ids=[source_filename])
        if res_by_id.get("ids"):
            ids_to_delete = res_by_id.get("ids")
        else:
            res_by_session = vector_db.get(where={"session_id": source_filename})
            ids_to_delete = res_by_session.get("ids", [])

        if not ids_to_delete:
            print(f"⚠️ 在当前集合 [{collection_name}] 中未找到唯一标识为 [{source_filename}] 的记录。")
            return

        confirm = input(f"🚨 警告: 确认从集合 [{collection_name}] 中删除记录 [{source_filename}] 吗？(y/n): ")
        if confirm.lower() == 'y':
            vector_db.delete(ids=ids_to_delete)
            print(f"✅ 成功从 [{collection_name}] 移除目标记录！")
        else:
            print("已取消删除操作。")


# =========================================================================
# 🌟 4. 清空当前集合的数据 (级别 1：数据清零，保留集合名)
# =========================================================================
def clear_single_collection_data(client: chromadb.PersistentClient, collection_name: str):
    """
    【粒度 1：清空当前集合数据】
    - 只把当前 Collection 内的向量与切片数据清零；
    - 集合名称依然保留在数据库中，后续可直接继续写入；
    - 库中其他 Collection 完全不受影响。
    """
    print(f"\n⚠️ 【操作范围提示】: 您即将清空集合 【 {collection_name} 】 内部的数据。")
    print(f"   该操作只会清空该集合的切片内容，集合名称仍会保留，其他集合的数据不受任何影响。")
    confirm = input(f"🚨 确认将集合 [{collection_name}] 的数据清零吗？(输入 yes 确认): ").strip()

    if confirm.lower() == "yes":
        try:
            # 删除后立即原地重建同名集合，实现数据完全清空但保留集合骨架
            client.delete_collection(name=collection_name)
            client.get_or_create_collection(name=collection_name)
            print(f"✅ 集合 [{collection_name}] 内部数据已全部清空！(集合已重置为空)")
        except Exception as e:
            print(f"❌ 清空集合 [{collection_name}] 失败: {e}")
    else:
        print("已取消清空操作。")


# =========================================================================
# 🌟 5. 彻底注销/删除当前集合 (级别 2：连同集合本身一起删除 Drop)
# =========================================================================
def drop_single_collection(client: chromadb.PersistentClient, collection_name: str) -> str:
    """
    【粒度 2：彻底注销/删除当前集合】
    - 将指定的 Collection 连同名称、元数据以及所有数据从 ChromaDB 中完全抹除 (Drop)；
    - 执行后在集合列表中将不再看到它；
    - 删除后会自动帮用户平滑切换到其他现存集合；若已无其他集合，则自动初始化默认的 tk_knowledge。
    """
    print(f"\n🚨 【极度危险操作提示】: 您即将从数据库中【彻底注销/删除】集合 【 {collection_name} 】！")
    print(f"   该集合及其名称将被完全拔除，列表里将不再有它！")
    confirm = input(f"确认永久注销集合 [{collection_name}] 吗？(请输入该集合全名以确认): ").strip()

    if confirm == collection_name:
        try:
            client.delete_collection(name=collection_name)
            print(f"💥 集合 [{collection_name}] 已从 ChromaDB 中彻底注销删除！")

            # 重新扫描现存集合并自动切换活动上下文
            remaining = list_available_collections(client)
            if remaining:
                new_col = remaining[0]
                print(f"🔄 自动切换当前工作集合至: 【 {new_col} 】")
                return new_col
            else:
                print("⚠️ 库中已无任何集合，自动为您初始化默认集合 [tk_knowledge]...")
                client.get_or_create_collection("tk_knowledge")
                return "tk_knowledge"
        except Exception as e:
            print(f"❌ 注销集合 [{collection_name}] 遇到异常: {e}")
            return collection_name
    else:
        print("输入的集合名称不匹配，已取消注销删除操作。")
        return collection_name


# =========================================================================
# 🌟 集合切换工具
# =========================================================================
def select_working_collection(client: chromadb.PersistentClient, current_name: str) -> str:
    """交互式切换工作集合"""
    col_names = list_available_collections(client)

    if not col_names:
        print("⚠️ 数据库内目前尚无任何集合，已自动为您创建默认集合 [tk_knowledge]。")
        client.get_or_create_collection("tk_knowledge")
        return "tk_knowledge"

    print("\n" + "─" * 45)
    print("🎯 请选择要切换的目标 Collection：")
    print("─" * 45)
    for idx, name in enumerate(col_names, 1):
        active_mark = "  ⭐ (当前处于操作状态)" if name == current_name else ""
        print(f"  [{idx}] {name}{active_mark}")
    print("─" * 45)

    while True:
        choice = input(f"请输入集合对应序号 (1-{len(col_names)}) 或直接按回车保持当前: ").strip()
        if not choice:
            return current_name
        if choice.isdigit() and 1 <= int(choice) <= len(col_names):
            chosen_name = col_names[int(choice) - 1]
            print(f"✅ 已成功切换工作上下文至 -> 【 {chosen_name} 】")
            return chosen_name
        print("❌ 输入序号无效，请重新输入正确的数字。")


# =========================================================================
# 主交互控制台 (清晰标明操作范围与级别)
# =========================================================================
def main_menu():
    client = get_chroma_client()
    existing_cols = list_available_collections(client)

    # 启动时智能初始化默认活动集合：优先选择 tk_knowledge
    if "tk_knowledge" in existing_cols:
        active_col = "tk_knowledge"
    elif existing_cols:
        active_col = existing_cols[0]
    else:
        client.get_or_create_collection("tk_knowledge")
        active_col = "tk_knowledge"

    while True:
        print("\n" + "🛠️" * 10 + " ChromaDB 运维管理面板 " + "🛠️" * 10)
        print(f"📌 当前操作集合: 【 {active_col} 】")
        print("─" * 55)
        print("【针对当前集合的操作】")
        print("  1. 查看当前集合文件与切片统计")
        print("  2. 抽查当前集合的切片/卡片内容")
        print("  3. 按文件名/标识精准删除指定记录")
        print("  4. 🧹 清空当前集合的数据 (数据清零，但保留集合结构)")
        print("  5. 🗑️ 彻底注销/删除当前集合 (连同集合名一并注销 Drop)")
        print("─" * 55)
        print("【跨集合与全库操作】")
        print("  6. 🔄 切换目标 Collection (在知识库与经验库间切换)")
        print("  7. 🌐 查看全库所有 Collection 规模总览对比")
        print("  8. 💥 全库物理级格式化 (彻底删除所有集合及全部物理文件)")
        print("  0. 🚪 退出面板")
        print("═" * 55)

        choice = input("请选择操作功能 (0-8): ").strip()

        # 1. 查看知识库文件与切片统计
        if choice == "1":
            inspect_database(collection_name=active_col)

        # 2. 抽查/预览指定文件的切片内容
        elif choice == "2":
            file_name = input(f"请输入要抽查的文件名或卡片标识 (所属集合: {active_col}): ").strip()
            if file_name:
                preview_file_chunks(source_filename=file_name, collection_name=active_col)
            else:
                print("文件名不可为空。")

        # 3. 按文件名精准删除数据
        elif choice == "3":
            file_name = input(f"请输入要删除的文件名或记录标识 (所属集合: {active_col}): ").strip()
            if file_name:
                delete_file_from_db(source_filename=file_name, collection_name=active_col)
            else:
                print("输入不可为空。")

        # 🌟 4. 粒度 1：清空当前集合的数据 (保留空集合)
        elif choice == "4":
            clear_single_collection_data(client, collection_name=active_col)

        # 🌟 5. 粒度 2：彻底注销/删除当前集合 (连根拔起)
        elif choice == "5":
            active_col = drop_single_collection(client, collection_name=active_col)

        # 6. 切换目标 Collection
        elif choice == "6":
            active_col = select_working_collection(client, current_name=active_col)

        # 7. 查看全库所有 Collection 总览对比
        elif choice == "7":
            names = list_available_collections(client)
            print("\n" + "═" * 55)
            print("🌐 全库各 Collection 规模概览：")
            print("─" * 55)
            for idx, n in enumerate(names, 1):
                c = client.get_collection(n).count()
                tag = " (当前操作中)" if n == active_col else ""
                print(f"  [{idx}] {n:<26} : {c} 条记录{tag}")
            print("═" * 55 + "\n")

        # 🌟 8. 粒度 3：物理级重置清空数据库 (彻底删除所有集合及物理文件)
        elif choice == "8":
            print("\n💥 【全库最高危险级别警示】！")
            print("   此操作将调用 clear_vector_database() 彻底销毁全部物理索引，清空库内【所有】集合！")
            confirm = input("确认物理清空数据库所有集合与内容？(请输入 'DELETE ALL' 确认): ").strip()
            if confirm == "DELETE ALL":
                clear_vector_database()  # 调用底层原有物理级清空
                print("💥 全库所有数据与底层文件已彻底清除！")
                active_col = select_working_collection(client, current_name="tk_knowledge")
            else:
                print("已取消全库物理清空操作。")

        # 0. 退出
        elif choice == "0":
            print("👋 运维管理工具已安全退出。")
            break
        else:
            print("❌ 输入有误，请输入 0 到 8 之间的数字选项。")


if __name__ == "__main__":
    main_menu()