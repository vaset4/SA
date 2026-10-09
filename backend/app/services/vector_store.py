"""
本地 FAISS 向量存储服务。

这里实现的是项目早期/兼容保留的本地向量检索层，不是当前主 RAG 路径默认使用的
向量后端。当前在线主检索链已经切到 `milvus_store.py`，但本文件仍保留完整的
本地 FAISS + JSON 落盘实现，便于兼容旧调用、离线调试、以及部分历史路径复用。

它不是传统意义上的业务数据库，职责更准确地说是：

  - 把论文内容转换成可检索的向量与倒排式元数据
  - 在本地磁盘维护 FAISS + JSON 混合索引
  - 对外提供“搜索预览”和“按块阅读全文”两种能力

与 `paper_db` 的分工如下：
  - `paper_db`：保存论文元数据、分析结果、笔记等业务事实
  - `vector_store`：保存论文内容如何被定位、召回、展开阅读

落盘结构：
  - `index.faiss`：句子级向量索引，每条向量对应一个句子
  - `id_map.json`：FAISS 内部整数 id -> sentence_id
  - `metadata.json`：sentence_id -> 命中句所在 chunk/parent/paper 等定位信息
  - `chunks.json`：chunk_id -> 父块全文与页码等阅读信息

因此，这里不是直接把整篇论文塞进向量库，而是把同一篇论文拆成三层：
  `paper_id` -> `parent_id` -> `chunk_id` -> `sentence_id`

其中：
  - `paper_id` 连接业务库与向量库
  - `parent_id` 代表可一次阅读的大块上下文
  - `chunk_id` 代表可检索/可预览的子块
  - `sentence_id` 是 FAISS 实际索引粒度

当前架构关系：
  - `paper_db`：业务事实与元数据
  - `milvus_store`：当前默认主检索实现
  - `vector_store`：本地 FAISS 兼容实现，仍提供相同风格的搜索/阅读接口
"""

# 导入 JSON，用于序列化 id_map、metadata、chunks。
import json
# 保留 pickle 导入，兼容历史实现痕迹；当前文件未直接使用。
import pickle
# 导入正则工具，用于句子切分。
import re
# 导入线程锁与后台线程，保证索引加载和重建的并发安全。
import threading
# 保留 time 导入，兼容历史实现痕迹；当前文件未直接使用。
import time
# 导入 Path，用于拼接和创建向量库目录。
from pathlib import Path
# 导入类型标注，明确输入输出结构。
from typing import Any, Dict, List, Optional, Set

try:
    # 尝试导入 numpy，用于向量矩阵拼接与类型转换。
    import numpy as np
    # 导入成功时标记可用。
    HAS_NUMPY = True
except ImportError:
    # 缺失依赖时显式标记不可用。
    HAS_NUMPY = False

try:
    # 尝试导入 faiss，负责相似度索引与搜索。
    import faiss
    # 导入成功时标记可用。
    HAS_FAISS = True
except ImportError:
    # 缺失依赖时显式标记不可用。
    HAS_FAISS = False

# 导入 Chunk 类型定义，约束 add_paper 的输入数据结构。
from app.models.models import Chunk
# 导入配置读取函数。
from app.config import get_embedding_identity, get_settings
from app.models.research_contracts import RetrievalScopeError
from app.services.embedding_client import get_embedding_client
# 导入业务数据库服务，用于标记论文已向量化。
import app.services.paper_db as paper_db

# 读取项目全局配置。
settings = get_settings()

# 当前架构下，如果你看到 `vector_store.py`，应把它理解为：
#   - “旧/本地兼容向量库”
#   - 对外接口尽量模拟主链需要的 search/read 形状
#   - 方便历史代码不必一次性全部切到 Milvus


# ============================================================
# 存储路径
# ============================================================

def _store_dir() -> Path:
    """返回当前向量身份的 FAISS 目录；无参数，local 保留历史目录。"""
    p = Path(settings.data_dir) / 'vector_store'
    # 1. 【检索】【向量隔离】API 向量不能读取同维度的本地或其他服务索引
    if settings.embedding_mode == "api":
        p = p / get_embedding_identity(settings)
    # 目录不存在时自动创建。
    p.mkdir(parents=True, exist_ok=True)
    # 返回最终路径对象。
    return p


def _index_path():
    # 返回 FAISS 索引文件路径。
    return str(_store_dir() / 'index.faiss')


def _id_map_path():
    # 返回 FAISS id 到 sentence_id 映射文件路径。
    return str(_store_dir() / 'id_map.json')


def _metadata_path():
    # 返回 sentence 元数据文件路径。
    return str(_store_dir() / 'metadata.json')


def _chunks_path():
    # 返回 chunk 父块全文文件路径。
    return str(_store_dir() / 'chunks.json')


# ============================================================
# VectorStore 类
# ============================================================

class VectorStore:
    """
    句子级本地向量存储。

    索引结构：
      index.faiss   每条向量对应一个句子（非 chunk）
      id_map.json   faiss 内部 int id → sentence_id
      metadata.json sentence_id → {chunk_id, parent_id, sentence, paper_id, section, page_num}
      chunks.json   chunk_id → {full_text, parent_id, paper_id, section, page_num}

    这个类实现的是本地 FAISS 版本的统一检索接口。
    当前主链虽然默认改用 Milvus，但这里的方法命名和返回结构仍保持与上层
    Agent/RAG 工具链兼容，方便历史代码平滑切换。
    """

    def __init__(self):
        """初始化 FAISS 内存状态；无参数，模型编码由共享客户端负责。"""
        # 锁用于保护加载、写盘、重建索引等共享状态操作。
        self._lock = threading.Lock()
        # FAISS 索引采用延迟加载，初始为空。
        self._index: Optional['faiss.Index'] = None
        # id_map 缓存在内存中，键是 faiss 内部整数 id。
        self._id_map: Dict[int, str] = {}
        # metadata 缓存在内存中，键是 sentence_id。
        self._metadata: Dict[str, dict] = {}
        # chunks 缓存在内存中，键是 chunk_id，对应父块全文信息。
        self._chunks: Dict[str, dict] = {}
        # _loaded 表示磁盘索引是否已经恢复到内存。
        self._loaded = False

    # ── 延迟加载 ──

    def _ensure_loaded(self):
        """懒加载索引与元数据，避免服务启动时强行加载大模型/大索引。"""
        # 已加载完成时直接返回。
        if self._loaded:
            return
        # 首次加载过程加锁，防止并发重复加载。
        with self._lock:
            # 双重检查，避免锁等待后重复执行加载。
            if self._loaded:
                return
            # 真正从磁盘恢复索引与元数据。
            self._load_from_disk()
            # 标记加载完成。
            self._loaded = True

    def _load_from_disk(self):
        """
        从磁盘恢复 FAISS 与 JSON 辅助索引。

        这一步只恢复“长期记忆”状态，不依赖 LangGraph checkpoint。
        因此即使执行态会话丢失，只要这些文件还在，知识库仍然可检索。

        这里恢复的是本地磁盘索引，不会读取 Milvus，也不参与 PostgreSQL checkpoint。
        """
        # 缺少核心依赖时直接返回，让上层决定是否报错或降级。
        if not HAS_NUMPY or not HAS_FAISS:
            return
        # 获取默认 embedding 维度，用于初始化空索引。
        dim = self._embedding_dim()
        # 读取索引文件路径。
        index_p = _index_path()
        # 磁盘上已有索引时直接加载。
        if Path(index_p).exists():
            self._index = faiss.read_index(index_p)
        else:
            # 没有索引文件时创建一个空的内积索引。
            self._index = faiss.IndexFlatIP(dim)
        # 读取 id_map 文件路径。
        id_map_p = _id_map_path()
        # id_map 文件存在时加载并把 key 转回 int。
        if Path(id_map_p).exists():
            raw = json.loads(Path(id_map_p).read_text(encoding='utf-8'))
            self._id_map = {int(k): v for k, v in raw.items()}
        # 读取 metadata 文件路径。
        meta_p = _metadata_path()
        # metadata 文件存在时直接加载。
        if Path(meta_p).exists():
            self._metadata = json.loads(Path(meta_p).read_text(encoding='utf-8'))
        # 读取 chunks 文件路径。
        chunks_p = _chunks_path()
        # chunks 文件存在时直接加载。
        if Path(chunks_p).exists():
            self._chunks = json.loads(Path(chunks_p).read_text(encoding='utf-8'))

    def _save_to_disk(self):
        """
        将当前内存索引完整落盘，保证重启后可恢复。

        这里选择“FAISS + JSON 辅助文件”的组合，是因为 FAISS 只擅长向量索引，
        而 chunk/full_text/定位信息仍需要独立元数据文件承载。
        """
        # 把当前 FAISS 索引写回磁盘。
        faiss.write_index(self._index, _index_path())
        # 把 id_map 序列化成 JSON，注意 key 需要转成字符串。
        Path(_id_map_path()).write_text(
            json.dumps({str(k): v for k, v in self._id_map.items()}, ensure_ascii=False),
            encoding='utf-8'
        )
        # 把 sentence 元数据写回磁盘。
        Path(_metadata_path()).write_text(
            json.dumps(self._metadata, ensure_ascii=False),
            encoding='utf-8'
        )
        # 把 chunk 父块全文映射写回磁盘。
        Path(_chunks_path()).write_text(
            json.dumps(self._chunks, ensure_ascii=False),
            encoding='utf-8'
        )

    # ── Embedding 模型 ──

    def _embedding_dim(self) -> int:
        """返回配置的向量维度，用于初始化 FAISS 空索引；无参数。"""
        return int(settings.embedding_dimension)

    def _encode(self, texts: List[str]) -> 'np.ndarray':
        """接收有序文本，复用共享编码端口并返回 FAISS 所需的二维 float32 矩阵。"""
        if not texts:
            return np.empty((0, self._embedding_dim()), dtype=np.float32)
        return np.asarray(get_embedding_client().embed_texts(texts), dtype=np.float32)

    # ── 句子切分 ──

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        将子块文本继续切成句子。

        这里的目标不是生成展示层语义单元，而是把 FAISS 的索引粒度
        再压低一级，让召回更精准。真正阅读时仍回到父块全文。
        """
        # 基于简单标点规则拆句。
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        # 过滤太短的片段，减少低质量向量。
        return [p.strip() for p in parts if len(p.strip()) > 10]

    # ============================================================
    # 写入
    # ============================================================

    def add_paper(self, paper_id: str, chunks: List[Chunk]) -> None:
        """
        将一篇论文的所有 Chunk 写入本地向量记忆。

        步骤：
          1. 遍历统一文档解析器生成的子块列表
          2. 对每个子块的 `text` 再切句，得到 sentence_id 粒度索引
          3. 把句向量写进 FAISS，把定位信息写进 metadata/id_map
          4. 把父块全文写进 chunks.json，供 `read_chunk()` 使用
          5. 落盘后在 `paper_db` 标记 `is_vectorized`

        这里最容易混淆的一点是：
        - FAISS 里存的是“句子”
        - 搜索结果展示的是“子块预览”
        - 真正展开阅读的是“父块全文”

        需要注意：这只会写入本地 FAISS 索引，不会同步写入 Milvus。
        当前生产主路径若需要在线检索能力，通常由 `milvus_store.add_paper()` 负责。
        """
        # 缺依赖时直接抛错，明确暴露问题根因。
        if not HAS_NUMPY or not HAS_FAISS:
            raise RuntimeError(
                "faiss-cpu / numpy 未安装，无法使用向量存储"
            )
        # 保证磁盘索引与缓存已经恢复。
        self._ensure_loaded()
        # 保存所有待写入的句子文本。
        sentences: List[str] = []
        # 保存与每个句子一一对应的元数据。
        sentence_metas: List[dict] = []
        # 遍历一篇论文拆出的所有 chunk。
        for chunk in chunks:
            # 先用子块文本切句，作为向量检索粒度。
            sents = self._split_sentences(chunk['text'])
            # 没切出句子时，用截断文本兜底，避免整块不可检索。
            if not sents:
                sents = [chunk['text'][:200]]
            # 逐句写入 sentence 列表与对应元数据。
            for sent in sents:
                # sentence_id 全局唯一，由 chunk_id 与当前全局顺序拼出。
                sent_id = f"{chunk['chunk_id']}__s{len(sentences)}"
                # 记录句子原文。
                sentences.append(sent)
                # 记录句子的定位元数据。
                sentence_metas.append({
                    'sentence_id': sent_id,
                    'chunk_id': chunk['chunk_id'],
                    'parent_id': chunk['parent_id'],
                    'sentence': sent,
                    'paper_id': paper_id,
                    'section': chunk['section'],
                    'page_num': chunk['page_num'],
                })
            # 把父块全文缓存到 chunks 映射，供 read_chunk 直接展开。
            self._chunks[chunk['chunk_id']] = {
                'full_text': chunk['full_text'],
                'parent_id': chunk['parent_id'],
                'paper_id': paper_id,
                'section': chunk['section'],
                'page_num': chunk['page_num'],
            }
        # 没有任何句子时直接返回，不写入空索引。
        if not sentences:
            return
        # 批量编码全部句子。
        embs = self._encode(sentences)
        # 加锁保护对索引与映射表的写入。
        with self._lock:
            # 记录新增向量开始前的 ntotal，后续用于计算 faiss_id。
            start_id = self._index.ntotal
            # 把向量追加进 FAISS 索引。
            self._index.add(embs.astype('float32'))
            # 遍历每个句子的元数据，建立 faiss_id 到 sentence_id 的映射。
            for i, meta in enumerate(sentence_metas):
                # 计算当前句子的 FAISS 内部 id。
                faiss_id = start_id + i
                # 写入 id_map。
                self._id_map[faiss_id] = meta['sentence_id']
                # 写入 sentence 元数据表。
                self._metadata[meta['sentence_id']] = meta
            # 所有内存结构更新完成后统一落盘。
            self._save_to_disk()
        # 通知业务库这篇论文已经具备本地向量检索能力。
        paper_db.mark_vectorized(paper_id)

    def exists(self, paper_id: str) -> bool:
        """判断某篇论文是否已经存在于本地 FAISS 兼容索引中。"""
        # 确保索引与缓存已加载。
        self._ensure_loaded()
        # 只要 chunks 映射里存在对应 paper_id，就认为向量记忆已存在。
        return any(
            v.get('paper_id') == paper_id for v in self._chunks.values()
        )

    def delete_paper(self, paper_id: str) -> None:
        """
        删除一篇论文对应的本地向量记忆，并在后台重建索引。

        之所以不直接在原 FAISS 索引里做精确删除，是因为当前实现采用
        `IndexFlatIP`，更简单可靠的做法是：
          1. 先从 metadata / chunks / id_map 的内存视图中移除目标论文
          2. 启动后台线程，按剩余句子重新编码并重建 FAISS

        这样删除路径更容易保持一致性，代价是删除后会有一次异步重建成本。
        """
        # 确保当前索引和缓存已经加载。
        self._ensure_loaded()
        # 先找出属于目标论文的全部 chunk_id。
        chunk_ids = {
            cid for cid, v in self._chunks.items() if v.get('paper_id') == paper_id
        }
        # 再找出属于目标论文的全部 sentence_id。
        sent_ids_to_remove = {
            sid for sid, meta in self._metadata.items()
            if meta.get('paper_id') == paper_id
        }
        # 加锁，保证内存结构删除的一致性。
        with self._lock:
            # 删除属于该论文的 sentence 元数据。
            for sid in sent_ids_to_remove:
                self._metadata.pop(sid, None)
            # 删除属于该论文的 chunk 全文缓存。
            for cid in chunk_ids:
                self._chunks.pop(cid, None)
            # 重建 id_map，去掉所有被删除句子的映射。
            self._id_map = {
                k: v for k, v in self._id_map.items()
                if v not in sent_ids_to_remove
            }
        # 异步启动后台重建，避免阻塞当前调用线程。
        threading.Thread(target=self._rebuild_index, daemon=True).start()

    def _rebuild_index(self):
        """
        后台重建本地 FAISS 索引。

        删除论文后，JSON 元数据已经先行删除；这里再根据剩余 sentence_id
        重新生成一个干净的 FAISS 索引，保证：
          - FAISS 内部 id 连续
          - id_map 与 metadata 一一对应
          - 磁盘文件保持一致
        """
        # 加锁，避免与其他写入操作并发冲突。
        with self._lock:
            # 读取当前剩余的全部 sentence_id。
            remaining_sids = list(self._metadata.keys())
            # 没有剩余句子时，直接重建为空索引并清空映射。
            if not remaining_sids:
                dim = self._embedding_dim()
                self._index = faiss.IndexFlatIP(dim)
                self._id_map = {}
                self._save_to_disk()
                return
            # 按剩余 sentence_id 顺序取回原句子文本。
            sentences = [self._metadata[sid]['sentence'] for sid in remaining_sids]
            # 重新编码剩余句子。
            embs = self._encode(sentences)
            # 从编码结果读取真实维度。
            dim = embs.shape[1]
            # 创建新的空内积索引。
            new_index = faiss.IndexFlatIP(dim)
            # 把全部剩余向量写入新索引。
            new_index.add(embs.astype('float32'))
            # 按新顺序构造连续的 id_map。
            new_id_map = {i: sid for i, sid in enumerate(remaining_sids)}
            # 用新索引替换旧索引。
            self._index = new_index
            # 用新映射替换旧映射。
            self._id_map = new_id_map
            # 将重建后的状态写回磁盘。
            self._save_to_disk()

    # ============================================================
    # 工具函数一：语义搜索（返回预览片段）
    # 来自 arag semantic_search + agentic-rag-for-dummies search_child_chunks
    # ============================================================

    def local_rag_search(
        self,
        query: str,
        top_k: int = None,
        filter_paper_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        在本地 FAISS 索引上执行语义搜索，返回 top_k 个子块预览（非全文）。

        索引粒度是句子，聚合到 chunk 后取最高句子得分作为 chunk 得分（来自 arag）。
        Preview 故意截断（约200字符），强制 Agent 通过 read_chunk 取全文。

        典型调用链：
          用户问题 -> query embedding -> 句子召回 -> 聚合成 chunk 结果
          -> Agent 决定是否调用 `read_chunk()` 展开全文

        当前它更适合：
          - 本地/测试环境下的兼容检索
          - 仍显式依赖 `VectorStore` 的历史调用
        不应误认为这是当前默认的线上主检索实现。

        Returns:
            [{chunk_id, parent_id, preview, paper_id, section, page_num, score}]
        """
        scoped_paper_ids = {
            str(item or "").strip()
            for item in filter_paper_ids or []
            if str(item or "").strip()
        }
        if not scoped_paper_ids:
            raise RetrievalScopeError()
        # 缺依赖时无法执行语义检索，直接返回空结果。
        if not HAS_NUMPY or not HAS_FAISS:
            return []
        # 确保索引和缓存已恢复。
        self._ensure_loaded()
        # 空索引时直接返回空结果。
        if self._index.ntotal == 0:
            return []
        # 如果未显式给 top_k，则使用配置默认值。
        top_k = top_k or settings.rag_top_k
        # 搜索时先多取一些句子结果，便于后续聚合成 chunk。
        k_search = min(top_k * 10, self._index.ntotal)
        # 对查询文本编码成单条向量。
        query_emb = self._encode([query]).astype('float32')
        # 在 FAISS 中执行相似度搜索。
        scores, faiss_ids = self._index.search(query_emb, k_search)
        # 取出第一条查询对应的分数数组。
        scores = scores[0]
        # 取出第一条查询对应的 id 数组。
        faiss_ids = faiss_ids[0]
        # 用 chunk_id 聚合多个句子命中。
        chunk_best: Dict[str, dict] = {}
        # 逐条遍历搜索返回的句子命中。
        for score, fid in zip(scores, faiss_ids):
            # fid 为负时表示无效命中，直接跳过。
            if fid < 0:
                continue
            # 根据 faiss_id 找到 sentence_id。
            sid = self._id_map.get(int(fid))
            # 映射不存在时跳过该结果。
            if not sid:
                continue
            # 根据 sentence_id 读取元数据。
            meta = self._metadata.get(sid)
            # 元数据丢失时跳过该结果。
            if not meta:
                continue
            # 指定了 paper_id 过滤时，不在白名单里的结果直接跳过。
            if meta['paper_id'] not in scoped_paper_ids:
                continue
            # 读取当前句子所属 chunk_id。
            cid = meta['chunk_id']
            # 同一 chunk 命中多个句子时，只保留最高分的一条。
            if cid not in chunk_best or score > chunk_best[cid]['score']:
                chunk_best[cid] = {
                    'chunk_id': cid,
                    'parent_id': meta['parent_id'],
                    'preview': self._make_preview(meta['sentence'], meta),
                    'paper_id': meta['paper_id'],
                    'section': meta['section'],
                    'page_num': meta['page_num'],
                    'score': float(score),
                }
        # 按相似度从高到低排序 chunk 结果。
        results = sorted(chunk_best.values(), key=lambda x: x['score'], reverse=True)
        # 仅返回前 top_k 条结果。
        return results[:top_k]

    def _make_preview(self, hit_sentence: str, meta: dict) -> str:
        """生成命中句子的展示预览，故意不泄露整段全文。"""
        # 去掉命中句首尾空白。
        text = hit_sentence.strip()
        # 过长文本截断为 200 字并追加省略号。
        if len(text) > 200:
            text = text[:200] + '...'
        # 在预览前缀里附带节标题和页码，方便回答阶段引用。
        return f'[{meta["section"]} p.{meta["page_num"]}] {text}'

    # ============================================================
    # 工具函数二：精确关键词搜索（返回预览片段）
    # 来自 arag keyword_search
    # ============================================================

    def keyword_search(
        self,
        keywords: List[str],
        top_k: int = None,
    ) -> List[dict]:
        """
        在本地全文缓存上做精确关键词匹配，适合缩写、术语、人名等 embedding 不稳定的查询。

        Score = Σ count(keyword, full_text) × len(keyword)（来自 arag）

        该函数与 `local_rag_search()` 互补：
        - `local_rag_search()` 擅长语义相近表达
        - `keyword_search()` 擅长必须原词命中的专有名词
        - 两者最终都只返回 preview，真正全文仍由 `read_chunk()` 提供

        Returns:
            [{chunk_id, parent_id, preview, paper_id, section, page_num, score}]
        """
        # 确保缓存与索引已加载。
        self._ensure_loaded()
        # 未指定 top_k 时使用默认配置。
        top_k = top_k or settings.rag_top_k
        # 没关键词或没任何 chunk 时，直接返回空结果。
        if not keywords or not self._chunks:
            return []
        # 统一转成小写，方便做不区分大小写的匹配。
        keywords_lower = [kw.lower() for kw in keywords]
        # 用于收集全部关键词搜索结果。
        results = []
        # 遍历每个 chunk 的父块全文。
        for chunk_id, data in self._chunks.items():
            # 全文也转成小写，便于后续 count 和 find。
            full_text_lower = data['full_text'].lower()
            # 根据关键词出现次数与长度计算启发式分数。
            score = sum(
                full_text_lower.count(kw) * len(kw)
                for kw in keywords_lower
            )
            # 没有任何命中时直接跳过当前 chunk。
            if score == 0:
                continue
            # 初始化第一个命中位置为未找到。
            first_pos = -1
            # 遍历所有关键词，寻找最早命中的位置。
            for kw in keywords_lower:
                pos = full_text_lower.find(kw)
                if pos >= 0 and (first_pos < 0 or pos < first_pos):
                    first_pos = pos
            # 找到命中位置时围绕命中位置截取局部片段。
            if first_pos >= 0:
                start = max(0, first_pos - 50)
                snippet = data['full_text'][start:start + 200]
                if start > 0:
                    snippet = '...' + snippet
                if start + 200 < len(data['full_text']):
                    snippet = snippet + '...'
            else:
                # 理论上不会走到这里，但仍提供截断全文作为兜底预览。
                snippet = data['full_text'][:200] + '...'
            # 把当前 chunk 结果加入结果集。
            results.append({
                'chunk_id': chunk_id,
                'parent_id': data['parent_id'],
                'preview': f'[{data["section"]} p.{data["page_num"]}] {snippet}',
                'paper_id': data['paper_id'],
                'section': data['section'],
                'page_num': data['page_num'],
                'score': float(score),
            })
        # 按分数从高到低排序。
        results.sort(key=lambda x: x['score'], reverse=True)
        # 返回前 top_k 条高分结果。
        # 返回结构故意和 `local_rag_search()` 保持一致，方便上层统一处理。
        return results[:top_k]

    # ============================================================
    # 工具函数三：按 chunk_id 读取完整父块
    # 来自 arag read_chunk（含 already_read 去重）
    # ============================================================

    def read_chunk(self, chunk_id: str, already_read: Set[str]) -> dict:
        """
        返回 chunk_id 对应的完整父块文本。

        already_read 去重（来自 arag context.read_chunk_ids）：
          已读过的 chunk 返回占位符，不重复消耗 token。

        这里读取的是本地 `chunks.json` 中缓存的父块全文，
        不会回查 Milvus，也不会访问业务数据库正文。

        Args:
            chunk_id:     来自 local_rag_search / keyword_search 结果的 chunk_id
            already_read: 当前子图生命周期内已读的 chunk_id 集合（可变，会被修改）

        Returns:
            {chunk_id, full_text, paper_id, section, page_num, skipped}
        """
        # 保证 chunk 缓存已加载。
        self._ensure_loaded()
        # 根据 chunk_id 读取父块全文缓存。
        data = self._chunks.get(chunk_id)
        # 找不到 chunk 时返回显式未找到占位。
        if not data:
            return {
                'chunk_id': chunk_id,
                'full_text': f'(chunk {chunk_id} not found)',
                'paper_id': '',
                'section': '',
                'page_num': 0,
                'skipped': True,
            }
        parent_id = str(data.get('parent_id') or '')
        parent_key = f'parent:{parent_id}' if parent_id else ''
        # 已经读过的 chunk 或同一父块直接返回跳过占位结果。
        if chunk_id in already_read or (parent_key and parent_key in already_read):
            return {
                'chunk_id': chunk_id,
                'parent_id': parent_id,
                'full_text': '(already read parent chunk, skipped to save tokens)',
                'paper_id': '',
                'section': '',
                'page_num': 0,
                'skipped': True,
            }
        # 在真正返回全文前先把 chunk 标记为已读。
        already_read.add(chunk_id)
        if parent_key:
            already_read.add(parent_key)
        # 返回完整父块内容与位置信息。
        return {
            'chunk_id': chunk_id,
            'parent_id': parent_id,
            'full_text': data['full_text'],
            'paper_id': data['paper_id'],
            'section': data['section'],
            'page_num': data['page_num'],
            'skipped': False,
        }


# ============================================================
# 模块级单例
# ============================================================

# 模块级缓存的本地 VectorStore 单例，整个进程共享同一套 FAISS 索引缓存。
_vs_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    返回模块级单例，保证整个进程共享同一套本地向量索引缓存。

    这个入口返回的是 `vector_store.py` 中的本地 FAISS 实现，
    不等同于当前主链默认使用的 `milvus_store.get_milvus_store()`。
    """
    # 需要修改模块级全局变量时显式声明 global。
    global _vs_instance
    # 首次调用时创建单例实例。
    if _vs_instance is None:
        _vs_instance = VectorStore()
    # 返回共享单例。
    return _vs_instance
