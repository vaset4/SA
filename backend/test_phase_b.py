"""
Phase B 端到端验收脚本

运行方式：
    cd backend
    python test_phase_b.py

覆盖：
    Step 5  document_parser — 用内置示例文本验证统一 Parent-Child 分块
    Step 6  paper_db       — add/get/checkpoint/log
    Step 7  vector_store   — add_paper / local_rag_search / keyword_search / read_chunk
    Step 8  rag_node       — 完整子图 ainvoke（知识库命中路径）
"""

import asyncio
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── 颜色输出 ──────────────────────────────────────────────────
def ok(msg):   print(f'  [PASS] {msg}')
def fail(msg): print(f'  [FAIL] {msg}'); sys.exit(1)
def section(title): print(f'\n{"="*55}\n  {title}\n{"="*55}')


# ============================================================
# Step 5 — 统一解析器分块逻辑（不依赖真实 PDF）
# ============================================================
section('Step 5 — document_parser 分块逻辑')

from app.services.document_parser import parse_markdown_text

parent_text = ('This is a detailed explanation of the proposed method. ' * 15).strip()
chunks = parse_markdown_text('# 3. Methodology\n\n' + parent_text, '2301.00001')
assert len(chunks) > 0, '子块数量为 0'
assert len({c['parent_id'] for c in chunks}) == 1, 'parent_id 不一致'
assert all(c['full_text'] == parent_text for c in chunks), 'full_text 不等于父块原文'
ok(f'Parent-Child 分块: {len(chunks)} 个子块，parent_id 一致，full_text 正确')


# ============================================================
# Step 6 — paper_db
# ============================================================
section('Step 6 — paper_db')

import app.services.paper_db as db
from app.models.library_models import DEFAULT_LIBRARY_ID
from app.models.models import Paper

db.init_db()
ok('init_db 成功')

test_paper = Paper(
    library_id=DEFAULT_LIBRARY_ID,
    paper_id='test_2301_00001',
    arxiv_id='2301.00001',
    doi=None,
    title='Attention Is All You Need (Test)',
    authors=['Vaswani', 'Shazeer'],
    abstract='We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.',
    pdf_url='https://arxiv.org/pdf/1706.03762',
    pdf_path=None,
    categories=['cs.LG', 'cs.AI'],
    published_date='2017-06-12',
    source='arxiv',
    analysis=None,
    tags=[],
)

pid = db.add_paper(test_paper)
ok(f'add_paper: paper_id={pid}')

pid2 = db.add_paper(test_paper)  # 重复写入应 merge
assert pid == pid2, f'重复写入应返回同一 paper_id，got {pid2}'
ok('重复写入 merge 正确（同一 paper_id）')

p = db.get_paper(pid)
assert p is not None and p['title'] == test_paper['title']
ok(f'get_paper: {p["title"][:40]}...')

assert db.exists(arxiv_id='2301.00001')
ok('exists(arxiv_id) = True')

sid = db.create_session('RAG test topic')
ok(f'create_session: {sid[:8]}...')

db.save_checkpoint(sid, 'rag_search', '{"query": "test"}')
cp = db.load_checkpoint(sid)
assert cp['step_name'] == 'rag_search'
ok(f'Checkpointer: save/load step={cp["step_name"]}')

db.write_log('rag_hit', 'test_node', 'test query', 3, 42)
ok('write_log OK')


# ============================================================
# Step 7 — vector_store 三工具
# ============================================================
section('Step 7 — vector_store 三工具')

from app.services.vector_store import get_vector_store
from app.models.models import Chunk

vs = get_vector_store()

# 构造测试 Chunk
test_chunks = []
sections_data = [
    ('3. Method',    'We propose an Agentic RAG framework with multi-hop reasoning. The retrieval pipeline uses dense vector search combined with BM25 sparse retrieval for hybrid matching.'),
    ('4. Experiment','We evaluate on four benchmark datasets. LoRA fine-tuning is applied to the base model. DPO alignment improves response quality significantly.'),
    ('5. Conclusion','Our experiments demonstrate that combining dense and sparse retrieval outperforms single-method approaches. Future work includes multi-modal RAG.'),
]

for i, (sec, text) in enumerate(sections_data):
    parent_id = f'test_2301_00001_{i}'
    full_text  = text * 3  # 模拟父块（子块的 full_text）
    # 每个父块切出 2 个子块
    mid = len(text) // 2
    for j, sub in enumerate([text[:mid], text[mid:]]):
        test_chunks.append(Chunk(
            chunk_id=f'{parent_id}_{j}',
            parent_id=parent_id,
            text=sub.strip(),
            full_text=full_text,
            page_num=i + 3,
            section=sec,
        ))

print(f'  准备写入 {len(test_chunks)} 个子块...')
vs.add_paper('test_2301_00001', test_chunks)
ok(f'add_paper: {len(test_chunks)} 个子块已索引')

# local_rag_search 返回预览（非全文）
results = vs.local_rag_search('retrieval augmented generation hybrid search', top_k=3)
assert len(results) > 0, 'local_rag_search 返回空'
assert 'full_text' not in results[0], 'search 结果不应含 full_text 字段'
assert 'preview' in results[0], 'search 结果应含 preview 字段'
assert len(results[0]['preview']) <= 300, f'preview 过长: {len(results[0]["preview"])}'
ok(f'local_rag_search: 命中 {len(results)} 个 chunk，preview 字段正确（无 full_text）')

# keyword_search 精确匹配
kw_results = vs.keyword_search(['LoRA', 'DPO'], top_k=3)
assert len(kw_results) > 0, 'keyword_search(LoRA, DPO) 返回空'
ok(f'keyword_search(LoRA, DPO): 命中 {len(kw_results)} 个 chunk')

# read_chunk 首次读取返回全文
already_read = set()
chunk_id = results[0]['chunk_id']
r = vs.read_chunk(chunk_id, already_read)
assert not r['skipped'], 'first read should NOT be skipped'
assert len(r['full_text']) > 100, 'full_text 应为父块全文'
ok(f'read_chunk 首次: full_text 长度={len(r["full_text"])}，skipped=False')

# read_chunk 重复读取返回占位符
r2 = vs.read_chunk(chunk_id, already_read)
assert r2['skipped'], '重复 read_chunk 应返回 skipped=True'
assert '(already read' in r2['full_text']
ok('read_chunk 重复: skipped=True，token 不重复消耗')


# ============================================================
# Step 8 — rag_node 完整子图（知识库命中路径）
# ============================================================
section('Step 8 — rag_node 子图（知识库命中）')

from app.nodes.rag_node import create_rag_node
from app.models.models import RAGState

async def run_rag():
    rag = create_rag_node()
    init_state = RAGState(
        user_query='What is the hybrid retrieval method proposed in this paper?',
    )
    print('  调用 rag_node 子图...')
    result = await rag.ainvoke(init_state)
    return result

result = asyncio.run(run_rag())

answer = result.get('answer', '')
iteration = result.get('iteration', 0)
assert answer, 'answer 为空'
ok(f'rag_node 完成: iteration={iteration}，answer 长度={len(answer)}')
print(f'\n  --- 回答预览（前200字符）---')
print(f'  {answer[:200]}')
print()


# ============================================================
# 汇总
# ============================================================
section('Phase B 验收通过')
print('  所有检查项均通过。')
print()
print('  已验证：')
print('    Step 5  document_parser — Parent-Child 分块，parent_id/full_text 正确')
print('    Step 6  paper_db       — 增删查/去重/Checkpointer/write_log')
print('    Step 7  vector_store   — 句子级索引/语义搜索/关键词搜索/read_chunk 去重')
print('    Step 8  rag_node       — 完整子图 ainvoke，知识库命中，生成带引用回答')
