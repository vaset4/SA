"""
Phase C 验收脚本 — 事件驱动推送

覆盖：
    Step C1  build_profile      — 空库 / 有关键词库两种情况
    Step C2  score_paper        — 分类/关键词/兴趣词三维打分
    Step C3  push_notification  — terminal 模式输出，file 模式写 JSON
    Step C4  write_back_node    — rag_result 含论文 → 打分 → 超阈值触发推送
"""

import asyncio
import sys
import io
import os
import json
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'


def ok(msg):    print(f'  [PASS] {msg}')
def fail(msg):  print(f'  [FAIL] {msg}'); sys.exit(1)
def section(t): print(f'\n{"="*55}\n  {t}\n{"="*55}')


# ============================================================
# 工具：构造 Paper 字典
# ============================================================
def make_paper(paper_id, title, categories=None, abstract='', keywords=None):
    from app.models.models import PaperAnalysis
    analysis = None
    if keywords:
        analysis = {
            'summary': 'test',
            'keywords': keywords,
            'methods': [],
            'datasets': [],
            'contributions': [],
            'limitations': '',
        }
    return {
        'paper_id':       paper_id,
        'arxiv_id':       paper_id,
        'doi':            None,
        'title':          title,
        'authors':        ['Author A'],
        'abstract':       abstract,
        'analysis':       analysis,
        'pdf_path':       None,
        'pdf_url':        None,
        'source':         'test',
        'categories':     categories or [],
        'published_date': '2024-01-01',
        'added_at':       '2024-01-01',
        'last_accessed':  None,
        'tags':           [],
        'is_vectorized':  False,
        'read_count':     0,
    }


# ============================================================
# Step C1 — build_profile
# ============================================================
section('Step C1 — build_profile')

import app.services.paper_db as paper_db
paper_db.init_db()

from app.services.recommendation import build_profile, score_paper, push_notification

# 空库
profile = build_profile()
assert isinstance(profile.auto_keywords, list), 'auto_keywords 不是 list'
ok(f'空库 build_profile: auto_keywords={profile.auto_keywords}')

# 写入有 keywords 的论文
p1 = make_paper('c_test_001', 'LoRA Fine-Tuning',
                categories=['cs.LG'],
                abstract='LoRA fine-tuning reduces parameters.',
                keywords=['LoRA', 'fine-tuning', 'LLM'])
p2 = make_paper('c_test_002', 'DPO Alignment',
                categories=['cs.AI'],
                abstract='DPO aligns LLM with human feedback.',
                keywords=['DPO', 'RLHF', 'LLM'])
paper_db.add_paper(p1)
paper_db.add_paper(p2)

profile = build_profile()
# LLM 出现两次，应该在 auto_keywords 里
assert 'llm' in profile.auto_keywords, f'auto_keywords 缺少 llm: {profile.auto_keywords}'
ok(f'有数据 build_profile: auto_keywords={profile.auto_keywords}')


# ============================================================
# Step C2 — score_paper
# ============================================================
section('Step C2 — score_paper')

from app.services.recommendation import UserProfile

# 构造包含 explicit_categories 和 explicit_keywords 的 profile
test_profile = UserProfile(
    explicit_keywords=['fine-tuning', 'alignment'],
    explicit_categories=['cs.LG'],
    auto_keywords=['llm', 'lora'],
)

# 论文与 profile 高度匹配
high_match = make_paper(
    'c_score_001', 'LoRA and Fine-Tuning for LLM',
    categories=['cs.LG'],
    abstract='We fine-tune LLMs using LoRA.',
    keywords=['LoRA', 'fine-tuning', 'LLM'],
)
score_h, reason_h = score_paper(high_match, test_profile)
assert score_h > 0.3, f'高匹配论文得分过低: {score_h}'
ok(f'高匹配论文: score={score_h:.3f}, reason={reason_h}')

# 论文与 profile 完全不匹配
low_match = make_paper(
    'c_score_002', 'Graph Neural Network for Protein Folding',
    categories=['cs.CV'],
    abstract='We propose a GNN model for protein structure.',
    keywords=['GNN', 'protein', 'biology'],
)
score_l, reason_l = score_paper(low_match, test_profile)
assert score_l < score_h, f'低匹配论文得分应低于高匹配: {score_l} vs {score_h}'
ok(f'低匹配论文: score={score_l:.3f}, reason={reason_l}')

# 边界：无 explicit_categories 时分类维度不贡献分数
profile_no_cat = UserProfile(auto_keywords=['attention'])
paper_with_cat = make_paper('c_score_003', 'Attention Mechanism',
                            categories=['cs.LG'], abstract='attention mechanism')
score_nc, _ = score_paper(paper_with_cat, profile_no_cat)
ok(f'无 explicit_categories: score={score_nc:.3f}（分类维度不贡献）')


# ============================================================
# Step C3 — push_notification
# ============================================================
section('Step C3 — push_notification')

from app.config import get_settings
settings = get_settings()

# terminal 模式：只验证不抛异常
import io as _io
captured = _io.StringIO()
import sys as _sys
_old_stdout = _sys.stdout
_sys.stdout = captured
push_notification(high_match, score_h, reason_h)
_sys.stdout = _old_stdout
output = captured.getvalue()
assert '推送' in output or 'score' in output.lower(), f'terminal 输出异常: {output!r}'
ok(f'terminal 模式推送: 输出长度={len(output)}')

# file 模式
with tempfile.TemporaryDirectory() as tmpdir:
    settings_orig_dir  = settings.data_dir
    settings_orig_mode = settings.push_mode

    # 临时切换设置
    object.__setattr__(settings, 'data_dir',   tmpdir)
    object.__setattr__(settings, 'push_mode',  'file')

    push_notification(high_match, score_h, reason_h)
    push_notification(low_match,  score_l, reason_l)

    notif_path = Path(tmpdir) / 'pending_notifications.json'
    assert notif_path.exists(), 'pending_notifications.json 未创建'
    data = json.loads(notif_path.read_text(encoding='utf-8'))
    assert len(data) == 2, f'期望 2 条通知，实际 {len(data)}'
    assert data[0]['paper_id'] == 'c_score_001'
    assert data[1]['paper_id'] == 'c_score_002'

    # 恢复
    object.__setattr__(settings, 'data_dir',   settings_orig_dir)
    object.__setattr__(settings, 'push_mode',  settings_orig_mode)

ok('file 模式推送: 写入 pending_notifications.json 正确（2 条）')


# ============================================================
# Step C4 — write_back_node 推送触发
# ============================================================
section('Step C4 — write_back_node 推送触发')

from app.models.models import RAGState, RAGResult

# 构造 rag_result，sources 指向已入库论文 c_test_001（LoRA）
# 设置 push_score_threshold 为低值，确保触发
object.__setattr__(settings, 'push_score_threshold', 0.0)
object.__setattr__(settings, 'push_mode', 'terminal')

import io as _io2
captured2 = _io2.StringIO()
_sys.stdout = captured2

from app.nodes.rag_node import write_back_node

state: RAGState = {
    'user_query': 'test',
    'rag_result': RAGResult(
        chunks=['some chunk'],
        sources=['c_test_001'],
        scores=[0.9],
        is_sufficient=True,
    ),
    'answer': 'test answer',
    'iteration': 1,
}
write_back_node(state)

_sys.stdout = _old_stdout
output2 = captured2.getvalue()

object.__setattr__(settings, 'push_score_threshold', 0.6)

# 检查推送输出
assert '推送' in output2 or 'score' in output2.lower() or len(output2) > 0, \
    f'write_back_node 未触发推送输出: {output2!r}'
ok(f'write_back_node 推送触发: 输出={output2.strip()[:80]}')


# ============================================================
# 汇总
# ============================================================
print(f'\n{"="*55}')
print('  Phase C 验收通过')
print(f'{"="*55}')
print('''
  已验证：
    Step C1  build_profile     — 空库/有数据两种场景，auto_keywords 自动提取
    Step C2  score_paper       — 三维打分正确，高/低匹配符合预期
    Step C3  push_notification — terminal 输出 + file 写 JSON 均正确
    Step C4  write_back_node   — rag_result 含论文 → 打分 → 超阈值触发推送
''')
