"""
重新摄入 LightRAG 失败的论文（专门针对 524 错误修复后的验证）

用法:
  python backend/scripts/reingest_lightrag_failed.py <paper_id1> [paper_id2] ... [--library-id=LIBRARY_ID]

示例:
  python backend/scripts/reingest_lightrag_failed.py upload_abdedbbc upload_a9e0d332
  python backend/scripts/reingest_lightrag_failed.py upload_abdedbbc --library-id=custom_lib

作者: xxx
创建时间: 2026-06-21 15:45:00
"""
import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# 设置环境变量指向后端数据目录
import os
os.environ["DATA_DIR"] = str(BACKEND_ROOT / "data")

# 常量配置
DEFAULT_LIBRARY_ID = "default"
UPLOADS_DIR = "uploads"
MINERU_SUFFIX = ".mineru"
MINERU_SOURCE_FILENAME = "source.md"


async def reingest_papers(paper_ids: list[str], library_id: str) -> None:
    """重新摄入指定的论文到 LightRAG"""
    from app.services.ingest.document_graph_service import insert_to_lightrag

    uploads_dir = BACKEND_ROOT / "data" / UPLOADS_DIR

    for paper_id in paper_ids:
        print(f"\n{'='*60}")
        print(f"正在处理: {paper_id}")
        print(f"{'='*60}")

        # 1. 查找 MinerU 解析结果
        mineru_dir = uploads_dir / f"{paper_id}{MINERU_SUFFIX}"
        source_md = mineru_dir / MINERU_SOURCE_FILENAME

        if not source_md.exists():
            print(f"[ERROR] 找不到 {source_md}")
            continue

        # 2. 读取全文
        full_text = source_md.read_text(encoding="utf-8")
        print(f"[INFO] 文本长度: {len(full_text)} 字符")

        # 3. 重新摄入
        try:
            print(f"[INFO] 开始摄入到 LightRAG (library_id={library_id})...")
            await insert_to_lightrag(
                paper_id=paper_id,
                full_text=full_text,
                library_id=library_id
            )
            print(f"[SUCCESS] 摄入成功: {paper_id}")
        except Exception as e:
            print(f"[ERROR] 摄入失败: {paper_id}")
            print(f"        错误信息: {e}")
            import traceback
            traceback.print_exc()


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python backend/scripts/reingest_lightrag_failed.py <paper_id1> [paper_id2] ... [--library-id=LIBRARY_ID]")
        print("示例: python backend/scripts/reingest_lightrag_failed.py upload_abdedbbc upload_a9e0d332")
        return 1

    # 解析参数：支持 --library-id 可选参数
    paper_ids = []
    library_id = DEFAULT_LIBRARY_ID

    for arg in sys.argv[1:]:
        if arg.startswith("--library-id="):
            library_id = arg.split("=", 1)[1]
        else:
            paper_ids.append(arg)

    if not paper_ids:
        print("[ERROR] 未提供任何 paper_id")
        return 1

    print(f"准备重新摄入 {len(paper_ids)} 篇论文: {', '.join(paper_ids)}")
    print(f"目标库: {library_id}")

    asyncio.run(reingest_papers(paper_ids, library_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
