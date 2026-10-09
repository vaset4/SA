"""业务库与 Milvus chunk_id 一致性检查脚本。

用法:
  python backend/scripts/check_chunks_consistency.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings

settings = get_settings()


def main():
    from app.services.milvus_store import get_milvus_store
    import app.services.paper_db as paper_db

    store = get_milvus_store()

    print("[consistency] checking Milvus v2 collection...")
    milvus_paper_ids = set(store.list_paper_ids())
    print(f"[consistency] Milvus papers: {len(milvus_paper_ids)}")

    print("[consistency] checking business database...")
    try:
        db_papers = paper_db.list_papers()
        db_paper_ids = {p.get("paper_id", "") for p in db_papers if p.get("paper_id")}
    except Exception as exc:
        print(f"[consistency] WARNING: cannot read business db: {exc}")
        db_paper_ids = set()

    print(f"[consistency] business db papers: {len(db_paper_ids)}")

    in_milvus_not_db = milvus_paper_ids - db_paper_ids
    in_db_not_milvus = db_paper_ids - milvus_paper_ids

    if in_milvus_not_db:
        print(f"[consistency] WARNING: {len(in_milvus_not_db)} papers in Milvus but not in business db:")
        for pid in sorted(in_milvus_not_db)[:10]:
            print(f"  - {pid}")

    if in_db_not_milvus:
        print(f"[consistency] WARNING: {len(in_db_not_milvus)} papers in business db but not in Milvus:")
        for pid in sorted(in_db_not_milvus)[:10]:
            print(f"  - {pid}")

    if not in_milvus_not_db and not in_db_not_milvus:
        print("[consistency] OK: Milvus and business db are consistent")
    else:
        print(f"[consistency] INCONSISTENT: {len(in_milvus_not_db)} orphan Milvus, {len(in_db_not_milvus)} orphan DB")
        sys.exit(1)


if __name__ == "__main__":
    main()
