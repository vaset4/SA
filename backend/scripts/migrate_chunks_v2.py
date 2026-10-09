"""Milvus chunks collection 一次性迁移脚本。

用法:
  python backend/scripts/migrate_chunks_v2.py --dry-run
  python backend/scripts/migrate_chunks_v2.py --backup-path /path/to/backup.jsonl
  python backend/scripts/migrate_chunks_v2.py --confirm-destroy --backup-path /path/to/backup.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings

settings = get_settings()


def _get_client():
    from pymilvus import MilvusClient
    return MilvusClient(uri=settings.milvus_uri)


def dry_run(client) -> dict:
    old_name = settings.milvus_collection
    new_name = settings.milvus_collection_v2
    report = {"old_collection": old_name, "new_collection": new_name}

    if client.has_collection(collection_name=new_name):
        report["v2_exists"] = True
        report["action"] = "skip - v2 already exists"
        return report

    report["v2_exists"] = False
    if not client.has_collection(collection_name=old_name):
        report["old_exists"] = False
        report["old_count"] = 0
    else:
        report["old_exists"] = True
        stats = client.get_collection_stats(collection_name=old_name)
        report["old_count"] = stats.get("row_count", 0)

    report["action"] = "would create v2, drop old, clean business db"
    return report


def backup_collection(client, backup_path: str) -> int:
    old_name = settings.milvus_collection
    if not client.has_collection(collection_name=old_name):
        print(f"[migrate] old collection '{old_name}' does not exist, nothing to backup")
        return 0

    path = Path(backup_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    iterator = client.query_iterator(
        collection_name=old_name,
        filter="chunk_id != \"\"",
        output_fields=["chunk_id", "paper_id", "parent_id", "text", "full_text",
                       "section", "page_num", "authors", "venue", "category",
                       "file_type", "year"],
        batch_size=500,
    )
    with open(path, "w", encoding="utf-8") as f:
        while True:
            rows = iterator.next()
            if not rows:
                break
            for row in rows:
                serializable = {k: v for k, v in row.items() if k != "dense_vector"}
                f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
                count += 1
    iterator.close()
    print(f"[migrate] backed up {count} rows to {path}")
    return count


def confirm_destroy(client, backup_path: str) -> None:
    old_name = settings.milvus_collection
    new_name = settings.milvus_collection_v2

    if client.has_collection(collection_name=new_name):
        print(f"[migrate] v2 collection '{new_name}' already exists, aborting (idempotent)")
        return

    if not Path(backup_path).exists():
        print(f"[migrate] ERROR: backup file not found at {backup_path}")
        print("[migrate] Run with --backup-path first")
        sys.exit(1)

    print(f"[migrate] creating v2 collection '{new_name}'...")
    from app.services.milvus_store import get_milvus_store
    store = get_milvus_store()
    store.ensure_collection()
    print(f"[migrate] v2 collection created")

    print("[migrate] cleaning business database...")
    _clean_business_db()

    if client.has_collection(collection_name=old_name):
        print(f"[migrate] dropping old collection '{old_name}'...")
        client.drop_collection(collection_name=old_name)
        print(f"[migrate] old collection dropped")

    print("[migrate] migration complete")
    print("[migrate] NOTE: all papers need to be re-uploaded")


def _clean_business_db() -> None:
    import app.services.paper_db as paper_db
    try:
        papers = paper_db.list_papers()
        for paper in papers:
            paper_id = paper.get("paper_id", "")
            if paper_id:
                paper_db.mark_needs_reindex(paper_id)
        print(f"[migrate] marked {len(papers)} papers as needs_reindex")
    except Exception as exc:
        print(f"[migrate] WARNING: business db cleanup partial: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Migrate Milvus chunks to v2 schema")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Check current state without modifications (default)")
    parser.add_argument("--backup-path", type=str, default="",
                        help="Dump old collection to this JSONL path")
    parser.add_argument("--confirm-destroy", action="store_true",
                        help="Execute migration (requires --backup-path to have succeeded)")
    args = parser.parse_args()

    client = _get_client()

    if args.confirm_destroy:
        if not args.backup_path:
            print("[migrate] ERROR: --confirm-destroy requires --backup-path")
            sys.exit(1)
        confirm_destroy(client, args.backup_path)
    elif args.backup_path:
        backup_collection(client, args.backup_path)
    else:
        report = dry_run(client)
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
