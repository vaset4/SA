"""上传索引的元数据转换，写入及发布由 UploadBatchIndexer 统一负责，作者：xxx。"""
from __future__ import annotations

from app.models.models import Paper


def extract_year(paper: Paper) -> int:
    """读取论文日期中的年份；参数为论文记录，返回可用于索引的年份或未识别值 0。"""
    year = (paper.get("published_date") or "")[:4]
    return int(year) if year.isdigit() else 0


def build_upload_metadata(paper: Paper, file_type: str) -> dict:
    """将论文记录和文件类型转换为索引元数据，不直接写业务库或索引。"""
    return {
        "year": extract_year(paper),
        "authors": ", ".join(paper.get("authors", [])),
        "venue": "",
        "category": ", ".join(paper.get("categories", [])),
        "file_type": file_type,
    }
