# 测试与评测

## 运行测试

在项目根目录激活 Python 虚拟环境，安装后端开发依赖：

```powershell
python -m pip install -r backend/requirements-dev.txt
```

进入 `backend` 目录，按修改范围选择测试。例如，验证向量和重排接口：

```powershell
python -m pytest tests/unit/test_embedding_client.py tests/unit/test_reranker.py --timeout=60 --timeout-method=thread -q
```

上述参数将单个测试限制为 60 秒。模型单元测试通过本机 HTTP 服务和测试模型检查请求、响应与错误处理。连接真实模型服务、数据库的集成测试需要独立的测试配置和资料。

前端在 `frontend` 目录运行：

```powershell
npm ci
npm test
npm run build
```

## 准备评测数据

`backend/evals/` 提供检索问答、证据引用和研究流程的评测工具。运行前需要准备自己的资料、问题和标注，并指定数据集与结果输出路径。

在 `backend` 目录查看具体模块的参数：

```powershell
python -m evals.ragas.evals --help
python -m evals.evidence.run_eval --help
```

通过 `--dataset` 指定数据集、`--output-dir` 指定结果目录；其余必填参数与数据格式以所选模块为准。若提示数据文件不存在，检查传入路径。测试真实检索质量时，应使用与部署环境一致的模型和解析配置。

## 生成 PDF 原文快照

原文快照保存评测所依据的 PDF 文字和页码，便于核对回答引用的证据。将可提取文字的 PDF 放在 `backend/paper/source.pdf`，在 `backend` 目录运行以下 Python 代码：

```python
from pathlib import Path
from evals.evidence.corpus import build_native_source_snapshot, validate_native_source

paper_root = Path("paper").resolve()
source = build_native_source_snapshot("my-paper", paper_root / "source.pdf", paper_dir=paper_root)
validate_native_source(source, paper_dir=paper_root)
Path("source-snapshot.json").write_text(source.model_dump_json(indent=2), encoding="utf-8")
```

生成的 `source-snapshot.json` 使用 `pypdf_pdf_page_text_v1` 格式。每个有文字的页面形成一个块，编号为 `pypdf-page-000001` 等；空白页不会改变后续页码。快照记录原文件哈希、字节数、文字、页码和 pypdf 版本。

还需为数据集准备问题、预期证据和人工复核的标注，原文快照本身不包含这些内容。扫描 PDF 应先通过 OCR 或 MinerU 提取文字。

如果已有快照使用 `pymupdf_pdf_text_blocks` 格式，或解析器版本与当前环境不一致，验证时会报错。请重新生成快照，并核对引用位置和证据标注；仅修改格式名称不能完成转换。
