# StudyAssistant 学习助手

StudyAssistant 是一个面向论文阅读与课题研究的开源学习工作台。你可以围绕研究主题组织资料，阅读论文并提问，对比多篇论文中的方法与结论，再将有价值的判断和来源保存到项目中，供后续研究继续使用。

项目把资料管理、AI 分析、研究对话和证据核对放在同一个工作区。内部研究围绕已有资料展开，外部研究用于搜索和补充资料；研究结果可以结合原文、引用和研究过程进行复核。

原创代码采用 [MIT](LICENSE)，依赖与图标遵循[各自许可及署名](THIRD_PARTY_NOTICES.md)。

## 适用场景

- **读懂一篇论文**：梳理研究问题、核心方法、主要结论与局限，并回到原文核对细节。
- **围绕课题整理文献**：将相关论文放入同一项目，比较不同工作的思路、实验结果与适用条件。
- **持续推进一项研究**：记录待解决的问题、已有判断及其证据，后续阅读新资料时继续补充和修订。

## 主要功能

| 功能 | 可以做什么 |
| --- | --- |
| 项目工作台 | 按研究主题组织论文、笔记和研究问题，查看资料分析状态 |
| 论文阅读与分析 | 上传和解析资料，查看原文与 AI 分析，了解论文的方法、贡献和局限 |
| 资料内问答 | 围绕选定项目或论文提问，结合检索到的资料生成回答并查看引用证据 |
| 跨论文研究 | 围绕同一个问题比较多篇论文，汇总不同来源的观点和依据 |
| 外部研究 | 按主题搜索外部资料，查看研究报告、来源和研究过程；联网检索需配置相应服务 |
| 研究记忆 | 在项目中保存研究判断、关联论文和证据，支持人工确认、修改和查看历史版本 |
| 知识图谱（可选） | 浏览资料中的实体与关联，辅助理解关系和查找证据；需要启用图谱服务 |

## 基本使用流程

1. **建立项目**：为课题创建项目，明确要研究的问题。
2. **准备资料**：在知识库上传论文，将相关资料加入项目，等待解析完成。
3. **阅读与分析**：查看论文原文，运行 AI 分析，了解研究问题、方法和结论。
4. **提出研究问题**：进入内部研究，对选定资料提问或开展跨论文比较；需要扩展资料时使用外部研究。
5. **核对并保存**：检查回答引用的原文，将有用的内容整理为项目笔记或研究记忆。

例如，研究“检索增强问答中的文档分块”时，可以先把相关论文加入项目，再询问“这些论文如何划分文档？各自依据什么实验评估效果？”，结合返回的证据阅读原文，并记录自己的判断。

## 技术组成

前端使用 Next.js 和 React，后端使用 FastAPI 和 LangGraph。PostgreSQL 保存账号、项目与研究记录，Milvus 提供向量检索，Neo4j 提供可选图谱能力。PDF 可通过 MinerU 在线解析，也可使用基础本地文字提取，详见 [PDF 处理说明](docs/pdf-processing.md)。

检索向量和重排**默认通过 API 调用**，首次使用无需下载本地 BGE 模型权重；也可按需启用本地模型。回答生成、检索模型与文档解析分别配置，具体参数见下方安装说明和 [模型配置](docs/model-api.md)。

## 安装与启动

可以使用 Docker 启动整套服务，或按 [本地开发](#本地开发) 分别运行前后端。更多说明见 [文档目录](docs/README.md)。

## Docker 启动

准备 Docker Engine/Desktop 和 Docker Compose 2.24+。Windows 使用 Linux 容器。首次构建需要联网下载镜像和依赖。

1. 在项目根目录复制配置模板：

   ```powershell
   Copy-Item .env.example .env
   ```

   macOS/Linux 可使用 `cp .env.example .env`。

2. 编辑 `.env`，填写以下配置：

   | 用途 | 配置 | 如何填写 |
   | --- | --- | --- |
   | 登录凭证签名 | `JWT_SECRET_KEY` | 用于签发和校验用户的登录凭证，首次部署时自行生成 |
   | 业务数据库 | `POSTGRES_PASSWORD` | PostgreSQL 的密码，后端用它连接数据库；首次部署时自行设置 |
   | 检索数据存储 | `MINIO_ROOT_USER`、`MINIO_ROOT_PASSWORD` | MinIO 的账号和密码，自行设置 |
   | 图数据库 | `NEO4J_USERNAME`、`NEO4J_PASSWORD` | 用户名保留 `neo4j`，密码自行设置 |
   | 对话和研究 | `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL_NAME` | 从模型服务商获取，选择支持 chat-completions 接口的模型 |
   | 检索向量 | `EMBEDDING_API_KEY` | 从硅基流动获取，默认模型为 `BAAI/bge-m3` |
   | 检索重排 | `RERANK_API_KEY` | 从硅基流动获取，默认模型为 `BAAI/bge-reranker-v2-m3` |
   | PDF 在线解析 | `MINERU_API_TOKEN` | 从 MinerU 获取；使用本地解析时设置 `MINERU_ENABLED=false` |
   | 可选联网检索 | `TAVILY_API_KEY` | 使用联网检索时从 Tavily 获取 |

   JWT 密钥和数据库密码不需要向服务商申请。若本机已安装 Python，可运行 `python -c "import secrets; print(secrets.token_hex(24))"` 生成随机值，每项密钥或密码分别生成一次。生成结果也适合直接用作 PostgreSQL 密码，无需处理连接地址中的特殊字符。

   当前 Docker 模板会启动 Neo4j 容器，后端也会等待它就绪。`NEO4J_ENABLED=false` 只关闭应用里的图谱功能，因此仍需填写 Neo4j 密码。

   模型 API 密钥需要逐项填写；如果服务商允许同一个密钥访问向量和重排模型，可以在两处填入同一密钥。更换服务商见 [模型配置](docs/model-api.md)。

3. 核对配置并启动。`config --quiet` 不会将完整凭证打印到终端：

   ```powershell
   docker compose -f docker-compose.yml config --quiet
   docker compose -f docker-compose.yml up -d --build
   docker compose -f docker-compose.yml logs -f backend
   ```

4. 访问 <http://localhost:3000>，注册账号后上传自己的资料；健康接口为 <http://localhost:8000/health>，接口文档为 <http://localhost:8000/docs>。PDF 上传、检索、问答分别需要有效的解析、向量和 LLM 配置。

宿主端口默认只绑定本机。Compose 前端运行开发服务器，适合本机使用；公网部署需配置生产前端和入口代理，见 [部署说明](docs/deployment-security.md)。

## 本地开发

建议 Python 3.11 和 Node.js 20。在根 `.env` 填好上述参数后，启动基础服务并安装后端依赖：

```powershell
docker compose -f docker-compose.yml up -d postgres etcd minio milvus neo4j
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements-dev.txt
python backend/scripts/prepare_chunk_tokenizer.py
```

在根 `.env` 另加 `DATABASE_URL=postgresql://studyassistant:你的数据库密码@127.0.0.1:5432/studyassistant`、`MILVUS_URI=http://127.0.0.1:19530`；启用 Neo4j 时加 `NEO4J_URI=bolt://127.0.0.1:7687`。分别启动后端和前端：

```powershell
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

```powershell
cd frontend
npm ci
npm run dev
```

后端加载项目自己的 `backend/.env` 和根 `.env`，根配置覆盖同名文件项，进程环境变量优先。Compose 中显式列出的参数来自根 `.env`；修改配置后重启后端。

API 模式的文档分块使用约 17 MB 的 BGE 分词文件，用于计算每段文字的长度，无需下载本地模型权重。上面的准备命令会下载并校验该文件；Docker 构建会自动准备。配置与故障处理见 [PDF 与分块说明](docs/pdf-processing.md)。

页面中激活 LLM 配置立即影响当前后端。本地 Python 运行会写入根 `.env`；Compose 重建后仍使用宿主根 `.env`，需同步修改该文件以长期保留，见 [配置持久化说明](docs/deployment-security.md)。

## 可选：运行本地模型

本地模型指在自己的电脑或服务器上运行检索向量模型和重排模型，需要额外下载模型权重并准备计算资源。使用默认 API 模式可以跳过本节；回答问题的 LLM 仍单独配置。

本地 Python 环境额外安装 `python -m pip install -r backend/requirements-local.txt`，设置 `EMBEDDING_MODE=local` 和 `RERANK_MODE=local`，再将 `EMBEDDING_MODEL` 和 `RERANK_MODEL` 设为模型目录或仓库名称。

Docker 用户准备好权重目录后使用显式覆盖文件：

```powershell
docker compose -f docker-compose.yml -f docker-compose.local-models.yml up -d --build
```

默认读取 `models/bge-m3` 与 `models/bge-reranker-v2-m3`，可在 `.env` 指定 `LOCAL_EMBEDDING_MODEL_DIR` 和 `LOCAL_RERANK_MODEL_DIR`。目录必须存在，挂载为只读。

## 切换向量模型后处理已有资料

已经上传并处理过的资料，其检索数据与当时的向量模型配置有关。切换本地/API 模式、向量服务地址或模型后，需要重新处理这些资料，才能按新配置检索。原文件会保留，详见 [模型切换说明](docs/model-api.md#模式切换与数据)。

## 运行测试

安装开发依赖后，可在 `backend` 运行目标测试：

```powershell
python -m pytest tests/unit/test_embedding_client.py tests/unit/test_reranker.py --timeout=60 --timeout-method=thread -q
```

前端在 `frontend` 目录运行 `npm test` 和 `npm run build`。部分集成测试需要独立的数据库或模型服务，数据准备和运行方式见 [测试与评测](docs/evaluation.md)。
