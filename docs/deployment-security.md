# 部署与配置

`docker-compose.yml` 用于本机开发和使用，需要 Docker Compose 2.24 或更高版本。服务端口默认只允许本机通过 `127.0.0.1` 访问，容器之间通过 Docker 网络连接。

## 本地启动

1. 将根目录 `.env.example` 复制为 `.env`。
2. 自行生成 `JWT_SECRET_KEY`，用来签发和校验登录凭证；设置 PostgreSQL、MinIO 和 Neo4j 的账号或密码。Neo4j 用户名保留 `neo4j`。
3. 填写模型服务商提供的对话、向量和重排 API 密钥；使用 MinerU 在线解析时填写其令牌。具体字段见 [README](../README.md)。
4. 运行 `docker compose -f docker-compose.yml config --quiet` 检查配置，再运行 `docker compose -f docker-compose.yml up -d --build` 启动。

当前模板始终启动 Neo4j 容器，后端会等待该容器就绪。`NEO4J_ENABLED=false` 关闭的是应用中的图谱功能，因此 Neo4j 密码仍为必填项。

默认镜像通过 API 调用检索模型。需要本地运行模型时，组合使用 `docker-compose.local-models.yml`，并准备模型目录，见 [本地模型说明](../README.md#可选运行本地模型)。

## 配置优先级与保存

本地 Python 运行时，配置优先级从低到高为 `backend/.env`、根 `.env`、进程环境变量。连接 Docker 中的数据库和向量服务时，使用 `127.0.0.1` 的地址，示例见 [本地开发](../README.md#本地开发)。

Docker Compose 明确列出的配置来自根 `.env`，其他扩展参数可以放在 `backend/.env`。修改根 `.env` 后，重新运行 `docker compose -f docker-compose.yml up -d`，让容器应用新的环境变量。

页面中激活 LLM 配置会立即影响当前后端，长期保存方式取决于启动方式：

| 启动方式 | 保存位置与操作 |
| --- | --- |
| 本地 Python | 写入项目根 `.env` |
| Docker Compose | 写入容器内 `/app/.env`；重建容器后不保留，需要同步修改宿主机的根 `.env` |

`.env` 包含密码和密钥，请保存在部署环境中，不要提交到源码仓库。

## 测试环境与生产部署

- 设置 `APP_ENV=staging` 或 `APP_ENV=prod`，并由部署环境提供登录密钥及数据库密码。启动时会检查 JWT、数据库连接配置和已启用的 Neo4j 凭证。
- MinIO 为 Milvus 保存检索数据。使用 Compose 时，`MINIO_ROOT_USER` 和 `MINIO_ROOT_PASSWORD` 会同时配置到 MinIO 与 Milvus。
- 使用反向代理或平台网关提供公网入口，数据库和存储服务端口保持内网访问。
- Compose 中的前端使用开发服务器。生产部署需在 `frontend` 执行 `npm run build` 后使用 `npm start`，或部署到支持 Next.js 的平台，并配置前端 API 地址和后端 CORS。
