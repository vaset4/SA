# 检索模型配置

检索使用两类模型：embedding 将资料和问题转换成向量，用于查找相关内容；rerank 对候选内容重新评分，让更相关的段落排在前面。它们与生成回答的 LLM 分别配置。

默认通过硅基流动 API 调用，无需下载本地模型权重。填写 `.env` 中的 `EMBEDDING_API_KEY` 和 `RERANK_API_KEY` 后重启后端；更换服务商时，还需修改地址、模型名称和向量维度。

| 配置 | 默认值 / 含义 |
| --- | --- |
| `EMBEDDING_MODE` | `api`；设置为 `local` 时使用本地模型 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3`；local 时可填本机目录 |
| `EMBEDDING_API_BASE_URL` | `https://api.siliconflow.cn/v1`，追加 `/embeddings` |
| `EMBEDDING_API_KEY` | 从向量模型服务商获取的 API 密钥 |
| `EMBEDDING_API_TIMEOUT_SECONDS` | `120` 秒 |
| `EMBEDDING_BATCH_SIZE` | `32` 条文本一批 |
| `EMBEDDING_DIMENSION` | `1024`，须与服务返回的向量维度一致 |
| `RERANK_MODE` | `api`；设置为 `local` 时使用本地模型 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` |
| `RERANK_ENDPOINT` | `https://api.siliconflow.cn/v1/rerank`，完整地址 |
| `RERANK_API_KEY` | 从重排模型服务商获取的 API 密钥 |
| `RERANK_API_TIMEOUT_SECONDS` | `120` 秒 |
| `RERANK_ENABLED` | `true`；设置为 `false` 时关闭重排 |

如果设置了 `VECTOR_RERANK_ENABLED`、`VECTOR_RERANK_MODEL` 或 `VECTOR_RERANK_TOP_K`，它们优先于对应的 `RERANK_*` 配置。修改重排模型时也需检查这些覆盖项。

## 接口兼容要求

- Bearer 认证。Embedding 发送 `model`、文本数组 `input`、`encoding_format=float`；按 `data[].index` 恢复顺序，检查数量、维度和有限非零向量，再归一化。
- Rerank 发送 `model`、`query`、全部候选 `documents`、完整候选数 `top_n`、`return_documents=false`；按 `results[].index` 将 `relevance_score` 对应回原材料，再排序取 top-k。
- Embedding 请求不发送 `dimensions` 参数，服务返回的维度必须与 `EMBEDDING_DIMENSION` 一致。
- 接入其他服务后，应使用自己的资料检查检索结果；不同模型的重排分数可能不同。

接口参考：[Embedding](https://docs.siliconflow.cn/docs/api/embeddings-post)、[Rerank](https://docs.siliconflow.cn/docs/api/rerank-post)。当前客户端支持上述文本接口；其他服务商需满足相同的请求和响应格式。

## 调用失败时检查什么

| 现象 | 检查项 |
| --- | --- |
| 缺少密钥或认证失败 | 对应功能的 API 密钥是否填写、有效，是否有模型访问权限 |
| HTTP 错误或超时 | 服务地址、网络、服务商状态及超时配置 |
| 返回内容无法解析或缺少条目 | 服务是否符合上述接口格式，是否返回了完整结果 |
| 向量维度不匹配 | `EMBEDDING_DIMENSION` 是否与所选模型实际输出一致 |

具体错误可在后端日志中查看。修改配置后重启后端，再重试对应操作。

## 模式切换与数据

系统根据向量模型的运行模式、服务地址、模型名称和维度区分检索配置。切换这些配置后，已有资料需要重新处理才能用于新配置下的检索；原文件和已有数据会保留。仅更换 API 密钥无需因此重新处理。

即使服务商使用相同的模型名称和维度，也不能仅凭这两项判断生成的向量可以混用。FAISS 和 LightRAG 的 API 文件索引按配置分别存储；Milvus 使用配置的集合名称，遇到已有集合维度不匹配时，需要选择新集合后重新处理资料。

需要在自己的电脑或服务器上运行检索模型时，使用 `requirements-local.txt` 安装额外依赖，并准备模型权重。操作步骤见 [运行本地模型](../README.md#可选运行本地模型)。默认 API 模式使用 `requirements.txt`，开发测试使用 `requirements-dev.txt`。
