# 第三方许可原文与来源

本目录提供第三方许可原文与来源记录，覆盖 Windows / Python 3.11 的 API、开发依赖，以及按前端锁文件安装的包。`python-packages.json` 和 `javascript-packages.json` 记录版本、来源、包元数据和许可文本的 SHA-256。

- `python/`：实际安装分发包提供的 LICENSE、COPYING、NOTICE、AUTHORS 文本，包含开发测试工具。包文件名末尾统一加 `.txt`，不改变内容。
- `javascript/`：实际安装前端包根目录声明，以及 Next 自带的 `dist/compiled` 声明。
- `lucide.txt`：完整 Lucide ISC 和继承 Feather MIT 声明。
- `tokenizers-apache-2.0.txt`：Tokenizers v0.22.2 上游 Apache-2.0 完整许可；该 wheel 未携带单独许可文件，来源为 https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.2/LICENSE
- `bge-m3-notice.md` / `bge-m3-model-card.txt`：固定版本分词资产的来源、哈希与上游 MIT 声明。
- `cc-by-4.0.txt`：Solar 等适用素材的 Creative Commons Attribution 4.0 International 法律文本。作者和作品链接见 [第三方声明](../../THIRD_PARTY_NOTICES.md)。
- `minio-agpl-3.0.txt` / `neo4j-gpl-3.0.txt`：Compose 引用的独立服务上游许可。
- `upstream-sources.json`：官方文本来源与获取记录。
- `upstream/` 和 `supplemental-license-sources.json`：wheel 未单独提供许可文件时补充的上游原文与获取记录。LangChain Core、Text Splitters、LangSmith 和 Tokenizers 的来源为对应版本标签；LangChain Community 与 nano-vectordb 的来源为上游主分支许可，并附安装包的许可元数据。

前端清单标记了未安装的其他平台包。空许可文件列表表示安装包没有提供标准名称的许可文件，仍需遵守该组件的许可。清单不完整覆盖 wheel 内部原生组件、可选本地模型或 Linux 容器中的依赖。

再分发已安装依赖或构建产物时必须保留它们自带的全部适用声明，本目录不能替代最终分发产物的许可核查。项目原创代码的 MIT 不覆盖第三方文件。
