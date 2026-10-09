# 第三方组件与素材声明

StudyAssistant 的原创代码使用根目录 [MIT License](LICENSE)。第三方代码、依赖、图标、字体、模型与独立服务保留其原有版权和许可；本项目的 MIT 不重新授权这些内容。

## 图标素材

- **Solar** — 作者 **480 Design**，采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。[原作品](https://www.figma.com/community/file/1166831539721848736)及 [Iconify 图标目录](https://icon-sets.iconify.design/solar/)。本项目通过 Iconify 使用图标，以界面样式调整显示尺寸和颜色，未修改路径图形；作者未对本项目作认可或背书。
- **Lucide** — Copyright (c) 2026 Lucide Icons and Contributors，[项目与作品](https://lucide.dev/)，采用 ISC。其继承的 Feather 图标保留 Copyright (c) 2013-present Cole Bemis 及 MIT 条款。完整上游声明见 [Lucide/Feather 许可](docs/licenses/lucide.txt)。
- Iconify React 渲染库的许可与图标作品许可不同，不能用渲染库的 MIT 替代图标署名。网站 `/credits` 页面保留上述可访问署名。

## 文档处理与主要依赖

| 组件 | 用途及许可 |
| --- | --- |
| pypdf 6.9.1 | PDF 页数、文字、元数据和密码可读性；BSD-3-Clause |
| Pillow 12.3.0 | 实际解码 PDF 解析产物中的 PNG/JPEG；MIT-CMU，随 wheel 的其他组件声明一并保留 |
| Tokenizers 0.22.2 | 无模型权重的分词与计数；Apache-2.0 |
| BGE-M3 分词文件 | 仅在部署时准备 tokenizer.json；固定版本官方模型卡声明 MIT，来源见 [分词文件声明](docs/licenses/bge-m3-notice.md) |
| ReportLab 4.5.1 | 开发测试中生成样例 PDF；BSD 许可及其随包声明 |
| psycopg / psycopg-binary / psycopg-pool | PostgreSQL 客户端；LGPL-3.0-only，二进制包内其他组件保留各自声明 |
| certifi | CA 证书数据；MPL-2.0 |
| orjson | MPL-2.0 AND (Apache-2.0 OR MIT)，不同文件适用不同条款 |
| tqdm | MPL-2.0 AND MIT |
| caniuse-lite | 前端构建使用的浏览器兼容数据；CC BY 4.0 |

版本化依赖清单、随包许可和来源记录位于 [docs/licenses](docs/licenses/README.md)。清单覆盖 Windows API/开发环境，其他平台、可选本地模型及 Linux 镜像的许可需按实际使用的组件核对。

## 再分发

分发源码时保留本文件、MIT 及适用第三方声明。分发已安装依赖、构建产物或镜像时，也需保留其中的版权、完整许可及 bundled notices；本目录不能替代实际产物内的声明。

LGPL 库的再分发需遵守其源码提供、库修改和适用重链接权利等条件；MPL 覆盖文件的修改与分发遵守相应源码义务。未分发的构建工具与实际打包进浏览器的代码应按最终产物区分。

## 独立服务、模型与外部 API

- Compose 引用的 **MinIO** 服务为 AGPLv3；**Neo4j Community** 服务为 GPLv3。它们是独立服务，与本应用之间使用网络协议。对应镜像和服务自身的许可义务分别适用，不能仅因网络通信就将整个应用认定为同一许可。
- 分发镜像或对外提供修改过的 AGPL 服务时，应履行相应源码和声明义务。仓库里的 Compose 配置不是完整容器许可证清单。
- 可选本地模型代码、模型权重及远程模型/MinerU 服务受其各自许可或服务条款约束。
