# 文档入口

- [项目介绍](../README.md)：项目定位、适用场景、主要功能和使用流程
- [安装与启动](../README.md#安装与启动)：首次配置、Docker 启动、本地开发
- [模型配置](model-api.md)：API 参数、本地模型与已有资料处理
- [部署与配置](deployment-security.md)：密码设置、配置保存和公网部署
- [PDF 与文档分块](pdf-processing.md)：解析方式、分词文件和常见问题
- [测试与评测](evaluation.md)：运行测试、准备数据和生成 PDF 原文快照
- [第三方许可](../THIRD_PARTY_NOTICES.md)：MIT 范围、组件许可与素材署名

代码阅读可从 [后端入口](../backend/app/main.py)、[生命周期](../backend/app/runtime/lifespan.py)、[API 路由](../backend/app/api/routes/) 和 [前端应用](../frontend/app/) 开始。
