# BGE-M3 分词文件

API 分块使用 [BAAI/BGE-M3](https://huggingface.co/BAAI/bge-m3) 的 tokenizer.json，来自 Beijing Academy of Artificial Intelligence 的官方模型仓库。

- 固定提交：`5617a9f61b028005a4858fdac845db406aefb181`
- 文件 SHA-256：`21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08`
- 官方模型卡声明许可：MIT，原文保存在 [bge-m3-model-card.txt](bge-m3-model-card.txt)
- 文件仅在部署准备时下载，未修改词表；运行时关闭填充和截断，不添加特殊 token 进行长度计数
- 源码仓库不附带模型权重，项目 MIT 不重新授予第三方资产的权利

若另行分发该分词文件，保留上游许可声明和本来源记录；自定义模型或词表需检查其自身许可。
