"""
Study Assistant — 主入口

当前文件是后端进程入口，负责在真正启动 FastAPI 之前完成最小启动准备。

需要特别区分：
  - 这里初始化的是业务 PostgreSQL 相关表和前端会话展示库
  - LangGraph 官方 PostgreSQL checkpointer 不在这里直接创建
  - checkpointer 的 compile 入口在 `agents/graph_factory.py`，
    生命周期托管在 `api/agent_singleton.py`

运行：
  python -m app.main
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

# 导入 os，用于在导入 aiohttp/ssl 前清理可能污染环境的变量。
import os

# 导入运行时辅助函数；Windows 下需要尽早切换事件循环策略，避免 psycopg 侧兼容问题。
from app.runtime import configure_windows_event_loop_policy, get_uvicorn_loop_config

# 在导入更多异步/网络依赖前优先设置事件循环策略。
configure_windows_event_loop_policy()

# 某些本地抓包环境会注入不可写的 SSLKEYLOGFILE，导致 aiohttp/ssl 导入期失败。
os.environ.pop("SSLKEYLOGFILE", None)

# 导入 uvicorn，供本地直接 `python -m app.main` 启动服务时使用。
import uvicorn

# 导入 FastAPI app 实例；真正的路由、生命周期与依赖注入都在 `server.py`。
from app.api.server import app
# 导入配置读取函数，统一从 settings 取 host/port/debug 等运行参数。
from app.config import get_settings

# 读取全局配置，供启动打印和 uvicorn 参数复用。
settings = get_settings()

# 只有直接运行本文件时才启动内置 uvicorn；被 `uvicorn app.main:app` 导入时不会进入这里。
if __name__ == "__main__":
    # 这里只服务本地直接启动调试；生产或标准开发流程通常由 `uvicorn app.main:app` 负责导入 app。
    # 打印基础访问地址，便于本地开发时快速确认端口。
    print(f"Starting Study Assistant on http://{settings.host}:{settings.port}")
    # docs 地址固定指向本机文档页，便于开发环境快速打开。
    print(f"API Docs:  http://127.0.0.1:{settings.port}/docs")
    # 交给 uvicorn 启动 FastAPI 应用；reload 仅在 debug 模式下开启。
    uvicorn.run(
        # 使用导入字符串而不是对象直传，便于 reload 模式重新加载模块。
        "app.main:app",
        # 绑定监听地址。
        host=settings.host,
        # 绑定监听端口。
        port=settings.port,
        # 调试模式下开启热重载。
        reload=settings.debug,
        # Windows 下强制使用 Selector 事件循环，避免 psycopg async 与 Proactor 不兼容。
        loop=get_uvicorn_loop_config(),
    )
