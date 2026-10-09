"""聚合 API router。"""

from fastapi import APIRouter

from app.api.routes.auth_api import router as auth_router
from app.api.routes.chat_api import router as chat_router
from app.api.routes.citation_api import router as citation_router
from app.api.routes.conversation_api import router as conversation_router
from app.api.routes.graph_api import router as graph_router
from app.api.routes.graph_inspector_api import router as graph_inspector_router
from app.api.routes.knowledge_inspector_api import router as knowledge_inspector_router
from app.api.routes.llm_config_api import router as llm_config_router
from app.api.routes.observability_api import router as observability_router
from app.api.routes.library_api import router as library_router
from app.api.routes.papers_api import router as papers_router
from app.api.routes.project_api import router as project_router
from app.api.routes.rag_run_api import router as rag_run_router
from app.api.routes.research_api import router as research_router
from app.api.routes.research_memory_api import router as research_memory_router
from app.api.routes.stream_api import router as stream_router
from app.api.routes.system_api import router as system_router
from app.api.routes.task_api import router as task_router

router = APIRouter(prefix="/api/v1")
router.include_router(system_router)
router.include_router(auth_router)
router.include_router(library_router)
router.include_router(project_router)
router.include_router(research_router)
router.include_router(research_memory_router)
router.include_router(stream_router)
router.include_router(papers_router)
router.include_router(task_router)
router.include_router(chat_router)
router.include_router(citation_router)
router.include_router(conversation_router)
router.include_router(rag_run_router)
router.include_router(graph_router)
router.include_router(graph_inspector_router)
router.include_router(observability_router)
router.include_router(knowledge_inspector_router)
router.include_router(llm_config_router)
