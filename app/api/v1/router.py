from fastapi import APIRouter

from app.api.v1.endpoints import admin, agent, analytics, chat, documents, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["Sağlık"])
api_router.include_router(documents.router, prefix="/documents", tags=["Belgeler"])
api_router.include_router(chat.router, prefix="/chat", tags=["Sohbet"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analitik"])
api_router.include_router(agent.router, prefix="/agent", tags=["Multi-Agent"])
api_router.include_router(admin.router, prefix="/admin", tags=["Yönetim"])
