from fastapi import APIRouter

from .endpoints import exams, generate

api = APIRouter()
api.include_router(generate.router)
api.include_router(exams.router)
