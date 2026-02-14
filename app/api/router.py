from fastapi import APIRouter

from .endpoints import exams, generate
from .endpoints_v2 import generate as generate_v2

api = APIRouter()
api.include_router(generate.router)
api.include_router(exams.router)
api.include_router(router=generate_v2.router, prefix="/v2")
