from fastapi import APIRouter

from .endpoints import generate

api = APIRouter()
api.include_router(generate.router)
