"""
Authentication endpoints.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/test")
async def auth_test():
    return {"message": "Auth endpoints placeholder"}
