"""
User management endpoints.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/test")
async def users_test():
    return {"message": "Users endpoints placeholder"}
