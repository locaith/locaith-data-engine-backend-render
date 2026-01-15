from fastapi import APIRouter, HTTPException, Depends
from typing import List

from models.schemas import APIKeyCreate, APIKeyResponse, APIKeyCreated
from services.api_key_service import api_key_service
from routers.auth import get_current_user

router = APIRouter(prefix="/keys", tags=["API Keys"])

@router.post("/", response_model=APIKeyCreated)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new API key linked to a Document Space"""
    result = api_key_service.create_key(
        user_id=current_user["id"],
        name=key_data.name,
        scopes=[s.value for s in key_data.scopes],
        space_id=key_data.space_id,
        expires_in_days=key_data.expires_in_days
    )
    
    return APIKeyCreated(**result)

@router.get("/", response_model=List[APIKeyResponse])
async def list_api_keys(current_user: dict = Depends(get_current_user)):
    """List all API keys for current user"""
    keys = api_key_service.get_keys(current_user["id"])
    return [APIKeyResponse(**k) for k in keys]

@router.get("/{key_id}/usage")
async def get_key_usage(
    key_id: str,
    days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    """Get usage statistics for an API key"""
    usage = api_key_service.get_key_usage(key_id, current_user["id"], days)
    if not usage:
        raise HTTPException(status_code=404, detail="API key không tồn tại")
    
    return usage

@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Revoke an API key"""
    success = api_key_service.revoke_key(key_id, current_user["id"])
    if not success:
        raise HTTPException(status_code=404, detail="API key không tồn tại")
    
    return {"message": "API key đã được thu hồi"}
