from fastapi import APIRouter, HTTPException, Depends
from typing import List

from models.schemas import QueryRequest, QueryResult
from services.lakehouse import lakehouse_service
from services.billing_service import billing_service
from routers.auth import get_current_user

router = APIRouter(prefix="/query", tags=["Query Engine"])

@router.post("/execute", response_model=QueryResult)
async def execute_query(
    query: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Execute a SQL query"""
    # Check query limits
    limits = billing_service.check_limits(current_user["id"], current_user["plan"])
    if not limits["queries_ok"]:
        raise HTTPException(
            status_code=429, 
            detail="Bạn đã đạt giới hạn queries trong tháng. Vui lòng nâng cấp gói dịch vụ."
        )
    
    result, status_code = lakehouse_service.execute_query(query.sql, current_user["id"])
    
    if status_code != 200:
        raise HTTPException(status_code=400, detail=result.get("error", "Query error"))
    
    return QueryResult(**result)

@router.get("/history")
async def get_query_history(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get query history"""
    history = lakehouse_service.get_query_history(current_user["id"], limit)
    return history

@router.get("/tables")
async def list_tables(current_user: dict = Depends(get_current_user)):
    """List available tables (datasets) for querying"""
    datasets = lakehouse_service.get_datasets(current_user["id"])
    return [
        {
            "table_name": ds["name"].replace(" ", "_").lower(),
            "original_name": ds["name"],
            "row_count": ds["row_count"],
            "schema": ds["schema_json"]
        }
        for ds in datasets
    ]
