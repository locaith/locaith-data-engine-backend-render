from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any

from routers.auth import get_current_user
from services.lakehouse import lakehouse_service
from services.ai_service import ai_service

router = APIRouter(prefix="/ai", tags=["AI Verification"])

@router.get("/status")
async def get_ai_status():
    """Check if AI service is available"""
    return {
        "available": ai_service.is_available(),
        "model": "gemini-3-flash-preview",
        "message": "AI service sẵn sàng" if ai_service.is_available() else "Chưa cấu hình GEMINI_API_KEY"
    }

@router.post("/verify/{dataset_id}")
async def verify_dataset(
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Verify data quality of a dataset using AI"""
    # Get dataset
    dataset = lakehouse_service.get_dataset(dataset_id, current_user["id"])
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset không tồn tại")
    
    # Get preview data
    preview = lakehouse_service.preview_dataset(dataset_id, current_user["id"], limit=50)
    if not preview:
        raise HTTPException(status_code=400, detail="Không thể đọc dữ liệu dataset")
    
    # Get schema
    import json
    schema = json.loads(dataset.get("schema_json", "{}"))
    
    # Verify with AI
    result = await ai_service.verify_data_quality(preview["data"], schema)
    
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset["name"],
        "row_count": dataset["row_count"],
        "verification": result
    }

@router.post("/normalize/{dataset_id}")
async def normalize_dataset(
    dataset_id: str,
    target_schema: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Normalize dataset to match target schema using AI"""
    # Get dataset
    dataset = lakehouse_service.get_dataset(dataset_id, current_user["id"])
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset không tồn tại")
    
    # Get preview data
    preview = lakehouse_service.preview_dataset(dataset_id, current_user["id"], limit=50)
    if not preview:
        raise HTTPException(status_code=400, detail="Không thể đọc dữ liệu dataset")
    
    # Normalize with AI
    result = await ai_service.normalize_data(preview["data"], target_schema)
    
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset["name"],
        "normalization": result
    }

@router.post("/analyze-pdf")
async def analyze_pdf_content(
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Analyze PDF content using AI for better extraction"""
    dataset = lakehouse_service.get_dataset(dataset_id, current_user["id"])
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset không tồn tại")
    
    if dataset["file_type"] != "pdf":
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ phân tích file PDF")
    
    # Get the original PDF path (stored in parquet but we need original)
    # For now, return the extracted data preview
    preview = lakehouse_service.preview_dataset(dataset_id, current_user["id"], limit=100)
    
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset["name"],
        "extracted_data": preview,
        "ai_analysis": {
            "available": ai_service.is_available(),
            "message": "Dữ liệu đã được trích xuất từ PDF thành công"
        }
    }
