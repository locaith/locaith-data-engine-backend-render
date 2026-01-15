from fastapi import APIRouter, HTTPException, Depends, Header, Request
from typing import Optional
from pydantic import BaseModel

from routers.auth import get_current_user
from services.lakehouse import lakehouse_service
from services.ai_service import ai_service
from services.api_key_service import api_key_service
import json

router = APIRouter(prefix="/rag", tags=["RAG - AI Assistant"])

# API Key Authentication Dependency - reads from middleware or validates header
async def get_api_key_user(request: Request, x_api_key: str = Header(None, alias="X-API-Key")):
    """Validate API Key from header and return user info"""
    # First check if middleware already validated
    if hasattr(request.state, 'api_key_info') and request.state.api_key_info:
        return request.state.api_key_info
    
    # Fallback: validate header directly
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API Key không được cung cấp. Vui lòng thêm header X-API-Key"
        )
    key_info = api_key_service.validate_key(x_api_key)
    if not key_info:
        raise HTTPException(
            status_code=401,
            detail="API Key không hợp lệ hoặc đã hết hạn"
        )
    return key_info

class RAGQuery(BaseModel):
    question: str
    dataset_id: Optional[str] = None  # Nếu None, tìm trong tất cả datasets

class RAGQueryExternal(BaseModel):
    question: str
    system_prompt: Optional[str] = None  # Custom system prompt cho chatbot
    dataset_id: Optional[str] = None

class RAGResponse(BaseModel):
    answer: str
    sources: list
    confidence: float

@router.post("/query")
async def query_rag(
    query: RAGQuery,
    current_user: dict = Depends(get_current_user)
):
    """
    Truy vấn AI với context từ datasets đã upload
    AI sẽ trả lời dựa trên nội dung file
    """
    if not ai_service.is_available():
        raise HTTPException(
            status_code=503, 
            detail="AI service chưa được cấu hình. Vui lòng set GEMINI_API_KEY."
        )
    
    # Lấy datasets của user
    datasets = lakehouse_service.get_datasets(current_user["id"])
    
    if not datasets:
        raise HTTPException(
            status_code=400,
            detail="Chưa có dataset nào. Vui lòng upload file trước."
        )
    
    # Lọc theo dataset_id nếu có
    if query.dataset_id:
        datasets = [ds for ds in datasets if ds["id"] == query.dataset_id]
        if not datasets:
            raise HTTPException(status_code=404, detail="Dataset không tồn tại")
    
    # Lấy preview data từ các datasets làm context
    context_parts = []
    sources = []
    
    for ds in datasets[:10]:  # Tăng lên 10 datasets cho context rộng hơn
        preview = lakehouse_service.preview_dataset(ds["id"], current_user["id"], limit=100)
        if preview:
            context_parts.append(f"""
--- Dataset: {ds['name']} (Loại: {ds['file_type']}, {ds['row_count']} dòng) ---
Dữ liệu:
{json.dumps(preview['data'][:100], ensure_ascii=False, indent=2)}
""")
            sources.append({
                "dataset_id": ds["id"],
                "name": ds["name"],
                "file_type": ds["file_type"],
                "row_count": ds["row_count"]
            })
    
    context = "\n".join(context_parts)
    
    # Gọi AI để trả lời
    try:
        from google.genai.types import GenerateContentConfig
        
        prompt = f"""Bạn là trợ lý AI thông minh của Locaith Data Engine - một hệ thống Data Lakehouse chuyên nghiệp.

NHIỆM VỤ: Trả lời câu hỏi của người dùng một cách CHÍNH XÁC 100% dựa trên dữ liệu được cung cấp bên dưới.

=== DỮ LIỆU CONTEXT ===
{context}
=== KẾT THÚC DỮ LIỆU ===

CÂU HỎI: {query.question}

NGUYÊN TẮC TRẢ LỜI (TUÂN THỦ NGHIÊM NGẶT):
1. CHỈ trả lời dựa trên thông tin có trong DỮ LIỆU CONTEXT ở trên
2. Nếu thông tin KHÔNG CÓ trong context, phải nói rõ: "Không tìm thấy thông tin này trong dữ liệu đã upload"
3. Trích dẫn TÊN DATASET cụ thể khi đưa ra thông tin quan trọng
4. Với số liệu (tiền, ngày tháng, số lượng): trích xuất CHÍNH XÁC từ dữ liệu, KHÔNG làm tròn hay ước lượng
5. Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu
6. KHÔNG bịa đặt, KHÔNG suy luận ngoài dữ liệu

ĐỊNH DẠNG TRẢ LỜI:
- Trả lời trực tiếp câu hỏi trước
- Nêu nguồn dữ liệu (tên dataset) sau
- Giữ câu trả lời ngắn gọn nhưng đầy đủ

TRẢ LỜI:"""

        response = ai_service.client.models.generate_content(
            model=ai_service.model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2000
            )
        )
        
        answer = response.text.strip()
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": 0.95,
            "question": query.question
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi gọi AI: {str(e)}"
        )

@router.get("/datasets")
async def get_available_datasets(current_user: dict = Depends(get_current_user)):
    """Lấy danh sách datasets có thể query"""
    datasets = lakehouse_service.get_datasets(current_user["id"])
    return {
        "datasets": [
            {
                "id": ds["id"],
                "name": ds["name"],
                "file_type": ds["file_type"],
                "row_count": ds["row_count"]
            }
            for ds in datasets
        ],
        "total": len(datasets)
    }

# ============ EXTERNAL API (for third-party apps like phechat.com) ============

@router.post("/external/query")
async def query_rag_external(
    query: RAGQueryExternal,
    api_key_info: dict = Depends(get_api_key_user)
):
    """
    RAG Query dùng API Key thay vì JWT Token.
    Dành cho bên thứ 3 như phechat.com gọi API.
    
    Headers:
        X-API-Key: LcAi_xxx...
    
    Body:
        question: Câu hỏi cần trả lời
        system_prompt: (optional) Custom system prompt cho AI
        dataset_id: (optional) ID dataset cụ thể
    """
    if not ai_service.is_available():
        raise HTTPException(
            status_code=503, 
            detail="AI service chưa được cấu hình."
        )
    
    api_key_id = api_key_info["key_id"]
    
    # Get datasets isolated by API Key (NOT user_id - important for commercial!)
    datasets = lakehouse_service.get_datasets_by_api_key(api_key_id)
    
    if not datasets:
        raise HTTPException(
            status_code=400,
            detail="Chưa có dataset nào. Vui lòng upload file trước."
        )
    
    # Lọc theo dataset_id nếu có
    if query.dataset_id:
        datasets = [ds for ds in datasets if ds["id"] == query.dataset_id]
        if not datasets:
            raise HTTPException(status_code=404, detail="Dataset không tồn tại")
    
    # Lấy context từ datasets
    context_parts = []
    sources = []
    
    for ds in datasets[:10]:
        preview = lakehouse_service.preview_dataset_by_api_key(ds["id"], api_key_id, limit=100)
        if preview and 'data' in preview:
            context_parts.append(f"""
--- Dataset: {ds['name']} (Loại: {ds['file_type']}, {ds['row_count']} dòng) ---
Dữ liệu:
{json.dumps(preview['data'][:100], ensure_ascii=False, indent=2)}
""")
            sources.append({
                "dataset_id": ds["id"],
                "name": ds["name"],
                "file_type": ds["file_type"],
                "row_count": ds["row_count"]
            })
    
    context = "\n".join(context_parts)
    
    # Build prompt với custom system_prompt nếu có
    default_system = f"""Bạn là trợ lý AI thông minh của Locaith Data Engine.

NHIỆM VỤ: Trả lời câu hỏi CHÍNH XÁC 100% dựa trên dữ liệu được cung cấp.

=== DỮ LIỆU CONTEXT ===
{context}
=== KẾT THÚC DỮ LIỆU ==="""

    if query.system_prompt:
        # Custom system prompt từ Document Space
        full_prompt = f"""{query.system_prompt}

=== DỮ LIỆU TỪ LOCAITH DATA ENGINE ===
{context}
=== KẾT THÚC DỮ LIỆU ===

CÂU HỎI: {query.question}

TRẢ LỜI:"""
    else:
        full_prompt = f"""{default_system}

CÂU HỎI: {query.question}

NGUYÊN TẮC TRẢ LỜI:
1. CHỈ trả lời dựa trên thông tin có trong context
2. Nếu không có thông tin, nói rõ "Không tìm thấy"
3. Trích dẫn nguồn dataset khi đưa ra thông tin
4. Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu

TRẢ LỜI:"""

    try:
        from google.genai.types import GenerateContentConfig
        
        response = ai_service.client.models.generate_content(
            model=ai_service.model,
            contents=full_prompt,
            config=GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2000
            )
        )
        
        return {
            "answer": response.text.strip(),
            "sources": sources,
            "confidence": 0.95,
            "question": query.question
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi gọi AI: {str(e)}"
        )

@router.get("/external/datasets")
async def get_datasets_external(api_key_info: dict = Depends(get_api_key_user)):
    """Lấy danh sách datasets dùng API Key (isolated)"""
    api_key_id = api_key_info["key_id"]
    datasets = lakehouse_service.get_datasets_by_api_key(api_key_id)
    return {
        "datasets": [
            {
                "id": ds["id"],
                "name": ds["name"],
                "file_type": ds["file_type"],
                "row_count": ds["row_count"]
            }
            for ds in datasets
        ],
        "total": len(datasets)
    }
