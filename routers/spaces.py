"""
Document Spaces Router - LlamaIndex-style API flow
Upload data → Test AI → Generate API Key
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid

from database import get_db
from routers.auth import get_current_user

router = APIRouter(prefix="/spaces", tags=["Document Spaces"])


# ===== Schemas =====
class SpaceCreate(BaseModel):
    name: str
    description: Optional[str] = None

class SpaceResponse(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str]
    status: str
    file_count: int
    total_size_mb: float
    created_at: datetime
    
class SpaceWithFiles(SpaceResponse):
    files: List[dict] = []

class ChatRequest(BaseModel):
    question: str
    
class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []


# ===== CRUD Endpoints =====

@router.get("/", response_model=List[SpaceResponse])
async def list_spaces(current_user: dict = Depends(get_current_user)):
    """List all document spaces for current user"""
    with get_db() as conn:
        result = conn.execute("""
            SELECT id, user_id, name, description, status, file_count, total_size_mb, created_at
            FROM document_spaces
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, [current_user["id"]]).fetchall()
        
        return [
            SpaceResponse(
                id=row[0], user_id=row[1], name=row[2], 
                description=row[3], status=row[4],
                file_count=row[5], total_size_mb=row[6], 
                created_at=row[7]
            ) for row in result
        ]


@router.post("/", response_model=SpaceResponse, status_code=status.HTTP_201_CREATED)
async def create_space(space: SpaceCreate, current_user: dict = Depends(get_current_user)):
    """Create a new document space"""
    space_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    with get_db() as conn:
        conn.execute("""
            INSERT INTO document_spaces (id, user_id, name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [space_id, current_user["id"], space.name, space.description, now, now])
        
        return SpaceResponse(
            id=space_id,
            user_id=current_user["id"],
            name=space.name,
            description=space.description,
            status="active",
            file_count=0,
            total_size_mb=0,
            created_at=now
        )


@router.get("/{space_id}", response_model=SpaceWithFiles)
async def get_space(space_id: str, current_user: dict = Depends(get_current_user)):
    """Get space details with files"""
    with get_db() as conn:
        # Get space
        space = conn.execute("""
            SELECT id, user_id, name, description, status, file_count, total_size_mb, created_at
            FROM document_spaces
            WHERE id = ? AND user_id = ?
        """, [space_id, current_user["id"]]).fetchone()
        
        if not space:
            raise HTTPException(status_code=404, detail="Space not found")
        
        # Get files in space
        files = conn.execute("""
            SELECT id, name, file_type, file_size, row_count, created_at
            FROM datasets
            WHERE space_id = ?
            ORDER BY created_at DESC
        """, [space_id]).fetchall()
        
        return SpaceWithFiles(
            id=space[0], user_id=space[1], name=space[2],
            description=space[3], status=space[4],
            file_count=space[5], total_size_mb=space[6],
            created_at=space[7],
            files=[{
                "id": f[0], "name": f[1], "file_type": f[2],
                "file_size": f[3], "row_count": f[4], "created_at": str(f[5])
            } for f in files]
        )


@router.delete("/{space_id}")
async def delete_space(space_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a document space and all its files"""
    with get_db() as conn:
        # Verify ownership
        space = conn.execute("""
            SELECT id FROM document_spaces WHERE id = ? AND user_id = ?
        """, [space_id, current_user["id"]]).fetchone()
        
        if not space:
            raise HTTPException(status_code=404, detail="Space not found")
        
        # Delete files in space
        conn.execute("DELETE FROM datasets WHERE space_id = ?", [space_id])
        
        # Delete space
        conn.execute("DELETE FROM document_spaces WHERE id = ?", [space_id])
        
        return {"message": "Space deleted successfully"}


# ===== AI Chat Test Endpoint =====

@router.post("/{space_id}/chat", response_model=ChatResponse)
async def chat_with_space(
    space_id: str, 
    chat: ChatRequest, 
    current_user: dict = Depends(get_current_user)
):
    """Test AI chat with documents in this space - Uses Gold Layer for 100% accuracy"""
    from services.rag_service import rag_service
    
    with get_db() as conn:
        # Verify ownership
        space = conn.execute("""
            SELECT id, name FROM document_spaces WHERE id = ? AND user_id = ?
        """, [space_id, current_user["id"]]).fetchone()
        
        if not space:
            raise HTTPException(status_code=404, detail="Space not found")
        
        # Get file paths in this space
        files = conn.execute("""
            SELECT file_path FROM datasets WHERE space_id = ?
        """, [space_id]).fetchall()
        
        if not files:
            return ChatResponse(
                answer="⚠️ Space này chưa có file nào. Vui lòng upload file trước khi test AI.",
                sources=[]
            )
        
        file_paths = [f[0] for f in files]
    
    # Query RAG with Gold Layer support
    try:
        result = await rag_service.query(
            question=chat.question,
            file_paths=file_paths,
            space_name=space[1],
            space_id=space_id  # Enable Gold Layer routing
        )
        
        answer = result.get("answer", "Không tìm thấy câu trả lời.")
        
        # Add source indicator (plain text, no markdown)
        if result.get("is_gold_query"):
            answer += "\n\n📊 Nguồn: Gold Layer SQL (100% chính xác)"
        
        return ChatResponse(
            answer=answer,
            sources=result.get("sources", [])
        )
    except Exception as e:
        return ChatResponse(
            answer=f"❌ Lỗi AI: {str(e)}",
            sources=[]
        )

