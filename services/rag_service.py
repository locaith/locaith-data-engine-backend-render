"""
RAG Service - Retrieval Augmented Generation for Document Spaces
Query AI with context from uploaded documents
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class RAGService:
    """Simple RAG service using Gemini for document Q&A"""
    
    def __init__(self):
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Gemini client"""
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
                print("[RAG Service] Initialized successfully")
            else:
                print("[RAG Service] No GEMINI_API_KEY found")
        except Exception as e:
            print(f"[RAG Service] Init error: {e}")
    
    def is_available(self) -> bool:
        """Check if service is available"""
        return self.client is not None
    
    async def query(
        self, 
        question: str, 
        file_paths: List[str],
        space_name: str = "Document Space"
    ) -> Dict[str, Any]:
        """
        Query AI with context from documents
        """
        if not self.is_available():
            return {
                "answer": "❌ AI service chưa được cấu hình. Vui lòng kiểm tra GEMINI_API_KEY.",
                "sources": []
            }
        
        # Read document contents
        context_parts = []
        sources = []
        
        for file_path in file_paths[:5]:  # Limit to 5 files
            try:
                content = await self._read_file_content(file_path)
                if content:
                    file_name = Path(file_path).name
                    context_parts.append(f"=== {file_name} ===\n{content[:5000]}")  # Limit content
                    sources.append({"file": file_name, "path": file_path})
            except Exception as e:
                print(f"[RAG] Error reading {file_path}: {e}")
        
        if not context_parts:
            return {
                "answer": "⚠️ Không thể đọc nội dung các file trong space này.",
                "sources": []
            }
        
        # Build prompt
        context = "\n\n".join(context_parts)
        prompt = f"""Bạn là AI assistant cho "{space_name}". Trả lời câu hỏi dựa trên nội dung tài liệu được cung cấp.

TÀI LIỆU:
{context}

CÂU HỎI: {question}

Hãy trả lời dựa trên nội dung tài liệu trên. Nếu không tìm thấy thông tin, hãy nói rõ."""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            return {
                "answer": response.text,
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"❌ Lỗi khi gọi AI: {str(e)}",
                "sources": []
            }
    
    async def _read_file_content(self, file_path: str) -> Optional[str]:
        """Read content from various file types"""
        path = Path(file_path)
        
        if not path.exists():
            return None
        
        ext = path.suffix.lower()
        
        try:
            # Text files
            if ext in ['.txt', '.csv', '.json', '.md']:
                return path.read_text(encoding='utf-8', errors='ignore')[:10000]
            
            # PDF
            elif ext == '.pdf':
                import pdfplumber
                text = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages[:10]:  # Limit pages
                        text.append(page.extract_text() or "")
                return "\n".join(text)[:10000]
            
            # Word docs
            elif ext == '.docx':
                from docx import Document
                doc = Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
                return text[:10000]
            
            # Excel
            elif ext in ['.xlsx', '.xls']:
                import pandas as pd
                df = pd.read_excel(file_path, nrows=100)
                return df.to_string()[:10000]
            
            # PowerPoint
            elif ext == '.pptx':
                from pptx import Presentation
                prs = Presentation(file_path)
                text = []
                for slide in prs.slides[:10]:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            text.append(shape.text)
                return "\n".join(text)[:10000]
            
            # Parquet
            elif ext == '.parquet':
                import pandas as pd
                df = pd.read_parquet(file_path).head(100)
                return df.to_string()[:10000]
            
            else:
                return None
                
        except Exception as e:
            print(f"[RAG] Read error {file_path}: {e}")
            return None


# Singleton instance
rag_service = RAGService()
