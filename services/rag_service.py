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
        self.model_name = "gemini-2.0-flash"
        self._init_client()
    
    def _init_client(self):
        """Initialize Gemini client with proper SDK"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[RAG Service] No GEMINI_API_KEY found")
            return
        
        try:
            # Try google-generativeai SDK first
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_name)
            self.use_genai = True
            print(f"[RAG Service] Initialized with google-generativeai, model: {self.model_name}")
        except ImportError:
            try:
                # Fallback to google.genai SDK
                from google import genai
                self.client = genai.Client(api_key=api_key)
                self.use_genai = False
                print(f"[RAG Service] Initialized with google.genai, model: {self.model_name}")
            except Exception as e:
                print(f"[RAG Service] Init error: {e}")
    
    def is_available(self) -> bool:
        """Check if service is available"""
        return self.client is not None
    
    def _generate(self, prompt: str) -> str:
        """Generate content with proper SDK"""
        if self.use_genai:
            response = self.client.generate_content(prompt)
            return response.text
        else:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
    
    async def query(
        self, 
        question: str, 
        file_paths: List[str],
        space_name: str = "Document Space"
    ) -> Dict[str, Any]:
        """
        Query AI with context from ALL documents
        """
        if not self.is_available():
            return {
                "answer": "❌ AI service chưa được cấu hình. Vui lòng kiểm tra GEMINI_API_KEY.",
                "sources": []
            }
        
        # Read ALL document contents (no limit)
        context_parts = []
        sources = []
        all_docs = []  # Track all documents for listing
        
        for file_path in file_paths:  # Process ALL files, no [:5] limit
            try:
                file_name = Path(file_path).stem  # Get name without extension
                content = await self._read_file_content(file_path)
                
                # Track all documents
                doc_info = {"file": file_name, "path": file_path, "has_content": content is not None}
                all_docs.append(doc_info)
                
                if content:
                    # Limit content per file to avoid token overflow, but still include all files
                    max_chars = min(8000, 50000 // max(len(file_paths), 1))
                    context_parts.append(f"=== {file_name} ===\n{content[:max_chars]}")
                    sources.append({"file": file_name, "path": file_path})
            except Exception as e:
                print(f"[RAG] Error reading {file_path}: {e}")
                all_docs.append({"file": Path(file_path).stem, "path": file_path, "has_content": False, "error": str(e)})
        
        if not context_parts:
            return {
                "answer": "⚠️ Không thể đọc nội dung các file trong space này.",
                "sources": []
            }
        
        # Build improved prompt with document listing
        context = "\n\n".join(context_parts)
        doc_list = "\n".join([f"- {d['file']}" for d in all_docs])
        
        prompt = f"""Bạn là AI assistant cho "{space_name}". Trả lời câu hỏi dựa trên nội dung tài liệu được cung cấp.

DANH SÁCH TÀI LIỆU ({len(all_docs)} files):
{doc_list}

NỘI DUNG TÀI LIỆU:
{context}

CÂU HỎI: {question}

YÊU CẦU:
1. Trả lời chính xác dựa trên nội dung tài liệu
2. Nếu hỏi về danh sách tài liệu, liệt kê ĐẦY ĐỦ tất cả {len(all_docs)} file
3. Nếu không tìm thấy thông tin, nói rõ ràng
4. Format câu trả lời dễ đọc, dùng bullet points khi cần"""

        try:
            answer = self._generate(prompt)
            
            return {
                "answer": answer,
                "sources": sources,
                "total_documents": len(all_docs)
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
