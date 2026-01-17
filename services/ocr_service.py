"""
OCR Service - Extract text from scanned PDFs and images
Critical for documents like 35-KHĐU.pdf that are image-based

Uses PyMuPDF (fitz) instead of pdf2image - NO poppler dependency!
"""

import os
import io
from typing import Optional, Dict, Any, List
from pathlib import Path
import base64


class OCRService:
    """
    Production-grade OCR service for scanned documents
    Uses PyMuPDF for PDF→Image (no external dependencies)
    Uses Gemini Vision for accurate Vietnamese OCR
    """
    
    def __init__(self):
        self.gemini_available = self._check_gemini()
        self.fitz_available = self._check_fitz()
        print(f"[OCR] PyMuPDF: {self.fitz_available}, Gemini Vision: {self.gemini_available}")
    
    def _check_fitz(self) -> bool:
        """Check if PyMuPDF is available"""
        try:
            import fitz
            return True
        except:
            return False
    
    def _check_gemini(self) -> bool:
        """Check if Gemini Vision is available"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            available = api_key is not None and len(api_key) > 10
            if available:
                print(f"[OCR] GEMINI_API_KEY found: {api_key[:10]}...")
            else:
                print(f"[OCR] GEMINI_API_KEY not found or invalid")
            return available
        except Exception as e:
            print(f"[OCR] Gemini check error: {e}")
            return False
    
    async def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a scanned PDF using Gemini Vision
        """
        print(f"[OCR] Starting OCR for: {pdf_path}")
        
        # Check if Gemini is available
        if not self.gemini_available:
            return {
                "success": False,
                "error": "GEMINI_API_KEY chưa được cấu hình trên server. Vui lòng thêm vào Render Environment.",
                "text": "",
                "pages": []
            }
        
        # Check if PyMuPDF is available
        if not self.fitz_available:
            return {
                "success": False,
                "error": "PyMuPDF (fitz) chưa được cài đặt.",
                "text": "",
                "pages": []
            }
        
        try:
            import fitz  # PyMuPDF
            
            # Open PDF
            doc = fitz.open(pdf_path)
            pages = []
            all_text = []
            
            print(f"[OCR] PDF has {len(doc)} pages")
            
            for page_num in range(min(len(doc), 10)):  # Max 10 pages
                page = doc[page_num]
                
                # Render page to image
                pix = page.get_pixmap(dpi=150)
                img_data = pix.tobytes("png")
                image_base64 = base64.b64encode(img_data).decode('utf-8')
                
                print(f"[OCR] Processing page {page_num + 1}...")
                
                # OCR with Gemini Vision
                text = await self._call_gemini_vision(image_base64)
                
                if text:
                    pages.append({"page": page_num + 1, "text": text})
                    all_text.append(f"[Trang {page_num + 1}]\n{text}")
                    print(f"[OCR] Page {page_num + 1}: extracted {len(text)} chars")
            
            doc.close()
            
            if all_text:
                return {
                    "success": True,
                    "text": "\n\n".join(all_text),
                    "pages": pages,
                    "method": "gemini_vision",
                    "total_pages": len(doc)
                }
            else:
                return {
                    "success": False,
                    "error": "Không thể OCR bất kỳ trang nào",
                    "text": "",
                    "pages": []
                }
            
        except Exception as e:
            print(f"[OCR] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"OCR error: {str(e)}",
                "text": "",
                "pages": []
            }
    
    async def _call_gemini_vision(self, image_base64: str) -> Optional[str]:
        """Call Gemini Vision API for OCR"""
        try:
            # Try google-generativeai SDK first
            try:
                import google.generativeai as genai
                
                api_key = os.getenv("GEMINI_API_KEY")
                genai.configure(api_key=api_key)
                
                # Use latest Gemini 3 Flash model for superior OCR reasoning
                model = genai.GenerativeModel("gemini-3-flash-preview")
                
                # Create proper image part for vision
                image_part = {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_base64
                    }
                }
                
                prompt = """Bạn là OCR expert. Hãy trích xuất CHÍNH XÁC từng chữ trong ảnh này.

QUY TẮC:
1. Giữ nguyên format, xuống dòng như văn bản gốc
2. KHÔNG thêm bất kỳ thông tin nào không có trong ảnh
3. Nếu có bảng, format thành dạng text rõ ràng
4. Chỉ trả về văn bản trích xuất, không giải thích
5. Nếu ảnh rõ ràng, trích xuất 100% chính xác

VĂN BẢN:"""
                
                response = model.generate_content([prompt, image_part])
                text = response.text.strip()
                
                return text if text else None
                
            except Exception as sdk_error:
                print(f"[OCR] google-generativeai SDK error: {sdk_error}")
                
                # Fallback to google.genai
                try:
                    from google import genai
                    
                    api_key = os.getenv("GEMINI_API_KEY")
                    client = genai.Client(api_key=api_key)
                    
                    # Try with image bytes
                    image_bytes = base64.b64decode(image_base64)
                    
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=[
                            "Trích xuất CHÍNH XÁC toàn bộ văn bản trong ảnh này. Chỉ trả về văn bản, không giải thích:",
                            {"mime_type": "image/png", "data": image_bytes}
                        ]
                    )
                    
                    return response.text.strip() if response.text else None
                    
                except Exception as fallback_error:
                    print(f"[OCR] Fallback SDK error: {fallback_error}")
                    return None
            
        except Exception as e:
            print(f"[OCR] Gemini Vision API error: {e}")
            return None
    
    async def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract text from a single image"""
        if not self.gemini_available:
            return {
                "success": False,
                "error": "GEMINI_API_KEY chưa được cấu hình",
                "text": ""
            }
        
        try:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            text = await self._call_gemini_vision(image_data)
            if text:
                return {
                    "success": True,
                    "text": text,
                    "method": "gemini_vision"
                }
            return {
                "success": False,
                "error": "Không thể OCR ảnh",
                "text": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }


# Singleton instance
ocr_service = OCRService()
