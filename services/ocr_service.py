"""
OCR Service - Extract text from scanned PDFs and images
Critical for documents like 35-KHĐU.pdf that are image-based

Uses multiple fallback methods:
1. pdf2image + pytesseract (basic OCR)
2. Google Vision AI (if available)
3. Gemini Vision for complex layouts
"""

import os
import tempfile
from typing import Optional, Dict, Any, List
from pathlib import Path
import base64


class OCRService:
    """
    Production-grade OCR service for scanned documents
    Handles: scanned PDFs, images, complex layouts
    """
    
    def __init__(self):
        self.tesseract_available = self._check_tesseract()
        self.gemini_available = self._check_gemini()
        print(f"[OCR] Tesseract: {self.tesseract_available}, Gemini Vision: {self.gemini_available}")
    
    def _check_tesseract(self) -> bool:
        """Check if pytesseract is available"""
        try:
            import pytesseract
            # Try to get version to verify it works
            pytesseract.get_tesseract_version()
            return True
        except:
            return False
    
    def _check_gemini(self) -> bool:
        """Check if Gemini Vision is available"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            return api_key is not None and len(api_key) > 10
        except:
            return False
    
    async def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a potentially scanned PDF
        
        Returns:
            {
                "success": bool,
                "text": str,
                "pages": [{page: int, text: str}],
                "method": "tesseract" | "gemini" | "fallback"
            }
        """
        # Try different methods in order
        
        # Method 1: Gemini Vision (best for complex layouts)
        if self.gemini_available:
            result = await self._ocr_with_gemini(pdf_path)
            if result and result.get("success"):
                return result
        
        # Method 2: Tesseract (traditional OCR)
        if self.tesseract_available:
            result = await self._ocr_with_tesseract(pdf_path)
            if result and result.get("success"):
                return result
        
        # Fallback: Return error
        return {
            "success": False,
            "error": "Không có OCR engine khả dụng. Cài đặt tesseract hoặc cấu hình GEMINI_API_KEY.",
            "text": "",
            "pages": []
        }
    
    async def _ocr_with_tesseract(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """OCR using pytesseract"""
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=200)
            
            pages = []
            all_text = []
            
            for page_num, image in enumerate(images, 1):
                # OCR with Vietnamese language
                text = pytesseract.image_to_string(image, lang='vie+eng')
                
                if text.strip():
                    pages.append({"page": page_num, "text": text.strip()})
                    all_text.append(f"[Trang {page_num}]\n{text.strip()}")
            
            return {
                "success": True,
                "text": "\n\n".join(all_text),
                "pages": pages,
                "method": "tesseract",
                "total_pages": len(images)
            }
            
        except Exception as e:
            print(f"[OCR] Tesseract error: {e}")
            return None
    
    async def _ocr_with_gemini(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """OCR using Gemini Vision API"""
        try:
            # Convert PDF pages to images
            from pdf2image import convert_from_path
            
            images = convert_from_path(pdf_path, dpi=150)  # Lower DPI for API
            
            pages = []
            all_text = []
            
            for page_num, image in enumerate(images[:10], 1):  # Limit to 10 pages
                # Convert to base64
                import io
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Call Gemini Vision
                text = await self._call_gemini_vision(image_data)
                
                if text:
                    pages.append({"page": page_num, "text": text})
                    all_text.append(f"[Trang {page_num}]\n{text}")
            
            if all_text:
                return {
                    "success": True,
                    "text": "\n\n".join(all_text),
                    "pages": pages,
                    "method": "gemini_vision",
                    "total_pages": len(images)
                }
            
            return None
            
        except Exception as e:
            print(f"[OCR] Gemini Vision error: {e}")
            return None
    
    async def _call_gemini_vision(self, image_base64: str) -> Optional[str]:
        """Call Gemini Vision API for OCR"""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            
            # Use vision-capable model
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            # Create image part
            image_part = {
                "mime_type": "image/png",
                "data": image_base64
            }
            
            prompt = """Bạn là OCR expert. Hãy trích xuất CHÍNH XÁC từng chữ trong ảnh này.

QUY TẮC:
1. Giữ nguyên format, xuống dòng như văn bản gốc
2. KHÔNG thêm bất kỳ thông tin nào không có trong ảnh
3. Nếu có bảng, format thành dạng text rõ ràng với | ngăn cách
4. Chỉ trả về văn bản trích xuất, không giải thích

VĂN BẢN TRONG ẢNH:"""
            
            response = model.generate_content([prompt, image_part])
            
            return response.text.strip()
            
        except Exception as e:
            print(f"[OCR] Gemini Vision API error: {e}")
            return None
    
    async def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract text from a single image"""
        # Try Gemini first
        if self.gemini_available:
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
            except Exception as e:
                print(f"[OCR] Image Gemini error: {e}")
        
        # Fallback to tesseract
        if self.tesseract_available:
            try:
                from PIL import Image
                import pytesseract
                
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image, lang='vie+eng')
                
                return {
                    "success": True,
                    "text": text.strip(),
                    "method": "tesseract"
                }
            except Exception as e:
                print(f"[OCR] Image tesseract error: {e}")
        
        return {
            "success": False,
            "error": "OCR không khả dụng",
            "text": ""
        }


# Singleton instance
ocr_service = OCRService()
