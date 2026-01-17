"""
AI Verification Service - Using Gemini 2.0 Flash
Verifies and validates data quality using AI
"""
import os
from typing import Dict, Any, List, Optional

# Prioritize the new google-genai SDK (V2)
try:
    from google import genai
    from google.genai.types import GenerateContentConfig
    USE_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as google_generativeai
        USE_NEW_SDK = False
    except ImportError:
        USE_NEW_SDK = None

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

class AIVerificationService:
    def __init__(self):
        self.client = None
        self.model_name = "gemini-3-flash-preview"
        self.init_error = None
        
        if not GEMINI_API_KEY:
            self.init_error = "GEMINI_API_KEY not set"
            print(f"[AI Service] Warning: {self.init_error}")
            return
        
        try:
            if USE_NEW_SDK:
                # Use the new google-genai SDK
                self.client = genai.Client(api_key=GEMINI_API_KEY)
                print(f"[AI Service] Initialized with New google-genai SDK, model: {self.model_name}")
            elif USE_NEW_SDK is False:
                # Use legacy google-generativeai SDK
                google_generativeai.configure(api_key=GEMINI_API_KEY)
                self.client = google_generativeai.GenerativeModel(self.model_name)
                print(f"[AI Service] Initialized with Legacy google-generativeai SDK, model: {self.model_name}")
            else:
                self.init_error = "No Gemini SDK available"
                print(f"[AI Service] Error: {self.init_error}")
        except Exception as e:
            self.init_error = str(e)
            print(f"[AI Service] Initialization failed: {e}")
    
    def is_available(self) -> bool:
        """Check if AI service is available"""
        return self.client is not None
    
    def _generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """Helper method to generate content with either SDK"""
        if USE_NEW_SDK:
            # Use the new google.genai SDK
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
        elif USE_NEW_SDK is False:
            # Legacy google-generativeai SDK
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            return response.text
        return ""
    
    async def verify_data_quality(self, data_sample: List[Dict], schema: Dict) -> Dict[str, Any]:
        """
        Verify data quality using AI
        Returns quality score and issues found
        """
        if not self.is_available():
            return {
                "available": False,
                "message": "AI service not configured. Set GEMINI_API_KEY environment variable."
            }
        
        try:
            # Prepare prompt for data verification
            prompt = f"""Analyze the following data sample for quality issues.

Schema: {schema}

Data Sample (first 10 rows):
{data_sample[:10]}

Please analyze and return a JSON response with:
1. quality_score: 0-100 score
2. issues: list of issues found (missing values, type mismatches, inconsistencies)
3. suggestions: list of improvement suggestions
4. summary: brief summary in Vietnamese

Return ONLY valid JSON, no markdown formatting."""

            result_text = self._generate(prompt, temperature=0.3, max_tokens=2000).strip()
            
            # Parse response
            import json
            result_text = response.text.strip()
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()
            
            result = json.loads(result_text)
            result["available"] = True
            return result
            
        except Exception as e:
            return {
                "available": True,
                "error": str(e),
                "quality_score": None,
                "issues": [],
                "suggestions": [],
                "summary": f"Không thể phân tích: {str(e)}"
            }
    
    async def extract_pdf_data(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract and structure data from PDF using AI
        """
        if not self.is_available():
            return {
                "available": False,
                "message": "AI service not configured"
            }
        
        try:
            import pdfplumber
            
            # Extract text from PDF
            text_content = ""
            tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text_content += page.extract_text() or ""
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
            
            # Use AI to structure the data
            prompt = f"""Analyze this PDF content and extract structured data.

Text Content:
{text_content[:5000]}

Tables Found: {len(tables)}
First Table Sample: {tables[0] if tables else 'No tables'}

Please return a JSON with:
1. document_type: type of document (invoice, report, form, etc.)
2. extracted_data: key-value pairs of important data extracted
3. table_data: list of tables with headers and rows if applicable
4. summary: brief summary in Vietnamese

Return ONLY valid JSON."""

            result_text = self._generate(prompt, temperature=0.2, max_tokens=4000).strip()
            
            import json
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()
            
            result = json.loads(result_text)
            result["available"] = True
            result["raw_text"] = text_content[:1000]
            result["tables_count"] = len(tables)
            return result
            
        except Exception as e:
            return {
                "available": True,
                "error": str(e),
                "summary": f"Không thể xử lý PDF: {str(e)}"
            }
    
    async def normalize_data(self, data: List[Dict], target_schema: Dict) -> Dict[str, Any]:
        """
        Normalize data according to target schema using AI
        """
        if not self.is_available():
            return {"available": False, "message": "AI service not configured"}
        
        try:
            prompt = f"""Normalize the following data to match the target schema.

Current Data Sample:
{data[:5]}

Target Schema:
{target_schema}

Please:
1. Map current fields to target fields
2. Convert data types as needed
3. Handle missing values appropriately
4. Return a JSON with:
   - field_mapping: dict mapping current to target fields
   - normalized_sample: first 3 rows normalized
   - issues: list of normalization issues
   - success_rate: estimated percentage of data that can be normalized

Return ONLY valid JSON."""

            result_text = self._generate(prompt, temperature=0.2, max_tokens=3000).strip()
            
            import json
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()
            
            result = json.loads(result_text)
            result["available"] = True
            return result
            
        except Exception as e:
            return {
                "available": True,
                "error": str(e)
            }

# Singleton instance
ai_service = AIVerificationService()
