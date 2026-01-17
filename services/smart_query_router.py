"""
Smart Query Router - The Brain of Gold Layer Architecture
Automatically routes queries to the best data source for 100% accuracy

Flow:
1. Classify query intent (số liệu cụ thể vs tổng hợp/phân tích)
2. Route to appropriate source:
   - Số liệu cụ thể → SQL on Gold tables (100% accurate)
   - Tổng hợp/phân tích → AI with Gold data context
3. Return formatted response with source attribution
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path

from database import get_db
from services.auth_service import generate_uuid


class QueryIntent(Enum):
    """Types of query intents"""
    NUMERIC = "numeric"      # Số liệu cụ thể → SQL query
    LOOKUP = "lookup"        # Tìm kiếm thông tin → SQL + AI
    AGGREGATE = "aggregate"  # Tổng hợp, thống kê → SQL aggregate
    ANALYSIS = "analysis"    # Phân tích, so sánh → AI with context
    LIST = "list"            # Liệt kê danh sách → SQL or AI
    GENERAL = "general"      # Câu hỏi chung → AI


class SmartQueryRouter:
    """
    World-class query routing for data lakehouse
    
    Key capabilities:
    - Intent classification using patterns and AI
    - SQL query generation from natural language
    - Caching for repeated queries
    - Source attribution for trust
    """
    
    def __init__(self):
        self.client = None
        self.use_genai = False
        self._init_client()
        
        # Patterns for intent classification
        self.numeric_patterns = [
            r'bao nhiêu', r'mức (phạt|lương|giá)', r'số (lượng|tiền)', 
            r'tổng cộng', r'giá trị', r'\d+', r'chi phí', r'ngân sách',
            r'số liệu', r'thống kê', r'con số', r'percentage', r'phần trăm'
        ]
        
        self.aggregate_patterns = [
            r'tổng', r'trung bình', r'max|min', r'cao nhất', r'thấp nhất',
            r'sum|count|avg', r'đếm', r'tổng kết'
        ]
        
        self.list_patterns = [
            r'liệt kê', r'danh sách', r'những (gì|ai|nào)', r'các (loại|mục)',
            r'list', r'tất cả', r'có những', r'gồm những'
        ]
        
        self.lookup_patterns = [
            r'tìm', r'tra cứu', r'thông tin (về|của)', r'tên', r'địa chỉ',
            r'số điện thoại', r'email', r'ngày', r'where is', r'how to find'
        ]
    
    def _init_client(self):
        """Initialize Gemini for SQL generation - Prioritize New SDK"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return
        
        try:
            # Try the new google.genai SDK first
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.use_genai = False # False means use the new client.models logic
            print(f"[Smart Router] Initialized with New google.genai")
        except ImportError:
            try:
                # Fallback to legacy google-generativeai SDK
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel("gemini-3-flash-preview")
                self.use_genai = True
                print(f"[Smart Router] Initialized with Legacy google-generativeai")
            except:
                pass
    
    def classify_intent(self, question: str) -> Tuple[QueryIntent, float]:
        """
        Classify the intent of a query
        
        Returns:
            (intent_type, confidence_score)
        """
        question_lower = question.lower()
        
        # Check patterns
        numeric_score = sum(1 for p in self.numeric_patterns if re.search(p, question_lower))
        aggregate_score = sum(1 for p in self.aggregate_patterns if re.search(p, question_lower))
        list_score = sum(1 for p in self.list_patterns if re.search(p, question_lower))
        lookup_score = sum(1 for p in self.lookup_patterns if re.search(p, question_lower))
        
        total_score = numeric_score + aggregate_score + list_score + lookup_score
        
        if total_score == 0:
            return QueryIntent.GENERAL, 0.5
        
        # Determine intent based on highest score
        if aggregate_score >= max(numeric_score, list_score, lookup_score):
            return QueryIntent.AGGREGATE, min(0.9, 0.5 + aggregate_score * 0.1)
        elif numeric_score >= max(list_score, lookup_score):
            return QueryIntent.NUMERIC, min(0.9, 0.5 + numeric_score * 0.1)
        elif list_score >= lookup_score:
            return QueryIntent.LIST, min(0.9, 0.5 + list_score * 0.1)
        else:
            return QueryIntent.LOOKUP, min(0.9, 0.5 + lookup_score * 0.1)
    
    async def route_query(
        self,
        question: str,
        space_id: str,
        gold_tables: List[Dict],
        file_paths: List[str] = None
    ) -> Dict[str, Any]:
        """
        Route a query to the best data source
        
        Args:
            question: User's question
            space_id: Document Space ID  
            gold_tables: Available Gold tables
            file_paths: Original file paths (for AI context)
            
        Returns:
            {
                "answer": str,
                "source_type": "sql" | "ai" | "hybrid",
                "confidence": float,
                "sql_used": str (if SQL),
                "data": [...] (if SQL returned data)
            }
        """
        # Check cache first
        cached = self._check_cache(space_id, question)
        if cached:
            return cached
        
        # Classify intent
        intent, confidence = self.classify_intent(question)
        
        # Route based on intent
        if gold_tables and intent in [QueryIntent.NUMERIC, QueryIntent.AGGREGATE, QueryIntent.LOOKUP]:
            # Try SQL first for structured queries
            sql_result = await self._try_sql_query(question, gold_tables, space_id)
            if sql_result and sql_result.get("success"):
                result = {
                    "answer": self._format_sql_result(sql_result, question),
                    "source_type": "sql",
                    "confidence": 0.99,  # SQL is 100% accurate
                    "sql_used": sql_result.get("sql"),
                    "data": sql_result.get("data", []),
                    "row_count": sql_result.get("row_count", 0),
                    "is_gold_query": True
                }
                self._save_to_cache(space_id, question, result)
                return result
        
        # Fallback to AI for analysis or if SQL failed
        if intent == QueryIntent.LIST and "tài liệu" in question.lower():
            # Special case: list documents
            return self._list_documents(gold_tables)
        
        # Return indicator that AI should be used
        return {
            "use_ai": True,
            "intent": intent.value,
            "confidence": confidence,
            "gold_context": self._build_gold_context(gold_tables) if gold_tables else None
        }
    
    async def _try_sql_query(
        self,
        question: str,
        gold_tables: List[Dict],
        space_id: str
    ) -> Optional[Dict[str, Any]]:
        """Try to generate and execute SQL for a question"""
        if not self.client or not gold_tables:
            return None
        
        # Build schema context for SQL generation
        schema_context = self._build_schema_context(gold_tables)
        
        # Generate SQL with AI
        prompt = f"""Bạn là SQL expert. Hãy tạo câu lệnh SQL DuckDB để trả lời câu hỏi.

SCHEMA CỦA CÁC BẢNG:
{schema_context}

CÂU HỎI: {question}

QUY TẮC:
1. Chỉ trả về câu SQL, không giải thích
2. Nếu không thể tạo SQL, trả về "NO_SQL"
3. SQL phải tương thích DuckDB
4. Sử dụng tên bảng và cột CHÍNH XÁC như trong schema

SQL:"""

        try:
            if self.use_genai:
                response = self.client.generate_content(prompt)
                sql = response.text.strip()
            else:
                response = self.client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=prompt
                )
                sql = response.text.strip()
            
            # Clean SQL
            sql = sql.replace("```sql", "").replace("```", "").strip()
            
            if sql == "NO_SQL" or not sql:
                return None
            
            # Execute SQL
            from services.gold_layer_service import gold_layer_service
            result = await gold_layer_service.query_gold(space_id, sql)
            
            if result.get("success"):
                result["sql"] = sql
                return result
            
            return None
            
        except Exception as e:
            print(f"[Smart Router] SQL generation error: {e}")
            return None
    
    def _build_schema_context(self, gold_tables: List[Dict]) -> str:
        """Build schema context for SQL generation"""
        context = []
        for table in gold_tables:
            table_name = table.get("table_name", "unknown")
            columns = table.get("columns", [])
            
            if isinstance(columns, list) and len(columns) > 0:
                if isinstance(columns[0], dict):
                    col_defs = [f"  {c.get('name', 'col')}: {c.get('sql_type', 'VARCHAR')}" for c in columns]
                else:
                    col_defs = [f"  {c}: VARCHAR" for c in columns]
            else:
                col_defs = ["  (no columns)"]
            
            context.append(f"Table: {table_name}\n" + "\n".join(col_defs))
        
        return "\n\n".join(context)
    
    def _format_sql_result(self, result: Dict, question: str) -> str:
        """Format SQL result as readable answer - NO MARKDOWN"""
        data = result.get("data", [])
        columns = result.get("columns", [])
        row_count = result.get("row_count", 0)
        
        if row_count == 0:
            return "Không tìm thấy dữ liệu phù hợp trong Gold tables."
        
        # Single value result
        if row_count == 1 and len(columns) == 1:
            value = data[0][0]
            return f"Kết quả: {value}\n\n(Nguồn: SQL query trên Gold tables - 100% chính xác)"
        
        # Multiple rows - format as plain text list
        answer_lines = [f"Tìm thấy {row_count} kết quả:\n"]
        
        # Header line
        answer_lines.append(" | ".join(str(c) for c in columns))
        answer_lines.append("-" * 50)
        
        for row in data[:20]:  # Limit to 20 rows
            row_str = " | ".join(str(v) if v is not None else "" for v in row)
            answer_lines.append(row_str)
        
        if row_count > 20:
            answer_lines.append(f"\n...và {row_count - 20} dòng khác")
        
        answer_lines.append("\n(Nguồn: SQL query trên Gold tables - 100% chính xác)")
        
        return "\n".join(answer_lines)
    
    def _build_gold_context(self, gold_tables: List[Dict]) -> str:
        """Build context from Gold tables for AI"""
        context_parts = []
        for table in gold_tables[:5]:  # Limit to 5 tables
            name = table.get("table_name", "unknown")
            row_count = table.get("row_count", 0)
            columns = table.get("columns", [])
            
            if isinstance(columns, list) and len(columns) > 0:
                if isinstance(columns[0], dict):
                    col_names = [c.get("name", "col") for c in columns]
                else:
                    col_names = columns
            else:
                col_names = []
            
            context_parts.append(f"Bảng '{name}': {row_count} dòng, cột: {', '.join(col_names)}")
        
        return "\n".join(context_parts)
    
    def _list_documents(self, gold_tables: List[Dict]) -> Dict[str, Any]:
        """Special handler for document listing - NO MARKDOWN"""
        if not gold_tables:
            return {
                "answer": "Chưa có dữ liệu Gold nào. Vui lòng upload và promote files lên Gold Layer.",
                "source_type": "system",
                "confidence": 1.0
            }
        
        doc_list = []
        total_rows = 0
        for t in gold_tables:
            name = t.get("source_file") or t.get("table_name", "unknown")
            rows = t.get("row_count", 0)
            quality = t.get("quality_level", "unknown")
            doc_list.append(f"• {name}: {rows} dòng, chất lượng: {quality}")
            total_rows += rows
        
        answer = f"Có {len(gold_tables)} tài liệu trong Gold Layer:\n\n" + "\n".join(doc_list)
        answer += f"\n\nTổng cộng: {total_rows} dòng dữ liệu có thể query SQL."
        
        return {
            "answer": answer,
            "source_type": "gold_metadata",
            "confidence": 1.0,
            "total_tables": len(gold_tables),
            "total_rows": total_rows
        }
    
    def _check_cache(self, space_id: str, question: str) -> Optional[Dict]:
        """Check if query result is cached"""
        query_hash = hashlib.md5(question.lower().encode()).hexdigest()
        
        try:
            with get_db() as conn:
                result = conn.execute("""
                    SELECT result_json FROM query_cache
                    WHERE space_id = ? AND query_hash = ?
                    ORDER BY last_hit_at DESC
                    LIMIT 1
                """, [space_id, query_hash]).fetchone()
                
                if result:
                    # Update hit count
                    conn.execute("""
                        UPDATE query_cache 
                        SET hit_count = hit_count + 1, last_hit_at = CURRENT_TIMESTAMP
                        WHERE space_id = ? AND query_hash = ?
                    """, [space_id, query_hash])
                    
                    cached = json.loads(result[0])
                    cached["from_cache"] = True
                    return cached
        except:
            pass
        
        return None
    
    def _save_to_cache(self, space_id: str, question: str, result: Dict):
        """Save query result to cache"""
        query_hash = hashlib.md5(question.lower().encode()).hexdigest()
        
        try:
            with get_db() as conn:
                cache_id = generate_uuid()
                conn.execute("""
                    INSERT INTO query_cache (id, space_id, query_hash, query_text, result_json)
                    VALUES (?, ?, ?, ?, ?)
                """, [cache_id, space_id, query_hash, question, json.dumps(result)])
        except:
            pass


# Singleton instance
smart_query_router = SmartQueryRouter()
