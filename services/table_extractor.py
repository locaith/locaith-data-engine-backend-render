"""
Table Extraction Service - Core of Gold Layer Architecture
Bóc tách bảng từ PDF/Excel/CSV thành structured data

Theo kiến trúc Medallion:
- Bronze: File gốc
- Silver: Extracted tables (service này)
- Gold: Validated & stored structured data
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime


class TableExtractor:
    """
    World-class table extraction service
    - PDF: Extract tables with pdfplumber
    - Excel/CSV: Smart schema detection
    - Auto-detect headers and data types
    """
    
    def __init__(self):
        self.supported_extensions = ['.pdf', '.xlsx', '.xls', '.csv', '.json', '.parquet']
    
    async def extract_tables(self, file_path: str) -> Dict[str, Any]:
        """
        Extract all tables from a file
        
        Returns:
            {
                "success": bool,
                "tables": [
                    {
                        "name": str,
                        "dataframe": pd.DataFrame,
                        "schema": {...},
                        "row_count": int,
                        "source_page": int (for PDF)
                    }
                ],
                "metadata": {...}
            }
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext not in self.supported_extensions:
            return {
                "success": False,
                "error": f"Unsupported file type: {ext}",
                "tables": []
            }
        
        try:
            if ext == '.pdf':
                return await self._extract_pdf_tables(file_path)
            elif ext in ['.xlsx', '.xls']:
                return await self._extract_excel_tables(file_path)
            elif ext == '.csv':
                return await self._extract_csv_tables(file_path)
            elif ext == '.json':
                return await self._extract_json_tables(file_path)
            elif ext == '.parquet':
                return await self._extract_parquet_tables(file_path)
            else:
                return {"success": False, "error": "Unknown format", "tables": []}
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tables": []
            }
    
    async def _extract_pdf_tables(self, file_path: str) -> Dict[str, Any]:
        """Extract tables from PDF using pdfplumber"""
        import pdfplumber
        
        tables = []
        metadata = {
            "file_name": Path(file_path).name,
            "file_type": "pdf",
            "extraction_time": datetime.now().isoformat()
        }
        
        with pdfplumber.open(file_path) as pdf:
            metadata["total_pages"] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables from page
                page_tables = page.extract_tables()
                
                for table_idx, table in enumerate(page_tables):
                    if not table or len(table) < 2:
                        continue
                    
                    # First row as headers
                    headers = self._clean_headers(table[0])
                    
                    # Create DataFrame
                    data_rows = table[1:]
                    df = pd.DataFrame(data_rows, columns=headers)
                    
                    # Detect and convert data types
                    df = self._infer_types(df)
                    
                    # Generate schema
                    schema = self._generate_schema(df)
                    
                    tables.append({
                        "name": f"table_p{page_num}_t{table_idx + 1}",
                        "dataframe": df,
                        "schema": schema,
                        "row_count": len(df),
                        "source_page": page_num,
                        "source_type": "pdf_table"
                    })
                
                # Also extract text content as potential key-value data
                text = page.extract_text() or ""
                kv_data = self._extract_key_value_pairs(text)
                if kv_data:
                    kv_df = pd.DataFrame([kv_data])
                    tables.append({
                        "name": f"metadata_p{page_num}",
                        "dataframe": kv_df,
                        "schema": self._generate_schema(kv_df),
                        "row_count": 1,
                        "source_page": page_num,
                        "source_type": "pdf_kv_extract"
                    })
        
        return {
            "success": True,
            "tables": tables,
            "metadata": metadata,
            "total_tables": len(tables)
        }
    
    async def _extract_excel_tables(self, file_path: str) -> Dict[str, Any]:
        """Extract all sheets from Excel as tables"""
        tables = []
        metadata = {
            "file_name": Path(file_path).name,
            "file_type": "excel",
            "extraction_time": datetime.now().isoformat()
        }
        
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        metadata["sheet_names"] = excel_file.sheet_names
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            if df.empty:
                continue
            
            # Clean column names
            df.columns = self._clean_headers(list(df.columns))
            
            # Infer types
            df = self._infer_types(df)
            
            # Generate schema
            schema = self._generate_schema(df)
            
            tables.append({
                "name": self._sanitize_table_name(sheet_name),
                "dataframe": df,
                "schema": schema,
                "row_count": len(df),
                "source_sheet": sheet_name,
                "source_type": "excel_sheet"
            })
        
        return {
            "success": True,
            "tables": tables,
            "metadata": metadata,
            "total_tables": len(tables)
        }
    
    async def _extract_csv_tables(self, file_path: str) -> Dict[str, Any]:
        """Extract CSV as table with smart encoding detection"""
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except:
                continue
        
        if df is None:
            return {"success": False, "error": "Cannot read CSV with any encoding", "tables": []}
        
        # Clean headers
        df.columns = self._clean_headers(list(df.columns))
        
        # Infer types
        df = self._infer_types(df)
        
        return {
            "success": True,
            "tables": [{
                "name": Path(file_path).stem,
                "dataframe": df,
                "schema": self._generate_schema(df),
                "row_count": len(df),
                "source_type": "csv"
            }],
            "metadata": {
                "file_name": Path(file_path).name,
                "file_type": "csv"
            },
            "total_tables": 1
        }
    
    async def _extract_json_tables(self, file_path: str) -> Dict[str, Any]:
        """Extract JSON as table (supports arrays and nested objects)"""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            if any(isinstance(v, list) for v in data.values()):
                # Find the first list and normalize it
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        df = pd.json_normalize(value)
                        break
            else:
                df = pd.DataFrame([data])
        else:
            return {"success": False, "error": "Unsupported JSON structure", "tables": []}
        
        df.columns = self._clean_headers(list(df.columns))
        df = self._infer_types(df)
        
        return {
            "success": True,
            "tables": [{
                "name": Path(file_path).stem,
                "dataframe": df,
                "schema": self._generate_schema(df),
                "row_count": len(df),
                "source_type": "json"
            }],
            "metadata": {
                "file_name": Path(file_path).name,
                "file_type": "json"
            },
            "total_tables": 1
        }
    
    async def _extract_parquet_tables(self, file_path: str) -> Dict[str, Any]:
        """Extract Parquet as table (already structured)"""
        df = pd.read_parquet(file_path)
        
        # Filter out internal columns
        user_cols = [c for c in df.columns if not c.startswith('_')]
        df_clean = df[user_cols] if user_cols else df
        
        return {
            "success": True,
            "tables": [{
                "name": Path(file_path).stem,
                "dataframe": df_clean,
                "schema": self._generate_schema(df_clean),
                "row_count": len(df_clean),
                "source_type": "parquet"
            }],
            "metadata": {
                "file_name": Path(file_path).name,
                "file_type": "parquet"
            },
            "total_tables": 1
        }
    
    def _clean_headers(self, headers: List[Any]) -> List[str]:
        """Clean and normalize column headers"""
        cleaned = []
        for i, h in enumerate(headers):
            if h is None or str(h).strip() == "":
                cleaned.append(f"col_{i}")
            else:
                # Clean: lowercase, remove special chars, replace spaces
                clean = str(h).strip().lower()
                clean = re.sub(r'[^\w\s]', '', clean)
                clean = re.sub(r'\s+', '_', clean)
                if not clean:
                    clean = f"col_{i}"
                cleaned.append(clean)
        
        # Handle duplicates
        seen = {}
        result = []
        for h in cleaned:
            if h in seen:
                seen[h] += 1
                result.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                result.append(h)
        
        return result
    
    def _infer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer and convert data types intelligently"""
        for col in df.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Try to convert to numeric
            try:
                # Remove common number formatting
                if df[col].dtype == 'object':
                    cleaned = df[col].astype(str).str.replace(',', '').str.replace(' ', '')
                    cleaned = cleaned.str.replace('đ', '').str.replace('VND', '').str.replace('$', '')
                    numeric = pd.to_numeric(cleaned, errors='coerce')
                    
                    # If >70% converted, use numeric
                    if numeric.notna().sum() / len(numeric) > 0.7:
                        df[col] = numeric
                        continue
            except:
                pass
            
            # Try to convert to datetime
            try:
                date_col = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                if date_col.notna().sum() / len(date_col) > 0.7:
                    df[col] = date_col
                    continue
            except:
                pass
        
        return df
    
    def _generate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate schema info for a DataFrame"""
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            
            # Map pandas dtypes to SQL-like types
            if 'int' in dtype:
                sql_type = 'INTEGER'
            elif 'float' in dtype:
                sql_type = 'DOUBLE'
            elif 'datetime' in dtype:
                sql_type = 'TIMESTAMP'
            elif 'bool' in dtype:
                sql_type = 'BOOLEAN'
            else:
                sql_type = 'VARCHAR'
            
            columns.append({
                "name": col,
                "pandas_type": dtype,
                "sql_type": sql_type,
                "nullable": df[col].isna().any(),
                "sample_values": df[col].dropna().head(3).tolist()
            })
        
        return {
            "columns": columns,
            "column_count": len(columns),
            "primary_key_candidates": self._detect_primary_key(df)
        }
    
    def _detect_primary_key(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that could be primary keys"""
        candidates = []
        for col in df.columns:
            # Check uniqueness
            if df[col].nunique() == len(df) and df[col].notna().all():
                candidates.append(col)
        return candidates
    
    def _extract_key_value_pairs(self, text: str) -> Dict[str, Any]:
        """Extract key-value pairs from text (for metadata)"""
        patterns = [
            r'([A-Za-zÀ-ỹ\s]+):\s*([^\n]+)',  # Key: Value
            r'([A-Za-zÀ-ỹ\s]+)\s*[-–]\s*([^\n]+)',  # Key - Value
        ]
        
        kv = {}
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                if key and value and len(key) < 50 and len(value) < 500:
                    kv[key] = value
        
        return kv if len(kv) >= 2 else {}
    
    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize table name for SQL compatibility"""
        clean = re.sub(r'[^\w]', '_', str(name).lower())
        clean = re.sub(r'_+', '_', clean)
        return clean.strip('_')


# Singleton instance
table_extractor = TableExtractor()
