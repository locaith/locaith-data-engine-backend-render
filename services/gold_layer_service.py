"""
Gold Layer Service - Orchestrates the Medallion Architecture
Bronze → Silver → Gold data pipeline

Core functions:
1. promote_to_gold() - Transform raw data to structured tables
2. query_gold() - SQL queries on Gold tables
3. get_gold_tables() - List queryable tables
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd

from database import get_db
from services.auth_service import generate_uuid
from services.table_extractor import table_extractor
from services.data_validator import data_validator, DataQuality
from config import settings


class GoldLayerService:
    """
    World-class Gold Layer Service
    - Extracts structured data from Bronze/Silver layers  
    - Validates and cleans data
    - Stores in queryable Gold tables
    - Enables SQL queries for 100% accuracy
    """
    
    def __init__(self):
        self.gold_dir = os.path.join(settings.DATA_DIR, "gold")
        os.makedirs(self.gold_dir, exist_ok=True)
        print("[Gold Layer] Initialized")
    
    async def promote_to_gold(
        self, 
        dataset_id: str,
        file_path: str,
        space_id: Optional[str] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Promote a dataset from Bronze/Silver to Gold Layer
        
        Pipeline:
        1. Extract tables from file
        2. Validate each table
        3. Clean data based on validation
        4. Store as queryable Gold table
        
        Args:
            dataset_id: Source dataset ID
            file_path: Path to source file (parquet or original)
            space_id: Document Space ID
            force: Force re-promotion even if already exists
            
        Returns:
            {
                "success": bool,
                "gold_tables": [{...}],  # Created gold tables
                "quality_summary": {...},
                "total_rows": int
            }
        """
        # Check if already promoted
        if not force:
            existing = self._get_existing_gold_tables(dataset_id)
            if existing:
                return {
                    "success": True,
                    "gold_tables": existing,
                    "message": "Already promoted to Gold Layer",
                    "skipped": True
                }
        
        # Step 1: Extract tables
        extraction_result = await table_extractor.extract_tables(file_path)
        
        if not extraction_result["success"]:
            return {
                "success": False,
                "error": extraction_result.get("error", "Extraction failed"),
                "gold_tables": []
            }
        
        tables = extraction_result["tables"]
        if not tables:
            return {
                "success": False,
                "error": "No tables found in file",
                "gold_tables": []
            }
        
        # Step 2 & 3: Validate and clean each table
        gold_tables_created = []
        total_rows = 0
        quality_scores = []
        
        for table_info in tables:
            df = table_info["dataframe"]
            table_name = table_info["name"]
            
            # Validate
            validation = await data_validator.validate_table(df, table_name)
            
            # Clean if quality is acceptable
            if validation.quality_level != DataQuality.BRONZE or validation.overall_score >= 50:
                df_clean = await data_validator.clean_dataframe(df, validation)
            else:
                df_clean = df  # Keep original if validation failed badly
            
            # Step 4: Store as Gold table
            gold_table = await self._store_gold_table(
                df=df_clean,
                table_name=table_name,
                dataset_id=dataset_id,
                space_id=space_id,
                source_info=table_info,
                validation=validation
            )
            
            gold_tables_created.append(gold_table)
            total_rows += len(df_clean)
            quality_scores.append(validation.overall_score)
        
        # Calculate average quality
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "success": True,
            "gold_tables": gold_tables_created,
            "total_tables": len(gold_tables_created),
            "total_rows": total_rows,
            "quality_summary": {
                "average_score": round(avg_quality, 2),
                "quality_level": "gold" if avg_quality >= 95 else "silver" if avg_quality >= 80 else "bronze"
            },
            "message": f"Promoted {len(gold_tables_created)} tables to Gold Layer"
        }
    
    async def _store_gold_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        dataset_id: str,
        space_id: Optional[str],
        source_info: Dict,
        validation
    ) -> Dict[str, Any]:
        """Store a DataFrame as a Gold table"""
        gold_table_id = generate_uuid()
        
        # Save as parquet for efficient querying
        parquet_path = os.path.join(self.gold_dir, f"{gold_table_id}.parquet")
        df.to_parquet(parquet_path, index=False)
        
        # Build schema JSON
        schema_json = json.dumps({
            "columns": [
                {
                    "name": col.column_name,
                    "type": col.expected_type,
                    "sql_type": self._get_sql_type(col.expected_type),
                    "nullable": col.null_count > 0,
                    "valid_percentage": col.valid_percentage
                }
                for col in validation.columns
            ]
        })
        
        # Insert into gold_tables
        with get_db() as conn:
            conn.execute("""
                INSERT INTO gold_tables 
                (id, dataset_id, space_id, table_name, source_file, source_type, 
                 schema_json, row_count, column_count, quality_score, quality_level, 
                 parquet_path, is_queryable)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                gold_table_id, dataset_id, space_id, table_name,
                source_info.get("name", "unknown"),
                source_info.get("source_type", "unknown"),
                schema_json, len(df), len(df.columns),
                validation.overall_score, validation.quality_level.value,
                parquet_path, True
            ])
            
            # Insert column metadata
            for col in validation.columns:
                col_id = generate_uuid()
                sample_values = json.dumps(col.sample_invalid[:3]) if col.sample_invalid else "[]"
                conn.execute("""
                    INSERT INTO gold_columns
                    (id, gold_table_id, column_name, column_type, sql_type, 
                     is_nullable, is_primary_key, sample_values, valid_percentage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    col_id, gold_table_id, col.column_name, col.expected_type,
                    self._get_sql_type(col.expected_type),
                    col.null_count > 0, False, sample_values, col.valid_percentage
                ])
        
        return {
            "id": gold_table_id,
            "table_name": table_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "quality_score": validation.overall_score,
            "quality_level": validation.quality_level.value,
            "parquet_path": parquet_path,
            "columns": [c.column_name for c in validation.columns]
        }
    
    def _get_sql_type(self, python_type: str) -> str:
        """Map Python/Pandas types to SQL types"""
        mapping = {
            "numeric": "DOUBLE",
            "integer": "BIGINT",
            "text": "VARCHAR",
            "date": "TIMESTAMP",
            "datetime": "TIMESTAMP",
            "boolean": "BOOLEAN",
            "money": "DOUBLE",
            "phone": "VARCHAR",
            "email": "VARCHAR",
            "unknown": "VARCHAR"
        }
        return mapping.get(python_type, "VARCHAR")
    
    def _get_existing_gold_tables(self, dataset_id: str) -> List[Dict]:
        """Get existing Gold tables for a dataset"""
        with get_db() as conn:
            result = conn.execute("""
                SELECT id, table_name, row_count, quality_score, quality_level, parquet_path
                FROM gold_tables
                WHERE dataset_id = ?
            """, [dataset_id]).fetchall()
            
            return [
                {
                    "id": row[0],
                    "table_name": row[1],
                    "row_count": row[2],
                    "quality_score": row[3],
                    "quality_level": row[4],
                    "parquet_path": row[5]
                }
                for row in result
            ]
    
    async def query_gold(
        self, 
        space_id: str, 
        sql: str,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Execute SQL query on Gold tables in a space
        
        Returns:
            {
                "success": bool,
                "columns": [...],
                "data": [...],
                "row_count": int,
                "execution_time_ms": int
            }
        """
        import time
        start_time = time.time()
        
        try:
            with get_db() as conn:
                # Get all Gold tables for this space
                gold_tables = conn.execute("""
                    SELECT id, table_name, parquet_path
                    FROM gold_tables
                    WHERE space_id = ? AND is_queryable = TRUE
                """, [space_id]).fetchall()
                
                if not gold_tables:
                    return {
                        "success": False,
                        "error": "No Gold tables found in this space"
                    }
                
                # Register each Gold table as a view
                for gt in gold_tables:
                    table_name = gt[1]
                    parquet_path = gt[2]
                    if os.path.exists(parquet_path):
                        safe_path = parquet_path.replace('\\', '/')
                        conn.execute(f"""
                            CREATE OR REPLACE VIEW {table_name} AS 
                            SELECT * FROM read_parquet('{safe_path}')
                        """)
                
                # Execute the query
                result = conn.execute(f"{sql} LIMIT {limit}")
                columns = [desc[0] for desc in result.description]
                data = result.fetchall()
                
                execution_time = int((time.time() - start_time) * 1000)
                
                return {
                    "success": True,
                    "columns": columns,
                    "data": [list(row) for row in data],
                    "row_count": len(data),
                    "execution_time_ms": execution_time,
                    "is_gold_query": True
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_gold_tables(self, space_id: str) -> List[Dict[str, Any]]:
        """Get all Gold tables in a space with their schemas"""
        with get_db() as conn:
            tables = conn.execute("""
                SELECT id, table_name, source_file, row_count, column_count,
                       quality_score, quality_level, schema_json, created_at
                FROM gold_tables
                WHERE space_id = ? AND is_queryable = TRUE
                ORDER BY created_at DESC
            """, [space_id]).fetchall()
            
            result = []
            for t in tables:
                schema = json.loads(t[7]) if t[7] else {}
                result.append({
                    "id": t[0],
                    "table_name": t[1],
                    "source_file": t[2],
                    "row_count": t[3],
                    "column_count": t[4],
                    "quality_score": t[5],
                    "quality_level": t[6],
                    "columns": schema.get("columns", []),
                    "created_at": str(t[8])
                })
            
            return result
    
    async def get_table_preview(
        self, 
        gold_table_id: str, 
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get preview data from a Gold table"""
        with get_db() as conn:
            table = conn.execute("""
                SELECT parquet_path, table_name, row_count
                FROM gold_tables
                WHERE id = ?
            """, [gold_table_id]).fetchone()
            
            if not table:
                return {"success": False, "error": "Table not found"}
            
            parquet_path, table_name, total_rows = table
            
            if not os.path.exists(parquet_path):
                return {"success": False, "error": "Parquet file not found"}
            
            df = pd.read_parquet(parquet_path).head(limit)
            
            return {
                "success": True,
                "table_name": table_name,
                "columns": list(df.columns),
                "data": df.to_dict(orient='records'),
                "preview_rows": len(df),
                "total_rows": total_rows
            }


# Singleton instance
gold_layer_service = GoldLayerService()
