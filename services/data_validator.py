"""
Data Validation Service - Silver Layer Quality Control
Validate và clean data trước khi promote lên Gold Layer

Chức năng:
1. Schema validation với Pydantic
2. Type coercion và data cleaning
3. Missing value handling
4. Data quality scoring
"""

import re
from typing import List, Dict, Any, Optional, Type
from datetime import datetime
import pandas as pd
from pydantic import BaseModel, Field, validator, ValidationError
from enum import Enum


class DataQuality(Enum):
    """Data quality levels"""
    GOLD = "gold"      # 100% valid, ready for SQL queries
    SILVER = "silver"  # 90%+ valid, minor issues
    BRONZE = "bronze"  # <90% valid, needs review


class ColumnValidation(BaseModel):
    """Validation result for a column"""
    column_name: str
    expected_type: str
    actual_type: str
    valid_count: int
    invalid_count: int
    null_count: int
    valid_percentage: float
    issues: List[str] = []
    sample_invalid: List[Any] = []


class TableValidation(BaseModel):
    """Validation result for a table"""
    table_name: str
    total_rows: int
    total_columns: int
    quality_level: DataQuality
    overall_score: float  # 0-100
    columns: List[ColumnValidation]
    issues: List[str] = []
    recommendations: List[str] = []


class DataValidator:
    """
    Enterprise-grade data validation service
    - Type validation and coercion
    - Data quality scoring
    - Recommendation engine
    """
    
    def __init__(self):
        # Common Vietnamese data patterns
        self.patterns = {
            'phone': r'^(0|\+84)[0-9]{9,10}$',
            'email': r'^[\w\.-]+@[\w\.-]+\.\w+$',
            'date_vn': r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',
            'money_vn': r'^[\d\.,]+\s*(đ|VND|đồng)?$',
            'percentage': r'^\d+(\.\d+)?\s*%?$'
        }
    
    async def validate_table(
        self, 
        df: pd.DataFrame, 
        table_name: str = "table",
        expected_schema: Optional[Dict] = None
    ) -> TableValidation:
        """
        Validate a DataFrame and return quality assessment
        
        Args:
            df: DataFrame to validate
            table_name: Name for reporting
            expected_schema: Optional expected column types
            
        Returns:
            TableValidation with quality score and recommendations
        """
        columns_validation = []
        issues = []
        recommendations = []
        
        for col in df.columns:
            col_validation = self._validate_column(df, col, expected_schema)
            columns_validation.append(col_validation)
            
            if col_validation.valid_percentage < 90:
                issues.append(f"Column '{col}' has low validity: {col_validation.valid_percentage:.1f}%")
        
        # Calculate overall score
        if columns_validation:
            overall_score = sum(c.valid_percentage for c in columns_validation) / len(columns_validation)
        else:
            overall_score = 0
        
        # Determine quality level
        if overall_score >= 95:
            quality_level = DataQuality.GOLD
        elif overall_score >= 80:
            quality_level = DataQuality.SILVER
        else:
            quality_level = DataQuality.BRONZE
        
        # Generate recommendations
        recommendations = self._generate_recommendations(df, columns_validation, overall_score)
        
        return TableValidation(
            table_name=table_name,
            total_rows=len(df),
            total_columns=len(df.columns),
            quality_level=quality_level,
            overall_score=overall_score,
            columns=columns_validation,
            issues=issues,
            recommendations=recommendations
        )
    
    def _validate_column(
        self, 
        df: pd.DataFrame, 
        col: str,
        expected_schema: Optional[Dict] = None
    ) -> ColumnValidation:
        """Validate a single column"""
        series = df[col]
        total = len(series)
        null_count = series.isna().sum()
        non_null = series.dropna()
        
        # Detect expected type
        if expected_schema and col in expected_schema:
            expected_type = expected_schema[col]
        else:
            expected_type = self._infer_expected_type(series)
        
        actual_type = str(series.dtype)
        
        # Validate values
        valid_count = 0
        invalid_values = []
        issues = []
        
        for val in non_null:
            if self._is_valid_for_type(val, expected_type):
                valid_count += 1
            else:
                if len(invalid_values) < 5:
                    invalid_values.append(val)
        
        invalid_count = len(non_null) - valid_count
        
        # Add type-specific issues
        if expected_type == 'numeric' and invalid_count > 0:
            issues.append(f"{invalid_count} values cannot be converted to number")
        
        if null_count > total * 0.5:
            issues.append(f"High null rate: {null_count}/{total}")
        
        valid_percentage = (valid_count / total * 100) if total > 0 else 0
        
        return ColumnValidation(
            column_name=col,
            expected_type=expected_type,
            actual_type=actual_type,
            valid_count=valid_count,
            invalid_count=invalid_count,
            null_count=null_count,
            valid_percentage=valid_percentage,
            issues=issues,
            sample_invalid=invalid_values[:5]
        )
    
    def _infer_expected_type(self, series: pd.Series) -> str:
        """Infer the expected type for a column"""
        non_null = series.dropna()
        if len(non_null) == 0:
            return "unknown"
        
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return "integer"
            return "numeric"
        
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        
        # Sample non-null values for pattern matching
        samples = non_null.astype(str).head(100)
        
        # Try to detect patterns
        numeric_matches = samples.str.replace(',', '').str.match(r'^-?[\d.]+$').sum()
        if numeric_matches / len(samples) > 0.7:
            return "numeric"
        
        date_matches = samples.str.match(self.patterns['date_vn']).sum()
        if date_matches / len(samples) > 0.7:
            return "date"
        
        phone_matches = samples.str.match(self.patterns['phone']).sum()
        if phone_matches / len(samples) > 0.7:
            return "phone"
        
        email_matches = samples.str.match(self.patterns['email']).sum()
        if email_matches / len(samples) > 0.7:
            return "email"
        
        money_matches = samples.str.match(self.patterns['money_vn']).sum()
        if money_matches / len(samples) > 0.7:
            return "money"
        
        return "text"
    
    def _is_valid_for_type(self, value: Any, expected_type: str) -> bool:
        """Check if a value is valid for expected type"""
        if pd.isna(value):
            return True  # Nulls handled separately
        
        str_val = str(value).strip()
        
        if expected_type == "numeric":
            try:
                cleaned = str_val.replace(',', '').replace(' ', '')
                float(cleaned)
                return True
            except:
                return False
        
        elif expected_type == "integer":
            try:
                int(float(str_val.replace(',', '')))
                return True
            except:
                return False
        
        elif expected_type == "date":
            return bool(re.match(self.patterns['date_vn'], str_val))
        
        elif expected_type == "phone":
            return bool(re.match(self.patterns['phone'], str_val))
        
        elif expected_type == "email":
            return bool(re.match(self.patterns['email'], str_val))
        
        elif expected_type == "money":
            return bool(re.match(self.patterns['money_vn'], str_val))
        
        elif expected_type == "boolean":
            return str_val.lower() in ['true', 'false', '0', '1', 'yes', 'no', 'có', 'không']
        
        # Text type accepts everything
        return True
    
    def _generate_recommendations(
        self, 
        df: pd.DataFrame,
        columns: List[ColumnValidation],
        score: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Low score recommendations
        if score < 80:
            recommendations.append("⚠️ Data quality below 80%. Review source data before using for queries.")
        
        # Column-specific recommendations
        for col in columns:
            if col.null_count > len(df) * 0.3:
                recommendations.append(f"📊 Column '{col.column_name}': Consider filling or removing null values")
            
            if col.invalid_count > 10:
                recommendations.append(f"🔧 Column '{col.column_name}': {col.invalid_count} values need cleaning")
        
        # Positive recommendations
        if score >= 95:
            recommendations.append("✅ Excellent data quality! Ready for Gold Layer promotion.")
        elif score >= 85:
            recommendations.append("👍 Good data quality. Minor cleanup recommended.")
        
        return recommendations
    
    async def clean_dataframe(
        self, 
        df: pd.DataFrame,
        validation: TableValidation
    ) -> pd.DataFrame:
        """
        Clean DataFrame based on validation results
        
        Applies:
        - Type coercion
        - Null handling
        - Format normalization
        """
        df_clean = df.copy()
        
        for col_val in validation.columns:
            col = col_val.column_name
            expected_type = col_val.expected_type
            
            if col not in df_clean.columns:
                continue
            
            # Apply type-specific cleaning
            if expected_type == "numeric":
                df_clean[col] = self._clean_numeric(df_clean[col])
            
            elif expected_type == "integer":
                df_clean[col] = self._clean_integer(df_clean[col])
            
            elif expected_type == "date":
                df_clean[col] = self._clean_date(df_clean[col])
            
            elif expected_type == "money":
                df_clean[col] = self._clean_money(df_clean[col])
            
            elif expected_type == "phone":
                df_clean[col] = self._clean_phone(df_clean[col])
        
        return df_clean
    
    def _clean_numeric(self, series: pd.Series) -> pd.Series:
        """Clean and convert to numeric"""
        cleaned = series.astype(str).str.replace(',', '').str.replace(' ', '')
        cleaned = cleaned.str.replace('đ', '').str.replace('VND', '').str.replace('$', '')
        return pd.to_numeric(cleaned, errors='coerce')
    
    def _clean_integer(self, series: pd.Series) -> pd.Series:
        """Clean and convert to integer"""
        numeric = self._clean_numeric(series)
        return numeric.round().astype('Int64')  # Nullable integer
    
    def _clean_date(self, series: pd.Series) -> pd.Series:
        """Clean and convert to datetime"""
        return pd.to_datetime(series, errors='coerce', dayfirst=True)
    
    def _clean_money(self, series: pd.Series) -> pd.Series:
        """Clean money values to numeric"""
        cleaned = series.astype(str).str.replace(',', '').str.replace('.', '')
        cleaned = cleaned.str.replace('đ', '').str.replace('VND', '').str.replace('đồng', '')
        cleaned = cleaned.str.replace(' ', '').str.strip()
        return pd.to_numeric(cleaned, errors='coerce')
    
    def _clean_phone(self, series: pd.Series) -> pd.Series:
        """Normalize phone numbers"""
        cleaned = series.astype(str).str.replace(' ', '').str.replace('-', '')
        cleaned = cleaned.str.replace('.', '')
        return cleaned


# Singleton instance
data_validator = DataValidator()
