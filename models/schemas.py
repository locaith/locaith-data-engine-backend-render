from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum

# Enums
class PlanType(str, Enum):
    STARTER = "starter"
    PRO = "pro"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"

class ScopeType(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

# Auth schemas
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: Optional[str]
    plan: str  # Changed from PlanType to str to match database
    is_active: bool
    created_at: datetime

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[str] = None
    scopes: List[str] = []

# Dataset schemas
class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None

class DatasetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    file_type: str
    file_size: int
    row_count: Optional[int]
    schema_json: Optional[str]
    created_at: Optional[datetime] = None

class DatasetPreview(BaseModel):
    columns: List[str]
    data: List[dict]
    total_rows: int

# Query schemas
class QueryRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=10000)

class QueryResult(BaseModel):
    columns: List[str]
    data: List[List[Any]]
    row_count: int
    execution_time_ms: int

# API Key schemas
class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    scopes: List[ScopeType] = [ScopeType.READ]
    space_id: Optional[str] = None  # Link to Document Space
    expires_in_days: Optional[int] = None

class APIKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str  # First 8 chars for identification
    scopes: str
    space_id: Optional[str] = None
    space_name: Optional[str] = None
    is_active: bool
    last_used_at: Optional[datetime]
    created_at: datetime
    expires_at: Optional[datetime]

class APIKeyCreated(BaseModel):
    id: str
    name: str
    key: str  # Full key, only shown once
    scopes: str
    created_at: datetime

# Usage schemas
class UsageStats(BaseModel):
    total_requests: int
    total_queries: int
    storage_used_mb: float
    period_start: datetime
    period_end: datetime

class UsageDetail(BaseModel):
    date: str
    requests: int
    queries: int
    bytes_transferred: int

# Billing schemas
class PlanInfo(BaseModel):
    name: str
    price: int  # VND
    queries_per_month: int
    storage_mb: int
    rate_limit_per_minute: int
    features: List[str]

class SubscriptionResponse(BaseModel):
    current_plan: PlanType
    usage_this_month: UsageStats
    next_billing_date: Optional[datetime]
