from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from database import get_db
from services.auth_service import generate_uuid, generate_api_key, hash_api_key, verify_api_key
from config import settings

class APIKeyService:
    def create_key(self, user_id: str, name: str, scopes: List[str], expires_in_days: int = None) -> Dict[str, Any]:
        """Create a new API key"""
        key_id = generate_uuid()
        api_key = generate_api_key()
        key_hash = hash_api_key(api_key)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        scopes_str = ",".join(scopes)
        
        with get_db() as conn:
            conn.execute("""
                INSERT INTO api_keys (id, user_id, key_hash, name, scopes, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [key_id, user_id, key_hash, name, scopes_str, expires_at])
        
        return {
            "id": key_id,
            "name": name,
            "key": api_key,  # Full key only shown once
            "scopes": scopes_str,
            "created_at": datetime.utcnow()
        }
    
    def get_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all API keys for a user"""
        with get_db() as conn:
            result = conn.execute("""
                SELECT id, name, scopes, is_active, last_used_at, created_at, expires_at
                FROM api_keys
                WHERE user_id = ?
                ORDER BY created_at DESC
            """, [user_id]).fetchall()
            
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "key_prefix": "LcAi_****",  # Hidden for security
                    "scopes": row[2],
                    "is_active": row[3],
                    "last_used_at": row[4],
                    "created_at": row[5],
                    "expires_at": row[6]
                }
                for row in result
            ]
    
    def validate_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return user info"""
        with get_db() as conn:
            # Get all active keys (need to check hash)
            result = conn.execute("""
                SELECT ak.id, ak.user_id, ak.key_hash, ak.scopes, ak.expires_at, u.plan
                FROM api_keys ak
                JOIN users u ON u.id = ak.user_id
                WHERE ak.is_active = TRUE
            """).fetchall()
            
            for row in result:
                key_id, user_id, key_hash, scopes, expires_at, plan = row
                
                if verify_api_key(api_key, key_hash):
                    # Check expiration
                    if expires_at and datetime.utcnow() > expires_at:
                        return None
                    
                    # Update last used
                    conn.execute("""
                        UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE id = ?
                    """, [key_id])
                    
                    return {
                        "key_id": key_id,
                        "user_id": user_id,
                        "scopes": scopes.split(","),
                        "plan": plan
                    }
            
            return None
    
    def revoke_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key"""
        with get_db() as conn:
            result = conn.execute("""
                UPDATE api_keys SET is_active = FALSE
                WHERE id = ? AND user_id = ?
            """, [key_id, user_id])
            
            return True
    
    def get_key_usage(self, key_id: str, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for an API key"""
        with get_db() as conn:
            # Verify ownership
            check = conn.execute("""
                SELECT id FROM api_keys WHERE id = ? AND user_id = ?
            """, [key_id, user_id]).fetchone()
            
            if not check:
                return None
            
            # Get usage stats
            start_date = datetime.utcnow() - timedelta(days=days)
            
            result = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(bytes_transferred) as total_bytes,
                    AVG(response_time_ms) as avg_response_time
                FROM usage_logs
                WHERE api_key_id = ? AND created_at >= ?
            """, [key_id, start_date]).fetchone()
            
            daily_result = conn.execute("""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as requests,
                    SUM(bytes_transferred) as bytes
                FROM usage_logs
                WHERE api_key_id = ? AND created_at >= ?
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """, [key_id, start_date]).fetchall()
            
            return {
                "total_requests": result[0] or 0,
                "total_bytes": result[1] or 0,
                "avg_response_time_ms": round(result[2] or 0, 2),
                "daily_usage": [
                    {"date": str(row[0]), "requests": row[1], "bytes": row[2]}
                    for row in daily_result
                ]
            }

api_key_service = APIKeyService()
