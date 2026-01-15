from datetime import datetime, timedelta
from typing import Dict, Any, List
from database import get_db
from services.auth_service import generate_uuid
from config import settings
import os

class BillingService:
    PLANS = {
        "starter": {
            "name": "Starter",
            "price": 0,
            "queries_per_month": settings.QUERY_LIMIT_STARTER,
            "storage_mb": settings.STORAGE_LIMIT_STARTER,
            "rate_limit_per_minute": settings.RATE_LIMIT_STARTER,
            "features": ["100MB lưu trữ", "1,000 queries/tháng", "Community support"]
        },
        "pro": {
            "name": "Pro",
            "price": 500000,
            "queries_per_month": settings.QUERY_LIMIT_PRO,
            "storage_mb": settings.STORAGE_LIMIT_PRO,
            "rate_limit_per_minute": settings.RATE_LIMIT_PRO,
            "features": ["5GB lưu trữ", "50,000 queries/tháng", "Email support 24h", "API access"]
        },
        "business": {
            "name": "Business",
            "price": 2000000,
            "queries_per_month": settings.QUERY_LIMIT_BUSINESS,
            "storage_mb": settings.STORAGE_LIMIT_BUSINESS,
            "rate_limit_per_minute": settings.RATE_LIMIT_BUSINESS,
            "features": ["50GB lưu trữ", "500,000 queries/tháng", "Chat support 12h", "Priority API", "Custom integrations"]
        },
        "enterprise": {
            "name": "Enterprise",
            "price": -1,  # Contact sales
            "queries_per_month": -1,
            "storage_mb": -1,
            "rate_limit_per_minute": settings.RATE_LIMIT_ENTERPRISE,
            "features": ["Không giới hạn lưu trữ", "Không giới hạn queries", "24/7 support", "SLA 99.9%", "On-premise option"]
        }
    }
    
    def get_plans(self) -> List[Dict[str, Any]]:
        """Get all available plans"""
        return [
            {"id": key, **value}
            for key, value in self.PLANS.items()
        ]
    
    def get_user_usage(self, user_id: str) -> Dict[str, Any]:
        """Get current month usage for a user"""
        start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        with get_db() as conn:
            # Get user plan
            user = conn.execute("""
                SELECT plan FROM users WHERE id = ?
            """, [user_id]).fetchone()
            
            plan = user[0] if user else "starter"
            
            # Get request count
            requests = conn.execute("""
                SELECT COUNT(*) FROM usage_logs
                WHERE user_id = ? AND created_at >= ?
            """, [user_id, start_of_month]).fetchone()[0]
            
            # Get query count
            queries = conn.execute("""
                SELECT COUNT(*) FROM query_history
                WHERE user_id = ? AND created_at >= ?
            """, [user_id, start_of_month]).fetchone()[0]
            
            # Get storage used
            storage_result = conn.execute("""
                SELECT COALESCE(SUM(file_size), 0) FROM datasets WHERE user_id = ?
            """, [user_id]).fetchone()[0]
            
            storage_mb = round(storage_result / (1024 * 1024), 2)
            
            plan_info = self.PLANS.get(plan, self.PLANS["starter"])
            
            return {
                "plan": plan,
                "plan_info": plan_info,
                "usage": {
                    "requests": requests,
                    "queries": queries,
                    "storage_mb": storage_mb,
                    "queries_limit": plan_info["queries_per_month"],
                    "storage_limit_mb": plan_info["storage_mb"]
                },
                "period_start": start_of_month.isoformat(),
                "period_end": (start_of_month + timedelta(days=32)).replace(day=1).isoformat()
            }
    
    def log_usage(self, user_id: str, api_key_id: str, endpoint: str, method: str, 
                  status_code: int, response_time_ms: int, bytes_transferred: int = 0):
        """Log an API usage"""
        with get_db() as conn:
            conn.execute("""
                INSERT INTO usage_logs (id, user_id, api_key_id, endpoint, method, status_code, response_time_ms, bytes_transferred)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [generate_uuid(), user_id, api_key_id, endpoint, method, status_code, response_time_ms, bytes_transferred])
    
    def check_limits(self, user_id: str, plan: str) -> Dict[str, bool]:
        """Check if user is within limits"""
        usage = self.get_user_usage(user_id)
        plan_info = self.PLANS.get(plan, self.PLANS["starter"])
        
        # Enterprise has no limits
        if plan == "enterprise":
            return {"queries_ok": True, "storage_ok": True}
        
        return {
            "queries_ok": usage["usage"]["queries"] < plan_info["queries_per_month"],
            "storage_ok": usage["usage"]["storage_mb"] < plan_info["storage_mb"]
        }
    
    def generate_vietqr_payment(self, amount: int, user_id: str, plan: str) -> Dict[str, Any]:
        """Generate VietQR payment info"""
        # Sepay/VietQR configuration
        bank_id = "970422"  # MB Bank (example)
        account_no = "0906888892"  # Your account number
        account_name = "LOCAITH SOLUTION TECH"
        
        # Generate unique transaction code
        transaction_code = f"LDE{user_id[:8].upper()}{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # VietQR URL format
        qr_content = f"https://img.vietqr.io/image/{bank_id}-{account_no}-compact.png?amount={amount}&addInfo={transaction_code}&accountName={account_name.replace(' ', '%20')}"
        
        return {
            "bank_name": "MB Bank",
            "account_no": account_no,
            "account_name": account_name,
            "amount": amount,
            "transaction_code": transaction_code,
            "qr_url": qr_content,
            "note": f"Nội dung chuyển khoản: {transaction_code}"
        }

billing_service = BillingService()
