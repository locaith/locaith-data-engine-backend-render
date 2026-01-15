from fastapi import APIRouter, HTTPException, Depends

from services.billing_service import billing_service
from routers.auth import get_current_user

router = APIRouter(prefix="/billing", tags=["Billing"])

@router.get("/plans")
async def get_plans():
    """Get all available plans"""
    return billing_service.get_plans()

@router.get("/usage")
async def get_usage(current_user: dict = Depends(get_current_user)):
    """Get current usage statistics"""
    return billing_service.get_user_usage(current_user["id"])

@router.post("/subscribe/{plan}")
async def subscribe_to_plan(
    plan: str,
    current_user: dict = Depends(get_current_user)
):
    """Subscribe to a plan - generates VietQR payment"""
    plans = billing_service.PLANS
    
    if plan not in plans:
        raise HTTPException(status_code=400, detail="Gói dịch vụ không hợp lệ")
    
    plan_info = plans[plan]
    
    if plan_info["price"] == 0:
        return {"message": "Bạn đang sử dụng gói Starter miễn phí"}
    
    if plan_info["price"] == -1:
        return {
            "message": "Vui lòng liên hệ để tư vấn gói Enterprise",
            "contact": {
                "telegram": "https://t.me/locaithsolution",
                "email": "support@locaith.com"
            }
        }
    
    # Generate VietQR payment
    payment = billing_service.generate_vietqr_payment(
        amount=plan_info["price"],
        user_id=current_user["id"],
        plan=plan
    )
    
    return {
        "plan": plan,
        "plan_info": plan_info,
        "payment": payment,
        "instructions": [
            "1. Quét mã QR hoặc chuyển khoản theo thông tin bên dưới",
            "2. Nhập đúng nội dung chuyển khoản để xác nhận thanh toán tự động",
            "3. Gói dịch vụ sẽ được kích hoạt trong vòng 5 phút sau khi thanh toán thành công"
        ]
    }

@router.get("/invoices")
async def get_invoices(current_user: dict = Depends(get_current_user)):
    """Get payment history"""
    # TODO: Implement invoice history from payment records
    return {
        "invoices": [],
        "message": "Chưa có hóa đơn nào"
    }
