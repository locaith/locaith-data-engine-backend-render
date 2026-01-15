from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from models.schemas import UserCreate, UserLogin, UserResponse, Token
from services.auth_service import (
    get_password_hash, verify_password, 
    create_access_token, create_refresh_token, decode_token, generate_uuid
)
from database import get_db

router = APIRouter(prefix="/auth", tags=["Authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from JWT token"""
    print(f"[DEBUG] Token received: {token[:30] if token else 'None'}...")
    payload = decode_token(token)
    print(f"[DEBUG] Decoded payload: {payload}")
    if not payload or payload.get("type") != "access":
        print(f"[DEBUG] Invalid payload - type: {payload.get('type') if payload else 'N/A'}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    with get_db() as conn:
        result = conn.execute("""
            SELECT id, email, username, full_name, plan, is_active, created_at
            FROM users WHERE id = ? AND is_active = TRUE
        """, [user_id]).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "id": result[0],
            "email": result[1],
            "username": result[2],
            "full_name": result[3],
            "plan": result[4],
            "is_active": result[5],
            "created_at": result[6]
        }

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    """Register a new user"""
    from datetime import datetime
    import traceback
    
    try:
        with get_db() as conn:
            # Check if email/username exists
            existing = conn.execute("""
                SELECT id FROM users WHERE email = ? OR username = ?
            """, [user.email, user.username]).fetchone()
            
            if existing:
                raise HTTPException(status_code=400, detail="Email hoặc username đã tồn tại")
            
            user_id = generate_uuid()
            hashed_password = get_password_hash(user.password)
            now = datetime.utcnow()
            
            conn.execute("""
                INSERT INTO users (id, email, username, hashed_password, full_name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [user_id, user.email, user.username, hashed_password, user.full_name, now, now])
            
            return UserResponse(
                id=user_id, 
                email=user.email, 
                username=user.username,
                full_name=user.full_name, 
                plan="starter", 
                is_active=True, 
                created_at=now
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in register: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access token"""
    with get_db() as conn:
        result = conn.execute("""
            SELECT id, hashed_password FROM users 
            WHERE (username = ? OR email = ?) AND is_active = TRUE
        """, [form_data.username, form_data.username]).fetchone()
        
        if not result or not verify_password(form_data.password, result[1]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tên đăng nhập hoặc mật khẩu không đúng",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = result[0]
        access_token = create_access_token(data={"sub": user_id})
        refresh_token = create_refresh_token(data={"sub": user_id})
        
        return Token(access_token=access_token, refresh_token=refresh_token)

@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    payload = decode_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    user_id = payload.get("sub")
    new_access_token = create_access_token(data={"sub": user_id})
    new_refresh_token = create_refresh_token(data={"sub": user_id})
    
    return Token(access_token=new_access_token, refresh_token=new_refresh_token)

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info"""
    return UserResponse(**current_user)
