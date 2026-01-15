from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from config import settings
from routers import auth, data, query, api_keys, billing, ai, rag
from services.api_key_service import api_key_service
from services.billing_service import billing_service

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    🚀 **Locaith Data Engine** - Nền tảng Data Lakehouse-as-a-Service
    
    Cho phép bạn:
    - 📁 Lưu trữ và quản lý dữ liệu lớn
    - 🔍 Truy vấn dữ liệu bằng SQL
    - 🔑 Tích hợp API RESTful cho ứng dụng bên thứ 3
    - 💰 Monetize thông qua API key và billing
    
    ---
    **Liên hệ hỗ trợ:** support@locaith.com | Telegram: @locaithsolution
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response

# API Key authentication for external API access
@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    # Skip auth for docs and internal endpoints
    if request.url.path in ["/", "/docs", "/redoc", "/openapi.json", "/health"]:
        return await call_next(request)
    
    # Skip for auth endpoints (they use JWT)
    if request.url.path.startswith("/api/v1/auth"):
        return await call_next(request)
    
    # Check for API key in header (for external API access)
    api_key = request.headers.get("X-API-Key")
    if api_key:
        key_info = api_key_service.validate_key(api_key)
        if not key_info:
            return JSONResponse(
                status_code=401,
                content={"detail": "API key không hợp lệ hoặc đã hết hạn"}
            )
        
        # Attach key info to request state
        request.state.api_key_info = key_info
        
        # Log usage
        response = await call_next(request)
        
        billing_service.log_usage(
            user_id=key_info["user_id"],
            api_key_id=key_info["key_id"],
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            response_time_ms=int(float(response.headers.get("X-Process-Time", 0)))
        )
        
        return response
    
    # No API key, proceed with normal JWT auth
    return await call_next(request)

# Include routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(data.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(api_keys.router, prefix="/api/v1")
app.include_router(billing.router, prefix="/api/v1")
app.include_router(ai.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "api_base": "/api/v1"
    }

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Health check for Render.com (under /api/v1)
@app.get("/api/v1/health")
async def health_check_v1():
    return {"status": "healthy", "version": settings.APP_VERSION, "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
