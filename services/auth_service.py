from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import uuid
import hashlib
import secrets
from config import settings

# Simple password hashing using SHA256 with salt (avoiding bcrypt Python 3.13 issues)
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    try:
        # Password format: salt:hash
        parts = hashed_password.split(":")
        if len(parts) != 2:
            return False
        salt, stored_hash = parts
        computed_hash = hashlib.sha256((salt + plain_password).encode()).hexdigest()
        return secrets.compare_digest(computed_hash, stored_hash)
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    """Hash a password with a random salt"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{password_hash}"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None

def generate_uuid() -> str:
    return str(uuid.uuid4())

def generate_api_key() -> str:
    """Generate a secure API key with LcAi_ prefix"""
    return f"LcAi_{uuid.uuid4().hex}{uuid.uuid4().hex[:16]}"

def hash_api_key(api_key: str) -> str:
    """Hash API key for storage using SHA256 (no length limit)"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Verify API key using SHA256"""
    return hashlib.sha256(plain_key.encode()).hexdigest() == hashed_key

