"""
Authentication API for memOS Server
Provides JWT-based authentication for Recovery Bot Android client
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
import jwt
import hashlib
import requests
import logging

from config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])
settings = get_settings()

class LoginRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    user_id: str
    username: str
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"

class RefreshRequest(BaseModel):
    refresh_token: str

class TokenResponse(BaseModel):
    access_token: str
    expires_in: int
    token_type: str = "Bearer"

class UserProfile(BaseModel):
    user_id: str
    username: str
    display_name: str
    email: Optional[str] = None
    preferences: dict = {}

@router.post("/login", response_model=AuthResponse)
async def login(credentials: LoginRequest):
    """
    Authenticate user and return JWT tokens
    """
    logger.info(f"Login attempt for user: {credentials.username}")
    
    # Step 1: Validate with PHP backend (if available)
    user_id = None
    username = credentials.username
    
    try:
        php_auth_url = f"{settings.recovery_bot_api_url}/auth_endpoints.php"
        response = requests.post(php_auth_url, json={
            "action": "login",
            "username": credentials.username,
            "password": credentials.password
        }, timeout=10)
        
        if response.status_code == 200:
            php_result = response.json()
            if php_result.get("success"):
                user_id = php_result.get("user_id", credentials.username)
                username = php_result.get("username", credentials.username)
                logger.info(f"PHP backend authentication successful for user: {username}")
            else:
                logger.warning(f"PHP backend authentication failed: {php_result.get('message')}")
                raise HTTPException(401, php_result.get("message", "Invalid credentials"))
        else:
            logger.warning(f"PHP backend returned status {response.status_code}")
            
    except requests.RequestException as e:
        logger.warning(f"PHP backend unavailable: {e}")
        # Continue to fallback authentication
    
    # Step 2: Fallback authentication for development
    if user_id is None:
        logger.info("Using fallback authentication")
        if credentials.username == "sparkone" and credentials.password == "Y3$hua3141":
            user_id = "test_user_001"
            username = "sparkone"
            logger.info("Fallback authentication successful")
        else:
            logger.error(f"Authentication failed for user: {credentials.username}")
            raise HTTPException(401, "Invalid credentials")
    
    # Step 3: Generate JWT tokens
    try:
        # Access token (1 hour)
        access_payload = {
            "user_id": user_id,
            "username": username,
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "type": "access"
        }
        access_token = jwt.encode(access_payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        
        # Refresh token (24 hours)
        refresh_payload = {
            "user_id": user_id,
            "username": username,
            "exp": datetime.utcnow() + timedelta(hours=24),
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        refresh_token = jwt.encode(refresh_payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        
        logger.info(f"JWT tokens generated successfully for user: {username}")
        
        return AuthResponse(
            success=True,
            user_id=user_id,
            username=username,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=3600,  # 1 hour in seconds
            token_type="Bearer"
        )
        
    except Exception as e:
        logger.error(f"Token generation failed: {e}")
        raise HTTPException(500, "Internal server error during token generation")

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest):
    """Refresh access token using refresh token"""
    try:
        payload = jwt.decode(request.refresh_token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "refresh":
            raise HTTPException(401, "Invalid token type")
            
        # Generate new access token
        access_payload = {
            "user_id": payload["user_id"],
            "username": payload["username"],
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "type": "access"
        }
        new_access_token = jwt.encode(access_payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        
        logger.info(f"Token refreshed for user: {payload['username']}")
        
        return TokenResponse(
            access_token=new_access_token,
            expires_in=3600,
            token_type="Bearer"
        )
        
    except jwt.ExpiredSignatureError:
        logger.warning("Refresh token expired")
        raise HTTPException(401, "Refresh token expired")
    except jwt.InvalidTokenError:
        logger.warning("Invalid refresh token")
        raise HTTPException(401, "Invalid refresh token")

@router.get("/profile", response_model=UserProfile)
async def get_profile(current_user: str = Depends(get_current_user_dependency)):
    """Get user profile information"""
    logger.info(f"Profile requested for user: {current_user}")
    
    return UserProfile(
        user_id=current_user,
        username=current_user,
        display_name=current_user.title(),
        preferences={}
    )

@router.post("/validate")
async def validate_token(authorization: str = Header(None)):
    """Validate JWT token and return user info"""
    try:
        current_user = await get_current_user_dependency(authorization)
        return {
            "valid": True,
            "user_id": current_user,
            "username": current_user
        }
    except HTTPException:
        return {
            "valid": False,
            "error": "Invalid or expired token"
        }

# JWT validation dependency
async def get_current_user_dependency(authorization: str = Header(None)) -> str:
    """Dependency to extract and validate current user from JWT token"""
    if not authorization:
        raise HTTPException(401, "Authorization header required")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header format")
    
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "access":
            raise HTTPException(401, "Invalid token type")
        
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(401, "Invalid token payload")
            
        return user_id
        
    except jwt.ExpiredSignatureError:
        logger.warning("Access token expired")
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        logger.warning("Invalid access token")
        raise HTTPException(401, "Invalid token")

# Export dependency for use in other modules
def get_current_user():
    """Get current user dependency for FastAPI"""
    return Depends(get_current_user_dependency)