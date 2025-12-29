"""
Authentication endpoints for Recovery Bot memOS server
Provides JWT-based authentication compatible with Android client
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import jwt
import requests
from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel

from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Response models
class AuthResponse(BaseModel):
    success: bool
    user_id: str
    username: str
    access_token: str
    refresh_token: str
    expires_in: int

class UserProfile(BaseModel):
    user_id: str
    username: str
    display_name: str
    preferences: Dict[str, Any]

class TokenValidationResponse(BaseModel):
    valid: bool
    user_id: Optional[str] = None
    username: Optional[str] = None

# JWT validation dependency - MOVED TO TOP
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

# Authentication endpoints
@router.post("/login", response_model=AuthResponse)
async def login(credentials: dict):
    """Authenticate user and return JWT tokens"""
    username = credentials.get("username")
    password = credentials.get("password")
    
    logger.info(f"Login attempt for user: {username}")
    
    if not username or not password:
        logger.warning("Missing username or password")
        raise HTTPException(400, "Username and password required")
    
    # Try PHP backend first
    try:
        php_response = requests.post(
            "https://deals.sparkonelabs.com/Recovery_Bot/api/auth/login",
            json={"username": username, "password": password},
            timeout=5
        )
        
        logger.info(f"PHP backend returned status {php_response.status_code}")
        
        if php_response.status_code == 200:
            php_data = php_response.json()
            if php_data.get("success"):
                logger.info("PHP backend authentication successful")
                return AuthResponse(
                    success=True,
                    user_id=php_data.get("user_id"),
                    username=username,
                    access_token=php_data.get("access_token"),
                    refresh_token=php_data.get("refresh_token"),
                    expires_in=3600
                )
    
    except Exception as e:
        logger.warning(f"PHP backend error: {e}")
    
    logger.warning(f"PHP backend returned status {php_response.status_code if 'php_response' in locals() else 'N/A'}")
    logger.info("Using fallback authentication")
    
    # Fallback authentication for development
    if username == "sparkone" and password == "Y3$hua3141":
        logger.info("Fallback authentication successful")
        
        # Generate JWT tokens
        access_payload = {
            "user_id": "test_user_001",
            "username": username,
            "type": "access",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
        }
        
        refresh_payload = {
            "user_id": "test_user_001", 
            "username": username,
            "type": "refresh",
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
            "iat": datetime.now(timezone.utc),
        }
        
        access_token = jwt.encode(access_payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        refresh_token = jwt.encode(refresh_payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        
        logger.info("JWT tokens generated successfully for user: " + username)
        
        return AuthResponse(
            success=True,
            user_id="test_user_001",
            username=username,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=3600
        )
    
    logger.warning(f"Authentication failed for user: {username}")
    raise HTTPException(401, "Invalid credentials")

@router.post("/refresh")
async def refresh_token(refresh_data: dict):
    """Refresh access token using refresh token"""
    token_value = refresh_data.get("refresh_token")

    if not token_value:
        raise HTTPException(400, "Refresh token required")

    try:
        payload = jwt.decode(token_value, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "refresh":
            raise HTTPException(401, "Invalid token type")
        
        # Generate new access token
        new_access_payload = {
            "user_id": payload.get("user_id"),
            "username": payload.get("username"),
            "type": "access",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
        }
        
        new_access_token = jwt.encode(new_access_payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        
        return {
            "access_token": new_access_token,
            "expires_in": 3600
        }
        
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

@router.post("/logout")
async def logout():
    """Logout user (client should discard tokens)"""
    return {"message": "Logged out successfully"}

# Export dependency for use in other modules
def get_current_user():
    """Get current user dependency for FastAPI"""
    return Depends(get_current_user_dependency)