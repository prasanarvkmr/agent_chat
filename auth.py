"""
Authentication module for Microsoft Entra ID (Azure AD) integration.
Can be enabled/disabled via environment variable.
"""

import os
from functools import wraps
from typing import Optional, Callable
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt import PyJWKClient
from config import config
from observability import get_logger

logger = get_logger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


class EntraAuthConfig:
    """Microsoft Entra ID authentication configuration."""
    
    ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    TENANT_ID: str = os.getenv("AZURE_TENANT_ID", "")
    CLIENT_ID: str = os.getenv("AZURE_CLIENT_ID", "")
    
    # Azure AD endpoints
    @classmethod
    def get_authority(cls) -> str:
        return f"https://login.microsoftonline.com/{cls.TENANT_ID}"
    
    @classmethod
    def get_jwks_url(cls) -> str:
        return f"https://login.microsoftonline.com/{cls.TENANT_ID}/discovery/v2.0/keys"
    
    @classmethod
    def get_issuer(cls) -> str:
        return f"https://sts.windows.net/{cls.TENANT_ID}/"
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate auth configuration if enabled."""
        if not cls.ENABLED:
            return []
        
        errors = []
        if not cls.TENANT_ID:
            errors.append("AZURE_TENANT_ID is required when AUTH_ENABLED=true")
        if not cls.CLIENT_ID:
            errors.append("AZURE_CLIENT_ID is required when AUTH_ENABLED=true")
        return errors


class TokenValidator:
    """Validates JWT tokens from Microsoft Entra ID."""
    
    def __init__(self):
        self._jwks_client = None
        
    @property
    def jwks_client(self) -> PyJWKClient:
        if self._jwks_client is None and EntraAuthConfig.ENABLED:
            self._jwks_client = PyJWKClient(EntraAuthConfig.get_jwks_url())
        return self._jwks_client
    
    def validate_token(self, token: str) -> dict:
        """
        Validate a JWT token and return the claims.
        
        Args:
            token: The JWT token to validate
            
        Returns:
            Dictionary of token claims
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            # Get the signing key
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            
            # Decode and validate the token
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=EntraAuthConfig.CLIENT_ID,
                issuer=EntraAuthConfig.get_issuer(),
                options={"verify_exp": True}
            )
            
            logger.info("Token validated successfully", extra={"extra_data": {
                "user": claims.get("preferred_username", claims.get("upn", "unknown"))
            }})
            
            return claims
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            raise HTTPException(status_code=401, detail="Authentication failed")


# Global token validator
token_validator = TokenValidator()


class User:
    """Represents an authenticated user."""
    
    def __init__(self, claims: dict = None):
        self.claims = claims or {}
        self.id = claims.get("oid", "anonymous") if claims else "anonymous"
        self.email = claims.get("preferred_username", claims.get("upn", "")) if claims else ""
        self.name = claims.get("name", "Anonymous User") if claims else "Anonymous User"
        self.is_authenticated = bool(claims)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "is_authenticated": self.is_authenticated
        }


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Dependency to get the current authenticated user.
    Returns anonymous user if auth is disabled.
    """
    # If auth is disabled, return anonymous user
    if not EntraAuthConfig.ENABLED:
        return User()
    
    # Auth is enabled, validate token
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    claims = token_validator.validate_token(credentials.credentials)
    return User(claims)


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication on a route."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if EntraAuthConfig.ENABLED:
            user = kwargs.get("user")
            if not user or not user.is_authenticated:
                raise HTTPException(status_code=401, detail="Authentication required")
        return await func(*args, **kwargs)
    return wrapper
