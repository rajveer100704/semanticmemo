"""FastAPI Router for SemanticMemo V2 Scoped Semantic Cache."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, Depends, Header, HTTPException, Security, status  # type: ignore
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # type: ignore

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class CacheQueryRequest(BaseModel):
    """Request payload for cache query."""

    prompt: str = Field(..., min_length=1, description="The query prompt")
    resource_id: str | None = Field(default=None, description="Optional scoped resource identifier")


class CacheQueryResponse(BaseModel):
    """Response payload for cache query."""

    query_id: str
    response: str
    was_cache_hit: bool
    cost_saved_usd: float
    latency_ms: float


def create_cache_router(
    cache_instance: Any,
    default_llm_function: Callable[[str], str] | None = None,
) -> Any:
    """Create a FastAPI router bound to a SemanticMemo instance."""
    if not FASTAPI_AVAILABLE:
        msg = "FastAPI is not installed. Install with `pip install fastapi`."
        raise ImportError(msg)

    router = APIRouter(prefix="/v2/cache", tags=["Semantic Cache"])
    security_bearer = HTTPBearer(auto_error=False)

    async def get_current_user_and_tenant(
        authorization: str | None = Header(default=None),
        credentials: HTTPAuthorizationCredentials | None = Security(security_bearer),
    ):
        """Extract and authenticate principal, deriving tenant identity server-side."""
        token = credentials.credentials if credentials else None
        if authorization and not token:
            parts = authorization.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
            elif len(parts) == 1:
                token = parts[0]

        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication credentials were not provided.",
            )

        # Format: user_id or user_id:tenant_id
        if ":" in token:
            user_id, _ = token.split(":", 1)
        else:
            user_id = token

        user = cache_instance.authz_service.authenticate_user(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Unauthenticated or unknown user: {user_id}",
            )
        return user

    @router.post("/query", response_model=CacheQueryResponse)
    async def query_cache(
        request: CacheQueryRequest,
        current_user: Any = Depends(get_current_user_and_tenant),
    ):
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt must not be empty.",
            )

        # Check API-level permission
        if not cache_instance.authz_service.check_permission(current_user.id, "cache:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Missing 'cache:read' permission.",
            )

        # Check resource-level RBAC if resource_id is provided
        if request.resource_id:
            has_res_read = cache_instance.authz_service.can_read_resource(
                current_user.id,
                request.resource_id,
            )
            if not has_res_read:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Unauthorized access to resource {request.resource_id}.",
                )

        llm_fn = default_llm_function or (lambda p: f"LLM generated answer for: {p}")
        try:
            result = await cache_instance.get_or_call(
                prompt=request.prompt,
                llm_function=llm_fn,
                user_id=current_user.id,
                resource_id=request.resource_id,
            )
        except PermissionError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(exc),
            ) from exc

        return CacheQueryResponse(
            query_id=str(result.query_id),
            response=result.response,
            was_cache_hit=result.was_cache_hit,
            cost_saved_usd=float(result.cost_saved_usd),
            latency_ms=result.latency_ms,
        )

    return router
