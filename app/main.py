"""
Semantic Cache Gateway: HTTP layer.

The request flow itself lives in app/gateway.py; this module only handles
HTTP concerns: validation, request IDs, response headers, error mapping,
start-up warm-up and logging.

Run it:
    uvicorn app.main:app --reload
"""

import logging
import secrets
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import gateway
from app.cache import get_cache
from app.config import settings
from app.llm_providers import LLMConfigurationError, LLMProviderError
from app.schemas import (
    ClearCacheResponse,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from app.stats import stats

VERSION = "1.0.0"

# The React dashboard, built by `npm run build` in frontend/ (see frontend/README.md).
# Optional: without a build, the API works as before and / goes to the API docs.
UI_DIR = Path(__file__).parent / "ui"

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("semantic_cache_gateway")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if settings.warmup_on_startup:
        log.info("warming up provider=%s model=%s", settings.llm_provider, settings.embedding_model)
        gateway.warm_up()
    yield


app = FastAPI(
    title="Semantic Cache Gateway",
    description=(
        "An LLM gateway that serves semantically similar questions from a cache "
        "and falls back to retrieval-augmented generation on a miss."
    ),
    version=VERSION,
    lifespan=lifespan,
    responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Cache", "X-Response-Time-Ms"],
)


@app.middleware("http")
async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    # Honour an upstream ID (load balancer, client) so logs correlate end to end.
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _error(request: Request, code: int, detail: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    body = ErrorResponse(detail=detail, request_id=request_id).model_dump()
    return JSONResponse(status_code=code, content=body)


@app.exception_handler(LLMProviderError)
async def llm_error_handler(request: Request, exc: LLMProviderError) -> JSONResponse:
    log.warning("llm_error request_id=%s error=%s", request.state.request_id, exc)
    return _error(request, status.HTTP_502_BAD_GATEWAY, "Upstream LLM provider failed.")


@app.exception_handler(LLMConfigurationError)
async def llm_config_handler(request: Request, exc: LLMConfigurationError) -> JSONResponse:
    log.error("llm_misconfigured error=%s", exc)
    return _error(request, status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/" if (UI_DIR / "index.html").is_file() else "/docs")


def mount_ui(target: FastAPI, ui_dir: Path) -> bool:
    """Serve the built dashboard at /ui if a build exists. Returns whether it was mounted."""
    if not (ui_dir / "index.html").is_file():
        return False
    target.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")
    return True


mount_ui(app, UI_DIR)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    cache = get_cache()
    try:
        cache.size()
        reachable = True
    except Exception:  # e.g. Redis down: report it rather than 500
        reachable = False
    return HealthResponse(
        status="ok" if reachable else "degraded",
        llm_provider=settings.llm_provider,
        cache_backend=cache.backend,
        cache_reachable=reachable,
        embedding_model=settings.embedding_model,
        version=VERSION,
    )


@app.post("/query", response_model=QueryResponse, tags=["gateway"])
def query(body: QueryRequest, request: Request, response: Response) -> QueryResponse:
    result = gateway.handle_query(body.query, use_cache=body.use_cache)
    request_id: str = request.state.request_id

    response.headers["X-Cache"] = result.cache_status.upper()
    response.headers["X-Response-Time-Ms"] = str(result.latency_ms)
    log.info(
        "query request_id=%s cache=%s similarity=%s vetoes=%s latency_ms=%s",
        request_id,
        result.cache_status,
        result.similarity,
        ",".join(result.guard_vetoes) or "-",
        result.latency_ms,
    )
    return QueryResponse(
        answer=result.answer,
        cache_status=result.cache_status,
        cache_hit=result.cache_hit,
        matched_query=result.matched_query,
        similarity=result.similarity,
        guard_vetoes=result.guard_vetoes,
        latency_ms=result.latency_ms,
        sources=result.sources,
        provider=result.provider,
        request_id=request_id,
    )


@app.get("/stats", response_model=StatsResponse, tags=["ops"])
def get_stats() -> StatsResponse:
    cache = get_cache()
    return StatsResponse(
        **stats.snapshot(),
        cache_size=cache.size(),
        cache_backend=cache.backend,
        similarity_threshold=cache.threshold,
    )


@app.delete("/cache", response_model=ClearCacheResponse, tags=["ops"])
def clear_cache(x_admin_token: str | None = Header(default=None)) -> ClearCacheResponse:
    """Empty the cache. Requires ADMIN_TOKEN to be configured and sent as X-Admin-Token."""
    if not settings.admin_token:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "Cache clearing is disabled (no ADMIN_TOKEN)"
        )
    if not x_admin_token or not secrets.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid admin token")
    removed = get_cache().clear()
    log.info("cache_cleared removed=%d", removed)
    return ClearCacheResponse(removed=removed)
