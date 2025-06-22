"""
FastAPI application for OSM Road Closures API with OpenLR integration.
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import time
import logging
from contextlib import asynccontextmanager

from app.config import settings
from app.core.database import init_database, close_database
from app.core.exceptions import APIException, ValidationException
from app.api import closures, users, auth
from app.api import openlr  # Import OpenLR endpoints


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL), format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting OSM Road Closures API...")
    try:
        await init_database()
        logger.info("Database initialized successfully")

        # Log OpenLR status
        if settings.OPENLR_ENABLED:
            logger.info(f"OpenLR service enabled - Format: {settings.OPENLR_FORMAT}")
        else:
            logger.info("OpenLR service disabled")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down OSM Road Closures API...")
    try:
        await close_database()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)


# Add middleware
if settings.ALLOWED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add response time header to all requests."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle custom API exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    """Handle validation exceptions."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Validation failed",
            "details": exc.errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle standard HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {exc}", exc_info=True)

    if settings.DEBUG:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": str(exc),
                "type": exc.__class__.__name__,
            },
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An internal server error occurred",
            },
        )


# Health check endpoints
@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        dict: Service health status
    """
    from app.core.database import db_manager

    db_healthy = db_manager.health_check()

    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "database": "connected" if db_healthy else "disconnected",
        "openlr": {
            "enabled": settings.OPENLR_ENABLED,
            "format": settings.OPENLR_FORMAT if settings.OPENLR_ENABLED else None,
        },
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with system information.

    Returns:
        dict: Detailed system health information
    """
    from app.core.database import db_manager
    import platform

    db_info = db_manager.get_database_info()
    db_healthy = "error" not in db_info

    try:
        import psutil

        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage("/").percent,
        }
    except ImportError:
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "note": "psutil not available for detailed system metrics",
        }

    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "environment": "development" if settings.DEBUG else "production",
        "database": db_info,
        "system": system_info,
        "openlr": {
            "enabled": settings.OPENLR_ENABLED,
            "format": settings.OPENLR_FORMAT if settings.OPENLR_ENABLED else None,
            "settings": settings.openlr_settings if settings.OPENLR_ENABLED else {},
        },
    }


@app.get("/")
async def root():
    """
    Root endpoint with API information.

    Returns:
        dict: API information
    """
    return {
        "message": "OSM Road Closures API",
        "version": settings.VERSION,
        "docs_url": f"{settings.API_V1_STR}/docs",
        "openapi_url": f"{settings.API_V1_STR}/openapi.json",
        "features": {
            "openlr_enabled": settings.OPENLR_ENABLED,
            "oauth_enabled": settings.OAUTH_ENABLED,
        },
        "endpoints": {
            "closures": f"{settings.API_V1_STR}/closures",
            "users": f"{settings.API_V1_STR}/users",
            "auth": f"{settings.API_V1_STR}/auth",
            "openlr": (
                f"{settings.API_V1_STR}/openlr" if settings.OPENLR_ENABLED else None
            ),
        },
    }


# Include routers
app.include_router(
    closures.router, prefix=f"{settings.API_V1_STR}/closures", tags=["closures"]
)

app.include_router(users.router, prefix=f"{settings.API_V1_STR}/users", tags=["users"])

app.include_router(
    auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["authentication"]
)

# Include OpenLR router only if enabled
if settings.OPENLR_ENABLED:
    app.include_router(
        openlr.router, prefix=f"{settings.API_V1_STR}/openlr", tags=["openlr"]
    )
    logger.info("OpenLR endpoints enabled")
else:
    logger.info("OpenLR endpoints disabled")


# Custom OpenAPI schema
def custom_openapi():
    """
    Custom OpenAPI schema with enhanced documentation.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=f"""
        {settings.DESCRIPTION}
        
        ## Features
        
        - 🗺️ **Geospatial Support**: Store and query road closures with PostGIS
        - 📍 **OpenLR Integration**: Location referencing compatible with navigation systems
        - 🔐 **Authentication**: Secure API access with JWT tokens and OAuth
        - 📊 **Statistics**: Closure analytics and reporting
        - 🚀 **High Performance**: Optimized for real-time navigation applications
        
        ## OpenLR Support
        
        {'✅ **Enabled**: Advanced location referencing with OpenLR format' if settings.OPENLR_ENABLED else '❌ **Disabled**: OpenLR functionality not available'}
        
        {f"- **Format**: {settings.OPENLR_FORMAT}" if settings.OPENLR_ENABLED else ""}
        {f"- **Accuracy Tolerance**: {settings.OPENLR_ACCURACY_TOLERANCE}m" if settings.OPENLR_ENABLED else ""}
        
        ## Usage
        
        1. **Authentication**: Obtain an access token via `/auth/login`
        2. **Submit Closures**: POST to `/closures` with GeoJSON geometry
        3. **Query Closures**: GET from `/closures` with spatial and temporal filters
        4. **OpenLR Integration**: Use OpenLR codes for cross-platform compatibility
        {'5. **OpenLR Operations**: Use `/openlr/*` endpoints for encoding/decoding' if settings.OPENLR_ENABLED else ''}
        
        ## Rate Limits
        
        - **Authenticated**: {settings.RATE_LIMIT_REQUESTS} requests per hour
        - **Public endpoints**: Limited rate for anonymous access
        
        ## Support
        
        For issues and questions, please visit the project repository or contact the development team.
        """,
        routes=app.routes,
    )

    # Add custom schema extensions
    openapi_schema["info"]["contact"] = {
        "name": "OSM Road Closures API Support",
        "url": "https://github.com/Archit1706/temporary-road-closures",
        "email": "arath21@uic.edu",
    }

    openapi_schema["info"]["license"] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }

    # Add server information
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.osmclosures.org", "description": "Production server"},
    ]

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
    }

    # Add OpenLR-specific documentation
    if settings.OPENLR_ENABLED:
        openapi_schema["info"]["x-openlr"] = {
            "enabled": True,
            "format": settings.OPENLR_FORMAT,
            "accuracy_tolerance_meters": settings.OPENLR_ACCURACY_TOLERANCE,
            "documentation": "https://www.openlr.org/",
        }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
