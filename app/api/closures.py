"""
API endpoints for closure management with Point and LineString support.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import math

from app.core.database import get_db
from app.api.deps import (
    get_current_active_user,
    get_current_user_optional,
    get_current_moderator,
    get_pagination_params,
)
from app.models.user import User
from app.models.closure import ClosureType, ClosureStatus
from app.schemas.closure import (
    ClosureCreate,
    ClosureUpdate,
    ClosureResponse,
    ClosureListResponse,
    ClosureQueryParams,
    ClosureStatsResponse,
)
from app.services.closure_service import ClosureService
from app.core.exceptions import NotFoundException, ValidationException


router = APIRouter()


@router.post(
    "/",
    response_model=ClosureResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new closure",
    description="Submit a new temporary road closure with Point or LineString geometry.",
)
async def create_closure(
    closure_data: ClosureCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new closure.

    **Supported Geometry Types:**
    - **Point**: For incidents at specific locations (accidents, small construction sites)
    - **LineString**: For road segments (lane closures, road maintenance)

    **Point Closures:**
    - **geometry**: GeoJSON Point with coordinates [longitude, latitude]
    - **radius_meters**: Affected radius around the point (default: 50m)
    - **Use cases**: Accidents, incidents, small construction sites, events

    **LineString Closures:**
    - **geometry**: GeoJSON LineString with multiple coordinate pairs
    - **openlr_code**: Automatically generated for navigation compatibility
    - **Use cases**: Lane closures, road maintenance, planned construction

    **Common Fields:**
    - **description**: Human-readable description of the closure
    - **closure_type**: Type of closure (construction, accident, event, etc.)
    - **start_time**: When the closure begins
    - **end_time**: When the closure ends (optional for indefinite closures)
    - **source**: Source of the closure information (optional)
    - **confidence_level**: Confidence in the information (1-10, optional)

    Returns the created closure with generated ID and OpenLR code (for LineString).
    """
    service = ClosureService(db)
    closure = service.create_closure(closure_data, current_user.id)

    # Get closure with geometry for response
    closure_dict = service.get_closure_with_geometry(closure.id)

    return ClosureResponse(**closure_dict)


@router.get(
    "/",
    response_model=ClosureListResponse,
    summary="Query closures",
    description="Query closures with spatial, temporal, geometry type, and other filters.",
)
async def query_closures(
    bbox: Optional[str] = Query(
        None,
        description="Bounding box filter: 'min_lon,min_lat,max_lon,max_lat'",
        example="-87.7,41.8,-87.6,41.9",
    ),
    valid_only: bool = Query(True, description="Return only currently valid closures"),
    closure_type: Optional[ClosureType] = Query(
        None, description="Filter by closure type"
    ),
    geometry_type: Optional[str] = Query(
        None, description="Filter by geometry type: 'Point' or 'LineString'"
    ),
    start_time: Optional[str] = Query(
        None, description="Filter closures starting after this time (ISO 8601)"
    ),
    end_time: Optional[str] = Query(
        None, description="Filter closures ending before this time (ISO 8601)"
    ),
    submitter_id: Optional[int] = Query(
        None, description="Filter by submitter user ID"
    ),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=1000, description="Page size"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Query closures with various filters.

    **Spatial Filtering:**
    - Use `bbox` parameter to get closures within a geographic area
    - Format: "min_longitude,min_latitude,max_longitude,max_latitude"
    - Works for both Point and LineString geometries

    **Geometry Type Filtering:**
    - `geometry_type=Point`: Only point-based closures (accidents, incidents)
    - `geometry_type=LineString`: Only line-based closures (road segments)
    - Omit parameter to get all geometry types

    **Temporal Filtering:**
    - `valid_only=true` (default): Only return currently valid closures
    - `start_time`: Filter closures that start after the specified time
    - `end_time`: Filter closures that end before the specified time

    **Other Filters:**
    - `closure_type`: Filter by type (construction, accident, event, etc.)
    - `submitter_id`: Get closures submitted by a specific user

    **Pagination:**
    - Use `page` and `size` parameters to paginate results
    - Maximum page size is 1000

    **Example Queries:**
    - All point closures: `?geometry_type=Point`
    - Active accidents: `?closure_type=accident&valid_only=true`
    - LineString construction: `?geometry_type=LineString&closure_type=construction`

    Returns paginated list of closures with metadata.
    """
    # Validate geometry_type parameter
    if geometry_type and geometry_type not in ["Point", "LineString"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="geometry_type must be 'Point' or 'LineString'",
        )

    # Parse datetime strings if provided
    start_datetime = None
    end_datetime = None

    if start_time:
        try:
            from datetime import datetime

            start_datetime = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid start_time format. Use ISO 8601 format.",
            )

    if end_time:
        try:
            from datetime import datetime

            end_datetime = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid end_time format. Use ISO 8601 format.",
            )

    # Create query parameters
    query_params = ClosureQueryParams(
        bbox=bbox,
        valid_only=valid_only,
        closure_type=closure_type,
        geometry_type=geometry_type,
        start_time=start_datetime,
        end_time=end_datetime,
        submitter_id=submitter_id,
        page=page,
        size=size,
    )

    service = ClosureService(db)
    closures, total = service.query_closures(query_params, current_user)

    # Convert closures to response format with geometry
    closure_dicts = service.get_closures_with_geometry(closures)
    closure_responses = [
        ClosureResponse(**closure_dict) for closure_dict in closure_dicts
    ]

    # Calculate pagination metadata
    pages = math.ceil(total / size) if total > 0 else 1

    return ClosureListResponse(
        items=closure_responses, total=total, page=page, size=size, pages=pages
    )


@router.get(
    "/{closure_id}",
    response_model=ClosureResponse,
    summary="Get closure by ID",
    description="Get detailed information about a specific closure.",
)
async def get_closure(
    closure_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Get a specific closure by ID.

    Returns detailed closure information including:
    - Full geometry as GeoJSON (Point or LineString)
    - Metadata and timestamps
    - OpenLR location reference code (for LineString geometries)
    - Current status and validity state
    - Geometry-specific fields (radius_meters for Points)
    """
    service = ClosureService(db)

    try:
        closure_dict = service.get_closure_with_geometry(closure_id)
        return ClosureResponse(**closure_dict)
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Closure with ID {closure_id} not found",
        )


@router.put(
    "/{closure_id}",
    response_model=ClosureResponse,
    summary="Update closure",
    description="Update an existing closure. Only the submitter or moderators can edit.",
)
async def update_closure(
    closure_id: int,
    closure_data: ClosureUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Update an existing closure.

    **Permissions:**
    - Users can update their own closures
    - Moderators can update any closure

    **Updatable Fields:**
    - Geometry (will regenerate OpenLR code for LineString)
    - Description and metadata
    - Start/end times
    - Status (for moderators)
    - Closure type
    - Radius (for Point geometries)

    **Geometry Type Changes:**
    - Can change from Point to LineString or vice versa
    - OpenLR code will be generated for new LineString geometries
    - OpenLR code will be cleared for new Point geometries
    - radius_meters will be validated based on new geometry type

    **Automatic Updates:**
    - `updated_at` timestamp is automatically set
    - OpenLR code is regenerated if LineString geometry changes
    - Status may be automatically updated based on timing
    """
    service = ClosureService(db)

    try:
        closure = service.update_closure(closure_id, closure_data, current_user)
        closure_dict = service.get_closure_with_geometry(closure.id)
        return ClosureResponse(**closure_dict)
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Closure with ID {closure_id} not found",
        )
    except ValidationException as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.delete(
    "/{closure_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete closure",
    description="Delete a closure. Only the submitter or moderators can delete.",
)
async def delete_closure(
    closure_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete a closure.

    **Permissions:**
    - Users can delete their own closures
    - Moderators can delete any closure

    **Note:** This is a hard delete operation. The closure and all its data
    will be permanently removed from the database.
    """
    service = ClosureService(db)

    try:
        service.delete_closure(closure_id, current_user)
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Closure with ID {closure_id} not found",
        )
    except ValidationException as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.get(
    "/statistics/summary",
    response_model=ClosureStatsResponse,
    summary="Get closure statistics",
    description="Get statistical summary of closures including geometry type breakdown.",
)
async def get_closure_statistics(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Get statistical summary of closures.

    Returns:
    - Total number of closures
    - Number of currently valid closures
    - Breakdown by closure type
    - Breakdown by status
    - Breakdown by geometry type (Point vs LineString)
    - Count of Point and LineString closures
    - Average closure duration
    - OpenLR encoding statistics (for LineString closures)

    This endpoint can be used for dashboards and monitoring.
    """
    service = ClosureService(db)
    stats = service.get_statistics()

    return ClosureStatsResponse(**stats)


@router.get(
    "/geometry-types",
    summary="Get supported geometry types",
    description="Get information about supported geometry types and their characteristics.",
)
async def get_supported_geometry_types():
    """
    Get information about supported geometry types.

    Returns detailed information about each supported geometry type,
    including use cases, required fields, and capabilities.
    """
    return {
        "supported_types": [
            {
                "type": "Point",
                "description": "Single coordinate point for location-specific closures",
                "use_cases": [
                    "Vehicle accidents",
                    "Incident responses",
                    "Small construction sites",
                    "Event-based closures",
                    "Emergency situations",
                ],
                "required_fields": ["coordinates"],
                "optional_fields": ["radius_meters"],
                "openlr_support": False,
                "default_radius_meters": 50,
                "coordinate_format": "[longitude, latitude]",
                "example": {
                    "type": "Point",
                    "coordinates": [-87.62940, 41.87830],
                },
            },
            {
                "type": "LineString",
                "description": "Linear feature for road segment closures",
                "use_cases": [
                    "Road construction",
                    "Lane closures",
                    "Maintenance work",
                    "Planned roadwork",
                    "Route diversions",
                ],
                "required_fields": ["coordinates"],
                "optional_fields": [],
                "openlr_support": True,
                "minimum_points": 2,
                "coordinate_format": "[[lon1, lat1], [lon2, lat2], ...]",
                "example": {
                    "type": "LineString",
                    "coordinates": [[-87.62980, 41.87810], [-87.62900, 41.87850]],
                },
            },
        ],
        "selection_guidelines": {
            "use_point_for": [
                "Specific location incidents",
                "Circular affected areas",
                "Temporary events",
                "Emergency responses",
            ],
            "use_linestring_for": [
                "Road segment closures",
                "Lane-specific restrictions",
                "Linear construction zones",
                "Route modifications",
            ],
        },
        "openlr_notes": {
            "point_geometries": "OpenLR encoding not applicable - use radius_meters instead",
            "linestring_geometries": "Automatic OpenLR encoding for navigation compatibility",
            "minimum_length": "LineString should be at least 15 meters for reliable OpenLR encoding",
        },
    }


@router.get(
    "/user/{user_id}",
    response_model=ClosureListResponse,
    summary="Get user's closures",
    description="Get closures submitted by a specific user.",
)
async def get_user_closures(
    user_id: int,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=1000, description="Page size"),
    valid_only: bool = Query(False, description="Return only valid closures"),
    geometry_type: Optional[str] = Query(
        None, description="Filter by geometry type: 'Point' or 'LineString'"
    ),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Get closures submitted by a specific user.

    **Privacy:**
    - Anyone can view closures by user ID
    - User information is not included in the response

    **Filtering:**
    - Use `valid_only=true` to see only currently valid closures
    - Use `geometry_type` to filter by Point or LineString geometries
    - Results are ordered by creation date (newest first)
    """
    # Validate geometry_type parameter
    if geometry_type and geometry_type not in ["Point", "LineString"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="geometry_type must be 'Point' or 'LineString'",
        )

    query_params = ClosureQueryParams(
        submitter_id=user_id,
        valid_only=valid_only,
        geometry_type=geometry_type,
        page=page,
        size=size,
    )

    service = ClosureService(db)
    closures, total = service.query_closures(query_params, current_user)

    # Convert to response format
    closure_dicts = service.get_closures_with_geometry(closures)
    closure_responses = [
        ClosureResponse(**closure_dict) for closure_dict in closure_dicts
    ]

    pages = math.ceil(total / size) if total > 0 else 1

    return ClosureListResponse(
        items=closure_responses, total=total, page=page, size=size, pages=pages
    )


@router.post(
    "/{closure_id}/status",
    response_model=ClosureResponse,
    summary="Update closure status",
    description="Update the status of a closure (moderators only).",
)
async def update_closure_status(
    closure_id: int,
    new_status: ClosureStatus,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_moderator),
):
    """
    Update closure status (moderators only).

    **Available Statuses:**
    - `active`: Closure is currently in effect
    - `expired`: Closure has ended naturally
    - `cancelled`: Closure was cancelled before completion
    - `planned`: Closure is scheduled for the future

    **Moderator Action:**
    This endpoint is restricted to moderators for status management.
    Regular users should use the general update endpoint.
    """
    service = ClosureService(db)

    try:
        # Create update object with just status
        update_data = ClosureUpdate(status=new_status)
        closure = service.update_closure(closure_id, update_data, current_user)

        closure_dict = service.get_closure_with_geometry(closure.id)
        return ClosureResponse(**closure_dict)
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Closure with ID {closure_id} not found",
        )


@router.get(
    "/geometry-type/{geometry_type}",
    response_model=ClosureListResponse,
    summary="Get closures by geometry type",
    description="Get all closures of a specific geometry type.",
)
async def get_closures_by_geometry_type(
    geometry_type: str,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=1000, description="Page size"),
    valid_only: bool = Query(True, description="Return only valid closures"),
    closure_type: Optional[ClosureType] = Query(
        None, description="Filter by closure type"
    ),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Get all closures of a specific geometry type.

    **Geometry Types:**
    - `Point`: Point-based closures (accidents, incidents, etc.)
    - `LineString`: Line-based closures (road segments, lanes, etc.)

    **Additional Filtering:**
    - Use `closure_type` to further filter by closure type
    - Use `valid_only` to get only currently active closures

    **Use Cases:**
    - Get all accident locations: `/geometry-type/Point?closure_type=accident`
    - Get all road construction: `/geometry-type/LineString?closure_type=construction`
    """
    # Validate geometry_type parameter
    if geometry_type not in ["Point", "LineString"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="geometry_type must be 'Point' or 'LineString'",
        )

    query_params = ClosureQueryParams(
        geometry_type=geometry_type,
        valid_only=valid_only,
        closure_type=closure_type,
        page=page,
        size=size,
    )

    service = ClosureService(db)
    closures, total = service.query_closures(query_params, current_user)

    # Convert to response format
    closure_dicts = service.get_closures_with_geometry(closures)
    closure_responses = [
        ClosureResponse(**closure_dict) for closure_dict in closure_dicts
    ]

    pages = math.ceil(total / size) if total > 0 else 1

    return ClosureListResponse(
        items=closure_responses, total=total, page=page, size=size, pages=pages
    )
