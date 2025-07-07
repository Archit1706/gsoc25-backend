"""
API endpoints for OpenLR encoding, decoding, and validation operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.api.deps import (
    get_current_active_user,
    get_current_user_optional,
    get_current_moderator,
)
from app.models.user import User
from app.schemas.closure import GeoJSONGeometry
from app.services.closure_service import ClosureService
from app.services.openlr_service import (
    create_openlr_service,
    encode_coordinates_to_openlr,
    decode_openlr_to_coordinates,
    check_geometry_openlr_suitability,
)
from app.core.exceptions import OpenLRException, GeospatialException
from app.config import settings


router = APIRouter()


# Pydantic schemas for OpenLR operations
class OpenLREncodeRequest(BaseModel):
    """Request schema for OpenLR encoding."""

    geometry: GeoJSONGeometry = Field(..., description="GeoJSON geometry to encode")
    validate_roundtrip: bool = Field(
        True, description="Validate encoding with roundtrip test"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "LineString Encoding",
                    "value": {
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-87.6298, 41.8781], [-87.6290, 41.8785]],
                        },
                        "validate_roundtrip": True,
                    },
                },
                {
                    "name": "Point Encoding (Not Applicable)",
                    "value": {
                        "geometry": {
                            "type": "Point",
                            "coordinates": [-87.6294, 41.8783],
                        },
                        "validate_roundtrip": False,
                    },
                },
            ]
        }
    }


class OpenLRDecodeRequest(BaseModel):
    """Request schema for OpenLR decoding."""

    openlr_code: str = Field(..., description="OpenLR code to decode")

    model_config = {"json_schema_extra": {"example": {"openlr_code": "CwRbWyNG/ztP"}}}


class OpenLROSMRequest(BaseModel):
    """Request schema for OSM way encoding."""

    way_id: int = Field(..., description="OSM way ID")
    start_node: Optional[int] = Field(None, description="Optional start node ID")
    end_node: Optional[int] = Field(None, description="Optional end node ID")

    model_config = {
        "json_schema_extra": {
            "example": {"way_id": 123456789, "start_node": 1001, "end_node": 1002}
        }
    }


class OpenLRResponse(BaseModel):
    """Response schema for OpenLR operations."""

    success: bool = Field(..., description="Whether operation was successful")
    applicable: bool = Field(
        True, description="Whether OpenLR is applicable to this geometry type"
    )
    geometry_type: Optional[str] = Field(None, description="Input geometry type")
    openlr_code: Optional[str] = Field(None, description="Generated OpenLR code")
    geometry: Optional[Dict[str, Any]] = Field(None, description="Decoded geometry")
    accuracy_meters: Optional[float] = Field(
        None, description="Roundtrip accuracy in meters"
    )
    valid: Optional[bool] = Field(
        None, description="Whether encoding/decoding is valid"
    )
    error: Optional[str] = Field(None, description="Error message if operation failed")
    warning: Optional[str] = Field(None, description="Warning message")
    note: Optional[str] = Field(None, description="Informational note")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    alternatives: Optional[Dict[str, Any]] = Field(
        None, description="Alternative methods for Point geometries"
    )


class OpenLRValidationResponse(BaseModel):
    """Response schema for OpenLR validation."""

    valid: bool = Field(..., description="Whether OpenLR code is valid")
    applicable: bool = Field(
        True, description="Whether OpenLR validation is applicable"
    )
    accuracy_meters: Optional[float] = Field(None, description="Accuracy in meters")
    tolerance_meters: float = Field(..., description="Configured tolerance")
    decoded_geometry: Optional[Dict[str, Any]] = Field(
        None, description="Decoded geometry"
    )
    openlr_code: str = Field(..., description="Validated OpenLR code")
    error: Optional[str] = Field(None, description="Error message if validation failed")
    geometry_type: Optional[str] = Field(None, description="Geometry type")


class OpenLRStatsResponse(BaseModel):
    """Response schema for OpenLR statistics."""

    enabled: bool = Field(..., description="Whether OpenLR is enabled")
    format: Optional[str] = Field(None, description="OpenLR format in use")
    total_encoded: Optional[int] = Field(
        None, description="Total LineString closures with OpenLR codes"
    )
    applicable_closures: Optional[int] = Field(
        None, description="Total LineString closures (OpenLR applicable)"
    )
    point_closures: Optional[int] = Field(
        None, description="Total Point closures (OpenLR not applicable)"
    )
    encoding_success_rate: Optional[float] = Field(
        None, description="Encoding success rate percentage for LineString closures"
    )
    accuracy_tolerance: float = Field(..., description="Configured accuracy tolerance")
    settings: Dict[str, Any] = Field(..., description="OpenLR configuration settings")


class RegenerationResponse(BaseModel):
    """Response schema for OpenLR code regeneration."""

    total_processed: int = Field(..., description="Total closures processed")
    applicable: int = Field(..., description="LineString closures processed")
    skipped: int = Field(..., description="Point closures skipped")
    successful: int = Field(..., description="Successfully regenerated codes")
    failed: int = Field(..., description="Failed regenerations")
    errors: List[str] = Field(..., description="List of error messages")
    error: Optional[str] = Field(None, description="General error message")


@router.get(
    "/info",
    response_model=OpenLRStatsResponse,
    summary="Get OpenLR information",
    description="Get OpenLR configuration and statistics including geometry type breakdown.",
)
async def get_openlr_info(
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Get OpenLR configuration and usage statistics.

    Returns information about:
    - OpenLR enabled status
    - Configuration settings
    - Encoding format and accuracy tolerance
    - Statistics for LineString closures (OpenLR applicable)
    - Count of Point closures (OpenLR not applicable)
    """
    openlr_settings = settings.openlr_settings

    return OpenLRStatsResponse(
        enabled=settings.OPENLR_ENABLED,
        format=settings.OPENLR_FORMAT if settings.OPENLR_ENABLED else None,
        accuracy_tolerance=settings.OPENLR_ACCURACY_TOLERANCE,
        settings=openlr_settings,
    )


@router.post(
    "/encode",
    response_model=OpenLRResponse,
    summary="Encode geometry to OpenLR",
    description="Encode a GeoJSON geometry to OpenLR format with geometry type awareness.",
)
async def encode_geometry(
    request: OpenLREncodeRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Encode a GeoJSON geometry to OpenLR format.

    **Geometry Type Support:**
    - **LineString**: Full OpenLR encoding and validation
    - **Point**: Not applicable - returns alternative location methods
    - **Polygon**: Not currently supported

    **For LineString geometries:**
    - Returns OpenLR encoded string
    - Provides validation results if requested
    - Includes accuracy metrics

    **For Point geometries:**
    - Explains why OpenLR is not applicable
    - Suggests alternative location referencing methods
    - Provides coordinate formats and recommendations

    **Note:** This endpoint can be used to test OpenLR encoding without
    creating a closure. Authentication is optional for testing.
    """
    if not settings.OPENLR_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenLR encoding is disabled",
        )

    try:
        openlr_service = create_openlr_service()
        geometry_dict = request.geometry.dict()
        geometry_type = geometry_dict.get("type")

        # Check geometry suitability for OpenLR
        suitability = openlr_service.is_geometry_suitable_for_openlr(geometry_dict)

        if not suitability.get("suitable", False):
            # Handle Point geometries and other non-suitable types
            response_data = {
                "success": False,
                "applicable": False,
                "geometry_type": geometry_type,
                "openlr_code": None,
                "error": suitability.get("reason"),
                "note": suitability.get("alternative", ""),
            }

            # Add alternative methods for Point geometries
            if geometry_type == "Point":
                alternatives = openlr_service.get_point_location_alternatives(
                    geometry_dict
                )
                response_data["alternatives"] = alternatives

            return OpenLRResponse(**response_data)

        # Process LineString geometries
        if request.validate_roundtrip:
            # Use roundtrip test for full validation
            result = openlr_service.test_encoding_roundtrip(geometry_dict)

            return OpenLRResponse(
                success=result.get("success", False),
                applicable=result.get("applicable", True),
                geometry_type=geometry_type,
                openlr_code=result.get("openlr_code"),
                geometry=result.get("decoded_geometry"),
                accuracy_meters=result.get("accuracy_meters"),
                valid=result.get("valid", False),
                error=result.get("error"),
                metadata={
                    "original_geometry": result.get("original_geometry"),
                    "tolerance_meters": settings.OPENLR_ACCURACY_TOLERANCE,
                },
            )
        else:
            # Simple encoding without validation
            openlr_code = openlr_service.encode_geometry(geometry_dict)

            return OpenLRResponse(
                success=openlr_code is not None,
                applicable=True,
                geometry_type=geometry_type,
                openlr_code=openlr_code,
                error="Encoding failed" if openlr_code is None else None,
            )

    except (OpenLRException, GeospatialException) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenLR encoding failed: {str(e)}",
        )


@router.post(
    "/decode",
    response_model=OpenLRResponse,
    summary="Decode OpenLR code",
    description="Decode an OpenLR code to GeoJSON geometry.",
)
async def decode_openlr_code(
    request: OpenLRDecodeRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Decode an OpenLR code to GeoJSON geometry.

    **Parameters:**
    - **openlr_code**: OpenLR encoded string (base64, hex, or XML)

    **Returns:**
    - Decoded GeoJSON geometry (always LineString)
    - Success status

    **Supported Formats:**
    - Base64 encoded binary OpenLR
    - Hexadecimal encoded binary OpenLR
    - XML OpenLR format

    **Note:** OpenLR codes always decode to LineString geometries since
    OpenLR is designed for linear location referencing.
    """
    if not settings.OPENLR_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenLR decoding is disabled",
        )

    try:
        openlr_service = create_openlr_service()
        decoded_geometry = openlr_service.decode_openlr(request.openlr_code)

        return OpenLRResponse(
            success=decoded_geometry is not None,
            applicable=True,
            geometry_type="LineString",
            geometry=decoded_geometry,
            error="Decoding failed" if decoded_geometry is None else None,
            metadata={
                "openlr_code": request.openlr_code,
                "format_detected": "auto",
                "note": "OpenLR always decodes to LineString geometries",
            },
        )

    except OpenLRException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenLR decoding failed: {str(e)}",
        )


@router.post(
    "/check-suitability",
    summary="Check OpenLR suitability",
    description="Check if a geometry is suitable for OpenLR encoding.",
)
async def check_openlr_suitability(
    request: OpenLREncodeRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Check if a geometry is suitable for OpenLR encoding.

    **Returns:**
    - Suitability assessment
    - Reason for suitability/unsuitability
    - Alternative methods for Point geometries
    - Recommendations based on geometry type

    **Use Cases:**
    - Validate geometry before creating closures
    - Get recommendations for Point-based location referencing
    - Understand OpenLR limitations and alternatives
    """
    geometry_dict = request.geometry.dict()
    suitability = check_geometry_openlr_suitability(geometry_dict)

    # Add alternatives for Point geometries
    if geometry_dict.get("type") == "Point":
        openlr_service = create_openlr_service()
        alternatives = openlr_service.get_point_location_alternatives(geometry_dict)
        suitability["alternatives"] = alternatives

    return suitability


@router.post(
    "/encode-osm-way",
    response_model=OpenLRResponse,
    summary="Encode OSM way to OpenLR",
    description="Encode an OpenStreetMap way to OpenLR format.",
)
async def encode_osm_way(
    request: OpenLROSMRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Encode an OpenStreetMap way to OpenLR format.

    **Parameters:**
    - **way_id**: OSM way ID to encode
    - **start_node**: Optional start node ID to limit the way segment
    - **end_node**: Optional end node ID to limit the way segment

    **Returns:**
    - OpenLR encoded string for the OSM way
    - Way geometry as GeoJSON LineString

    **Note:** This endpoint fetches way data from OpenStreetMap API.
    Large ways may take longer to process. OSM ways are always LineString
    geometries, so OpenLR encoding is always applicable.
    """
    if not settings.OPENLR_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenLR encoding is disabled",
        )

    try:
        openlr_service = create_openlr_service()
        openlr_code = openlr_service.encode_osm_way(
            request.way_id, request.start_node, request.end_node
        )

        # Also fetch geometry for response
        geometry = openlr_service._fetch_osm_way_geometry(
            request.way_id, request.start_node, request.end_node
        )

        return OpenLRResponse(
            success=openlr_code is not None,
            applicable=True,
            geometry_type="LineString",
            openlr_code=openlr_code,
            geometry=geometry,
            error="Encoding failed" if openlr_code is None else None,
            metadata={
                "way_id": request.way_id,
                "start_node": request.start_node,
                "end_node": request.end_node,
                "source": "OpenStreetMap",
            },
        )

    except OpenLRException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OSM way encoding failed: {str(e)}",
        )


@router.post(
    "/validate",
    response_model=OpenLRValidationResponse,
    summary="Validate OpenLR code",
    description="Validate an OpenLR code format and accuracy.",
)
async def validate_openlr_code(
    request: OpenLRDecodeRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Validate an OpenLR code format and check if it can be decoded.

    **Parameters:**
    - **openlr_code**: OpenLR code to validate

    **Returns:**
    - Validation status
    - Format validation results
    - Decoding success status

    **Validation Checks:**
    - Format validity (base64, hex, XML)
    - Decoding capability
    - Basic structure validation
    """
    if not settings.OPENLR_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenLR validation is disabled",
        )

    try:
        openlr_service = create_openlr_service()

        # Validate format
        is_valid = openlr_service.validate_openlr_code(request.openlr_code)

        # Try to decode for additional validation
        decoded_geometry = None
        error_msg = None

        if is_valid:
            try:
                decoded_geometry = openlr_service.decode_openlr(request.openlr_code)
            except Exception as e:
                is_valid = False
                error_msg = f"Decoding failed: {str(e)}"

        return OpenLRValidationResponse(
            valid=is_valid,
            applicable=True,
            tolerance_meters=settings.OPENLR_ACCURACY_TOLERANCE,
            decoded_geometry=decoded_geometry,
            openlr_code=request.openlr_code,
            error=error_msg,
            geometry_type="LineString" if decoded_geometry else None,
        )

    except Exception as e:
        return OpenLRValidationResponse(
            valid=False,
            applicable=True,
            tolerance_meters=settings.OPENLR_ACCURACY_TOLERANCE,
            openlr_code=request.openlr_code,
            error=str(e),
        )


@router.get(
    "/closure/{closure_id}/validate",
    response_model=OpenLRValidationResponse,
    summary="Validate closure OpenLR",
    description="Validate OpenLR code for a specific closure.",
)
async def validate_closure_openlr(
    closure_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Validate OpenLR encoding for a specific closure.

    **Parameters:**
    - **closure_id**: ID of closure to validate

    **Returns:**
    - Validation results
    - Accuracy metrics
    - Comparison with original geometry

    **Validation Process:**
    1. Retrieves closure and its geometry
    2. Checks if OpenLR is applicable to the geometry type
    3. For LineString: Decodes the OpenLR code and compares with original
    4. For Point: Explains why OpenLR is not applicable
    5. Calculates accuracy metrics for LineString geometries

    **Geometry Type Handling:**
    - **LineString**: Full OpenLR validation with accuracy metrics
    - **Point**: Returns not applicable with explanation
    """
    try:
        service = ClosureService(db)
        validation_result = service.validate_closure_openlr(closure_id)

        return OpenLRValidationResponse(
            valid=validation_result.get("valid", False),
            applicable=validation_result.get("applicable", True),
            accuracy_meters=validation_result.get("accuracy_meters"),
            tolerance_meters=validation_result.get(
                "tolerance_meters", settings.OPENLR_ACCURACY_TOLERANCE
            ),
            decoded_geometry=validation_result.get("decoded_geometry"),
            openlr_code=validation_result.get("openlr_code", ""),
            error=validation_result.get("error"),
            geometry_type=validation_result.get("geometry_type"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )


@router.post(
    "/regenerate",
    response_model=RegenerationResponse,
    summary="Regenerate OpenLR codes",
    description="Regenerate OpenLR codes for LineString closures (moderators only).",
)
async def regenerate_openlr_codes(
    force: bool = Query(
        False, description="Force regeneration for all LineString closures"
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_moderator),
):
    """
    Regenerate OpenLR codes for LineString closures that don't have them or have invalid codes.

    **Parameters:**
    - **force**: If true, regenerate codes for all LineString closures regardless of existing status

    **Authorization:**
    - Requires moderator privileges

    **Process:**
    1. Identifies LineString closures needing OpenLR codes
    2. Skips Point closures (OpenLR not applicable)
    3. Generates codes for each LineString geometry
    4. Validates new codes
    5. Updates database with successful codes

    **Returns:**
    - Processing statistics
    - Success/failure counts
    - Number of Point closures skipped
    - Error details for failed regenerations

    **Note:** Point closures are automatically skipped since OpenLR is not applicable.
    """
    if not settings.OPENLR_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="OpenLR is disabled"
        )

    try:
        service = ClosureService(db)
        results = service.regenerate_openlr_codes(force=force)

        return RegenerationResponse(**results)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Regeneration failed: {str(e)}",
        )


@router.get(
    "/statistics",
    response_model=OpenLRStatsResponse,
    summary="Get OpenLR statistics",
    description="Get detailed OpenLR usage statistics with geometry type breakdown (moderators only).",
)
async def get_openlr_statistics(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_moderator)
):
    """
    Get detailed OpenLR usage and performance statistics.

    **Authorization:**
    - Requires moderator privileges

    **Statistics Include:**
    - Total LineString closures with OpenLR codes
    - Total Point closures (OpenLR not applicable)
    - Encoding success rate for LineString closures
    - Configuration settings
    - Performance metrics

    **Returns:**
    - Comprehensive OpenLR statistics
    - Geometry type breakdown
    - Usage patterns
    - System health metrics
    """
    try:
        service = ClosureService(db)
        stats = service.get_statistics()

        openlr_stats = stats.get("openlr", {})

        return OpenLRStatsResponse(
            enabled=openlr_stats.get("enabled", False),
            format=openlr_stats.get("format"),
            total_encoded=openlr_stats.get("total_encoded"),
            applicable_closures=openlr_stats.get("applicable_closures"),
            point_closures=stats.get("point_closures", 0),
            encoding_success_rate=openlr_stats.get("encoding_success_rate"),
            accuracy_tolerance=settings.OPENLR_ACCURACY_TOLERANCE,
            settings=settings.openlr_settings,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}",
        )


# Utility endpoints for quick testing
@router.get(
    "/test/coordinates",
    response_model=OpenLRResponse,
    summary="Test coordinate encoding",
    description="Quick test endpoint for encoding coordinate pairs.",
)
async def test_coordinate_encoding(
    coordinates: str = Query(
        ..., description="Comma-separated coordinates: lon1,lat1,lon2,lat2,..."
    ),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Quick test endpoint for encoding coordinate pairs to OpenLR.

    **Parameters:**
    - **coordinates**: Comma-separated coordinate string (lon1,lat1,lon2,lat2,...)

    **Example:**
    `/test/coordinates?coordinates=-87.6298,41.8781,-87.6290,41.8785`

    **Returns:**
    - OpenLR encoded string for LineString
    - Validation results

    **Note:** This endpoint creates a LineString from the coordinates.
    For Point testing, use the main encode endpoint.
    """
    if not settings.OPENLR_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="OpenLR is disabled"
        )

    try:
        # Parse coordinates
        coord_values = [float(x.strip()) for x in coordinates.split(",")]
        if len(coord_values) % 2 != 0 or len(coord_values) < 4:
            raise ValueError("Must provide an even number of coordinates (at least 4)")

        # Group into coordinate pairs
        coord_pairs = []
        for i in range(0, len(coord_values), 2):
            coord_pairs.append([coord_values[i], coord_values[i + 1]])

        # Encode as LineString
        openlr_code = encode_coordinates_to_openlr(coord_pairs)

        return OpenLRResponse(
            success=openlr_code is not None,
            applicable=True,
            geometry_type="LineString",
            openlr_code=openlr_code,
            error="Encoding failed" if openlr_code is None else None,
            metadata={
                "input_coordinates": coord_pairs,
                "coordinate_count": len(coord_pairs),
                "note": "Coordinates encoded as LineString geometry",
            },
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid coordinates: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Encoding failed: {str(e)}",
        )


@router.get(
    "/geometry-support",
    summary="Get geometry type support information",
    description="Get information about OpenLR support for different geometry types.",
)
async def get_geometry_support_info():
    """
    Get information about OpenLR support for different geometry types.

    Returns detailed information about which geometry types are supported
    by OpenLR and what alternatives are available for unsupported types.
    """
    return {
        "openlr_enabled": settings.OPENLR_ENABLED,
        "supported_geometries": {
            "LineString": {
                "supported": True,
                "description": "Fully supported with encoding, decoding, and validation",
                "use_cases": [
                    "Road segment closures",
                    "Lane restrictions",
                    "Construction zones",
                    "Route diversions",
                ],
                "features": [
                    "Automatic encoding",
                    "Roundtrip validation",
                    "Accuracy metrics",
                    "Navigation compatibility",
                ],
            },
            "Point": {
                "supported": False,
                "description": "Not supported by OpenLR standard",
                "reason": "OpenLR is designed for linear location referencing",
                "alternatives": {
                    "radius_based": "Use radius_meters field to define affected area",
                    "coordinate_based": "Direct coordinate reference with precision",
                    "address_based": "Reference to nearest address or intersection",
                    "landmark_based": "Reference to nearby landmarks",
                },
                "use_cases": [
                    "Accident locations",
                    "Incident responses",
                    "Point-based events",
                    "Emergency situations",
                ],
                "recommended_fields": ["radius_meters", "confidence_level"],
            },
            "Polygon": {
                "supported": False,
                "description": "Not currently supported",
                "reason": "Complex area referencing not implemented",
                "alternatives": {
                    "boundary_linestring": "Use LineString for polygon boundary",
                    "multiple_points": "Use multiple Point closures",
                    "center_point_with_radius": "Use center Point with large radius",
                },
                "future_support": "May be added in future versions",
            },
        },
        "recommendations": {
            "for_road_segments": "Use LineString geometry for OpenLR compatibility",
            "for_intersections": "Use Point geometry with appropriate radius",
            "for_large_areas": "Consider multiple Point closures or boundary LineString",
            "for_navigation_apps": "LineString closures provide best integration",
        },
        "configuration": {
            "accuracy_tolerance_meters": settings.OPENLR_ACCURACY_TOLERANCE,
            "minimum_distance_meters": settings.OPENLR_MIN_DISTANCE,
            "format": settings.OPENLR_FORMAT,
            "validation_enabled": settings.OPENLR_VALIDATE_ROUNDTRIP,
        },
    }
