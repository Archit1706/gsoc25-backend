"""
Pydantic schemas for closure data validation and serialization with Point and LineString support.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum

from app.models.closure import ClosureType, ClosureStatus


class GeoJSONGeometry(BaseModel):
    """GeoJSON geometry schema supporting Point and LineString."""

    type: str = Field(..., description="Geometry type (Point or LineString)")
    coordinates: Union[List[float], List[List[float]]] = Field(
        ..., description="Coordinate array"
    )

    @field_validator("type")
    @classmethod
    def validate_geometry_type(cls, v):
        """Validate geometry type."""
        allowed_types = ["Point", "LineString", "Polygon"]
        if v not in allowed_types:
            raise ValueError(f"Geometry type must be one of {allowed_types}")
        return v

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, v, info):
        """Validate coordinates based on geometry type."""
        # Get the geometry type from the validated data
        data = info.data if hasattr(info, "data") else {}
        geometry_type = data.get("type")

        if geometry_type == "Point":
            # Point should have exactly 2 coordinates [lon, lat]
            if not isinstance(v, list) or len(v) != 2:
                raise ValueError("Point must have exactly 2 coordinates [lon, lat]")

            lon, lat = v
            if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
                raise ValueError("Point coordinates must be numbers")

            # Validate longitude and latitude ranges
            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude {lon} is out of range [-180, 180]")
            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude {lat} is out of range [-90, 90]")

            # Round to 5 decimal places
            return [round(lon, 5), round(lat, 5)]

        elif geometry_type == "LineString":
            if not isinstance(v, list) or len(v) < 2:
                raise ValueError("LineString must have at least 2 coordinates")

            # Round coordinates to 5 decimal places and validate ranges
            rounded_coords = []
            for coord in v:
                if not isinstance(coord, list) or len(coord) != 2:
                    raise ValueError(
                        "Each coordinate must have exactly 2 values [lon, lat]"
                    )
                # Validate longitude and latitude ranges
                lon, lat = coord
                if not isinstance(lon, (int, float)) or not isinstance(
                    lat, (int, float)
                ):
                    raise ValueError("Coordinates must be numbers")
                if not (-180 <= lon <= 180):
                    raise ValueError(f"Longitude {lon} is out of range [-180, 180]")
                if not (-90 <= lat <= 90):
                    raise ValueError(f"Latitude {lat} is out of range [-90, 90]")

                # Round to 5 decimal places
                rounded_coords.append([round(lon, 5), round(lat, 5)])

            return rounded_coords

        elif geometry_type == "Polygon":
            # Basic polygon validation (for future use)
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError("Polygon must have at least one ring")

            # Validate each ring
            rounded_rings = []
            for ring in v:
                if not isinstance(ring, list) or len(ring) < 4:
                    raise ValueError("Polygon ring must have at least 4 coordinates")

                rounded_ring = []
                for coord in ring:
                    if not isinstance(coord, list) or len(coord) != 2:
                        raise ValueError(
                            "Each coordinate must have exactly 2 values [lon, lat]"
                        )
                    lon, lat = coord
                    if not isinstance(lon, (int, float)) or not isinstance(
                        lat, (int, float)
                    ):
                        raise ValueError("Coordinates must be numbers")
                    if not (-180 <= lon <= 180):
                        raise ValueError(f"Longitude {lon} is out of range [-180, 180]")
                    if not (-90 <= lat <= 90):
                        raise ValueError(f"Latitude {lat} is out of range [-90, 90]")

                    rounded_ring.append([round(lon, 5), round(lat, 5)])

                # Ensure ring is closed
                if rounded_ring[0] != rounded_ring[-1]:
                    rounded_ring.append(rounded_ring[0])

                rounded_rings.append(rounded_ring)

            return rounded_rings

        return v


class ClosureBase(BaseModel):
    """Base closure schema with common fields."""

    description: str = Field(
        ..., min_length=10, max_length=1000, description="Closure description"
    )
    closure_type: ClosureType = Field(..., description="Type of closure")
    start_time: datetime = Field(..., description="Closure start time")
    end_time: Optional[datetime] = Field(None, description="Closure end time")
    source: Optional[str] = Field(
        None, max_length=100, description="Source of closure information"
    )
    confidence_level: Optional[int] = Field(
        None, ge=1, le=10, description="Confidence level (1-10)"
    )
    radius_meters: Optional[int] = Field(
        None,
        ge=1,
        le=10000,
        description="Affected radius in meters (for Point geometries only)",
    )

    @field_validator("end_time")
    @classmethod
    def validate_end_time(cls, v, info):
        """Validate that end_time is after start_time."""
        if v is not None and hasattr(info, "data") and "start_time" in info.data:
            if v <= info.data["start_time"]:
                raise ValueError("end_time must be after start_time")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v):
        """Validate description content."""
        if not v.strip():
            raise ValueError("Description cannot be empty or only whitespace")
        return v.strip()


class ClosureCreate(ClosureBase):
    """Schema for creating a new closure."""

    geometry: GeoJSONGeometry = Field(..., description="Closure geometry as GeoJSON")

    @model_validator(mode="after")
    def validate_geometry_specific_fields(self):
        """Validate geometry-specific field combinations."""
        geometry_type = self.geometry.type

        if geometry_type == "Point":
            # For Point geometries, radius_meters is recommended
            if self.radius_meters is None:
                # Set default radius for point closures
                self.radius_meters = 50  # 50 meter default radius
        else:
            # For non-Point geometries, radius_meters should not be set
            if self.radius_meters is not None:
                raise ValueError(
                    f"radius_meters can only be set for Point geometries, not {geometry_type}"
                )

        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "LineString Construction Closure",
                    "value": {
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [-87.62980, 41.87810],
                                [-87.62900, 41.87850],
                            ],
                        },
                        "description": "Water main repair blocking eastbound traffic on Madison Street",
                        "closure_type": "construction",
                        "start_time": "2025-07-06T08:00:00Z",
                        "end_time": "2025-07-06T18:00:00Z",
                        "source": "City of Chicago",
                        "confidence_level": 9,
                    },
                },
                {
                    "name": "Point Accident Closure",
                    "value": {
                        "geometry": {
                            "type": "Point",
                            "coordinates": [-87.62940, 41.87830],
                        },
                        "description": "Vehicle accident blocking intersection at Madison and Wells",
                        "closure_type": "accident",
                        "start_time": "2025-07-06T14:30:00Z",
                        "end_time": "2025-07-06T16:00:00Z",
                        "source": "Chicago Police Department",
                        "confidence_level": 10,
                        "radius_meters": 100,
                    },
                },
            ]
        }
    }


class ClosureUpdate(BaseModel):
    """Schema for updating an existing closure."""

    geometry: Optional[GeoJSONGeometry] = Field(None, description="Updated geometry")
    description: Optional[str] = Field(
        None, min_length=10, max_length=1000, description="Updated description"
    )
    closure_type: Optional[ClosureType] = Field(
        None, description="Updated closure type"
    )
    start_time: Optional[datetime] = Field(None, description="Updated start time")
    end_time: Optional[datetime] = Field(None, description="Updated end time")
    status: Optional[ClosureStatus] = Field(None, description="Updated status")
    source: Optional[str] = Field(None, max_length=100, description="Updated source")
    confidence_level: Optional[int] = Field(
        None, ge=1, le=10, description="Updated confidence level"
    )
    radius_meters: Optional[int] = Field(
        None,
        ge=1,
        le=10000,
        description="Updated radius in meters (for Point geometries only)",
    )

    @model_validator(mode="after")
    def validate_time_consistency(self):
        """Validate time consistency when both times are provided."""
        if self.start_time is not None and self.end_time is not None:
            if self.end_time <= self.start_time:
                raise ValueError("end_time must be after start_time")
        return self

    @model_validator(mode="after")
    def validate_geometry_radius_consistency(self):
        """Validate radius is only set for Point geometries."""
        if self.geometry and self.radius_meters is not None:
            if self.geometry.type != "Point":
                raise ValueError(
                    f"radius_meters can only be set for Point geometries, not {self.geometry.type}"
                )
        return self


class ClosureResponse(ClosureBase):
    """Schema for closure responses."""

    id: int = Field(..., description="Closure ID")
    geometry: Dict[str, Any] = Field(..., description="Closure geometry as GeoJSON")
    geometry_type: str = Field(..., description="Geometry type (Point or LineString)")
    status: ClosureStatus = Field(..., description="Current closure status")
    openlr_code: Optional[str] = Field(None, description="OpenLR location reference")
    submitter_id: int = Field(..., description="ID of user who submitted this closure")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    is_valid: bool = Field(..., description="Whether closure is currently valid")
    duration_hours: Optional[float] = Field(
        None, description="Closure duration in hours"
    )
    is_point_closure: bool = Field(..., description="Whether this is a point closure")
    is_linestring_closure: bool = Field(
        ..., description="Whether this is a linestring closure"
    )

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "name": "LineString Closure Response",
                    "value": {
                        "id": 123,
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [-87.62980, 41.87810],
                                [-87.62900, 41.87850],
                            ],
                        },
                        "geometry_type": "LineString",
                        "description": "Water main repair blocking eastbound traffic",
                        "closure_type": "construction",
                        "start_time": "2025-07-06T08:00:00Z",
                        "end_time": "2025-07-06T18:00:00Z",
                        "status": "active",
                        "openlr_code": "CwRbWyNG/ztP",
                        "submitter_id": 456,
                        "created_at": "2025-07-05T14:30:00Z",
                        "updated_at": "2025-07-05T14:30:00Z",
                        "is_valid": True,
                        "duration_hours": 10.0,
                        "source": "City of Chicago",
                        "confidence_level": 9,
                        "radius_meters": None,
                        "is_point_closure": False,
                        "is_linestring_closure": True,
                    },
                },
                {
                    "name": "Point Closure Response",
                    "value": {
                        "id": 124,
                        "geometry": {
                            "type": "Point",
                            "coordinates": [-87.62940, 41.87830],
                        },
                        "geometry_type": "Point",
                        "description": "Vehicle accident blocking intersection",
                        "closure_type": "accident",
                        "start_time": "2025-07-06T14:30:00Z",
                        "end_time": "2025-07-06T16:00:00Z",
                        "status": "active",
                        "openlr_code": None,
                        "submitter_id": 456,
                        "created_at": "2025-07-06T14:35:00Z",
                        "updated_at": "2025-07-06T14:35:00Z",
                        "is_valid": True,
                        "duration_hours": 1.5,
                        "source": "Chicago Police Department",
                        "confidence_level": 10,
                        "radius_meters": 100,
                        "is_point_closure": True,
                        "is_linestring_closure": False,
                    },
                },
            ]
        },
    }


class ClosureListResponse(BaseModel):
    """Schema for paginated closure list responses."""

    items: List[ClosureResponse] = Field(..., description="List of closures")
    total: int = Field(..., description="Total number of closures")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")


class ClosureQueryParams(BaseModel):
    """Schema for closure query parameters."""

    bbox: Optional[str] = Field(
        None, description="Bounding box as 'min_lon,min_lat,max_lon,max_lat'"
    )
    valid_only: bool = Field(True, description="Return only valid closures")
    closure_type: Optional[ClosureType] = Field(
        None, description="Filter by closure type"
    )
    geometry_type: Optional[str] = Field(
        None, description="Filter by geometry type (Point or LineString)"
    )
    start_time: Optional[datetime] = Field(
        None, description="Filter closures starting after this time"
    )
    end_time: Optional[datetime] = Field(
        None, description="Filter closures ending before this time"
    )
    submitter_id: Optional[int] = Field(None, description="Filter by submitter user ID")
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(50, ge=1, le=1000, description="Page size")


class ClosureStatsResponse(BaseModel):
    """Schema for closure statistics."""

    total_closures: int = Field(..., description="Total number of closures")
    valid_closures: int = Field(..., description="Number of valid closures")
    by_type: Dict[str, int] = Field(..., description="Closures by type")
    by_status: Dict[str, int] = Field(..., description="Closures by status")
    by_geometry_type: Dict[str, int] = Field(
        ..., description="Closures by geometry type"
    )
    avg_duration_hours: Optional[float] = Field(
        None, description="Average closure duration in hours"
    )
    point_closures: int = Field(..., description="Number of point closures")
    linestring_closures: int = Field(..., description="Number of linestring closures")
