"""
Enhanced business logic for closure management with full Point and LineString support plus OpenLR integration.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from geoalchemy2.functions import ST_AsGeoJSON, ST_GeomFromGeoJSON, ST_Intersects
from typing import List, Optional, Dict, Any, Tuple
import json
from datetime import datetime, timezone
import logging

from app.models.closure import Closure, ClosureType, ClosureStatus
from app.models.user import User
from app.schemas.closure import ClosureCreate, ClosureUpdate, ClosureQueryParams
from app.core.exceptions import (
    NotFoundException,
    ValidationException,
    GeospatialException,
    OpenLRException,
)
from app.services.openlr_service import OpenLRService, create_openlr_service
from app.services.spatial_service import SpatialService
from app.config import settings

logger = logging.getLogger(__name__)


class ClosureService:
    """
    Enhanced service class for closure-related business logic with Point and LineString support.
    """

    def __init__(self, db: Session):
        self.db = db
        self.openlr_service = create_openlr_service()
        self.spatial_service = SpatialService(db)
        self.openlr_enabled = settings.OPENLR_ENABLED
        self.validate_roundtrip = settings.OPENLR_VALIDATE_ROUNDTRIP

    def create_closure(self, closure_data: ClosureCreate, user_id: int) -> Closure:
        """
        Create a new closure with OpenLR encoding (for LineString) or point validation.

        Args:
            closure_data: Closure creation data
            user_id: ID of user creating the closure

        Returns:
            Closure: Created closure with OpenLR code (if applicable)

        Raises:
            ValidationException: If data is invalid
            GeospatialException: If geometry is invalid
            OpenLRException: If OpenLR encoding fails
        """
        try:
            # Validate geometry
            geometry_geojson = closure_data.geometry.dict()
            self._validate_geometry(geometry_geojson)

            # Round coordinates to 5 decimal places
            geometry_geojson = self._round_geometry_coordinates(geometry_geojson)

            # Convert GeoJSON to PostGIS geometry
            geometry_wkt = self.spatial_service.geojson_to_wkt(geometry_geojson)
            geometry_type = geometry_geojson.get("type")

            # Create closure instance
            closure = Closure(
                geometry=func.ST_GeomFromText(geometry_wkt, 4326),
                geometry_type=geometry_type,
                description=closure_data.description,
                closure_type=closure_data.closure_type.value,
                start_time=closure_data.start_time,
                end_time=closure_data.end_time,
                source=closure_data.source,
                confidence_level=closure_data.confidence_level,
                submitter_id=user_id,
                status=ClosureStatus.ACTIVE.value,
                radius_meters=closure_data.radius_meters,
            )

            # Generate OpenLR code only for LineString geometries
            if geometry_type == "LineString":
                openlr_result = self._encode_geometry_to_openlr(geometry_geojson)
                if openlr_result.get("success") and openlr_result.get("openlr_code"):
                    closure.openlr_code = openlr_result["openlr_code"]

                    # Log OpenLR encoding success
                    logger.info(
                        f"OpenLR encoding successful for LineString closure: {openlr_result.get('accuracy_meters', 'N/A')}m accuracy"
                    )

                    # Warn if accuracy is poor
                    accuracy = openlr_result.get("accuracy_meters", 0)
                    if accuracy > settings.OPENLR_ACCURACY_TOLERANCE:
                        logger.warning(
                            f"OpenLR encoding accuracy ({accuracy}m) exceeds tolerance ({settings.OPENLR_ACCURACY_TOLERANCE}m)"
                        )
                else:
                    # OpenLR encoding failed, but don't fail the entire operation
                    error_msg = openlr_result.get(
                        "error", "Unknown OpenLR encoding error"
                    )
                    logger.warning(f"OpenLR encoding failed: {error_msg}")
                    closure.openlr_code = None
            elif geometry_type == "Point":
                # Point geometries don't use OpenLR encoding
                closure.openlr_code = None
                logger.info("Point closure created without OpenLR encoding")

                # Validate radius for point closures
                if not closure_data.radius_meters:
                    logger.warning(
                        "Point closure created without radius - using default 50m"
                    )
            else:
                # For other geometry types (future expansion)
                closure.openlr_code = None
                logger.info(f"{geometry_type} closure created without OpenLR encoding")

            # Save to database
            self.db.add(closure)
            self.db.commit()
            self.db.refresh(closure)

            return closure

        except Exception as e:
            self.db.rollback()
            if isinstance(e, (ValidationException, GeospatialException)):
                raise
            raise ValidationException(f"Failed to create closure: {str(e)}")

    def update_closure(
        self, closure_id: int, closure_data: ClosureUpdate, user: User
    ) -> Closure:
        """
        Update an existing closure with OpenLR re-encoding if LineString geometry changes.

        Args:
            closure_id: Closure ID to update
            closure_data: Update data
            user: User performing the update

        Returns:
            Closure: Updated closure

        Raises:
            NotFoundException: If closure not found
            ValidationException: If user doesn't have permission or data is invalid
        """
        closure = self.get_closure_by_id(closure_id)

        # Check permissions
        if not self._can_edit_closure(closure, user):
            raise ValidationException("You don't have permission to edit this closure")

        try:
            # Update fields
            update_data = closure_data.dict(exclude_unset=True)
            geometry_updated = False

            # Handle geometry update
            if "geometry" in update_data and update_data["geometry"]:
                geometry_geojson = update_data["geometry"]
                self._validate_geometry(geometry_geojson)

                # Round coordinates to 5 decimal places
                geometry_geojson = self._round_geometry_coordinates(geometry_geojson)

                geometry_wkt = self.spatial_service.geojson_to_wkt(geometry_geojson)
                geometry_type = geometry_geojson.get("type")

                closure.geometry = func.ST_GeomFromText(geometry_wkt, 4326)
                closure.geometry_type = geometry_type
                geometry_updated = True

                # Handle OpenLR code based on new geometry type
                if geometry_type == "LineString":
                    # Regenerate OpenLR code for LineString geometry
                    openlr_result = self._encode_geometry_to_openlr(geometry_geojson)
                    if openlr_result.get("success") and openlr_result.get(
                        "openlr_code"
                    ):
                        closure.openlr_code = openlr_result["openlr_code"]
                        logger.info(
                            f"OpenLR code regenerated for updated LineString closure {closure_id}"
                        )
                    else:
                        logger.warning(
                            f"Failed to regenerate OpenLR code for closure {closure_id}: {openlr_result.get('error')}"
                        )
                        closure.openlr_code = None
                else:
                    # For Point and other geometries, clear OpenLR code
                    closure.openlr_code = None
                    logger.info(
                        f"OpenLR code cleared for {geometry_type} closure {closure_id}"
                    )

                del update_data["geometry"]

            # Validate radius_meters consistency with geometry type
            if "radius_meters" in update_data:
                if closure.geometry_type != "Point" and update_data["radius_meters"]:
                    raise ValidationException(
                        f"radius_meters can only be set for Point geometries, not {closure.geometry_type}"
                    )

            # Update other fields
            for field, value in update_data.items():
                if hasattr(closure, field):
                    setattr(closure, field, value)

            # Update status if needed
            closure.update_status_if_needed()

            self.db.commit()
            self.db.refresh(closure)

            return closure

        except Exception as e:
            self.db.rollback()
            if isinstance(e, ValidationException):
                raise
            raise ValidationException(f"Failed to update closure: {str(e)}")

    def get_closure_by_id(self, closure_id: int) -> Closure:
        """
        Get closure by ID.

        Args:
            closure_id: Closure ID

        Returns:
            Closure: Found closure

        Raises:
            NotFoundException: If closure not found
        """
        closure = self.db.query(Closure).filter(Closure.id == closure_id).first()

        if not closure:
            raise NotFoundException("Closure", closure_id)

        return closure

    def delete_closure(self, closure_id: int, user: User) -> None:
        """
        Delete a closure.

        Args:
            closure_id: Closure ID to delete
            user: User performing the deletion

        Raises:
            NotFoundException: If closure not found
            ValidationException: If user doesn't have permission
        """
        closure = self.get_closure_by_id(closure_id)

        # Check permissions
        if not self._can_delete_closure(closure, user):
            raise ValidationException(
                "You don't have permission to delete this closure"
            )

        try:
            self.db.delete(closure)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise ValidationException(f"Failed to delete closure: {str(e)}")

    def query_closures(
        self, params: ClosureQueryParams, user: Optional[User] = None
    ) -> Tuple[List[Closure], int]:
        """
        Query closures with filters and pagination.

        Args:
            params: Query parameters
            user: Optional user for permission filtering

        Returns:
            tuple: (closures, total_count)
        """
        query = self.db.query(Closure)

        # Apply filters
        if params.bbox:
            min_lon, min_lat, max_lon, max_lat = self._parse_bbox(params.bbox)
            bbox_geom = func.ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
            query = query.filter(ST_Intersects(Closure.geometry, bbox_geom))

        if params.valid_only:
            now = datetime.now(timezone.utc)
            query = query.filter(
                Closure.status == ClosureStatus.ACTIVE,
                Closure.start_time <= now,
                or_(Closure.end_time.is_(None), Closure.end_time > now),
            )

        if params.closure_type:
            query = query.filter(Closure.closure_type == params.closure_type)

        if params.geometry_type:
            query = query.filter(Closure.geometry_type == params.geometry_type)

        if params.start_time:
            query = query.filter(Closure.start_time >= params.start_time)

        if params.end_time:
            query = query.filter(
                or_(Closure.end_time.is_(None), Closure.end_time <= params.end_time)
            )

        if params.submitter_id:
            query = query.filter(Closure.submitter_id == params.submitter_id)

        # Get total count before pagination
        total = query.count()

        # Apply pagination
        skip = (params.page - 1) * params.size
        closures = query.offset(skip).limit(params.size).all()

        return closures, total

    def get_closure_with_geometry(self, closure_id: int) -> Dict[str, Any]:
        """
        Get closure with GeoJSON geometry and OpenLR validation.

        Args:
            closure_id: Closure ID

        Returns:
            dict: Closure data with GeoJSON geometry and OpenLR info
        """
        closure = self.get_closure_by_id(closure_id)

        # Get geometry as GeoJSON
        geometry_result = (
            self.db.query(ST_AsGeoJSON(Closure.geometry))
            .filter(Closure.id == closure_id)
            .first()
        )

        closure_dict = closure.to_dict()

        if geometry_result and geometry_result[0]:
            geometry = json.loads(geometry_result[0])

            # Round coordinates in the geometry
            geometry = self._round_geometry_coordinates(geometry)
            closure_dict["geometry"] = geometry

            # Add OpenLR validation info if OpenLR code exists (LineString only)
            if (
                closure.openlr_code
                and self.openlr_enabled
                and closure.geometry_type == "LineString"
            ):
                openlr_info = self._validate_openlr_code(closure.openlr_code, geometry)
                closure_dict["openlr_validation"] = openlr_info
            elif closure.geometry_type == "Point":
                # Add point-specific information
                closure_dict["openlr_validation"] = {
                    "applicable": False,
                    "reason": "OpenLR not applicable to Point geometries",
                }

        return closure_dict

    def get_closures_with_geometry(
        self, closures: List[Closure]
    ) -> List[Dict[str, Any]]:
        """
        Get multiple closures with GeoJSON geometry and OpenLR info.

        Args:
            closures: List of closures

        Returns:
            list: Closure data with GeoJSON geometry and OpenLR info
        """
        if not closures:
            return []

        closure_ids = [c.id for c in closures]

        # Get geometries for all closures
        geometry_results = (
            self.db.query(Closure.id, ST_AsGeoJSON(Closure.geometry))
            .filter(Closure.id.in_(closure_ids))
            .all()
        )

        geometry_map = {
            result[0]: json.loads(result[1]) if result[1] else None
            for result in geometry_results
        }

        # Convert closures to dict with geometry and OpenLR info
        result = []
        for closure in closures:
            closure_dict = closure.to_dict()
            geometry = geometry_map.get(closure.id)

            if geometry:
                # Round coordinates in the geometry
                geometry = self._round_geometry_coordinates(geometry)

            closure_dict["geometry"] = geometry

            # Add OpenLR validation info if enabled and applicable
            if (
                closure.openlr_code
                and self.openlr_enabled
                and geometry
                and closure.geometry_type == "LineString"
            ):
                try:
                    openlr_info = self._validate_openlr_code(
                        closure.openlr_code, geometry
                    )
                    closure_dict["openlr_validation"] = openlr_info
                except Exception as e:
                    logger.warning(
                        f"OpenLR validation failed for closure {closure.id}: {e}"
                    )
                    closure_dict["openlr_validation"] = {
                        "valid": False,
                        "error": str(e),
                    }
            elif closure.geometry_type == "Point":
                closure_dict["openlr_validation"] = {
                    "applicable": False,
                    "reason": "OpenLR not applicable to Point geometries",
                }

            result.append(closure_dict)

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get closure statistics including geometry type breakdown and OpenLR metrics.

        Returns:
            dict: Statistics data including geometry type and OpenLR metrics
        """
        now = datetime.now(timezone.utc)

        # Total closures
        total_closures = self.db.query(Closure).count()

        # Valid closures
        valid_closures = (
            self.db.query(Closure)
            .filter(
                Closure.status == ClosureStatus.ACTIVE,
                Closure.start_time <= now,
                or_(Closure.end_time.is_(None), Closure.end_time > now),
            )
            .count()
        )

        # Closures by type
        type_stats = (
            self.db.query(Closure.closure_type, func.count(Closure.id))
            .group_by(Closure.closure_type)
            .all()
        )
        by_type = {str(type_val): count for type_val, count in type_stats}

        # Closures by status
        status_stats = (
            self.db.query(Closure.status, func.count(Closure.id))
            .group_by(Closure.status)
            .all()
        )
        by_status = {str(status_val): count for status_val, count in status_stats}

        # Closures by geometry type
        geometry_type_stats = (
            self.db.query(Closure.geometry_type, func.count(Closure.id))
            .group_by(Closure.geometry_type)
            .all()
        )
        by_geometry_type = {
            str(geom_type): count for geom_type, count in geometry_type_stats
        }

        # Point and LineString counts
        point_closures = (
            self.db.query(Closure).filter(Closure.geometry_type == "Point").count()
        )
        linestring_closures = (
            self.db.query(Closure).filter(Closure.geometry_type == "LineString").count()
        )

        # Average duration
        avg_duration_result = (
            self.db.query(
                func.avg(
                    func.extract("epoch", Closure.end_time - Closure.start_time) / 3600
                )
            )
            .filter(Closure.end_time.isnot(None))
            .scalar()
        )
        avg_duration_hours = float(avg_duration_result) if avg_duration_result else None

        # OpenLR statistics
        openlr_stats = {}
        if self.openlr_enabled:
            # Count LineString closures with OpenLR codes
            linestring_with_openlr = (
                self.db.query(Closure)
                .filter(
                    Closure.geometry_type == "LineString",
                    Closure.openlr_code.isnot(None),
                )
                .count()
            )

            openlr_encoding_rate = (
                (linestring_with_openlr / linestring_closures * 100)
                if linestring_closures > 0
                else 0
            )

            openlr_stats = {
                "enabled": True,
                "total_encoded": linestring_with_openlr,
                "applicable_closures": linestring_closures,
                "encoding_success_rate": round(openlr_encoding_rate, 2),
                "format": settings.OPENLR_FORMAT,
                "note": "OpenLR only applicable to LineString geometries",
            }
        else:
            openlr_stats = {"enabled": False}

        return {
            "total_closures": total_closures,
            "valid_closures": valid_closures,
            "by_type": by_type,
            "by_status": by_status,
            "by_geometry_type": by_geometry_type,
            "point_closures": point_closures,
            "linestring_closures": linestring_closures,
            "avg_duration_hours": avg_duration_hours,
            "openlr": openlr_stats,
        }

    def validate_closure_openlr(self, closure_id: int) -> Dict[str, Any]:
        """
        Validate OpenLR encoding for a specific closure (LineString only).

        Args:
            closure_id: Closure ID to validate

        Returns:
            dict: Validation results
        """
        closure = self.get_closure_by_id(closure_id)

        if closure.geometry_type != "LineString":
            return {
                "valid": False,
                "applicable": False,
                "error": f"OpenLR validation not applicable to {closure.geometry_type} geometries",
                "closure_id": closure_id,
            }

        if not closure.openlr_code:
            return {
                "valid": False,
                "applicable": True,
                "error": "No OpenLR code available",
                "closure_id": closure_id,
            }

        # Get original geometry
        geometry_result = (
            self.db.query(ST_AsGeoJSON(Closure.geometry))
            .filter(Closure.id == closure_id)
            .first()
        )

        if not geometry_result or not geometry_result[0]:
            return {
                "valid": False,
                "applicable": True,
                "error": "No geometry available",
                "closure_id": closure_id,
            }

        geometry = json.loads(geometry_result[0])
        geometry = self._round_geometry_coordinates(geometry)
        validation_result = self._validate_openlr_code(closure.openlr_code, geometry)
        validation_result["applicable"] = True
        return validation_result

    def regenerate_openlr_codes(self, force: bool = False) -> Dict[str, Any]:
        """
        Regenerate OpenLR codes for LineString closures that don't have them or have invalid codes.

        Args:
            force: If True, regenerate all codes regardless of existing status

        Returns:
            dict: Regeneration results
        """
        if not self.openlr_enabled:
            return {"error": "OpenLR is disabled"}

        # Find LineString closures needing OpenLR codes
        query = self.db.query(Closure).filter(Closure.geometry_type == "LineString")
        if not force:
            query = query.filter(Closure.openlr_code.is_(None))

        closures = query.all()

        results = {
            "total_processed": len(closures),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
        }

        for closure in closures:
            try:
                # Skip non-LineString geometries
                if closure.geometry_type != "LineString":
                    results["skipped"] += 1
                    continue

                # Get geometry
                geometry_result = (
                    self.db.query(ST_AsGeoJSON(Closure.geometry))
                    .filter(Closure.id == closure.id)
                    .first()
                )

                if geometry_result and geometry_result[0]:
                    geometry = json.loads(geometry_result[0])
                    geometry = self._round_geometry_coordinates(geometry)

                    # Encode to OpenLR
                    openlr_result = self._encode_geometry_to_openlr(geometry)

                    if openlr_result.get("success") and openlr_result.get(
                        "openlr_code"
                    ):
                        closure.openlr_code = openlr_result["openlr_code"]
                        results["successful"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(
                            f"Closure {closure.id}: {openlr_result.get('error', 'Unknown error')}"
                        )
                else:
                    results["failed"] += 1
                    results["errors"].append(
                        f"Closure {closure.id}: No geometry available"
                    )

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Closure {closure.id}: {str(e)}")

        # Commit changes
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            results["error"] = f"Failed to commit changes: {str(e)}"

        return results

    def _round_geometry_coordinates(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Round all coordinates in a geometry to 5 decimal places.

        Args:
            geometry: GeoJSON geometry

        Returns:
            dict: Geometry with rounded coordinates
        """
        if not geometry or "coordinates" not in geometry:
            return geometry

        geometry_type = geometry.get("type")
        rounded_geometry = geometry.copy()

        if geometry_type == "Point":
            # Point: [lon, lat]
            coords = geometry["coordinates"]
            rounded_geometry["coordinates"] = [round(coords[0], 5), round(coords[1], 5)]
        elif geometry_type == "LineString":
            # LineString: [[lon, lat], [lon, lat], ...]
            coords = geometry["coordinates"]
            rounded_coords = []
            for coord in coords:
                rounded_coords.append([round(coord[0], 5), round(coord[1], 5)])
            rounded_geometry["coordinates"] = rounded_coords
        elif geometry_type == "Polygon":
            # Polygon: [[[lon, lat], ...], [[lon, lat], ...], ...]
            coords = geometry["coordinates"]
            rounded_rings = []
            for ring in coords:
                rounded_ring = []
                for coord in ring:
                    rounded_ring.append([round(coord[0], 5), round(coord[1], 5)])
                rounded_rings.append(rounded_ring)
            rounded_geometry["coordinates"] = rounded_rings

        return rounded_geometry

    def _encode_geometry_to_openlr(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode LineString geometry to OpenLR with validation.

        Args:
            geometry: GeoJSON LineString geometry

        Returns:
            dict: Encoding result with success status and details
        """
        try:
            # Only encode LineString geometries
            if geometry.get("type") != "LineString":
                return {
                    "success": False,
                    "error": f"OpenLR encoding only supports LineString, not {geometry.get('type')}",
                }

            # Encode to OpenLR
            openlr_code = self.openlr_service.encode_geometry(geometry)

            if not openlr_code:
                return {"success": False, "error": "OpenLR encoding returned None"}

            result = {"success": True, "openlr_code": openlr_code}

            # Validate roundtrip if enabled
            if self.validate_roundtrip:
                roundtrip_result = self.openlr_service.test_encoding_roundtrip(geometry)
                result.update(roundtrip_result)

                # Check if accuracy is acceptable
                accuracy = roundtrip_result.get("accuracy_meters", float("inf"))
                if accuracy > settings.OPENLR_ACCURACY_TOLERANCE:
                    result["warning"] = (
                        f"Accuracy ({accuracy}m) exceeds tolerance ({settings.OPENLR_ACCURACY_TOLERANCE}m)"
                    )

            return result

        except Exception as e:
            logger.error(f"OpenLR encoding failed: {e}")
            return {"success": False, "error": str(e)}

    def _validate_openlr_code(
        self, openlr_code: str, original_geometry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate an OpenLR code against original LineString geometry.

        Args:
            openlr_code: OpenLR code to validate
            original_geometry: Original GeoJSON LineString geometry

        Returns:
            dict: Validation results
        """
        try:
            # Decode OpenLR code
            decoded_geometry = self.openlr_service.decode_openlr(openlr_code)

            if not decoded_geometry:
                return {"valid": False, "error": "Failed to decode OpenLR code"}

            # Calculate accuracy
            accuracy = self._calculate_geometry_accuracy(
                original_geometry, decoded_geometry
            )

            return {
                "valid": accuracy <= settings.OPENLR_ACCURACY_TOLERANCE,
                "accuracy_meters": round(accuracy, 2),
                "tolerance_meters": settings.OPENLR_ACCURACY_TOLERANCE,
                "decoded_geometry": decoded_geometry,
                "openlr_code": openlr_code,
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _calculate_geometry_accuracy(
        self, geom1: Dict[str, Any], geom2: Dict[str, Any]
    ) -> float:
        """Calculate accuracy between two LineString geometries in meters."""
        if not geom1 or not geom2:
            return float("inf")

        coords1 = geom1.get("coordinates", [])
        coords2 = geom2.get("coordinates", [])

        if not coords1 or not coords2:
            return float("inf")

        # Calculate average distance between corresponding points
        total_distance = 0.0
        point_count = 0

        for i, coord1 in enumerate(coords1):
            if i < len(coords2):
                coord2 = coords2[i]
                distance = self._calculate_haversine_distance(coord1, coord2)
                total_distance += distance
                point_count += 1

        return total_distance / point_count if point_count > 0 else float("inf")

    def _calculate_haversine_distance(
        self, point1: List[float], point2: List[float]
    ) -> float:
        """Calculate distance between two points using Haversine formula."""
        import math

        R = 6371000  # Earth radius in meters
        lat1, lon1 = math.radians(point1[1]), math.radians(point1[0])
        lat2, lon2 = math.radians(point2[1]), math.radians(point2[0])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _validate_geometry(self, geometry: Dict[str, Any]) -> None:
        """
        Validate GeoJSON geometry for both Point and LineString.

        Args:
            geometry: GeoJSON geometry object

        Raises:
            GeospatialException: If geometry is invalid
        """
        if not isinstance(geometry, dict):
            raise GeospatialException("Geometry must be a GeoJSON object")

        if "type" not in geometry or "coordinates" not in geometry:
            raise GeospatialException(
                "Geometry must have 'type' and 'coordinates' fields"
            )

        geometry_type = geometry["type"]
        if geometry_type not in ["Point", "LineString", "Polygon"]:
            raise GeospatialException(f"Unsupported geometry type: {geometry_type}")

        coordinates = geometry["coordinates"]

        if geometry_type == "Point":
            if not isinstance(coordinates, list) or len(coordinates) != 2:
                raise GeospatialException("Point must have exactly 2 coordinates")

            lon, lat = coordinates
            if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                raise GeospatialException(f"Invalid Point coordinates: [{lon}, {lat}]")

        elif geometry_type == "LineString":
            if len(coordinates) < 2:
                raise GeospatialException("LineString must have at least 2 coordinates")

            # Check for minimum distance between points
            if settings.OPENLR_MIN_DISTANCE > 0:
                for i in range(len(coordinates) - 1):
                    distance = self._calculate_haversine_distance(
                        coordinates[i], coordinates[i + 1]
                    )
                    if distance < settings.OPENLR_MIN_DISTANCE:
                        logger.warning(
                            f"Points {i} and {i+1} are closer than minimum distance ({distance}m < {settings.OPENLR_MIN_DISTANCE}m)"
                        )

        elif geometry_type == "Polygon":
            if not isinstance(coordinates, list) or len(coordinates) == 0:
                raise GeospatialException("Polygon must have at least one ring")

            for ring in coordinates:
                if not isinstance(ring, list) or len(ring) < 4:
                    raise GeospatialException(
                        "Polygon ring must have at least 4 coordinates"
                    )

    def _parse_bbox(self, bbox: str) -> Tuple[float, float, float, float]:
        """
        Parse bounding box string and round coordinates.

        Args:
            bbox: Bounding box string "min_lon,min_lat,max_lon,max_lat"

        Returns:
            tuple: Parsed coordinates rounded to 5 decimal places

        Raises:
            ValidationException: If bbox format is invalid
        """
        try:
            coords = [round(float(x.strip()), 5) for x in bbox.split(",")]
            if len(coords) != 4:
                raise ValueError("Must have exactly 4 coordinates")
            return tuple(coords)
        except (ValueError, IndexError) as e:
            raise ValidationException(f"Invalid bounding box format: {e}")

    def _can_edit_closure(self, closure: Closure, user: User) -> bool:
        """
        Check if user can edit closure.

        Args:
            closure: Closure to check
            user: User to check

        Returns:
            bool: True if user can edit closure
        """
        # Moderators can edit any closure
        if user.is_moderator:
            return True

        # Users can edit their own closures
        return closure.submitter_id == user.id

    def _can_delete_closure(self, closure: Closure, user: User) -> bool:
        """
        Check if user can delete closure.

        Args:
            closure: Closure to check
            user: User to check

        Returns:
            bool: True if user can delete closure
        """
        # Same permissions as editing for now
        return self._can_edit_closure(closure, user)
