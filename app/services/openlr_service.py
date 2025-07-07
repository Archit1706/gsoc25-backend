"""
OpenLR (Open Location Referencing) service for encoding and decoding location references.

This service provides functionality to:
- Encode GeoJSON LineString geometries to OpenLR format
- Decode OpenLR codes back to geographic coordinates
- Validate OpenLR codes and geometries
- Handle different OpenLR formats (binary, base64, XML)
- Gracefully handle Point geometries (OpenLR not applicable)
"""

import base64
import json
import struct
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import math
from geojson import LineString
import requests

from app.config import settings
from app.core.exceptions import OpenLRException, GeospatialException


logger = logging.getLogger(__name__)


class OpenLRFormat(str, Enum):
    """OpenLR encoding formats."""

    BINARY = "binary"
    BASE64 = "base64"
    XML = "xml"


class FunctionalRoadClass(int, Enum):
    """Functional Road Class according to OpenLR specification."""

    MAIN_ROAD = 0
    FIRST_CLASS_ROAD = 1
    SECOND_CLASS_ROAD = 2
    THIRD_CLASS_ROAD = 3
    FOURTH_CLASS_ROAD = 4
    FIFTH_CLASS_ROAD = 5
    SIXTH_CLASS_ROAD = 6
    OTHER_ROAD = 7


class FormOfWay(int, Enum):
    """Form of Way according to OpenLR specification."""

    UNDEFINED = 0
    MOTORWAY = 1
    MULTIPLE_CARRIAGEWAY = 2
    SINGLE_CARRIAGEWAY = 3
    ROUNDABOUT = 4
    TRAFFICSQUARE = 5
    SLIPROAD = 6
    OTHER = 7


@dataclass
class OpenLRPoint:
    """Represents a point in OpenLR encoding."""

    longitude: float
    latitude: float
    functional_road_class: FunctionalRoadClass
    form_of_way: FormOfWay
    bearing: int  # 0-360 degrees
    distance_to_next: Optional[int] = None  # meters

    def to_coordinates(self) -> List[float]:
        """Convert to [longitude, latitude] format."""
        return [self.longitude, self.latitude]


@dataclass
class OpenLRLocationReference:
    """Complete OpenLR location reference."""

    points: List[OpenLRPoint]
    positive_offset: Optional[int] = None  # meters
    negative_offset: Optional[int] = None  # meters

    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON LineString."""
        coordinates = [point.to_coordinates() for point in self.points]
        return {"type": "LineString", "coordinates": coordinates}


class OpenLRService:
    """
    Service for OpenLR encoding and decoding operations.
    Handles LineString geometries and gracefully handles Point geometries.
    """

    def __init__(self):
        """Initialize OpenLR service with configuration."""
        self.enabled = settings.OPENLR_ENABLED
        self.format = OpenLRFormat.BASE64  # Default format
        self.map_version = getattr(settings, "OPENLR_MAP_VERSION", "latest")

        # OpenLR constants (adjusted for better compatibility)
        self.COORDINATE_FACTOR = 100000  # For coordinate precision
        self.BEARING_SECTORS = 32  # 32 sectors of 11.25 degrees each
        self.DISTANCE_INTERVALS = 256  # 256 distance intervals

        logger.info(f"OpenLR Service initialized - Enabled: {self.enabled}")

    def encode_geometry(self, geometry: Dict[str, Any]) -> Optional[str]:
        """
        Encode a GeoJSON geometry to OpenLR format.

        Args:
            geometry: GeoJSON geometry (LineString, Point, etc.)

        Returns:
            str: OpenLR encoded string (base64 by default) or None if not applicable

        Raises:
            OpenLRException: If encoding fails
            GeospatialException: If geometry is invalid
        """
        if not self.enabled:
            logger.warning("OpenLR encoding skipped - service disabled")
            return None

        try:
            # Validate geometry
            self._validate_geometry(geometry)
            geometry_type = geometry.get("type")

            # Handle different geometry types
            if geometry_type == "Point":
                logger.info("OpenLR encoding not applicable to Point geometries")
                return None
            elif geometry_type == "LineString":
                return self._encode_linestring(geometry)
            elif geometry_type == "Polygon":
                logger.info(
                    "OpenLR encoding not currently supported for Polygon geometries"
                )
                return None
            else:
                raise GeospatialException(f"Unsupported geometry type: {geometry_type}")

        except Exception as e:
            logger.error(f"OpenLR encoding failed: {e}")
            if isinstance(e, (OpenLRException, GeospatialException)):
                raise
            raise OpenLRException(f"Encoding failed: {str(e)}")

    def _encode_linestring(self, geometry: Dict[str, Any]) -> Optional[str]:
        """
        Encode a LineString geometry to OpenLR format.

        Args:
            geometry: GeoJSON LineString geometry

        Returns:
            str: OpenLR encoded string
        """
        # Extract coordinates
        coordinates = geometry.get("coordinates", [])
        if len(coordinates) < 2:
            raise GeospatialException("LineString must have at least 2 coordinates")

        # Simplified encoding for demonstration purposes
        # In a real implementation, you would use a proper OpenLR library

        # For now, create a simplified but valid OpenLR-like code
        encoded_data = self._create_simple_encoding(coordinates)

        # Convert to requested format
        if self.format == OpenLRFormat.BASE64:
            return base64.b64encode(encoded_data).decode("ascii")
        elif self.format == OpenLRFormat.BINARY:
            return encoded_data.hex()
        else:
            return self._encode_to_xml_simple(coordinates)

    def decode_openlr(self, openlr_code: str) -> Optional[Dict[str, Any]]:
        """
        Decode an OpenLR code to GeoJSON geometry.

        Args:
            openlr_code: OpenLR encoded string

        Returns:
            dict: GeoJSON LineString geometry

        Raises:
            OpenLRException: If decoding fails
        """
        if not self.enabled or not openlr_code:
            return None

        try:
            # Determine format and decode
            if self._is_base64(openlr_code):
                binary_data = base64.b64decode(openlr_code)
                coordinates = self._decode_simple_encoding(binary_data)
            elif self._is_hex(openlr_code):
                binary_data = bytes.fromhex(openlr_code)
                coordinates = self._decode_simple_encoding(binary_data)
            elif openlr_code.startswith("<"):
                coordinates = self._decode_from_xml_simple(openlr_code)
            else:
                raise OpenLRException("Unknown OpenLR format")

            # Convert to GeoJSON LineString
            return {"type": "LineString", "coordinates": coordinates}

        except Exception as e:
            logger.error(f"OpenLR decoding failed: {e}")
            if isinstance(e, OpenLRException):
                raise
            raise OpenLRException(f"Decoding failed: {str(e)}")

    def validate_openlr_code(self, openlr_code: str) -> bool:
        """
        Validate an OpenLR code format.

        Args:
            openlr_code: OpenLR code to validate

        Returns:
            bool: True if valid format
        """
        if not openlr_code:
            return False

        try:
            # Try to decode - if successful, it's valid
            result = self.decode_openlr(openlr_code)
            return result is not None
        except:
            return False

    def encode_osm_way(
        self, way_id: int, start_node: int = None, end_node: int = None
    ) -> Optional[str]:
        """
        Encode an OSM way to OpenLR format.

        Args:
            way_id: OSM way ID
            start_node: Optional start node ID
            end_node: Optional end node ID

        Returns:
            str: OpenLR encoded string

        Raises:
            OpenLRException: If encoding fails
        """
        if not self.enabled:
            return None

        try:
            # Fetch way geometry from OSM API
            geometry = self._fetch_osm_way_geometry(way_id, start_node, end_node)

            # Encode the geometry
            return self.encode_geometry(geometry)

        except Exception as e:
            logger.error(f"OSM way encoding failed: {e}")
            raise OpenLRException(f"OSM way encoding failed: {str(e)}")

    def test_encoding_roundtrip(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test encoding/decoding roundtrip for validation.

        Args:
            geometry: GeoJSON geometry to test

        Returns:
            dict: Test results with original, encoded, and decoded data
        """
        try:
            geometry_type = geometry.get("type")

            # Handle different geometry types
            if geometry_type == "Point":
                return {
                    "success": False,
                    "applicable": False,
                    "original_geometry": geometry,
                    "error": "OpenLR encoding not applicable to Point geometries",
                    "note": "Point geometries don't require OpenLR encoding",
                }
            elif geometry_type != "LineString":
                return {
                    "success": False,
                    "applicable": False,
                    "original_geometry": geometry,
                    "error": f"OpenLR encoding not supported for {geometry_type} geometries",
                }

            # Test LineString encoding/decoding
            encoded = self.encode_geometry(geometry)

            # Decode
            decoded = self.decode_openlr(encoded) if encoded else None

            # Calculate accuracy
            accuracy = (
                self._calculate_geometry_accuracy(geometry, decoded) if decoded else 0.0
            )

            return {
                "success": decoded is not None,
                "applicable": True,
                "original_geometry": geometry,
                "openlr_code": encoded,
                "decoded_geometry": decoded,
                "accuracy_meters": accuracy,
                "valid": (
                    accuracy < settings.OPENLR_ACCURACY_TOLERANCE if decoded else False
                ),
            }

        except Exception as e:
            return {
                "success": False,
                "applicable": geometry.get("type") == "LineString",
                "error": str(e),
                "original_geometry": geometry,
            }

    def is_geometry_suitable_for_openlr(
        self, geometry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if a geometry is suitable for OpenLR encoding.

        Args:
            geometry: GeoJSON geometry

        Returns:
            dict: Suitability assessment
        """
        geometry_type = geometry.get("type")

        if geometry_type == "Point":
            return {
                "suitable": False,
                "reason": "OpenLR is designed for line-based location referencing, not points",
                "alternative": "Use radius_meters field for point-based closures",
                "geometry_type": geometry_type,
            }
        elif geometry_type == "LineString":
            coordinates = geometry.get("coordinates", [])

            if len(coordinates) < 2:
                return {
                    "suitable": False,
                    "reason": "LineString must have at least 2 coordinates",
                    "geometry_type": geometry_type,
                }

            # Check minimum distance between points
            if len(coordinates) >= 2:
                total_length = 0
                for i in range(len(coordinates) - 1):
                    distance = self._calculate_haversine_distance(
                        coordinates[i], coordinates[i + 1]
                    )
                    total_length += distance

                min_length = settings.OPENLR_MIN_DISTANCE * (len(coordinates) - 1)

                if total_length < min_length:
                    return {
                        "suitable": False,
                        "reason": f"LineString too short for reliable OpenLR encoding (total: {total_length:.1f}m, min: {min_length:.1f}m)",
                        "geometry_type": geometry_type,
                        "total_length_meters": total_length,
                        "minimum_length_meters": min_length,
                    }

            return {
                "suitable": True,
                "reason": "LineString geometry is suitable for OpenLR encoding",
                "geometry_type": geometry_type,
                "total_length_meters": (
                    total_length if "total_length" in locals() else None
                ),
            }
        elif geometry_type == "Polygon":
            return {
                "suitable": False,
                "reason": "OpenLR encoding for Polygon geometries not currently supported",
                "alternative": "Consider using boundary LineString or multiple Point closures",
                "geometry_type": geometry_type,
            }
        else:
            return {
                "suitable": False,
                "reason": f"Unsupported geometry type: {geometry_type}",
                "geometry_type": geometry_type,
            }

    def get_point_location_alternatives(
        self, point_geometry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get alternative location referencing methods for Point geometries.

        Args:
            point_geometry: GeoJSON Point geometry

        Returns:
            dict: Alternative methods and recommendations
        """
        if point_geometry.get("type") != "Point":
            return {"error": "Geometry must be Point type"}

        coordinates = point_geometry.get("coordinates", [])
        if len(coordinates) != 2:
            return {"error": "Invalid Point coordinates"}

        lon, lat = coordinates

        return {
            "geometry_type": "Point",
            "coordinates": [lon, lat],
            "alternatives": {
                "radius_based": {
                    "description": "Use radius_meters field to define affected area",
                    "recommended_radius_meters": [25, 50, 100, 200],
                    "use_cases": ["Accidents", "Incidents", "Small construction sites"],
                },
                "address_based": {
                    "description": "Use nearest street address or intersection",
                    "example": "Intersection of Main St and 1st Ave",
                },
                "landmark_based": {
                    "description": "Reference to nearby landmarks",
                    "example": "Near City Hall entrance",
                },
                "coordinates": {
                    "description": "Direct coordinate reference",
                    "decimal_degrees": f"{lat:.5f}, {lon:.5f}",
                    "degrees_minutes_seconds": self._convert_to_dms(lat, lon),
                },
            },
            "recommendations": {
                "for_accidents": "Use radius_meters=50-100 with high confidence_level",
                "for_events": "Use radius_meters=100-200 depending on event size",
                "for_construction": "Consider using LineString if affects road segment",
            },
        }

    def _convert_to_dms(self, lat: float, lon: float) -> Dict[str, str]:
        """Convert decimal degrees to degrees, minutes, seconds format."""

        def dd_to_dms(dd):
            degrees = int(dd)
            minutes_float = (dd - degrees) * 60
            minutes = int(minutes_float)
            seconds = (minutes_float - minutes) * 60
            return f"{degrees}°{minutes}'{seconds:.2f}\""

        return {
            "latitude": dd_to_dms(abs(lat)) + ("N" if lat >= 0 else "S"),
            "longitude": dd_to_dms(abs(lon)) + ("E" if lon >= 0 else "W"),
        }

    def _create_simple_encoding(self, coordinates: List[List[float]]) -> bytes:
        """
        Create a simplified but valid encoding for demonstration.
        In production, use a proper OpenLR library like openlr-python.
        """
        data = bytearray()

        # Simplified header (just a marker for our custom format)
        data.append(0x42)  # Custom marker byte

        # Number of points
        data.append(len(coordinates))

        # Encode each coordinate pair
        for coord in coordinates:
            lon, lat = coord

            # Scale and pack coordinates (4 bytes each for precision)
            lon_scaled = int((lon + 180) * 1000000) % (1 << 32)
            lat_scaled = int((lat + 90) * 1000000) % (1 << 32)

            data.extend(struct.pack(">I", lon_scaled))  # Big-endian unsigned int
            data.extend(struct.pack(">I", lat_scaled))

        return bytes(data)

    def _decode_simple_encoding(self, binary_data: bytes) -> List[List[float]]:
        """
        Decode the simplified encoding back to coordinates.
        """
        if len(binary_data) < 2:
            raise OpenLRException("Binary data too short")

        data = bytearray(binary_data)

        # Check header
        if data[0] != 0x42:
            raise OpenLRException("Invalid encoding format")

        # Get number of points
        num_points = data[1]
        expected_length = 2 + (num_points * 8)  # Header + points * 8 bytes each

        if len(data) != expected_length:
            raise OpenLRException(
                f"Invalid data length: expected {expected_length}, got {len(data)}"
            )

        coordinates = []
        offset = 2

        for i in range(num_points):
            # Unpack coordinates
            lon_scaled = struct.unpack(">I", data[offset : offset + 4])[0]
            lat_scaled = struct.unpack(">I", data[offset + 4 : offset + 8])[0]

            # Scale back to degrees
            lon = (lon_scaled / 1000000.0) - 180
            lat = (lat_scaled / 1000000.0) - 90

            coordinates.append([lon, lat])
            offset += 8

        return coordinates

    def _encode_to_xml_simple(self, coordinates: List[List[float]]) -> str:
        """Encode coordinates to simplified XML format."""
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_parts.append("<OpenLR>")
        xml_parts.append("<LocationReference>")
        xml_parts.append("<GeometryType>LineString</GeometryType>")

        for i, coord in enumerate(coordinates):
            lon, lat = coord
            xml_parts.append(f'  <Point id="{i}">')
            xml_parts.append(f"    <Longitude>{lon}</Longitude>")
            xml_parts.append(f"    <Latitude>{lat}</Latitude>")
            xml_parts.append("  </Point>")

        xml_parts.append("</LocationReference>")
        xml_parts.append("</OpenLR>")

        return "\n".join(xml_parts)

    def _decode_from_xml_simple(self, xml_data: str) -> List[List[float]]:
        """Decode XML data to coordinates."""
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml_data)
            coordinates = []

            for point_elem in root.findall(".//Point"):
                longitude = float(point_elem.find("Longitude").text)
                latitude = float(point_elem.find("Latitude").text)
                coordinates.append([longitude, latitude])

            return coordinates

        except ET.ParseError as e:
            raise OpenLRException(f"Invalid XML format: {e}")

    def _validate_geometry(self, geometry: Dict[str, Any]) -> None:
        """Validate GeoJSON geometry for OpenLR processing."""
        if not isinstance(geometry, dict):
            raise GeospatialException("Geometry must be a dictionary")

        if "type" not in geometry or "coordinates" not in geometry:
            raise GeospatialException(
                "Geometry must have 'type' and 'coordinates' fields"
            )

        geometry_type = geometry.get("type")
        if geometry_type not in ["Point", "LineString", "Polygon"]:
            raise GeospatialException(f"Unsupported geometry type: {geometry_type}")

        coordinates = geometry.get("coordinates", [])

        if geometry_type == "Point":
            if not isinstance(coordinates, list) or len(coordinates) != 2:
                raise GeospatialException("Point must have exactly 2 coordinates")

            lon, lat = coordinates
            if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                raise GeospatialException(f"Invalid Point coordinates: [{lon}, {lat}]")

        elif geometry_type == "LineString":
            if not coordinates or len(coordinates) < 2:
                raise GeospatialException("LineString must have at least 2 coordinates")

            for coord in coordinates:
                if not isinstance(coord, list) or len(coord) != 2:
                    raise GeospatialException(
                        "Each coordinate must be [longitude, latitude]"
                    )

                lon, lat = coord
                if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                    raise GeospatialException(f"Invalid coordinates: [{lon}, {lat}]")

    def _fetch_osm_way_geometry(
        self, way_id: int, start_node: int = None, end_node: int = None
    ) -> Dict[str, Any]:
        """Fetch OSM way geometry from Overpass API."""
        overpass_url = "https://overpass-api.de/api/interpreter"

        query = f"""
        [out:json];
        (
          way({way_id});
        );
        (._;>;);
        out geom;
        """

        try:
            response = requests.post(overpass_url, data=query, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Extract way geometry
            way_element = None
            nodes = {}

            for element in data.get("elements", []):
                if element["type"] == "node":
                    nodes[element["id"]] = [element["lon"], element["lat"]]
                elif element["type"] == "way" and element["id"] == way_id:
                    way_element = element

            if not way_element:
                raise OpenLRException(f"OSM way {way_id} not found")

            # Build coordinate array
            coordinates = []
            node_ids = way_element.get("nodes", [])

            # Handle start/end node filtering
            if start_node:
                try:
                    start_idx = node_ids.index(start_node)
                    node_ids = node_ids[start_idx:]
                except ValueError:
                    logger.warning(f"Start node {start_node} not found in way {way_id}")

            if end_node:
                try:
                    end_idx = node_ids.index(end_node)
                    node_ids = node_ids[: end_idx + 1]
                except ValueError:
                    logger.warning(f"End node {end_node} not found in way {way_id}")

            for node_id in node_ids:
                if node_id in nodes:
                    coordinates.append(nodes[node_id])

            if len(coordinates) < 2:
                raise OpenLRException(f"Insufficient coordinates for way {way_id}")

            return {"type": "LineString", "coordinates": coordinates}

        except requests.RequestException as e:
            raise OpenLRException(f"Failed to fetch OSM data: {e}")

    def _calculate_geometry_accuracy(
        self, original: Dict[str, Any], decoded: Dict[str, Any]
    ) -> float:
        """Calculate accuracy between original and decoded geometries."""
        if not original or not decoded:
            return float("inf")

        orig_coords = original.get("coordinates", [])
        dec_coords = decoded.get("coordinates", [])

        if len(orig_coords) != len(dec_coords):
            return float("inf")

        total_distance = 0.0
        for orig, dec in zip(orig_coords, dec_coords):
            distance = self._calculate_haversine_distance(orig, dec)
            total_distance += distance

        return total_distance / len(orig_coords) if orig_coords else float("inf")

    def _calculate_haversine_distance(
        self, point1: List[float], point2: List[float]
    ) -> float:
        """Calculate distance between two points in meters."""
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

    def _is_base64(self, s: str) -> bool:
        """Check if string is valid base64."""
        try:
            base64.b64decode(s, validate=True)
            return True
        except:
            return False

    def _is_hex(self, s: str) -> bool:
        """Check if string is valid hexadecimal."""
        try:
            bytes.fromhex(s)
            return True
        except:
            return False


# Factory function for creating OpenLR service
def create_openlr_service() -> OpenLRService:
    """Create and configure OpenLR service."""
    return OpenLRService()


# Utility functions for external use
def encode_coordinates_to_openlr(coordinates: List[List[float]]) -> Optional[str]:
    """
    Utility function to encode coordinates directly to OpenLR.

    Args:
        coordinates: List of [longitude, latitude] pairs

    Returns:
        str: OpenLR encoded string or None if not applicable
    """
    service = create_openlr_service()
    geometry = {"type": "LineString", "coordinates": coordinates}
    return service.encode_geometry(geometry)


def decode_openlr_to_coordinates(openlr_code: str) -> Optional[List[List[float]]]:
    """
    Utility function to decode OpenLR directly to coordinates.

    Args:
        openlr_code: OpenLR encoded string

    Returns:
        list: List of [longitude, latitude] pairs
    """
    service = create_openlr_service()
    geometry = service.decode_openlr(openlr_code)
    return geometry.get("coordinates") if geometry else None


def check_geometry_openlr_suitability(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if a geometry is suitable for OpenLR encoding.

    Args:
        geometry: GeoJSON geometry

    Returns:
        dict: Suitability assessment
    """
    service = create_openlr_service()
    return service.is_geometry_suitable_for_openlr(geometry)
