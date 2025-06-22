"""
OpenLR (Open Location Referencing) service for encoding and decoding location references.

This service provides functionality to:
- Encode GeoJSON LineString geometries to OpenLR format
- Decode OpenLR codes back to geographic coordinates
- Validate OpenLR codes and geometries
- Handle different OpenLR formats (binary, base64, XML)
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
    """

    def __init__(self):
        """Initialize OpenLR service with configuration."""
        self.enabled = settings.OPENLR_ENABLED
        self.format = OpenLRFormat.BASE64  # Default format
        self.map_version = getattr(settings, "OPENLR_MAP_VERSION", "latest")

        # OpenLR constants
        self.COORDINATE_FACTOR = 100000  # For coordinate precision
        self.BEARING_FACTOR = 11.25  # Bearing sector size in degrees
        self.DISTANCE_FACTOR = 58.6  # Distance encoding factor

        logger.info(f"OpenLR Service initialized - Enabled: {self.enabled}")

    def encode_geometry(self, geometry: Dict[str, Any]) -> Optional[str]:
        """
        Encode a GeoJSON geometry to OpenLR format.

        Args:
            geometry: GeoJSON LineString geometry

        Returns:
            str: OpenLR encoded string (base64 by default)

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

            # Extract coordinates
            coordinates = geometry.get("coordinates", [])
            if len(coordinates) < 2:
                raise GeospatialException("LineString must have at least 2 coordinates")

            # Convert to OpenLR points
            openlr_points = self._coordinates_to_openlr_points(coordinates)

            # Create location reference
            location_ref = OpenLRLocationReference(points=openlr_points)

            # Encode to binary
            binary_data = self._encode_to_binary(location_ref)

            # Convert to requested format
            if self.format == OpenLRFormat.BASE64:
                return base64.b64encode(binary_data).decode("ascii")
            elif self.format == OpenLRFormat.BINARY:
                return binary_data.hex()
            else:
                return self._encode_to_xml(location_ref)

        except Exception as e:
            logger.error(f"OpenLR encoding failed: {e}")
            if isinstance(e, (OpenLRException, GeospatialException)):
                raise
            raise OpenLRException(f"Encoding failed: {str(e)}")

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
                location_ref = self._decode_from_binary(binary_data)
            elif self._is_hex(openlr_code):
                binary_data = bytes.fromhex(openlr_code)
                location_ref = self._decode_from_binary(binary_data)
            elif openlr_code.startswith("<"):
                location_ref = self._decode_from_xml(openlr_code)
            else:
                raise OpenLRException("Unknown OpenLR format")

            # Convert to GeoJSON
            return location_ref.to_geojson()

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
            # Encode
            encoded = self.encode_geometry(geometry)

            # Decode
            decoded = self.decode_openlr(encoded) if encoded else None

            # Calculate accuracy
            accuracy = (
                self._calculate_geometry_accuracy(geometry, decoded) if decoded else 0.0
            )

            return {
                "success": decoded is not None,
                "original_geometry": geometry,
                "openlr_code": encoded,
                "decoded_geometry": decoded,
                "accuracy_meters": accuracy,
                "valid": accuracy < 50.0 if decoded else False,  # Accept 50m tolerance
            }

        except Exception as e:
            return {"success": False, "error": str(e), "original_geometry": geometry}

    def _validate_geometry(self, geometry: Dict[str, Any]) -> None:
        """Validate GeoJSON geometry for OpenLR encoding."""
        if not isinstance(geometry, dict):
            raise GeospatialException("Geometry must be a dictionary")

        if geometry.get("type") != "LineString":
            raise GeospatialException("Only LineString geometries are supported")

        coordinates = geometry.get("coordinates", [])
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

    def _coordinates_to_openlr_points(
        self, coordinates: List[List[float]]
    ) -> List[OpenLRPoint]:
        """Convert coordinate array to OpenLR points."""
        points = []

        for i, coord in enumerate(coordinates):
            lon, lat = coord

            # Calculate bearing to next point
            bearing = 0
            distance = None

            if i < len(coordinates) - 1:
                next_coord = coordinates[i + 1]
                bearing = self._calculate_bearing(coord, next_coord)
                distance = int(self._calculate_distance(coord, next_coord))

            # Estimate road class and form (would need map data for accuracy)
            frc = self._estimate_functional_road_class(coord)
            fow = self._estimate_form_of_way(coord)

            point = OpenLRPoint(
                longitude=lon,
                latitude=lat,
                functional_road_class=frc,
                form_of_way=fow,
                bearing=int(bearing),
                distance_to_next=distance,
            )
            points.append(point)

        return points

    def _encode_to_binary(self, location_ref: OpenLRLocationReference) -> bytes:
        """Encode location reference to binary format."""
        data = bytearray()

        # Header: version (3 bits) + has attributes (1 bit) + address format (4 bits)
        header = 0x03  # Version 3, line location
        data.append(header)

        # Encode each point
        for i, point in enumerate(location_ref.points):
            # Coordinates (24 bits each, relative to previous for intermediate points)
            if i == 0:
                # First point: absolute coordinates
                lon_encoded = int(
                    (point.longitude + 180) * self.COORDINATE_FACTOR / 360
                )
                lat_encoded = int((point.latitude + 90) * self.COORDINATE_FACTOR / 180)
            else:
                # Relative coordinates
                prev_point = location_ref.points[i - 1]
                lon_encoded = int(
                    (point.longitude - prev_point.longitude) * self.COORDINATE_FACTOR
                )
                lat_encoded = int(
                    (point.latitude - prev_point.latitude) * self.COORDINATE_FACTOR
                )

            # Pack coordinates (3 bytes each)
            data.extend(lon_encoded.to_bytes(3, "big", signed=True))
            data.extend(lat_encoded.to_bytes(3, "big", signed=True))

            # Attributes byte: FRC (3 bits) + FOW (3 bits) + reserved (2 bits)
            attr = (point.functional_road_class.value << 5) | (
                point.form_of_way.value << 2
            )
            data.append(attr)

            # Bearing (1 byte)
            bearing_encoded = int(point.bearing / self.BEARING_FACTOR) % 32
            data.append(bearing_encoded)

            # Distance to next point (1 byte, except for last point)
            if i < len(location_ref.points) - 1 and point.distance_to_next:
                distance_encoded = min(
                    int(point.distance_to_next / self.DISTANCE_FACTOR), 255
                )
                data.append(distance_encoded)

        return bytes(data)

    def _decode_from_binary(self, binary_data: bytes) -> OpenLRLocationReference:
        """Decode binary data to location reference."""
        if len(binary_data) < 7:
            raise OpenLRException("Binary data too short for valid OpenLR")

        data = bytearray(binary_data)
        offset = 0

        # Parse header
        header = data[offset]
        offset += 1

        version = (header >> 5) & 0x07
        if version != 3:
            raise OpenLRException(f"Unsupported OpenLR version: {version}")

        points = []
        prev_lon, prev_lat = 0, 0

        # Parse points
        while offset < len(data) - 2:  # Need at least 8 bytes for a point
            # Coordinates (6 bytes)
            lon_bytes = data[offset : offset + 3]
            lat_bytes = data[offset + 3 : offset + 6]
            offset += 6

            lon_encoded = int.from_bytes(lon_bytes, "big", signed=True)
            lat_encoded = int.from_bytes(lat_bytes, "big", signed=True)

            if len(points) == 0:
                # First point: absolute coordinates
                longitude = (lon_encoded * 360 / self.COORDINATE_FACTOR) - 180
                latitude = (lat_encoded * 180 / self.COORDINATE_FACTOR) - 90
            else:
                # Relative coordinates
                longitude = prev_lon + (lon_encoded / self.COORDINATE_FACTOR)
                latitude = prev_lat + (lat_encoded / self.COORDINATE_FACTOR)

            prev_lon, prev_lat = longitude, latitude

            # Attributes
            attr = data[offset]
            offset += 1

            frc = FunctionalRoadClass((attr >> 5) & 0x07)
            fow = FormOfWay((attr >> 2) & 0x07)

            # Bearing
            bearing_encoded = data[offset]
            offset += 1

            bearing = (bearing_encoded * self.BEARING_FACTOR) % 360

            # Distance (if not last point)
            distance = None
            if offset < len(data):
                distance_encoded = data[offset]
                offset += 1
                distance = int(distance_encoded * self.DISTANCE_FACTOR)

            point = OpenLRPoint(
                longitude=longitude,
                latitude=latitude,
                functional_road_class=frc,
                form_of_way=fow,
                bearing=int(bearing),
                distance_to_next=distance,
            )
            points.append(point)

        return OpenLRLocationReference(points=points)

    def _encode_to_xml(self, location_ref: OpenLRLocationReference) -> str:
        """Encode location reference to XML format."""
        # Simplified XML encoding
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_parts.append("<OpenLR>")
        xml_parts.append("<LocationReference>")

        for i, point in enumerate(location_ref.points):
            xml_parts.append(f'  <Point id="{i}">')
            xml_parts.append(f"    <Longitude>{point.longitude}</Longitude>")
            xml_parts.append(f"    <Latitude>{point.latitude}</Latitude>")
            xml_parts.append(f"    <FRC>{point.functional_road_class.value}</FRC>")
            xml_parts.append(f"    <FOW>{point.form_of_way.value}</FOW>")
            xml_parts.append(f"    <Bearing>{point.bearing}</Bearing>")
            if point.distance_to_next:
                xml_parts.append(f"    <Distance>{point.distance_to_next}</Distance>")
            xml_parts.append("  </Point>")

        xml_parts.append("</LocationReference>")
        xml_parts.append("</OpenLR>")

        return "\n".join(xml_parts)

    def _decode_from_xml(self, xml_data: str) -> OpenLRLocationReference:
        """Decode XML data to location reference."""
        # Simplified XML parsing (would use proper XML parser in production)
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml_data)
            points = []

            for point_elem in root.findall(".//Point"):
                longitude = float(point_elem.find("Longitude").text)
                latitude = float(point_elem.find("Latitude").text)
                frc = FunctionalRoadClass(int(point_elem.find("FRC").text))
                fow = FormOfWay(int(point_elem.find("FOW").text))
                bearing = int(point_elem.find("Bearing").text)

                distance_elem = point_elem.find("Distance")
                distance = (
                    int(distance_elem.text) if distance_elem is not None else None
                )

                point = OpenLRPoint(
                    longitude=longitude,
                    latitude=latitude,
                    functional_road_class=frc,
                    form_of_way=fow,
                    bearing=bearing,
                    distance_to_next=distance,
                )
                points.append(point)

            return OpenLRLocationReference(points=points)

        except ET.ParseError as e:
            raise OpenLRException(f"Invalid XML format: {e}")

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

    def _calculate_bearing(self, point1: List[float], point2: List[float]) -> float:
        """Calculate bearing between two points."""
        lon1, lat1 = math.radians(point1[0]), math.radians(point1[1])
        lon2, lat2 = math.radians(point2[0]), math.radians(point2[1])

        dlon = lon2 - lon1

        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
            lat2
        ) * math.cos(dlon)

        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360

        return bearing

    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
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

    def _estimate_functional_road_class(
        self, coord: List[float]
    ) -> FunctionalRoadClass:
        """Estimate functional road class (would need map data for accuracy)."""
        # Simplified estimation - in practice would need access to road network data
        return FunctionalRoadClass.THIRD_CLASS_ROAD

    def _estimate_form_of_way(self, coord: List[float]) -> FormOfWay:
        """Estimate form of way (would need map data for accuracy)."""
        # Simplified estimation - in practice would need access to road network data
        return FormOfWay.SINGLE_CARRIAGEWAY

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
            distance = self._calculate_distance(orig, dec)
            total_distance += distance

        return total_distance / len(orig_coords) if orig_coords else float("inf")

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
        str: OpenLR encoded string
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
