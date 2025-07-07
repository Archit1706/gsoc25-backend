"""
Spatial operations service with Point and LineString support.
"""

import json
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class SpatialService:
    """Service for spatial operations and geometry conversions."""

    def __init__(self, db):
        self.db = db

    def geojson_to_wkt(self, geojson: Dict[str, Any]) -> Optional[str]:
        """
        Convert GeoJSON geometry to WKT format.

        Args:
            geojson: GeoJSON geometry object

        Returns:
            str: WKT representation of the geometry

        Raises:
            ValueError: If geometry type is not supported
        """
        if not geojson or "type" not in geojson or "coordinates" not in geojson:
            return None

        geometry_type = geojson.get("type")
        coordinates = geojson.get("coordinates")

        try:
            if geometry_type == "Point":
                return self._point_to_wkt(coordinates)
            elif geometry_type == "LineString":
                return self._linestring_to_wkt(coordinates)
            elif geometry_type == "Polygon":
                return self._polygon_to_wkt(coordinates)
            else:
                logger.warning(f"Unsupported geometry type: {geometry_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to convert {geometry_type} to WKT: {e}")
            return None

    def _point_to_wkt(self, coordinates: List[float]) -> str:
        """
        Convert Point coordinates to WKT.

        Args:
            coordinates: [longitude, latitude]

        Returns:
            str: WKT Point representation
        """
        if len(coordinates) != 2:
            raise ValueError("Point must have exactly 2 coordinates")

        lon, lat = coordinates
        return f"POINT({lon} {lat})"

    def _linestring_to_wkt(self, coordinates: List[List[float]]) -> str:
        """
        Convert LineString coordinates to WKT.

        Args:
            coordinates: [[lon, lat], [lon, lat], ...]

        Returns:
            str: WKT LineString representation
        """
        if len(coordinates) < 2:
            raise ValueError("LineString must have at least 2 coordinates")

        coord_pairs = []
        for coord in coordinates:
            if len(coord) != 2:
                raise ValueError("Each coordinate must have exactly 2 values")
            lon, lat = coord
            coord_pairs.append(f"{lon} {lat}")

        return f"LINESTRING({', '.join(coord_pairs)})"

    def _polygon_to_wkt(self, coordinates: List[List[List[float]]]) -> str:
        """
        Convert Polygon coordinates to WKT.

        Args:
            coordinates: [[[lon, lat], ...], [[lon, lat], ...], ...]

        Returns:
            str: WKT Polygon representation
        """
        if len(coordinates) == 0:
            raise ValueError("Polygon must have at least one ring")

        rings = []
        for ring in coordinates:
            if len(ring) < 4:
                raise ValueError("Polygon ring must have at least 4 coordinates")

            coord_pairs = []
            for coord in ring:
                if len(coord) != 2:
                    raise ValueError("Each coordinate must have exactly 2 values")
                lon, lat = coord
                coord_pairs.append(f"{lon} {lat}")

            rings.append(f"({', '.join(coord_pairs)})")

        return f"POLYGON({', '.join(rings)})"

    def wkt_to_geojson(self, wkt: str) -> Optional[Dict[str, Any]]:
        """
        Convert WKT to GeoJSON geometry (basic implementation).

        Args:
            wkt: WKT string

        Returns:
            dict: GeoJSON geometry object
        """
        if not wkt:
            return None

        try:
            wkt = wkt.strip().upper()

            if wkt.startswith("POINT"):
                return self._wkt_point_to_geojson(wkt)
            elif wkt.startswith("LINESTRING"):
                return self._wkt_linestring_to_geojson(wkt)
            elif wkt.startswith("POLYGON"):
                return self._wkt_polygon_to_geojson(wkt)
            else:
                logger.warning(f"Unsupported WKT geometry type: {wkt[:20]}...")
                return None

        except Exception as e:
            logger.error(f"Failed to convert WKT to GeoJSON: {e}")
            return None

    def _wkt_point_to_geojson(self, wkt: str) -> Dict[str, Any]:
        """Convert WKT Point to GeoJSON."""
        # Extract coordinates from "POINT(lon lat)"
        coords_str = wkt[wkt.find("(") + 1 : wkt.find(")")]
        lon, lat = map(float, coords_str.split())

        return {"type": "Point", "coordinates": [lon, lat]}

    def _wkt_linestring_to_geojson(self, wkt: str) -> Dict[str, Any]:
        """Convert WKT LineString to GeoJSON."""
        # Extract coordinates from "LINESTRING(lon1 lat1, lon2 lat2, ...)"
        coords_str = wkt[wkt.find("(") + 1 : wkt.find(")")]
        coord_pairs = coords_str.split(",")

        coordinates = []
        for pair in coord_pairs:
            lon, lat = map(float, pair.strip().split())
            coordinates.append([lon, lat])

        return {"type": "LineString", "coordinates": coordinates}

    def _wkt_polygon_to_geojson(self, wkt: str) -> Dict[str, Any]:
        """Convert WKT Polygon to GeoJSON."""
        # This is a simplified implementation
        # Extract coordinates from "POLYGON((ring1), (ring2), ...)"
        inner_content = wkt[wkt.find("(") + 1 : wkt.rfind(")")]

        # Find all rings
        rings = []
        ring_start = 0
        paren_count = 0

        for i, char in enumerate(inner_content):
            if char == "(":
                if paren_count == 0:
                    ring_start = i + 1
                paren_count += 1
            elif char == ")":
                paren_count -= 1
                if paren_count == 0:
                    ring_str = inner_content[ring_start:i]
                    ring_coords = []
                    for pair in ring_str.split(","):
                        lon, lat = map(float, pair.strip().split())
                        ring_coords.append([lon, lat])
                    rings.append(ring_coords)

        return {"type": "Polygon", "coordinates": rings}

    def calculate_point_distance(
        self, point1: List[float], point2: List[float]
    ) -> float:
        """
        Calculate distance between two points using Haversine formula.

        Args:
            point1: [longitude, latitude] of first point
            point2: [longitude, latitude] of second point

        Returns:
            float: Distance in meters
        """
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

    def point_in_polygon(
        self, point: List[float], polygon_coordinates: List[List[List[float]]]
    ) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.

        Args:
            point: [longitude, latitude]
            polygon_coordinates: Polygon coordinates from GeoJSON

        Returns:
            bool: True if point is inside polygon
        """
        x, y = point
        inside = False

        # Use first ring (exterior ring) of polygon
        if not polygon_coordinates or len(polygon_coordinates) == 0:
            return False

        ring = polygon_coordinates[0]  # exterior ring

        j = len(ring) - 1
        for i in range(len(ring)):
            xi, yi = ring[i]
            xj, yj = ring[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def calculate_linestring_length(self, coordinates: List[List[float]]) -> float:
        """
        Calculate the total length of a LineString in meters.

        Args:
            coordinates: LineString coordinates [[lon, lat], ...]

        Returns:
            float: Total length in meters
        """
        if len(coordinates) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(coordinates) - 1):
            distance = self.calculate_point_distance(coordinates[i], coordinates[i + 1])
            total_length += distance

        return total_length

    def simplify_linestring(
        self, coordinates: List[List[float]], tolerance: float = 10.0
    ) -> List[List[float]]:
        """
        Simplify a LineString using Douglas-Peucker algorithm.

        Args:
            coordinates: Original LineString coordinates
            tolerance: Simplification tolerance in meters

        Returns:
            list: Simplified coordinates
        """
        if len(coordinates) <= 2:
            return coordinates

        # Simple implementation - could be improved with proper Douglas-Peucker
        simplified = [coordinates[0]]  # Always keep first point

        for i in range(1, len(coordinates) - 1):
            # Calculate distance from current point to line between previous kept point and next point
            prev_point = simplified[-1]
            curr_point = coordinates[i]
            next_point = coordinates[i + 1]

            # Simple distance check (not true perpendicular distance)
            dist_to_prev = self.calculate_point_distance(prev_point, curr_point)

            if dist_to_prev > tolerance:
                simplified.append(curr_point)

        simplified.append(coordinates[-1])  # Always keep last point

        return simplified

    def create_buffer_around_point(
        self, point: List[float], radius_meters: float, num_points: int = 16
    ) -> List[List[float]]:
        """
        Create a circular buffer around a point.

        Args:
            point: [longitude, latitude]
            radius_meters: Buffer radius in meters
            num_points: Number of points to create the circle

        Returns:
            list: Coordinates forming a circle around the point
        """
        import math

        lon, lat = point
        lat_rad = math.radians(lat)

        # Convert radius from meters to degrees (approximate)
        radius_deg_lat = radius_meters / 111320.0  # meters per degree latitude
        radius_deg_lon = radius_meters / (
            111320.0 * math.cos(lat_rad)
        )  # adjusted for longitude

        coordinates = []
        for i in range(num_points + 1):  # +1 to close the polygon
            angle = 2 * math.pi * i / num_points
            point_lon = lon + radius_deg_lon * math.cos(angle)
            point_lat = lat + radius_deg_lat * math.sin(angle)
            coordinates.append([point_lon, point_lat])

        return coordinates

    def validate_geometry_coordinates(self, geometry: Dict[str, Any]) -> bool:
        """
        Validate that geometry coordinates are within valid ranges.

        Args:
            geometry: GeoJSON geometry

        Returns:
            bool: True if all coordinates are valid
        """
        if not geometry or "coordinates" not in geometry:
            return False

        geometry_type = geometry.get("type")
        coordinates = geometry["coordinates"]

        try:
            if geometry_type == "Point":
                return self._validate_point_coordinates(coordinates)
            elif geometry_type == "LineString":
                return self._validate_linestring_coordinates(coordinates)
            elif geometry_type == "Polygon":
                return self._validate_polygon_coordinates(coordinates)
            else:
                return False
        except (ValueError, TypeError, IndexError):
            return False

    def _validate_point_coordinates(self, coordinates: List[float]) -> bool:
        """Validate Point coordinates."""
        if len(coordinates) != 2:
            return False

        lon, lat = coordinates
        return -180 <= lon <= 180 and -90 <= lat <= 90

    def _validate_linestring_coordinates(self, coordinates: List[List[float]]) -> bool:
        """Validate LineString coordinates."""
        if len(coordinates) < 2:
            return False

        for coord in coordinates:
            if not self._validate_point_coordinates(coord):
                return False

        return True

    def _validate_polygon_coordinates(
        self, coordinates: List[List[List[float]]]
    ) -> bool:
        """Validate Polygon coordinates."""
        if len(coordinates) == 0:
            return False

        for ring in coordinates:
            if len(ring) < 4:
                return False

            for coord in ring:
                if not self._validate_point_coordinates(coord):
                    return False

            # Check if ring is closed
            if ring[0] != ring[-1]:
                return False

        return True
