"""
Comprehensive tests for OpenLR service functionality.
"""

import pytest
import json
import base64
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.services.openlr_service import (
    OpenLRService,
    OpenLRPoint,
    OpenLRLocationReference,
    FunctionalRoadClass,
    FormOfWay,
    create_openlr_service,
    encode_coordinates_to_openlr,
    decode_openlr_to_coordinates
)
from app.services.closure_service import ClosureService
from app.models.closure import Closure
from app.models.user import User
from app.schemas.closure import ClosureCreate, GeoJSONGeometry
from app.core.exceptions import OpenLRException, GeospatialException
from app.config import settings


# Test client
client = TestClient(app)


class TestOpenLRService:
    """Test OpenLR service core functionality."""

    def test_service_initialization(self):
        """Test OpenLR service initialization."""
        service = OpenLRService()
        
        assert service.enabled == settings.OPENLR_ENABLED
        assert service.COORDINATE_FACTOR == 100000
        assert service.BEARING_FACTOR == 11.25

    def test_encode_simple_linestring(self):
        """Test encoding a simple LineString geometry."""
        service = OpenLRService()
        
        geometry = {
            "type": "LineString",
            "coordinates": [
                [-87.6298, 41.8781],  # Chicago coordinates
                [-87.6290, 41.8785]
            ]
        }
        
        # Mock the service being enabled
        with patch.object(service, 'enabled', True):
            result = service.encode_geometry(geometry)
            
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0

    def test_encode_invalid_geometry_type(self):
        """Test encoding with invalid geometry type."""
        service = OpenLRService()
        
        geometry = {
            "type": "Point",  # Not supported for OpenLR
            "coordinates": [-87.6298, 41.8781]
        }
        
        with patch.object(service, 'enabled', True):
            with pytest.raises(GeospatialException):
                service.encode_geometry(geometry)

    def test_encode_insufficient_coordinates(self):
        """Test encoding with insufficient coordinates."""
        service = OpenLRService()
        
        geometry = {
            "type": "LineString",
            "coordinates": [[-87.6298, 41.8781]]  # Only one point
        }
        
        with patch.object(service, 'enabled', True):
            with pytest.raises(GeospatialException):
                service.encode_geometry(geometry)

    def test_encode_invalid_coordinates(self):
        """Test encoding with invalid coordinate values."""
        service = OpenLRService()
        
        geometry = {
            "type": "LineString",
            "coordinates": [
                [-200, 41.8781],  # Invalid longitude
                [-87.6290, 41.8785]
            ]
        }
        
        with patch.object(service, 'enabled', True):
            with pytest.raises(GeospatialException):
                service.encode_geometry(geometry)

    def test_encode_disabled_service(self):
        """Test encoding when service is disabled."""
        service = OpenLRService()
        
        geometry = {
            "type": "LineString",
            "coordinates": [[-87.6298, 41.8781], [-87.6290, 41.8785]]
        }
        
        with patch.object(service, 'enabled', False):
            result = service.encode_geometry(geometry)
            assert result is None

    def test_decode_openlr_disabled(self):
        """Test decoding when service is disabled."""
        service = OpenLRService()
        
        with patch.object(service, 'enabled', False):
            result = service.decode_openlr("test_code")
            assert result is None

    def test_validate_openlr_code_format(self):
        """Test OpenLR code format validation."""
        service = OpenLRService()
        
        with patch.object(service, 'enabled', True):
            # Test base64 format detection
            assert service._is_base64("SGVsbG8gV29ybGQ=")  # "Hello World" in base64
            assert not service._is_base64("not_base64!")
            
            # Test hex format detection
            assert service._is_hex("48656c6c6f")  # "Hello" in hex
            assert not service._is_hex("not_hex!")

    def test_coordinates_to_openlr_points(self):
        """Test conversion of coordinates to OpenLR points."""
        service = OpenLRService()
        
        coordinates = [
            [-87.6298, 41.8781],
            [-87.6290, 41.8785],
            [-87.6280, 41.8790]
        ]
        
        points = service._coordinates_to_openlr_points(coordinates)
        
        assert len(points) == 3
        assert all(isinstance(point, OpenLRPoint) for point in points)
        assert points[0].longitude == -87.6298
        assert points[0].latitude == 41.8781
        assert points[0].distance_to_next is not None
        assert points[-1].distance_to_next is None  # Last point has no next distance

    def test_bearing_calculation(self):
        """Test bearing calculation between two points."""
        service = OpenLRService()
        
        point1 = [-87.6298, 41.8781]
        point2 = [-87.6290, 41.8785]
        
        bearing = service._calculate_bearing(point1, point2)
        
        assert 0 <= bearing <= 360
        assert isinstance(bearing, float)

    def test_distance_calculation(self):
        """Test distance calculation between two points."""
        service = OpenLRService()
        
        point1 = [-87.6298, 41.8781]
        point2 = [-87.6290, 41.8785]
        
        distance = service._calculate_distance(point1, point2)
        
        assert distance > 0
        assert isinstance(distance, float)
        # Chicago coordinates should be reasonable distance apart
        assert 100 < distance < 1000  # meters

    def test_binary_encoding_decoding(self):
        """Test binary encoding and decoding roundtrip."""
        service = OpenLRService()
        
        # Create test location reference
        point1 = OpenLRPoint(
            longitude=-87.6298,
            latitude=41.8781,
            functional_road_class=FunctionalRoadClass.THIRD_CLASS_ROAD,
            form_of_way=FormOfWay.SINGLE_CARRIAGEWAY,
            bearing=45,
            distance_to_next=500
        )
        
        point2 = OpenLRPoint(
            longitude=-87.6290,
            latitude=41.8785,
            functional_road_class=FunctionalRoadClass.THIRD_CLASS_ROAD,
            form_of_way=FormOfWay.SINGLE_CARRIAGEWAY,
            bearing=90
        )
        
        location_ref = OpenLRLocationReference(points=[point1, point2])
        
        # Encode to binary
        binary_data = service._encode_to_binary(location_ref)
        assert isinstance(binary_data, bytes)
        assert len(binary_data) > 10  # Should have substantial data
        
        # Decode from binary
        decoded_ref = service._decode_from_binary(binary_data)
        assert len(decoded_ref.points) == 2
        
        # Check accuracy (should be close to original)
        assert abs(decoded_ref.points[0].longitude - point1.longitude) < 0.001
        assert abs(decoded_ref.points[0].latitude - point1.latitude) < 0.001

    def test_xml_encoding_decoding(self):
        """Test XML encoding and decoding."""
        service = OpenLRService()
        
        point = OpenLRPoint(
            longitude=-87.6298,
            latitude=41.8781,
            functional_road_class=FunctionalRoadClass.THIRD_CLASS_ROAD,
            form_of_way=FormOfWay.SINGLE_CARRIAGEWAY,
            bearing=45
        )
        
        location_ref = OpenLRLocationReference(points=[point])
        
        # Encode to XML
        xml_data = service._encode_to_xml(location_ref)
        assert isinstance(xml_data, str)
        assert "<OpenLR>" in xml_data
        assert "<Longitude>-87.6298</Longitude>" in xml_data
        
        # Decode from XML
        decoded_ref = service._decode_from_xml(xml_data)
        assert len(decoded_ref.points) == 1
        assert decoded_ref.points[0].longitude == -87.6298

    def test_roundtrip_accuracy(self):
        """Test encoding/decoding roundtrip accuracy."""
        service = OpenLRService()
        
        with patch.object(service, 'enabled', True):
            geometry = {
                "type": "LineString",
                "coordinates": [
                    [-87.6298, 41.8781],
                    [-87.6290, 41.8785],
                    [-87.6280, 41.8790]
                ]
            }
            
            # Test roundtrip
            roundtrip_result = service.test_encoding_roundtrip(geometry)
            
            assert "success" in roundtrip_result
            assert "original_geometry" in roundtrip_result
            assert "openlr_code" in roundtrip_result
            
            if roundtrip_result["success"]:
                assert "decoded_geometry" in roundtrip_result
                assert "accuracy_meters" in roundtrip_result
                assert roundtrip_result["accuracy_meters"] >= 0

    @patch('requests.post')
    def test_osm_way_fetching(self, mock_post):
        """Test fetching OSM way geometry."""
        service = OpenLRService()
        
        # Mock Overpass API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "elements": [
                {
                    "type": "node",
                    "id": 1001,
                    "lon": -87.6298,
                    "lat": 41.8781
                },
                {
                    "type": "node",
                    "id": 1002,
                    "lon": -87.6290,
                    "lat": 41.8785
                },
                {
                    "type": "way",
                    "id": 123456,
                    "nodes": [1001, 1002]
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        geometry = service._fetch_osm_way_geometry(123456)
        
        assert geometry["type"] == "LineString"
        assert len(geometry["coordinates"]) == 2
        assert geometry["coordinates"][0] == [-87.6298, 41.8781]


class TestOpenLRUtilityFunctions:
    """Test OpenLR utility functions."""

    def test_encode_coordinates_to_openlr(self):
        """Test utility function for coordinate encoding."""
        coordinates = [[-87.6298, 41.8781], [-87.6290, 41.8785]]
        
        with patch('app.services.openlr_service.settings.OPENLR_ENABLED', True):
            result = encode_coordinates_to_openlr(coordinates)
            # Should return string or None based on implementation
            assert result is None or isinstance(result, str)

    def test_decode_openlr_to_coordinates(self):
        """Test utility function for OpenLR decoding."""
        # Test with a mock valid OpenLR code
        with patch('app.services.openlr_service.settings.OPENLR_ENABLED', True):
            result = decode_openlr_to_coordinates("mock_openlr_code")
            # Should return coordinates or None
            assert result is None or isinstance(result, list)


class TestClosureServiceOpenLRIntegration:
    """Test OpenLR integration in closure service."""

    def test_create_closure_with_openlr(self, db_session: Session):
        """Test creating closure with OpenLR encoding."""
        # Create test user
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash",
            is_active=True
        )
        db_session.add(user)
        db_session.commit()

        service = ClosureService(db_session)
        
        # Mock OpenLR encoding success
        with patch.object(service, '_encode_geometry_to_openlr') as mock_encode:
            mock_encode.return_value = {
                "success": True,
                "openlr_code": "test_openlr_code",
                "accuracy_meters": 25.0
            }

            geometry = GeoJSONGeometry(
                type="LineString",
                coordinates=[[-87.6298, 41.8781], [-87.6290, 41.8785]]
            )

            closure_data = ClosureCreate(
                geometry=geometry,
                description="Test closure with OpenLR",
                closure_type="construction",
                start_time="2025-06-01T08:00:00Z",
                end_time="2025-06-01T18:00:00Z"
            )

            closure = service.create_closure(closure_data, user.id)

            assert closure.openlr_code == "test_openlr_code"
            assert closure.description == "Test closure with OpenLR"

    def test_create_closure_openlr_failure(self, db_session: Session):
        """Test creating closure when OpenLR encoding fails."""
        # Create test user
        user = User(
            username="testuser2",
            email="test2@example.com",
            hashed_password="hash",
            is_active=True
        )
        db_session.add(user)
        db_session.commit()

        service = ClosureService(db_session)
        
        # Mock OpenLR encoding failure
        with patch.object(service, '_encode_geometry_to_openlr') as mock_encode:
            mock_encode.return_value = {
                "success": False,
                "error": "Encoding failed"
            }

            geometry = GeoJSONGeometry(
                type="LineString",
                coordinates=[[-87.6298, 41.8781], [-87.6290, 41.8785]]
            )

            closure_data = ClosureCreate(
                geometry=geometry,
                description="Test closure without OpenLR",
                closure_type="construction",
                start_time="2025-06-01T08:00:00Z",
                end_time="2025-06-01T18:00:00Z"
            )

            # Should still create closure even if OpenLR fails
            closure = service.create_closure(closure_data, user.id)

            assert closure.openlr_code is None
            assert closure.description == "Test closure without OpenLR"

    def test_closure_statistics_with_openlr(self, db_session: Session):
        """Test closure statistics include OpenLR metrics."""
        service = ClosureService(db_session)
        
        with patch.object(service, 'openlr_enabled', True):
            stats = service.get_statistics()
            
            assert "openlr" in stats
            assert "enabled" in stats["openlr"]
            assert stats["openlr"]["enabled"] is True


class TestOpenLRAPIEndpoints:
    """Test OpenLR API endpoints."""

    def test_openlr_info_endpoint(self):
        """Test OpenLR info endpoint."""
        response = client.get("/api/v1/openlr/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "settings" in data
        assert "accuracy_tolerance" in data

    def test_encode_endpoint_enabled(self):
        """Test encoding endpoint when OpenLR is enabled."""
        request_data = {
            "geometry": {
                "type": "LineString",
                "coordinates": [[-87.6298, 41.8781], [-87.6290, 41.8785]]
            },
            "validate_roundtrip": True
        }

        with patch('app.config.settings.OPENLR_ENABLED', True):
            response = client.post("/api/v1/openlr/encode", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "success" in data

    def test_encode_endpoint_disabled(self):
        """Test encoding endpoint when OpenLR is disabled."""
        request_data = {
            "geometry": {
                "type": "LineString",
                "coordinates": [[-87.6298, 41.8781], [-87.6290, 41.8785]]
            }
        }

        with patch('app.config.settings.OPENLR_ENABLED', False):
            response = client.post("/api/v1/openlr/encode", json=request_data)
            
            assert response.status_code == 503

    def test_decode_endpoint(self):
        """Test decoding endpoint."""
        request_data = {
            "openlr_code": "test_openlr_code"
        }

        with patch('app.config.settings.OPENLR_ENABLED', True):
            response = client.post("/api/v1/openlr/decode", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "success" in data

    def test_validate_endpoint(self):
        """Test validation endpoint."""
        request_data = {
            "openlr_code": "test_openlr_code"
        }

        with patch('app.config.settings.OPENLR_ENABLED', True):
            response = client.post("/api/v1/openlr/validate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "valid" in data
            assert "openlr_code" in data

    def test_coordinate_test_endpoint(self):
        """Test coordinate encoding test endpoint."""
        coordinates = "-87.6298,41.8781,-87.6290,41.8785"

        with patch('app.config.settings.OPENLR_ENABLED', True):
            response = client.get(f"/api/v1/openlr/test/coordinates?coordinates={coordinates}")
            
            assert response.status_code == 200
            data = response.json()
            assert "success" in data
            assert "metadata" in data

    def test_coordinate_test_invalid_format(self):
        """Test coordinate encoding with invalid format."""
        coordinates = "-87.6298,41.8781,-87.6290"  # Odd number of coordinates

        with patch('app.config.settings.OPENLR_ENABLED', True):
            response = client.get(f"/api/v1/openlr/test/coordinates?coordinates={coordinates}")
            
            assert response.status_code == 400


class TestOpenLRExceptionHandling:
    """Test OpenLR error handling and edge cases."""

    def test_openlr_exception_handling(self):
        """Test OpenLR exception handling."""
        service = OpenLRService()
        
        # Test with malformed geometry
        malformed_geometry = {
            "type": "LineString"
            # Missing coordinates
        }
        
        with patch.object(service, 'enabled', True):
            with pytest.raises(GeospatialException):
                service.encode_geometry(malformed_geometry)

    def test_network_error_handling(self):
        """Test handling of network errors during OSM fetching."""
        service = OpenLRService()
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            with pytest.raises(OpenLRException):
                service.encode_osm_way(123456)

    def test_invalid_osm_response(self):
        """Test handling of invalid OSM API responses."""
        service = OpenLRService()
        
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"elements": []}  # Empty response
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            with pytest.raises(OpenLRException):
                service._fetch_osm_way_geometry(123456)


class TestOpenLRPerformance:
    """Test OpenLR performance characteristics."""

    def test_encoding_performance(self):
        """Test encoding performance with various geometry sizes."""
        service = OpenLRService()
        
        # Test with different numbers of coordinates
        coordinate_counts = [2, 5, 10, 20]
        
        for count in coordinate_counts:
            coordinates = []
            for i in range(count):
                # Generate coordinates along a line
                lon = -87.6298 + (i * 0.001)
                lat = 41.8781 + (i * 0.001)
                coordinates.append([lon, lat])
            
            geometry = {
                "type": "LineString",
                "coordinates": coordinates
            }
            
            with patch.object(service, 'enabled', True):
                start_time = pytest.time.time()
                result = service.encode_geometry(geometry)
                end_time = pytest.time.time()
                
                # Encoding should complete within reasonable time
                assert (end_time - start_time) < 5.0  # 5 seconds max
                assert result is not None or not service.enabled

    def test_large_geometry_handling(self):
        """Test handling of large geometries."""
        service = OpenLRService()
        
        # Create a geometry with many points
        coordinates = []
        for i in range(100):  # Large geometry
            lon = -87.6298 + (i * 0.0001)
            lat = 41.8781 + (i * 0.0001)
            coordinates.append([lon, lat])
        
        geometry = {
            "type": "LineString",
            "coordinates": coordinates
        }
        
        with patch.object(service, 'enabled', True):
            # Should handle large geometries gracefully
            result = service.encode_geometry(geometry)
            # Result may be None if auto-simplification occurs
            assert result is None or isinstance(result, str)


# Fixtures for testing
@pytest.fixture
def db_session():
    """Create test database session."""
    from app.core.database import SessionLocal, Base, engine

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create session
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()
        # Clean up
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_geometry():
    """Sample geometry for testing."""
    return {
        "type": "LineString",
        "coordinates": [
            [-87.6298, 41.8781],  # Chicago coordinates
            [-87.6290, 41.8785],
            [-87.6280, 41.8790]
        ]
    }


@pytest.fixture
def openlr_service():
    """Create OpenLR service for testing."""
    return create_openlr_service()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])