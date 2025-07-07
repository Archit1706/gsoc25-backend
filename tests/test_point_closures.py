"""
Test suite for Point closure functionality.
"""

import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.core.database import get_db
from app.models.closure import Closure, ClosureType, ClosureStatus
from app.models.user import User
from app.schemas.closure import ClosureCreate, GeoJSONGeometry
from app.services.closure_service import ClosureService
from app.services.openlr_service import create_openlr_service


client = TestClient(app)


@pytest.fixture
def sample_point_geometry():
    """Sample Point geometry for testing."""
    return {"type": "Point", "coordinates": [-87.6294, 41.8783]}


@pytest.fixture
def sample_linestring_geometry():
    """Sample LineString geometry for testing."""
    return {
        "type": "LineString",
        "coordinates": [[-87.6298, 41.8781], [-87.6290, 41.8785]],
    }


@pytest.fixture
def test_user(db: Session):
    """Create a test user."""
    user = User(
        username="test_point_user",
        email="test@example.com",
        hashed_password="hashed_password_123",
        is_active=True,
        is_verified=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


class TestPointGeometryValidation:
    """Test Point geometry validation."""

    def test_valid_point_coordinates(self, sample_point_geometry):
        """Test valid Point coordinates."""
        geometry = GeoJSONGeometry(**sample_point_geometry)
        assert geometry.type == "Point"
        assert geometry.coordinates == [-87.6294, 41.8783]

    def test_point_coordinate_rounding(self):
        """Test Point coordinate rounding to 5 decimal places."""
        geometry_data = {"type": "Point", "coordinates": [-87.629456789, 41.878312345]}
        geometry = GeoJSONGeometry(**geometry_data)
        assert geometry.coordinates == [-87.62946, 41.87831]

    def test_invalid_point_coordinates(self):
        """Test invalid Point coordinates."""
        # Test out of range longitude
        with pytest.raises(ValueError, match="Longitude .* is out of range"):
            GeoJSONGeometry(type="Point", coordinates=[200.0, 41.8783])

        # Test out of range latitude
        with pytest.raises(ValueError, match="Latitude .* is out of range"):
            GeoJSONGeometry(type="Point", coordinates=[-87.6294, 95.0])

        # Test wrong number of coordinates
        with pytest.raises(ValueError, match="Point must have exactly 2 coordinates"):
            GeoJSONGeometry(type="Point", coordinates=[-87.6294])

        # Test non-numeric coordinates
        with pytest.raises(ValueError, match="Point coordinates must be numbers"):
            GeoJSONGeometry(type="Point", coordinates=["invalid", 41.8783])


class TestPointClosureCreation:
    """Test Point closure creation."""

    def test_create_point_closure_success(
        self, db: Session, test_user, sample_point_geometry
    ):
        """Test successful Point closure creation."""
        service = ClosureService(db)

        closure_data = ClosureCreate(
            geometry=GeoJSONGeometry(**sample_point_geometry),
            description="Vehicle accident at intersection",
            closure_type=ClosureType.ACCIDENT,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(hours=2),
            radius_meters=100,
            confidence_level=10,
        )

        closure = service.create_closure(closure_data, test_user.id)

        assert closure.id is not None
        assert closure.geometry_type == "Point"
        assert closure.radius_meters == 100
        assert closure.openlr_code is None  # OpenLR not applicable to Points
        assert closure.closure_type == ClosureType.ACCIDENT.value

    def test_create_point_closure_default_radius(
        self, db: Session, test_user, sample_point_geometry
    ):
        """Test Point closure creation with default radius."""
        service = ClosureService(db)

        closure_data = ClosureCreate(
            geometry=GeoJSONGeometry(**sample_point_geometry),
            description="Emergency incident",
            closure_type=ClosureType.ACCIDENT,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(hours=1),
            # No radius_meters specified
        )

        closure = service.create_closure(closure_data, test_user.id)

        assert closure.radius_meters == 50  # Default radius

    def test_create_point_closure_invalid_radius(self, sample_point_geometry):
        """Test Point closure creation with invalid radius."""
        # Test negative radius
        with pytest.raises(ValueError):
            ClosureCreate(
                geometry=GeoJSONGeometry(**sample_point_geometry),
                description="Test closure",
                closure_type=ClosureType.ACCIDENT,
                start_time=datetime.now(timezone.utc),
                radius_meters=-10,
            )

        # Test radius too large
        with pytest.raises(ValueError):
            ClosureCreate(
                geometry=GeoJSONGeometry(**sample_point_geometry),
                description="Test closure",
                closure_type=ClosureType.ACCIDENT,
                start_time=datetime.now(timezone.utc),
                radius_meters=15000,
            )


class TestPointClosureQuerying:
    """Test Point closure querying."""

    def test_query_point_closures_only(
        self, db: Session, test_user, sample_point_geometry, sample_linestring_geometry
    ):
        """Test querying only Point closures."""
        service = ClosureService(db)

        # Create Point closure
        point_closure_data = ClosureCreate(
            geometry=GeoJSONGeometry(**sample_point_geometry),
            description="Point closure",
            closure_type=ClosureType.ACCIDENT,
            start_time=datetime.now(timezone.utc),
            radius_meters=75,
        )
        service.create_closure(point_closure_data, test_user.id)

        # Create LineString closure
        line_closure_data = ClosureCreate(
            geometry=GeoJSONGeometry(**sample_linestring_geometry),
            description="LineString closure",
            closure_type=ClosureType.CONSTRUCTION,
            start_time=datetime.now(timezone.utc),
        )
        service.create_closure(line_closure_data, test_user.id)

        # Query only Point closures
        from app.schemas.closure import ClosureQueryParams

        query_params = ClosureQueryParams(geometry_type="Point", page=1, size=10)
        closures, total = service.query_closures(query_params)

        assert total >= 1
        for closure in closures:
            assert closure.geometry_type == "Point"

    def test_point_closure_bbox_query(self, db: Session, test_user):
        """Test Point closure bounding box query."""
        service = ClosureService(db)

        # Create Point closure within bbox
        inside_point = {
            "type": "Point",
            "coordinates": [-87.6250, 41.8800],  # Within Chicago downtown
        }
        closure_data = ClosureCreate(
            geometry=GeoJSONGeometry(**inside_point),
            description="Point inside bbox",
            closure_type=ClosureType.ACCIDENT,
            start_time=datetime.now(timezone.utc),
            radius_meters=50,
        )
        closure = service.create_closure(closure_data, test_user.id)

        # Query with bounding box that includes the point
        from app.schemas.closure import ClosureQueryParams

        bbox = "-87.63,41.87,-87.62,41.89"  # Chicago downtown area
        query_params = ClosureQueryParams(bbox=bbox, page=1, size=10)
        closures, total = service.query_closures(query_params)

        # Should find the closure
        closure_ids = [c.id for c in closures]
        assert closure.id in closure_ids


class TestPointClosureValidation:
    """Test Point closure validation and constraints."""

    def test_point_closure_radius_constraint(
        self, db: Session, test_user, sample_linestring_geometry
    ):
        """Test that radius_meters is only allowed for Point geometries."""
        service = ClosureService(db)

        # This should fail: LineString with radius_meters
        with pytest.raises(
            ValueError, match="radius_meters can only be set for Point geometries"
        ):
            ClosureCreate(
                geometry=GeoJSONGeometry(**sample_linestring_geometry),
                description="LineString with invalid radius",
                closure_type=ClosureType.CONSTRUCTION,
                start_time=datetime.now(timezone.utc),
                radius_meters=100,  # Should not be allowed for LineString
            )

    def test_point_closure_update_geometry_type(
        self, db: Session, test_user, sample_point_geometry, sample_linestring_geometry
    ):
        """Test updating closure geometry type from Point to LineString."""
        service = ClosureService(db)

        # Create Point closure
        closure_data = ClosureCreate(
            geometry=GeoJSONGeometry(**sample_point_geometry),
            description="Originally a Point closure",
            closure_type=ClosureType.ACCIDENT,
            start_time=datetime.now(timezone.utc),
            radius_meters=100,
        )
        closure = service.create_closure(closure_data, test_user.id)

        # Update to LineString geometry
        from app.schemas.closure import ClosureUpdate

        update_data = ClosureUpdate(
            geometry=GeoJSONGeometry(**sample_linestring_geometry),
            description="Now a LineString closure",
        )

        updated_closure = service.update_closure(closure.id, update_data, test_user)

        assert updated_closure.geometry_type == "LineString"
        assert updated_closure.radius_meters is None  # Should be cleared
        assert updated_closure.openlr_code is not None  # Should have OpenLR code


class TestOpenLRPointIntegration:
    """Test OpenLR integration with Point geometries."""

    def test_openlr_not_applicable_to_points(self, sample_point_geometry):
        """Test that OpenLR encoding is not applicable to Point geometries."""
        openlr_service = create_openlr_service()

        # Test encoding Point (should return None)
        result = openlr_service.encode_geometry(sample_point_geometry)
        assert result is None

    def test_openlr_suitability_check_point(self, sample_point_geometry):
        """Test OpenLR suitability check for Point geometries."""
        openlr_service = create_openlr_service()

        suitability = openlr_service.is_geometry_suitable_for_openlr(
            sample_point_geometry
        )

        assert suitability["suitable"] is False
        assert suitability["geometry_type"] == "Point"
        assert "not applicable" in suitability["reason"].lower()

    def test_point_location_alternatives(self, sample_point_geometry):
        """Test alternative location methods for Point geometries."""
        openlr_service = create_openlr_service()

        alternatives = openlr_service.get_point_location_alternatives(
            sample_point_geometry
        )

        assert alternatives["geometry_type"] == "Point"
        assert "alternatives" in alternatives
        assert "radius_based" in alternatives["alternatives"]
        assert "coordinates" in alternatives["alternatives"]
        assert "recommendations" in alternatives

    def test_openlr_roundtrip_point(self, sample_point_geometry):
        """Test OpenLR roundtrip test with Point geometry."""
        openlr_service = create_openlr_service()

        result = openlr_service.test_encoding_roundtrip(sample_point_geometry)

        assert result["success"] is False
        assert result["applicable"] is False
        assert "not applicable" in result["error"].lower()


class TestPointClosureAPI:
    """Test Point closure API endpoints."""

    def test_api_create_point_closure(self, sample_point_geometry):
        """Test API endpoint for creating Point closures."""
        # First register and login
        register_response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "point_test_user",
                "email": "pointtest@example.com",
                "password": "TestPass123",
            },
        )
        assert register_response.status_code == 201

        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "point_test_user",
                "password": "TestPass123",
            },
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Create Point closure
        closure_data = {
            "geometry": sample_point_geometry,
            "description": "API test Point closure",
            "closure_type": "accident",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            "radius_meters": 75,
            "confidence_level": 10,
        }

        response = client.post(
            "/api/v1/closures/",
            json=closure_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 201
        closure = response.json()
        assert closure["geometry_type"] == "Point"
        assert closure["radius_meters"] == 75
        assert closure["is_point_closure"] is True
        assert closure["is_linestring_closure"] is False
        assert closure["openlr_code"] is None

    def test_api_query_point_closures(self):
        """Test API endpoint for querying Point closures."""
        response = client.get("/api/v1/closures/?geometry_type=Point")
        assert response.status_code == 200

        data = response.json()
        assert "items" in data

        # Check that all returned closures are Point type
        for closure in data["items"]:
            assert closure["geometry_type"] == "Point"

    def test_api_geometry_type_validation(self):
        """Test API geometry type parameter validation."""
        # Invalid geometry type
        response = client.get("/api/v1/closures/?geometry_type=Invalid")
        assert response.status_code == 400
        assert "must be 'Point' or 'LineString'" in response.json()["detail"]

    def test_api_openlr_point_check(self, sample_point_geometry):
        """Test OpenLR API with Point geometry."""
        response = client.post(
            "/api/v1/openlr/check-suitability", json={"geometry": sample_point_geometry}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["suitable"] is False
        assert data["geometry_type"] == "Point"
        assert "alternatives" in data

    def test_api_statistics_with_points(self):
        """Test statistics API includes Point closure counts."""
        response = client.get("/api/v1/closures/statistics/summary")
        assert response.status_code == 200

        stats = response.json()
        assert "by_geometry_type" in stats
        assert "point_closures" in stats
        assert "linestring_closures" in stats


class TestPointClosureIntegration:
    """Integration tests for Point closures."""

    def test_full_point_closure_lifecycle(
        self, db: Session, test_user, sample_point_geometry
    ):
        """Test complete Point closure lifecycle."""
        service = ClosureService(db)

        # Create
        closure_data = ClosureCreate(
            geometry=GeoJSONGeometry(**sample_point_geometry),
            description="Full lifecycle test Point closure",
            closure_type=ClosureType.ACCIDENT,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(hours=2),
            radius_meters=100,
            confidence_level=9,
        )

        # 1. Create closure
        closure = service.create_closure(closure_data, test_user.id)
        assert closure.geometry_type == "Point"

        # 2. Retrieve with geometry
        closure_dict = service.get_closure_with_geometry(closure.id)
        assert closure_dict["geometry"]["type"] == "Point"
        assert closure_dict["radius_meters"] == 100

        # 3. Update description and radius
        from app.schemas.closure import ClosureUpdate

        update_data = ClosureUpdate(
            description="Updated Point closure description",
            radius_meters=150,
        )
        updated_closure = service.update_closure(closure.id, update_data, test_user)
        assert updated_closure.radius_meters == 150

        # 4. Query and find
        from app.schemas.closure import ClosureQueryParams

        query_params = ClosureQueryParams(geometry_type="Point", page=1, size=10)
        closures, total = service.query_closures(query_params)
        closure_ids = [c.id for c in closures]
        assert closure.id in closure_ids

        # 5. Delete
        service.delete_closure(closure.id, test_user)

        # 6. Verify deletion
        with pytest.raises(Exception):  # Should raise NotFoundException
            service.get_closure_by_id(closure.id)

    def test_mixed_geometry_queries(
        self, db: Session, test_user, sample_point_geometry, sample_linestring_geometry
    ):
        """Test queries with mixed Point and LineString geometries."""
        service = ClosureService(db)

        # Create both types
        point_closure = service.create_closure(
            ClosureCreate(
                geometry=GeoJSONGeometry(**sample_point_geometry),
                description="Point closure",
                closure_type=ClosureType.ACCIDENT,
                start_time=datetime.now(timezone.utc),
                radius_meters=50,
            ),
            test_user.id,
        )

        line_closure = service.create_closure(
            ClosureCreate(
                geometry=GeoJSONGeometry(**sample_linestring_geometry),
                description="LineString closure",
                closure_type=ClosureType.CONSTRUCTION,
                start_time=datetime.now(timezone.utc),
            ),
            test_user.id,
        )

        # Test queries
        from app.schemas.closure import ClosureQueryParams

        # All closures
        all_params = ClosureQueryParams(page=1, size=10)
        all_closures, all_total = service.query_closures(all_params)
        all_ids = [c.id for c in all_closures]
        assert point_closure.id in all_ids
        assert line_closure.id in all_ids

        # Only Points
        point_params = ClosureQueryParams(geometry_type="Point", page=1, size=10)
        point_closures, point_total = service.query_closures(point_params)
        point_ids = [c.id for c in point_closures]
        assert point_closure.id in point_ids
        assert line_closure.id not in point_ids

        # Only LineStrings
        line_params = ClosureQueryParams(geometry_type="LineString", page=1, size=10)
        line_closures, line_total = service.query_closures(line_params)
        line_ids = [c.id for c in line_closures]
        assert line_closure.id in line_ids
        assert point_closure.id not in line_ids


if __name__ == "__main__":
    pytest.main([__file__])
