#!/usr/bin/env python3
"""
Database initialization script for OSM Road Closures API with Point and LineString support.

This script:
1. Creates database tables if they don't exist
2. Creates sample users
3. Creates sample closures with both Point and LineString geometries
4. Demonstrates OpenLR encoding for LineString geometries
5. Shows radius-based location referencing for Point geometries
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

# Add the parent directory to the path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import db_manager, SessionLocal, init_database
from app.models.user import User
from app.models.closure import Closure, ClosureType, ClosureStatus
from app.services.user_service import UserService
from app.services.closure_service import ClosureService
from app.schemas.user import UserCreate
from app.schemas.closure import ClosureCreate, GeoJSONGeometry
from app.config import settings


def create_sample_users(db: Session) -> dict:
    """
    Create sample users for testing.

    Returns:
        dict: Dictionary of created users
    """
    print("Creating sample users...")

    user_service = UserService(db)
    users = {}

    # Sample users data
    sample_users = [
        {
            "username": "chicago_mapper",
            "email": "mapper@chicago.gov",
            "password": "SecurePass123",
            "full_name": "Chicago City Mapper",
            "is_moderator": True,
        },
        {
            "username": "traffic_reporter",
            "email": "traffic@chicago.gov",
            "password": "ReportPass456",
            "full_name": "Traffic Control Reporter",
            "is_moderator": False,
        },
        {
            "username": "construction_admin",
            "email": "construction@chicagodot.org",
            "password": "BuildPass789",
            "full_name": "Construction Administrator",
            "is_moderator": False,
        },
        {
            "username": "emergency_dispatch",
            "email": "dispatch@chicagopd.org",
            "password": "EmergencyPass321",
            "full_name": "Emergency Dispatcher",
            "is_moderator": False,
        },
    ]

    for user_data in sample_users:
        # Check if user already exists
        existing_user = User.get_by_username(db, user_data["username"])
        if existing_user:
            print(f"  User {user_data['username']} already exists, skipping...")
            users[user_data["username"]] = existing_user
            continue

        try:
            # Create user
            user_create = UserCreate(
                username=user_data["username"],
                email=user_data["email"],
                password=user_data["password"],
                full_name=user_data["full_name"],
            )

            user = user_service.create_user(user_create)

            # Set moderator status
            if user_data.get("is_moderator"):
                user.is_moderator = True

            # Mark as verified for sample data
            user.is_verified = True
            user.email_verified_at = datetime.now(timezone.utc)

            db.commit()
            users[user_data["username"]] = user

            print(f"  Created user: {user.username} ({user.email})")

        except Exception as e:
            print(f"  Error creating user {user_data['username']}: {e}")
            db.rollback()

    return users


def create_sample_closures(db: Session, users: dict) -> list:
    """
    Create sample closures with both Point and LineString geometries.

    Args:
        db: Database session
        users: Dictionary of created users

    Returns:
        list: List of created closures
    """
    print("Creating sample closures...")

    closure_service = ClosureService(db)
    closures = []

    # Get user IDs
    chicago_mapper = users.get("chicago_mapper")
    traffic_reporter = users.get("traffic_reporter")
    construction_admin = users.get("construction_admin")
    emergency_dispatch = users.get("emergency_dispatch")

    if not all(
        [chicago_mapper, traffic_reporter, construction_admin, emergency_dispatch]
    ):
        print("  Warning: Not all sample users available, using first available user")
        default_user = next(iter(users.values()))
        chicago_mapper = chicago_mapper or default_user
        traffic_reporter = traffic_reporter or default_user
        construction_admin = construction_admin or default_user
        emergency_dispatch = emergency_dispatch or default_user

    # Sample closures data
    sample_closures = [
        # LineString closures (road segments)
        {
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [-87.6298, 41.8781],  # Madison Street
                    [-87.6290, 41.8785],
                    [-87.6282, 41.8789],
                ],
            },
            "description": "Water main repair blocking eastbound traffic on Madison Street between Wells and LaSalle",
            "closure_type": ClosureType.CONSTRUCTION,
            "start_time": datetime.now(timezone.utc) - timedelta(hours=2),
            "end_time": datetime.now(timezone.utc) + timedelta(hours=6),
            "source": "City of Chicago Water Department",
            "confidence_level": 9,
            "submitter": construction_admin,
        },
        {
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [-87.6347, 41.8857],  # Lake Shore Drive
                    [-87.6345, 41.8867],
                    [-87.6343, 41.8877],
                    [-87.6341, 41.8887],
                ],
            },
            "description": "Lane closure for bridge maintenance on Lake Shore Drive northbound",
            "closure_type": ClosureType.MAINTENANCE,
            "start_time": datetime.now(timezone.utc) + timedelta(days=1),
            "end_time": datetime.now(timezone.utc) + timedelta(days=3),
            "source": "Illinois Department of Transportation",
            "confidence_level": 10,
            "submitter": chicago_mapper,
        },
        # Point closures (specific locations)
        {
            "geometry": {
                "type": "Point",
                "coordinates": [-87.6294, 41.8783],  # Madison & Wells intersection
            },
            "description": "Vehicle accident blocking intersection at Madison and Wells",
            "closure_type": ClosureType.ACCIDENT,
            "start_time": datetime.now(timezone.utc) - timedelta(minutes=30),
            "end_time": datetime.now(timezone.utc) + timedelta(hours=1),
            "radius_meters": 75,
            "source": "Chicago Police Department",
            "confidence_level": 10,
            "submitter": emergency_dispatch,
        },
        {
            "geometry": {
                "type": "Point",
                "coordinates": [-87.6235, 41.8795],  # Near Millennium Park
            },
            "description": "Street festival causing temporary road closure around event area",
            "closure_type": ClosureType.EVENT,
            "start_time": datetime.now(timezone.utc) + timedelta(hours=12),
            "end_time": datetime.now(timezone.utc) + timedelta(hours=20),
            "radius_meters": 200,
            "source": "City of Chicago Events Department",
            "confidence_level": 8,
            "submitter": traffic_reporter,
        },
        {
            "geometry": {
                "type": "Point",
                "coordinates": [-87.6270, 41.8850],  # Near Chicago River
            },
            "description": "Emergency water main break - avoid area until repairs complete",
            "closure_type": ClosureType.MAINTENANCE,
            "start_time": datetime.now(timezone.utc) - timedelta(hours=1),
            "end_time": None,  # Indefinite
            "radius_meters": 150,
            "source": "Chicago Emergency Services",
            "confidence_level": 10,
            "submitter": emergency_dispatch,
        },
        # More LineString examples
        {
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [-87.6230, 41.8820],  # State Street
                    [-87.6230, 41.8830],
                    [-87.6230, 41.8840],
                ],
            },
            "description": "Planned road resurfacing on State Street - right lane closed",
            "closure_type": ClosureType.CONSTRUCTION,
            "start_time": datetime.now(timezone.utc) + timedelta(days=7),
            "end_time": datetime.now(timezone.utc) + timedelta(days=14),
            "source": "Chicago Department of Transportation",
            "confidence_level": 9,
            "submitter": construction_admin,
        },
        # Weather-related closure
        {
            "geometry": {
                "type": "Point",
                "coordinates": [-87.6050, 41.8750],  # Lower Wacker area
            },
            "description": "Flooding in underpass - road temporarily impassable",
            "closure_type": ClosureType.WEATHER,
            "start_time": datetime.now(timezone.utc) - timedelta(hours=4),
            "end_time": datetime.now(timezone.utc) + timedelta(hours=8),
            "radius_meters": 100,
            "source": "Chicago Traffic Management",
            "confidence_level": 8,
            "submitter": traffic_reporter,
        },
    ]

    for i, closure_data in enumerate(sample_closures):
        try:
            # Create GeoJSON geometry
            geometry = GeoJSONGeometry(**closure_data["geometry"])

            # Create closure request
            closure_create = ClosureCreate(
                geometry=geometry,
                description=closure_data["description"],
                closure_type=closure_data["closure_type"],
                start_time=closure_data["start_time"],
                end_time=closure_data["end_time"],
                source=closure_data["source"],
                confidence_level=closure_data["confidence_level"],
                radius_meters=closure_data.get("radius_meters"),
            )

            # Create closure
            closure = closure_service.create_closure(
                closure_create, closure_data["submitter"].id
            )

            closures.append(closure)

            geom_type = closure_data["geometry"]["type"]
            openlr_note = ""
            if geom_type == "LineString" and closure.openlr_code:
                openlr_note = f" (OpenLR: {closure.openlr_code[:16]}...)"
            elif geom_type == "Point":
                radius = closure_data.get("radius_meters", "N/A")
                openlr_note = f" (Radius: {radius}m)"

            print(
                f"  Created {geom_type} closure: {closure.description[:50]}...{openlr_note}"
            )

        except Exception as e:
            print(f"  Error creating closure {i+1}: {e}")
            db.rollback()

    return closures


def verify_database_setup(db: Session):
    """
    Verify that the database is properly set up with sample data.

    Args:
        db: Database session
    """
    print("\nVerifying database setup...")

    # Check users
    user_count = db.query(User).count()
    moderator_count = db.query(User).filter(User.is_moderator == True).count()
    print(f"  Users: {user_count} total, {moderator_count} moderators")

    # Check closures
    total_closures = db.query(Closure).count()
    point_closures = db.query(Closure).filter(Closure.geometry_type == "Point").count()
    linestring_closures = (
        db.query(Closure).filter(Closure.geometry_type == "LineString").count()
    )

    print(f"  Closures: {total_closures} total")
    print(f"    Point geometries: {point_closures}")
    print(f"    LineString geometries: {linestring_closures}")

    # Check OpenLR codes
    if settings.OPENLR_ENABLED:
        with_openlr = db.query(Closure).filter(Closure.openlr_code.isnot(None)).count()
        print(f"    With OpenLR codes: {with_openlr}")

    # Check by status
    active_closures = (
        db.query(Closure).filter(Closure.status == ClosureStatus.ACTIVE).count()
    )
    planned_closures = (
        db.query(Closure).filter(Closure.status == ClosureStatus.PLANNED).count()
    )

    print(f"  Active closures: {active_closures}")
    print(f"  Planned closures: {planned_closures}")

    # Check PostGIS
    try:
        result = db.execute(func.PostGIS_Version()).scalar()
        print(f"  PostGIS version: {result}")
    except Exception as e:
        print(f"  PostGIS check failed: {e}")


def print_sample_api_calls():
    """
    Print sample API calls for testing the Point geometry support.
    """
    print("\n" + "=" * 60)
    print("SAMPLE API CALLS FOR TESTING")
    print("=" * 60)

    print("\n1. Get all closures:")
    print("   GET /api/v1/closures/")

    print("\n2. Get only Point closures:")
    print("   GET /api/v1/closures/?geometry_type=Point")

    print("\n3. Get only LineString closures:")
    print("   GET /api/v1/closures/?geometry_type=LineString")

    print("\n4. Get accident locations (Point closures):")
    print("   GET /api/v1/closures/?geometry_type=Point&closure_type=accident")

    print("\n5. Get construction road segments (LineString closures):")
    print("   GET /api/v1/closures/?geometry_type=LineString&closure_type=construction")

    print("\n6. Get closures in downtown Chicago (bounding box):")
    print("   GET /api/v1/closures/?bbox=-87.65,41.87,-87.62,41.89")

    print("\n7. Get OpenLR information:")
    print("   GET /api/v1/openlr/info")

    print("\n8. Check OpenLR suitability for a geometry:")
    print("   POST /api/v1/openlr/check-suitability")
    print("   Body: {")
    print('     "geometry": {"type": "Point", "coordinates": [-87.6294, 41.8783]}')
    print("   }")

    print("\n9. Create a Point closure:")
    print("   POST /api/v1/closures/")
    print("   Body: {")
    print('     "geometry": {"type": "Point", "coordinates": [-87.630, 41.880]},')
    print('     "description": "Emergency incident at intersection",')
    print('     "closure_type": "accident",')
    print('     "start_time": "2025-07-06T15:00:00Z",')
    print('     "end_time": "2025-07-06T17:00:00Z",')
    print('     "radius_meters": 100,')
    print('     "confidence_level": 10')
    print("   }")

    print("\n10. Create a LineString closure:")
    print("    POST /api/v1/closures/")
    print("    Body: {")
    print('      "geometry": {')
    print('        "type": "LineString",')
    print('        "coordinates": [[-87.640, 41.885], [-87.635, 41.890]]')
    print("      },")
    print('      "description": "Road construction blocking traffic",')
    print('      "closure_type": "construction",')
    print('      "start_time": "2025-07-07T08:00:00Z",')
    print('      "end_time": "2025-07-07T18:00:00Z",')
    print('      "confidence_level": 9')
    print("    }")

    print("\n" + "=" * 60)


async def main():
    """
    Main initialization function.
    """
    print(
        "Initializing OSM Road Closures Database with Point and LineString support..."
    )
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Database: {settings.DATABASE_URL}")
    print(f"OpenLR enabled: {settings.OPENLR_ENABLED}")

    # Check database connection
    if not db_manager.health_check():
        print("❌ Database connection failed!")
        return False

    print("✅ Database connection successful")

    # Initialize database (create tables, extensions, etc.)
    try:
        await init_database()
        print("✅ Database tables initialized")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

    # Create sample data
    db = SessionLocal()
    try:
        # Create users
        users = create_sample_users(db)
        if not users:
            print("❌ No users created")
            return False

        # Create closures
        closures = create_sample_closures(db, users)
        if not closures:
            print("❌ No closures created")
            return False

        # Verify setup
        verify_database_setup(db)

        print("\n✅ Database initialization completed successfully!")
        print(f"Created {len(users)} users and {len(closures)} closures")

        # Print sample API calls
        print_sample_api_calls()

        return True

    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        db.rollback()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
