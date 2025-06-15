from app.core.database import SessionLocal
from app.models.user import User
from app.core.security import hash_password

db = SessionLocal()

# Check if test user exists
existing_user = db.query(User).filter(User.username == "testuser").first()

if not existing_user:
    # Create test user
    hashed_password = hash_password("testpass123")
    test_user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=hashed_password,
        full_name="Test User",
        is_active=True,
    )

    db.add(test_user)
    db.commit()
    db.refresh(test_user)
    print(f"Created test user: {test_user.username} (ID: {test_user.id})")
else:
    print(
        f"Test user already exists: {existing_user.username} (ID: {existing_user.id})"
    )

db.close()
