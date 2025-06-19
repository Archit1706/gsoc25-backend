"""
Tests for OAuth authentication functionality.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from unittest.mock import patch, MagicMock, AsyncMock
import json
from datetime import datetime, timezone, timedelta

from app.main import app
from app.core.database import get_db
from app.models.user import User
from app.models.auth import AuthEvent, AuthSession, OAuthState
from app.services.oauth_service import (
    OAuthService,
    GoogleOAuthProvider,
    GitHubOAuthProvider,
)
from app.services.user_service import UserService
from app.schemas.user import OAuthUser
from app.config import settings


# Test client
client = TestClient(app)


class TestOAuthService:
    """Test OAuth service functionality."""

    def test_get_authorization_url_google(self):
        """Test Google OAuth authorization URL generation."""
        oauth_service = OAuthService()

        auth_url, state = oauth_service.get_authorization_url("google")

        assert "accounts.google.com/o/oauth2/auth" in auth_url
        assert "client_id=" in auth_url
        assert "scope=" in auth_url
        assert "state=" in auth_url
        assert len(state) > 20  # State should be secure random string

    def test_get_authorization_url_github(self):
        """Test GitHub OAuth authorization URL generation."""
        oauth_service = OAuthService()

        auth_url, state = oauth_service.get_authorization_url("github")

        assert "github.com/login/oauth/authorize" in auth_url
        assert "client_id=" in auth_url
        assert "scope=" in auth_url
        assert "state=" in auth_url

    def test_unsupported_provider(self):
        """Test error handling for unsupported OAuth provider."""
        oauth_service = OAuthService()

        with pytest.raises(Exception):
            oauth_service.get_authorization_url("unsupported_provider")

    @pytest.mark.asyncio
    async def test_exchange_code_for_token_success(self):
        """Test successful OAuth code exchange."""
        oauth_service = OAuthService()

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "test_token"}
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            token = await oauth_service.exchange_code_for_token(
                "google", "test_code", "test_state"
            )

            assert token == "test_token"

    @pytest.mark.asyncio
    async def test_get_user_info_google(self):
        """Test getting user info from Google."""
        oauth_service = OAuthService()

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "123456",
            "email": "test@gmail.com",
            "name": "Test User",
            "picture": "https://example.com/avatar.jpg",
        }
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            user_info = await oauth_service.get_user_info("google", "test_token")

            assert user_info.provider == "google"
            assert user_info.provider_id == "123456"
            assert user_info.email == "test@gmail.com"
            assert user_info.name == "Test User"


class TestUserService:
    """Test user service OAuth functionality."""

    def test_create_oauth_user_new(self, db_session: Session):
        """Test creating new OAuth user."""
        user_service = UserService(db_session)

        oauth_user = OAuthUser(
            provider="google",
            provider_id="123456",
            email="new@gmail.com",
            name="New User",
            username="newuser",
        )

        user = user_service.create_or_get_oauth_user(oauth_user)

        assert user.provider == "google"
        assert user.provider_id == "123456"
        assert user.email == "new@gmail.com"
        assert user.full_name == "New User"
        assert user.is_verified is True
        assert user.hashed_password is None

    def test_create_oauth_user_existing_email(self, db_session: Session):
        """Test OAuth login with existing email."""
        # Create existing user
        existing_user = User(
            username="existing",
            email="existing@gmail.com",
            hashed_password="hash",
            is_verified=False,
        )
        db_session.add(existing_user)
        db_session.commit()

        user_service = UserService(db_session)

        oauth_user = OAuthUser(
            provider="google",
            provider_id="123456",
            email="existing@gmail.com",
            name="Existing User",
        )

        user = user_service.create_or_get_oauth_user(oauth_user)

        # Should return existing user, now verified
        assert user.id == existing_user.id
        assert user.is_verified is True
        assert user.full_name == "Existing User"

    def test_generate_unique_username(self, db_session: Session):
        """Test unique username generation."""
        # Create user with preferred username
        existing_user = User(username="testuser", email="test1@example.com")
        db_session.add(existing_user)
        db_session.commit()

        user_service = UserService(db_session)

        oauth_user = OAuthUser(
            provider="github",
            provider_id="789",
            email="test2@example.com",
            username="testuser",  # Username already taken
        )

        user = user_service.create_or_get_oauth_user(oauth_user)

        # Should generate unique username
        assert user.username != "testuser"
        assert user.username.startswith("testuser_")


class TestAuthEndpoints:
    """Test authentication endpoints."""

    def test_register_success(self, db_session: Session):
        """Test successful user registration."""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123",
            "full_name": "Test User",
        }

        response = client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert data["is_verified"] is False

    def test_register_duplicate_username(self, db_session: Session):
        """Test registration with duplicate username."""
        # Create existing user
        existing_user = User(
            username="duplicate", email="first@example.com", hashed_password="hash"
        )
        db_session.add(existing_user)
        db_session.commit()

        user_data = {
            "username": "duplicate",
            "email": "second@example.com",
            "password": "SecurePass123",
        }

        response = client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_login_success(self, db_session: Session):
        """Test successful login."""
        from app.core.security import hash_password

        # Create user
        user = User(
            username="logintest",
            email="login@example.com",
            hashed_password=hash_password("password123"),
            is_verified=True,
        )
        db_session.add(user)
        db_session.commit()

        login_data = {"username": "logintest", "password": "password123"}

        response = client.post("/api/v1/auth/login", json=login_data)

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["username"] == "logintest"

    def test_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        login_data = {"username": "nonexistent", "password": "wrongpass"}

        response = client.post("/api/v1/auth/login", json=login_data)

        assert response.status_code == 401
        assert "Invalid username or password" in response.json()["detail"]

    def test_oauth_login_redirect(self):
        """Test OAuth login redirect."""
        with patch.object(settings, "OAUTH_ENABLED", True):
            response = client.get("/api/v1/auth/oauth/google", allow_redirects=False)

            assert response.status_code == 302
            assert "accounts.google.com" in response.headers["location"]

    def test_oauth_disabled(self):
        """Test OAuth when disabled."""
        with patch.object(settings, "OAUTH_ENABLED", False):
            response = client.get("/api/v1/auth/oauth/google")

            assert response.status_code == 503
            assert "disabled" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_oauth_callback_success(self, db_session: Session):
        """Test successful OAuth callback."""
        # Mock OAuth service
        mock_oauth_user = OAuthUser(
            provider="google",
            provider_id="123456",
            email="oauth@gmail.com",
            name="OAuth User",
        )

        with patch("app.services.oauth_service.OAuthService") as mock_service:
            mock_instance = mock_service.return_value
            mock_instance.exchange_code_for_token = AsyncMock(return_value="test_token")
            mock_instance.get_user_info = AsyncMock(return_value=mock_oauth_user)

            # Create OAuth state
            oauth_state = OAuthState.create_state(
                db_session, state="test_state", provider="google"
            )

            # Set cookie
            client.cookies.set("oauth_state_google", "test_state")

            response = client.get(
                "/api/v1/auth/oauth/google/callback?code=test_code&state=test_state",
                allow_redirects=False,
            )

            assert response.status_code == 302
            assert "token=" in response.headers["location"]

    def test_oauth_callback_invalid_state(self):
        """Test OAuth callback with invalid state."""
        response = client.get(
            "/api/v1/auth/oauth/google/callback?code=test_code&state=invalid_state"
        )

        assert response.status_code == 302
        assert "error" in response.headers["location"]


class TestAuthModels:
    """Test authentication models."""

    def test_auth_session_creation(self, db_session: Session):
        """Test authentication session creation."""
        user = User(username="sessiontest", email="session@example.com")
        db_session.add(user)
        db_session.commit()

        session = AuthSession.create_session(
            db_session,
            user_id=user.id,
            ip_address="192.168.1.1",
            user_agent="Test Browser",
        )

        assert session.user_id == user.id
        assert session.ip_address == "192.168.1.1"
        assert session.is_active is True
        assert session.is_valid is True
        assert len(session.session_id) > 20

    def test_auth_session_expiration(self, db_session: Session):
        """Test session expiration."""
        user = User(username="expiretest", email="expire@example.com")
        db_session.add(user)
        db_session.commit()

        # Create expired session
        session = AuthSession(
            user_id=user.id, expires_at=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        db_session.add(session)
        db_session.commit()

        assert session.is_expired is True
        assert session.is_valid is False

    def test_auth_event_logging(self, db_session: Session):
        """Test authentication event logging."""
        user = User(username="eventtest", email="event@example.com")
        db_session.add(user)
        db_session.commit()

        event = AuthEvent.log_event(
            db_session,
            event_type="login",
            success=True,
            user_id=user.id,
            ip_address="192.168.1.1",
            details={"method": "password"},
        )

        assert event.event_type == "login"
        assert event.success is True
        assert event.user_id == user.id
        assert event.details["method"] == "password"

    def test_oauth_state_management(self, db_session: Session):
        """Test OAuth state creation and validation."""
        state = OAuthState.create_state(
            db_session,
            state="test_state_123",
            provider="google",
            ip_address="192.168.1.1",
        )

        assert state.state == "test_state_123"
        assert state.provider == "google"
        assert state.is_valid is True

        # Validate and consume state
        validated = OAuthState.validate_and_consume_state(
            db_session,
            state="test_state_123",
            provider="google",
            ip_address="192.168.1.1",
        )

        assert validated is not None
        assert validated.state == "test_state_123"

        # State should be consumed (deleted)
        remaining = (
            db_session.query(OAuthState)
            .filter(OAuthState.state == "test_state_123")
            .first()
        )
        assert remaining is None


class TestUserModel:
    """Test User model OAuth functionality."""

    def test_oauth_user_properties(self):
        """Test OAuth user properties."""
        oauth_user = User(
            username="oauthuser",
            email="oauth@example.com",
            provider="google",
            provider_id="123456",
        )

        assert oauth_user.is_oauth_user is True
        assert oauth_user.is_verified is True  # OAuth users are auto-verified
        assert oauth_user.hashed_password is None

    def test_regular_user_properties(self):
        """Test regular user properties."""
        regular_user = User(
            username="regularuser",
            email="regular@example.com",
            hashed_password="hashed_password",
        )

        assert regular_user.is_oauth_user is False
        assert regular_user.is_verified is False  # Needs email verification

    def test_account_locking(self, db_session: Session):
        """Test account locking after failed attempts."""
        user = User(
            username="locktest", email="lock@example.com", hashed_password="hash"
        )
        db_session.add(user)
        db_session.commit()

        # Increment login attempts
        for _ in range(5):
            user.increment_login_attempts(db_session)

        assert user.is_locked is True
        assert user.locked_until is not None
        assert user.can_login is False

    def test_unlock_expired_locks(self, db_session: Session):
        """Test automatic unlocking of expired locks."""
        user = User(
            username="unlocktest",
            email="unlock@example.com",
            locked_until=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        db_session.add(user)
        db_session.commit()

        # Should automatically unlock expired locks
        unlocked_count = User.cleanup_locked_accounts(db_session)

        assert unlocked_count == 1

        # Refresh user
        db_session.refresh(user)
        assert user.is_locked is False
        assert user.locked_until is None


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
def override_get_db(db_session):
    """Override database dependency for testing."""

    def _override_get_db():
        return db_session

    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


# Integration test for full OAuth flow
class TestOAuthIntegration:
    """Integration tests for complete OAuth flow."""

    @pytest.mark.asyncio
    async def test_complete_google_oauth_flow(
        self, db_session: Session, override_get_db
    ):
        """Test complete Google OAuth flow end-to-end."""

        # Step 1: Initiate OAuth flow
        with patch.object(settings, "OAUTH_ENABLED", True):
            response = client.get("/api/v1/auth/oauth/google", allow_redirects=False)

            assert response.status_code == 302

            # Extract state from cookie
            state_cookie = None
            for cookie in response.cookies:
                if cookie.name == "oauth_state_google":
                    state_cookie = cookie.value
                    break

            assert state_cookie is not None

        # Step 2: Mock OAuth callback
        mock_oauth_user = OAuthUser(
            provider="google",
            provider_id="test_user_123",
            email="integration@gmail.com",
            name="Integration Test User",
        )

        with patch("app.services.oauth_service.OAuthService") as mock_service:
            mock_instance = mock_service.return_value
            mock_instance.exchange_code_for_token = AsyncMock(return_value="mock_token")
            mock_instance.get_user_info = AsyncMock(return_value=mock_oauth_user)

            # Set the state cookie
            client.cookies.set("oauth_state_google", state_cookie)

            response = client.get(
                f"/api/v1/auth/oauth/google/callback?code=mock_code&state={state_cookie}",
                allow_redirects=False,
            )

            assert response.status_code == 302
            redirect_url = response.headers["location"]
            assert "token=" in redirect_url

            # Extract token from redirect
            token = redirect_url.split("token=")[1].split("&")[0]
            assert len(token) > 50  # JWT should be substantial

        # Step 3: Use token to access protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == 200
        user_data = response.json()
        assert user_data["email"] == "integration@gmail.com"
        assert user_data["full_name"] == "Integration Test User"
        assert user_data["is_verified"] is True

        # Step 4: Verify user was created in database
        user = (
            db_session.query(User).filter(User.email == "integration@gmail.com").first()
        )
        assert user is not None
        assert user.provider == "google"
        assert user.provider_id == "test_user_123"
        assert user.is_oauth_user is True


# Performance tests
class TestOAuthPerformance:
    """Performance tests for OAuth functionality."""

    @pytest.mark.asyncio
    async def test_oauth_service_performance(self):
        """Test OAuth service performance under load."""
        oauth_service = OAuthService()

        # Test multiple authorization URL generations
        start_time = datetime.now()

        for _ in range(100):
            auth_url, state = oauth_service.get_authorization_url("google")
            assert len(auth_url) > 50
            assert len(state) > 20

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should complete 100 operations in under 1 second
        assert duration < 1.0

    def test_user_lookup_performance(self, db_session: Session):
        """Test user lookup performance with many users."""
        # Create many users
        users = []
        for i in range(1000):
            user = User(
                username=f"user_{i}",
                email=f"user_{i}@example.com",
                provider="google" if i % 2 == 0 else None,
                provider_id=str(i) if i % 2 == 0 else None,
            )
            users.append(user)

        db_session.bulk_save_objects(users)
        db_session.commit()

        # Test lookup performance
        start_time = datetime.now()

        for i in range(100):
            user = User.get_by_email(db_session, f"user_{i}@example.com")
            assert user is not None

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should complete 100 lookups in under 0.5 seconds
        assert duration < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
