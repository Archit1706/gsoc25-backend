#!/usr/bin/env python3
"""
Test Swagger UI Authentication after fixes are applied
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"
API_V1 = f"{BASE_URL}/api/v1"


def test_oauth2_login():
    """Test OAuth2-compatible login (form data)"""
    print("🔐 Testing OAuth2 Login (Form Data)...")

    # OAuth2 login with form data (like Swagger UI uses)
    login_data = {"username": "chicago_mapper", "password": "SecurePass123"}

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        response = requests.post(
            f"{API_V1}/auth/login",
            data=login_data,  
            headers=headers,
        )

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            token = data["access_token"]
            print(f"   ✅ OAuth2 login successful!")
            print(f"   Token (first 50 chars): {token[:50]}...")
            return token
        else:
            print(f"   ❌ OAuth2 login failed: {response.text}")
            return None

    except Exception as e:
        print(f"   ❌ OAuth2 login error: {e}")
        return None


def test_json_login():
    """Test JSON login (backward compatibility)"""
    print("\n🔐 Testing JSON Login...")

    login_data = {"username": "chicago_mapper", "password": "SecurePass123"}

    try:
        response = requests.post(f"{API_V1}/auth/login-json", json=login_data)

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            token = data["access_token"]
            print(f"   ✅ JSON login successful!")
            print(f"   Token (first 50 chars): {token[:50]}...")
            return token
        else:
            print(f"   ❌ JSON login failed: {response.text}")
            return None

    except Exception as e:
        print(f"   ❌ JSON login error: {e}")
        return None


def test_authentication_with_token(token):
    """Test authentication using the token"""
    print(f"\n🔍 Testing Authentication with Token...")

    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(f"{API_V1}/auth/me", headers=headers)

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            user_data = response.json()
            print(f"   ✅ Authentication successful!")
            print(f"   User: {user_data['username']} (ID: {user_data['id']})")
            return True
        else:
            print(f"   ❌ Authentication failed: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Authentication error: {e}")
        return False


def test_protected_endpoint(token):
    """Test a protected endpoint"""
    print(f"\n🚧 Testing Protected Endpoint (Create Closure)...")

    headers = {"Authorization": f"Bearer {token}"}

    closure_data = {
        "geometry": {
            "type": "LineString",
            "coordinates": [[-87.6298, 41.8781], [-87.6290, 41.8785]],
        },
        "description": f"Test closure from authentication test - {datetime.now().strftime('%H:%M:%S')}",
        "closure_type": "construction",
        "start_time": "2025-07-03T08:00:00Z",
        "end_time": "2025-07-03T18:00:00Z",
        "source": "Authentication Test",
        "confidence_level": 10,
    }

    try:
        response = requests.post(
            f"{API_V1}/closures/", json=closure_data, headers=headers
        )

        print(f"   Status: {response.status_code}")

        if response.status_code == 201:
            closure = response.json()
            print(f"   ✅ Closure created successfully!")
            print(f"   Closure ID: {closure['id']}")
            print(f"   OpenLR Code: {closure.get('openlr_code', 'N/A')[:20]}...")
            return True
        else:
            print(f"   ❌ Closure creation failed: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Closure creation error: {e}")
        return False


def test_swagger_compatibility():
    """Test OpenAPI schema for Swagger compatibility"""
    print(f"\n📖 Testing OpenAPI Schema...")

    try:
        response = requests.get(f"{API_V1}/openapi.json")

        if response.status_code == 200:
            openapi_schema = response.json()

            # Check for OAuth2PasswordBearer scheme
            security_schemes = openapi_schema.get("components", {}).get(
                "securitySchemes", {}
            )

            print(f"   Available security schemes: {list(security_schemes.keys())}")

            if "OAuth2PasswordBearer" in security_schemes:
                print("   ✅ OAuth2PasswordBearer scheme found!")
                oauth2_scheme = security_schemes["OAuth2PasswordBearer"]
                token_url = (
                    oauth2_scheme.get("flows", {}).get("password", {}).get("tokenUrl")
                )
                print(f"   Token URL: {token_url}")
            else:
                print("   ⚠️ OAuth2PasswordBearer scheme not found")

            # Check global security requirements
            security = openapi_schema.get("security", [])
            print(f"   Global security requirements: {security}")

            return True
        else:
            print(f"   ❌ OpenAPI schema request failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"   ❌ OpenAPI schema error: {e}")
        return False


def main():
    """Run complete authentication test suite"""
    print("🧪 Swagger UI Authentication Test Suite")
    print("=" * 60)

    # Test 1: OAuth2 Login (for Swagger UI)
    oauth_token = test_oauth2_login()

    # Test 2: JSON Login (backward compatibility)
    json_token = test_json_login()

    # Use the OAuth2 token (preferred for Swagger UI)
    test_token = oauth_token or json_token

    if not test_token:
        print(
            "\n❌ No valid token obtained. Check your login credentials and API status."
        )
        return

    # Test 3: Authentication
    auth_success = test_authentication_with_token(test_token)

    # Test 4: Protected endpoint
    if auth_success:
        test_protected_endpoint(test_token)

    # Test 5: OpenAPI schema
    test_swagger_compatibility()

    # Summary and instructions
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)

    if oauth_token and auth_success:
        print("✅ OAuth2 authentication working!")
        print("✅ Swagger UI should now work properly!")

        print(f"\n🎯 FOR SWAGGER UI DEMO:")
        print(f"1. Go to: {BASE_URL}/api/v1/docs")
        print(f"2. Click the 🔒 'Authorize' button")
        print(f"3. In the OAuth2PasswordBearer section, enter:")
        print(f"   Username: chicago_mapper")
        print(f"   Password: SecurePass123")
        print(f"4. Click 'Authorize'")
        print(f"5. Test the /auth/me endpoint")
        print(f"6. Try creating a closure!")

        print(f"\n💡 ALTERNATIVE: Manual Token Entry")
        print(f"If the OAuth2 form doesn't work, you can:")
        print(f"1. Click 'Authorize'")
        print(f"2. In the 'Value' field, enter just the token:")
        print(f"   {test_token}")
        print(f"3. Click 'Authorize'")

    else:
        print("❌ Authentication issues detected!")
        print("\n🔧 Troubleshooting steps:")
        print("1. Make sure you've applied all the code fixes")
        print("2. Restart your API container: docker-compose restart api")
        print("3. Check that the user 'chicago_mapper' exists")
        print("4. Verify the password is 'SecurePass123'")

    print(f"\n🔗 Useful URLs:")
    print(f"📖 Swagger UI: {BASE_URL}/api/v1/docs")
    print(f"💚 Health Check: {BASE_URL}/health")
    print(f"🗄️ Database Admin: http://localhost:8080")


if __name__ == "__main__":
    main()
