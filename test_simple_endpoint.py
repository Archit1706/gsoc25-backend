import requests

# Test just getting all closures without filters
response = requests.get("http://localhost:8000/api/v1/closures/?active_only=false")
print(f"All closures (no filter): {response.status_code}")
print(f"Response: {response.text}")

# Test root endpoint
response = requests.get("http://localhost:8000/")
print(f"Root: {response.status_code} - {response.json()}")

# Test health
response = requests.get("http://localhost:8000/health")
print(f"Health: {response.status_code} - {response.json()}")
