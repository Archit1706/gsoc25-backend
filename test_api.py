import requests

BASE_URL = "http://localhost:8000"

print("=== Testing OSM Road Closures API ===")

# Test health
response = requests.get(f"{BASE_URL}/health")
print(f"Health: {response.status_code} - {response.text}")

# Test closures list
response = requests.get(f"{BASE_URL}/api/v1/closures/")
print(f"Closures: {response.status_code} - {response.text}")

# Test statistics
response = requests.get(f"{BASE_URL}/api/v1/closures/statistics/summary")
print(f"Stats: {response.status_code} - {response.text}")

# Test API info
response = requests.get(f"{BASE_URL}/")
print(f"API Info: {response.status_code} - {response.text}")

print("=== Tests Complete ===")
