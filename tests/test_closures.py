import requests
import json

# Test data
closure_data = {
    "geometry": {
        "type": "LineString",
        "coordinates": [[-87.6298, 41.8781], [-87.6290, 41.8785]],
    },
    "description": "Test road closure - water main repair on Michigan Avenue",
    "closure_type": "construction",
    "start_time": "2025-06-15T08:00:00Z",
    "end_time": "2025-06-15T18:00:00Z",
    "source": "Test API",
    "confidence_level": 8,
}

# Test without auth (will fail but show us the error)
response = requests.post("http://localhost:8000/api/v1/closures/", json=closure_data)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Test getting closures (should work without auth)
response = requests.get("http://localhost:8000/api/v1/closures/")
print(f"Get closures status: {response.status_code}")
print(f"Get closures response: {response.text}")

# Test statistics
response = requests.get("http://localhost:8000/api/v1/closures/statistics/summary")
print(f"Stats status: {response.status_code}")
print(f"Stats response: {response.text}")
