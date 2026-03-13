import urllib.request
import json
import time
import sys

# Test local endpoint if running, or use a provided URL
base_url = "http://localhost:8080"
if len(sys.argv) > 1:
    base_url = sys.argv[1].rstrip('/')

url = f"{base_url}/visualize"
payload = {
    "query": "taj mahal",
    "style": "realistic",
    "complexity": "high"
}
data = json.dumps(payload).encode("utf-8")
headers = {"Content-Type": "application/json"}

print(f"Testing /visualize endpoint at {url}...")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    start_time = time.time()
    
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
        elapsed = time.time() - start_time
        
        print(f"\n--- API Response (Latency: {elapsed:.2f}s) ---")
        print(json.dumps(result, indent=2))
        
        # Verify structure
        expected_fields = ["success", "type", "data", "processingTime", "source"]
        missing = [f for f in expected_fields if f not in result]
        if not missing:
            print("\n✅ Response structure follows VisualizeResponse contract.")
        else:
            print(f"\n❌ Response structure missing fields: {missing}")

except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} - {e.reason}")
    print(e.read().decode("utf-8"))
except Exception as e:
    print(f"Error: {e}")
    print("Is the server running at http://localhost:8080?")
