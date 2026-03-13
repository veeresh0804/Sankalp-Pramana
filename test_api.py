import urllib.request
import json
import time

url = "https://ai-3d-backend-526063550551.us-central1.run.app/search_model"
data = json.dumps({"concept": "tajmahal", "top_k": 3}).encode("utf-8")
headers = {"Content-Type": "application/json"}

for attempt in range(1, 10):
    try:
        print(f"Attempt {attempt}: Sending POST request to {url} ...")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        start_time = time.time()
        
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - start_time
            
            print(f"\n--- API Response (Latency: {elapsed:.2f}s) ---")
            print(json.dumps(result, indent=2))
            
            model_url = result.get("model_url")
            if model_url:
                print(f"\n--- Validating Model URL ---")
                print(f"Fetching: {model_url}")
                
                try:
                    headers_head = {"Range": "bytes=0-3"}
                    req_get = urllib.request.Request(model_url, method="GET", headers=headers_head)
                    with urllib.request.urlopen(req_get, timeout=15) as get_resp:
                        magic_bytes = get_resp.read()
                        print(f"Status: {get_resp.status}")
                        print(f"Content-Type: {get_resp.headers.get('Content-Type')}")
                        print(f"Magic bytes (glTF): {magic_bytes}")
                        if magic_bytes == b"glTF":
                            print("✅ Confirmed valid GLB file signature.")
                        else:
                            print("⚠️ Warning: File signature does not match standard GLB.")
                except Exception as e_head:
                    print(f"Validation request failed: {e_head}")
            
            break # Exit loop if successful
            
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        if "503" in str(e.code) or "models are loading" in body:
            print("Models are loading, retrying in 10s...")
            time.sleep(10)
        else:
            print(f"HTTP Error: {e.code} - {e.reason}")
            print(body)
            break
    except Exception as e:
        print(f"Error: {e}")
        break
