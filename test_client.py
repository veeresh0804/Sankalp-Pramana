import requests
import json

# Replace with your actual deployed URL
API_URL = "https://ai-3d-backend-526063550551.us-central1.run.app/visualize"

def test_visualize():
    print("=== AI 3D Backend Test Client ===")
    query = input("Enter a 3D concept (e.g., 'taj mahal', 'human heart'): ")
    
    payload = {
        "query": query,
        "style": "realistic",
        "complexity": "medium"
    }

    print(f"\n[Request] Sending query: '{query}' to {API_URL}...")
    
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        print("\n=== JSON RESPONSE ===")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"\n[Error] Failed to connect to backend: {e}")

if __name__ == "__main__":
    test_visualize()
