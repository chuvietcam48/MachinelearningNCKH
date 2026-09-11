import json
import os
import time
import urllib.request
import urllib.error

def test_key(api_key, model_name="gemini-flash-latest"):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    data = json.dumps({
        "contents": [{"parts": [{"text": "hi"}]}]
    }).encode('utf-8')
    
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    for attempt in range(2): # Try up to 2 times
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return "SUCCESS"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                return "QUOTA_EXCEEDED (429)"
            elif e.code == 403:
                return "FORBIDDEN (403)"
            elif e.code == 400:
                return "BAD_REQUEST (400)"
            elif e.code == 503:
                if attempt == 0:
                    time.sleep(2)
                    continue
                return "SERVICE_UNAVAILABLE (503)"
            elif e.code == 404:
                body = e.read().decode('utf-8')
                return f"NOT_FOUND (404): {body}"
            else:
                return f"HTTP_ERROR_{e.code}"
        except Exception as e:
            if "timed out" in str(e).lower() and attempt == 0:
                time.sleep(2)
                continue
            return f"ERROR: {str(e)[:50]}"

def main():
    try:
        with open('api_keys.json', 'r') as f:
            data = json.load(f)
            keys = data.get("keys", [])
            model_name = data.get("model", "gemini-flash-latest")
    except Exception as e:
        print(f"Could not read api_keys.json: {e}")
        return

    print(f"Found {len(keys)} keys. Checking...")
    print("-" * 50)
    
    valid_keys = []
    
    for i, key in enumerate(keys):
        status = test_key(key, model_name)
        masked_key = key[:10] + "..." + key[-5:] if len(key) > 15 else key
        print(f"Key {i+1}/{len(keys)} [{masked_key}]: {status}")
        if status == "SUCCESS":
            valid_keys.append(key)
        time.sleep(1) # Be nice to the API
        
    print("-" * 50)
    print(f"Summary: {len(valid_keys)}/{len(keys)} keys are working.")
    
    # Optionally save working keys to a file or print them
    if valid_keys:
        print("\nWorking keys:")
        for k in valid_keys:
            print(k)

if __name__ == "__main__":
    main()
