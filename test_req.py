import requests  # type: ignore
import json
import sys

# Test the home endpoint
try:
    res = requests.get('http://127.0.0.1:8000/')
    print("Home response:", res.json())
except Exception as e:
    print("Failed connection to home:", e)
    sys.exit(1)

# Test the predict endpoint
url = 'http://127.0.0.1:8000/predict'
file_path = 'confusion_matrix.png'

print(f"Uploading {file_path} to {url}...")
try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'image/png')}
        res = requests.post(url, files=files)
        
    print("Status Code:", res.status_code)
    try:
        print("Response JSON:", json.dumps(res.json(), indent=2))
    except:
        print("Response Text:", res.text)
except Exception as e:
    print("Error during predict request:", e)
