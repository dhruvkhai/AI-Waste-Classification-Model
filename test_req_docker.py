import requests
import json
import sys

# Test the predict endpoint
url = 'http://127.0.0.1:8080/predict'
file_path = 'confusion_matrix.png'

print(f"Testing Docker container at {url}...")
try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'image/png')}
        res = requests.post(url, files=files)
        
    print("Status Code:", res.status_code)
    print("Response JSON:", json.dumps(res.json(), indent=2))
except Exception as e:
    print("Error during predict request:", e)
