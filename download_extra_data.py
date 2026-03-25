import os
import json
import urllib.request
import urllib.parse
from pathlib import Path

# Target directories
TRAIN_IMG = "archive/Warp-D/train/images"
TRAIN_LBL = "archive/Warp-D/train/labels"
TEST_IMG = "archive/Warp-D/test/images"
TEST_LBL = "archive/Warp-D/test/labels"

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TRAIN_LBL, exist_ok=True)
os.makedirs(TEST_IMG, exist_ok=True)
os.makedirs(TEST_LBL, exist_ok=True)

# Wikimedia Commons API URL
API_URL = "https://commons.wikimedia.org/w/api.php"

def search_commons(query, limit=50):
    params = urllib.parse.urlencode({
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    })
    req_url = f"{API_URL}?{params}"
    
    req = urllib.request.Request(req_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to fetch {query}: {e}")
        return []

    pages = data.get("query", {}).get("pages", {})
    urls = []
    for pid, pdata in pages.items():
        if "imageinfo" in pdata:
            urls.append(pdata["imageinfo"][0]["url"])
    return urls

def fetch_and_label(urls, class_id, prefix=""):
    test_split_idx = int(len(urls) * 0.8) # 80% train, 20% test
    
    for idx, url in enumerate(urls):
        ext = url.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'webp']:
            ext = 'jpg'
        
        filename = f"{prefix}_{idx}.{ext}"
        is_train = idx < test_split_idx
        
        img_dir = TRAIN_IMG if is_train else TEST_IMG
        lbl_dir = TRAIN_LBL if is_train else TEST_LBL
        
        img_path = os.path.join(img_dir, filename)
        lbl_path = os.path.join(lbl_dir, filename.replace(f'.{ext}', '.txt'))
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(img_path, 'wb') as out_file:
                out_file.write(response.read())
            
            # create center bbox: class_id x_center y_center width height
            with open(lbl_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")
            
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

# Biodegradable (Class 0)
print("Fetching Biodegradable (Class 0)...")
bio_queries = ["food waste", "banana peel", "compost pile", "apple core"]
bio_urls = []
for q in bio_queries:
    bio_urls.extend(search_commons(q, 15))
fetch_and_label(bio_urls, class_id=0, prefix="bio")

# Hazardous (Class 2)
print("Fetching Hazardous (Class 2)...")
haz_queries = ["used battery", "medical waste syringe", "chemical container", "broken glass tube"]
haz_urls = []
for q in haz_queries:
    haz_urls.extend(search_commons(q, 15))
fetch_and_label(haz_urls, class_id=2, prefix="haz")

print("Download and labeling complete.")
