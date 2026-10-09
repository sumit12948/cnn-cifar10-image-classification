import os
import sys
import hashlib
import tarfile
import urllib.request
import time

DATA_DIR = r"D:\CNN-CIFAR10-Project\data"
os.makedirs(DATA_DIR, exist_ok=True)
tar_path = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
extract_dir = os.path.join(DATA_DIR, "cifar-10-batches-py")
EXPECTED_MD5 = "c58f30108f718f92721af3b95e74349a"

def verify_md5(path):
    if not os.path.exists(path):
        return False
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == EXPECTED_MD5

if not verify_md5(tar_path):
    print("Downloading CIFAR-10 from high-speed CDN mirror...")
    url = "https://github.com/Digital-Media/cv_data/releases/download/cifar-10/cifar-10-python.tar.gz"
    
    start = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(tar_path, "wb") as out_file:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            pct = (downloaded / total_size * 100) if total_size else 0
            mb = downloaded / (1024 * 1024)
            tot_mb = total_size / (1024 * 1024)
            print(f"\rProgress: {mb:.1f}MB / {tot_mb:.1f}MB ({pct:.1f}%)", end="", flush=True)
            
    print(f"\nDownload completed in {time.time() - start:.1f} seconds!")
else:
    print("Archive already present and MD5 verified.")

# Verify integrity
assert verify_md5(tar_path), "MD5 checksum mismatch!"
print(f"MD5 verified: {EXPECTED_MD5}")

# Extract
print("Extracting archive...")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=DATA_DIR)
print(f"Extraction complete! Files present in: {extract_dir}")

