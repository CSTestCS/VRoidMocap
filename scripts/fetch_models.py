"""Download the MediaPipe model files into vroidmocap/assets/models (run before building)."""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vroidmocap.core.detect import MODEL_FILES, MODEL_URLS  # noqa: E402

dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vroidmocap", "assets", "models")
os.makedirs(dest, exist_ok=True)
for key, name in MODEL_FILES.items():
    path = os.path.join(dest, name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print("ok      ", name)
        continue
    print("download", name)
    urllib.request.urlretrieve(MODEL_URLS[key], path + ".part")
    os.replace(path + ".part", path)
