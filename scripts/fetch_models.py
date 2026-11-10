"""Download the model files (run before building / running from source).

The MediaPipe models go into vroidmocap/assets/models and are packed into the
exe. The large whole-body model is not packed: the app fetches it on first use
into the user's data folder. `--all` fetches it there now."""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vroidmocap.core.detect import DOWNLOAD_ON_USE, MODEL_FILES, MODEL_URLS, ensure_model  # noqa: E402

dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vroidmocap", "assets", "models")
os.makedirs(dest, exist_ok=True)
for key, name in MODEL_FILES.items():
    if key in DOWNLOAD_ON_USE:
        continue
    path = os.path.join(dest, name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print("ok      ", name)
        continue
    print("download", name)
    urllib.request.urlretrieve(MODEL_URLS[key], path + ".part")
    os.replace(path + ".part", path)
if "--all" in sys.argv:
    for key in DOWNLOAD_ON_USE:
        print("fetch   ", MODEL_FILES[key])
        p = ensure_model(key)
        print("ok      " if p else "FAILED  ", p or MODEL_FILES[key])
        if not p:
            sys.exit(1)
