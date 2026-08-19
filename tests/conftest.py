import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"

for path in (PROJECT_ROOT, APP_DIR):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)
