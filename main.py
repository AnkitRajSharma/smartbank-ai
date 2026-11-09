"""SmartBank AI - Root entrypoint and runner.

Allows starting the application from the project root via:
    uvicorn main:app --host 0.0.0.0 --port $PORT
or
    python main.py
as well as the standard:
    uvicorn backend.main:app --host 0.0.0.0 --port $PORT
"""

import os
from pathlib import Path
import sys

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the FastAPI application instance from backend.main
from backend.main import app

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting SmartBank AI on http://{host}:{port} ...")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
