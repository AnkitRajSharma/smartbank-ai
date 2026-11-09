"""Streamlit entrypoint for SmartBank AI Assistant."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).resolve().parent / "app.py"
    runpy.run_path(str(app_path), run_name="__main__")
