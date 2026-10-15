"""Streamlit Community Cloud Root Entrypoint (streamlit_app.py).

Delegates execution to app.py so that any default main file path setting
('streamlit_app.py' or 'app.py') runs successfully.
"""
from pathlib import Path
import runpy

app_path = Path(__file__).resolve().parent / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
