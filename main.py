"""
Root entry point forwarding to backend.main:app.
This allows commands like `uvicorn main:app --reload` to continue working seamlessly
from the project root, keeping existing run and deploy scripts intact.
"""
import sys
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent / "backend")
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from backend.main import app  # noqa: F401
