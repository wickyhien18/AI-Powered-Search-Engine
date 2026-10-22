"""Backward-compatible test runner from root directory."""
import runpy
from pathlib import Path

target = Path(__file__).resolve().parent / "backend" / "test_backend.py"
runpy.run_path(str(target), run_name="__main__")
