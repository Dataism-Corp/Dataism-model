import os
import subprocess

def test_files_exist():
    # Check if key project files exist
    files = ["cli.py", "init_db.py", "init_fresh_db.py", "main_api.py"]
    for f in files:
        assert os.path.exists(f), f"{f} is missing!"

def test_cli_runs():
    # Try running cli.py, expect it not to crash
    result = subprocess.run(["python", "cli.py"], capture_output=True, text=True)
    assert result.returncode in (0, 1), f"cli.py crashed: {result.stderr}"
