import os
import sys
import subprocess
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parent

# Detect OS
IS_WINDOWS = platform.system().lower() == "windows"

# Correct venv path depending on OS
if IS_WINDOWS:
    VENV_PY = ROOT / "winvenv" / "Scripts" / "python.exe"
else:
    VENV_PY = ROOT / "venv" / "bin" / "python"

# If venv doesn't exist, tell user how to create it
if not VENV_PY.exists():
    print("❌ ERROR: Virtual environment not found.")
    print("Please create it:")
    if IS_WINDOWS:
        print("    python -m venv winvenv")
        print("    winvenv\\Scripts\\pip install -r requirements.txt")
    else:
        print("    python3 -m venv venv")
        print("    source venv/bin/activate")
        print("    pip install -r requirements.txt")
    sys.exit(1)

print(f"[INFO] Using Python interpreter: {VENV_PY}")

# 1. Run system checks (optional)
try:
    subprocess.run([str(VENV_PY), "scripts/check_system.py"], check=True)
except FileNotFoundError:
    print("[WARN] No check_system.py found.")

# 2. Launch the bot
subprocess.run(
    [str(VENV_PY), "examples/run_bot.py", *sys.argv[1:]],
    check=True
)
