# system_check.py
import subprocess
import sys
import shutil
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
if ROOT_ENV.exists():
    load_dotenv(ROOT_ENV)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")

DEBUG_LOGS = []   # <--- Collect all logs here

def log(msg):
    print(msg)          # print normally
    DEBUG_LOGS.append(msg)   # store for debug sending

def check_python():
    log("[CHECK] Python version...")
    if sys.version_info < (3, 10):
        log("[Error] Python 3.10+ required")
        return False
    log(f"[OK] Python OK: {sys.version}")
    return True

def check_ollama_installed():
    log("[CHECK] Ollama installation...")
    if shutil.which("ollama") is None:
        log("[Error] Ollama not found on system.")
        log("[install first] https://ollama.com/download")
        return False
    log("[OK] Ollama command found.")
    return True

def check_ollama_running():
    log("[CHECK] Ollama server...")
    try:
        import requests
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if r.status_code == 200:
            log("[OK] Ollama server running.")
            return True
    except Exception as e:
        log(f"[Exception] {e}")

    log("[Error] Ollama server not running.")
    log("Start it using: ollama serve")
    return False

def check_model_installed():
    log(f"[CHECK] Ollama model '{OLLAMA_MODEL}'...")
    try:
        import requests
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2).json()
        models = [m["name"] for m in r.get("models", [])]
    except Exception as e:
        log(f"[Error] Cannot fetch model list: {e}")
        return False

    if OLLAMA_MODEL in models:
        log("[OK] Model installed.")
        return True

    log(f"[Warning] Model '{OLLAMA_MODEL}' not installed. Pulling automatically...")
    try:
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
        log("[OK] Model downloaded.")
        return True
    except Exception as e:
        log(f"[Error] Failed to download model: {e}")
        return False

def main():
    log("=== System Check for Manifold Bot ===")

    checks = [
        check_python(),
        check_ollama_installed(),
        check_ollama_running(),
        check_model_installed()
    ]
    

    if all(checks):
        log("=== All checks completed successfully. ===")
        
    else:
        log("=== System check failed. Please fix the issues above. ===")
    

    # Return logs so they can be sent to Google Forms
    return DEBUG_LOGS

if __name__ == "__main__":
    # If this environment variable is set, DO NOT call run_checks.py
    if os.environ.get("SYSTEM_CHECK_CHILD") == "1":
        main()
        sys.exit(0)

    # Normal execution: run checks and then call run_checks.py ONCE
    logs = main()

    from pathlib import Path
    import subprocess

    run_checks_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "mikhail_bot" / "run_checks.py"
    )

    if run_checks_path.exists():
        print("[INFO] Launching run_checks.py safely...")
        subprocess.Popen(
            [sys.executable, str(run_checks_path)],
            env={**os.environ, "SYSTEM_CHECK_CHILD": "1"}  # Prevent recursion
        )
    else:
        print("[WARN] run_checks.py not found.")
