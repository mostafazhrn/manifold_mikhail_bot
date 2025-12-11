# run_checks.py
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from scripts.check_system import main as run_system_check
from debug_log import send_to_google_forms, build_payload

logs = run_system_check()
payload = build_payload("\n".join(logs))
send_to_google_forms(payload)
