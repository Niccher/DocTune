"""
run_web.py
----------
Launcher script for the FastAPI AdminLTE v4 Fine-Tuning Web Studio.

Usage:
    python run_web.py
    python run_web.py --host 0.0.0.0 --port 8080 --reload
"""

import errno
import sys
import argparse
import uvicorn
from pathlib import Path

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def start_server(host: str, port: int, reload: bool):
    print("=" * 70)
    print("PRIVATE LLM FINE-TUNING STUDIO (AdminLTE v4 + FastAPI)")
    print("=" * 70)
    print(f"Server URL : http://{host}:{port}")
    print(f"Dashboard  : http://{host}:{port}/")
    print(f"Data Prep  : http://{host}:{port}/data-prep")
    print(f"Training   : http://{host}:{port}/training")
    print(f"Models     : http://{host}:{port}/models")
    print(f"Playground : http://{host}:{port}/playground")
    print("=" * 70)

    try:
        uvicorn.run("web.main:app", host=host, port=port, reload=reload)
    except (PermissionError, OSError) as exc:
        is_permission_denied = isinstance(exc, PermissionError) or getattr(exc, "errno", None) == errno.EACCES
        if port < 1024 and is_permission_denied:
            fallback_port = 8080
            print(f"\n[NOTICE] Binding port {port} requires root/administrator privileges.")
            print(f"[NOTICE] Automatically falling back to standard unprivileged port: {fallback_port}")
            start_server(host=host, port=fallback_port, reload=reload)
        else:
            raise


def main():
    parser = argparse.ArgumentParser(description="Run the Fine-Tuning Studio Web App")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=80, help="Port to listen on (default: 80, falls back to 8080 if unprivileged)")
    parser.add_argument("--reload", action="store_true", help="Enable automatic code reload")
    args = parser.parse_args()

    start_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
