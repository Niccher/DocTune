"""
run_web.py
----------
Launcher script for the FastAPI AdminLTE v4 Fine-Tuning Web Studio.

Usage:
    python run_web.py
    python run_web.py --host 0.0.0.0 --port 8080 --reload
"""

import sys
import argparse
import uvicorn
from pathlib import Path

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run the Fine-Tuning Studio Web App")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=4444, help="Port to listen on (default: 4444)")
    parser.add_argument("--reload", action="store_true", help="Enable automatic code reload")
    args = parser.parse_args()

    print("=" * 70)
    print("PRIVATE LLM FINE-TUNING STUDIO (AdminLTE v4 + FastAPI)")
    print("=" * 70)
    print(f"Server URL : http://{args.host}:{args.port}")
    print(f"Dashboard  : http://{args.host}:{args.port}/")
    print(f"Data Prep  : http://{args.host}:{args.port}/data-prep")
    print(f"Training   : http://{args.host}:{args.port}/training")
    print(f"Models     : http://{args.host}:{args.port}/models")
    print(f"Playground : http://{args.host}:{args.port}/playground")
    print("=" * 70)

    uvicorn.run("web.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
