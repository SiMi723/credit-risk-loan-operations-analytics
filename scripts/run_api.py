"""
Run the FastAPI server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --port 8080
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Start the Credit Risk API server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", default=False)
    args = parser.parse_args()
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
