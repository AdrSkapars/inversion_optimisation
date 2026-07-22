"""
viewer.py — launch the bloom-viewer dev server.

Usage:
    python viewer.py                         # uses runs/ in the same directory
    python viewer.py runs_9/quantized_3turns # specific results folder
    python viewer.py runs_9 --port 5174      # custom port

Only requires Python stdlib — no uv, no venv, no extra packages.
npm must be available on PATH (the viewer itself is a Svelte app).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent.resolve()
VIEWER_DIR   = (SCRIPT_DIR / ".." / ".." / "src" / "bloom-viewer").resolve()
DEFAULT_PORT = 5173


def main():
    parser = argparse.ArgumentParser(description="Launch the bloom-viewer dev server.")
    parser.add_argument(
        "results_dir", nargs="?", default=None,
        help="Results folder to view (relative to this script or absolute). "
             "Defaults to the most recently modified runs_* folder.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    # Resolve results dir
    if args.results_dir:
        results_path = Path(args.results_dir)
        if not results_path.is_absolute():
            results_path = SCRIPT_DIR / results_path
        results_path = results_path.resolve()
    else:
        # Pick the most recently modified runs_* folder, falling back to SCRIPT_DIR
        candidates = sorted(SCRIPT_DIR.glob("runs*"), key=lambda p: p.stat().st_mtime, reverse=True)
        results_path = candidates[0] if candidates else SCRIPT_DIR
        print(f"No results dir given — using: {results_path}", flush=True)

    if not results_path.exists():
        print(f"ERROR: results dir does not exist: {results_path}", file=sys.stderr)
        sys.exit(1)

    if not VIEWER_DIR.exists():
        print(f"ERROR: bloom-viewer not found at {VIEWER_DIR}", file=sys.stderr)
        print("Expected: src/bloom-viewer/package.json two levels up from this script.", file=sys.stderr)
        sys.exit(1)

    print(f"Viewer:  {VIEWER_DIR}", flush=True)
    print(f"Results: {results_path}", flush=True)
    print(f"URL:     http://localhost:{args.port}", flush=True)
    print("Press Ctrl-C to stop.\n", flush=True)

    env = {**os.environ, "TRANSCRIPT_DIR": str(results_path)}

    # On Windows npm is npm.cmd; on Linux/macOS it's npm.
    npm = "npm.cmd" if sys.platform == "win32" else "npm"

    try:
        subprocess.run(
            [npm, "run", "dev", "--", "--port", str(args.port)],
            cwd=str(VIEWER_DIR),
            env=env,
        )
    except FileNotFoundError:
        print(f"ERROR: '{npm}' not found. Is Node/npm installed and on PATH?", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nViewer stopped.", flush=True)


if __name__ == "__main__":
    main()
