"""Optional local GUI for the PV detector, served only on 127.0.0.1.

python gui.py [--model PATH] [--labels PATH] [--port 8765] [--no-browser]

Without --model the GUI uses the same model location as `python inference.py`. A missing model disables
local CSV analysis but not the synthetic check. Ctrl-C stops the server and deletes the session's temporary
files; the submission interface remains `python inference.py path/to/test.csv`.
"""

from __future__ import annotations

import argparse
import signal
import sys
import webbrowser
from pathlib import Path

import inference
from pvdetect.gui import create_server


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--model", type=Path, default=inference.MODEL_PATH, help="trained model.json (default: the CLI's location)"
    )
    ap.add_argument(
        "--labels", type=Path, help="labels CSV; every analysed CSV is then diagnosed against it automatically"
    )
    ap.add_argument("--port", type=int, default=8765, help="loopback port (0 picks a free one; default 8765)")
    ap.add_argument("--no-browser", action="store_true", help="only print the address; do not open a browser")
    args = ap.parse_args(argv)
    if not 0 <= args.port <= 65535:
        ap.error("--port must be between 0 and 65535")
    try:
        server = create_server(args.model, port=args.port, labels_path=args.labels)
    except OSError:
        print(f"PV detector GUI: port {args.port} is not available; choose another with --port.", file=sys.stderr)
        return 1
    signal.signal(signal.SIGTERM, _interrupt)
    print(f"PV detector GUI: {server.url}", flush=True)
    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown_and_cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
