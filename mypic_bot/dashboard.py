from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

from .config import Settings


def _image_content_type(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed and guessed.startswith("image/"):
        return guessed
    signature = path.read_bytes()[:16]
    if signature.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if signature.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if signature.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if signature.startswith(b"RIFF") and signature[8:12] == b"WEBP":
        return "image/webp"
    if signature[4:12] in (b"ftypavif", b"ftypavis"):
        return "image/avif"
    return "application/octet-stream"


def _safe_image_path(value: object, images_dir: Path) -> Optional[Path]:
    if not isinstance(value, str) or not value:
        return None
    images_root = images_dir.resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = images_root.parent / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(images_root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _sanitize_record(record: dict[str, Any], images_dir: Path) -> dict[str, Any]:
    payload = deepcopy(record)
    selection_id = str(payload.get("selection_id", ""))
    result = payload.get("result")
    if isinstance(result, dict):
        image_path = _safe_image_path(result.pop("path", None), images_dir)
        if image_path is not None and selection_id:
            result["image_url"] = (
                f"/api/image/{quote(selection_id, safe='')}"
            )
    for candidate in payload.get("candidates", []):
        if isinstance(candidate, dict):
            candidate.pop("local_path", None)
    retrieval = payload.get("retrieval")
    if isinstance(retrieval, dict):
        for candidate in retrieval.get("candidates", []):
            if isinstance(candidate, dict):
                candidate.pop("local_path", None)
    return payload


def read_decisions(
    path: Path,
    limit: int,
    images_dir: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    recent_lines: deque[str] = deque(maxlen=max(1, min(limit, 1000)))
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                recent_lines.append(line)
    records = []
    for line in reversed(recent_lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(_sanitize_record(record, images_dir))
    return records


def resolve_image_for_selection(
    path: Path,
    selection_id: str,
    images_dir: Path,
) -> Optional[Path]:
    if not path.is_file() or not selection_id:
        return None
    with path.open("r", encoding="utf-8") as stream:
        lines = stream.readlines()
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(record.get("selection_id", "")) != selection_id:
            continue
        result = record.get("result")
        if not isinstance(result, dict):
            return None
        return _safe_image_path(result.get("path"), images_dir)
    return None


def create_handler(
    static_root: Path,
    decision_log_path: Path,
    images_dir: Path,
):
    class DashboardHandler(SimpleHTTPRequestHandler):
        def end_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; "
                "script-src 'self'; style-src 'self'; connect-src 'self'",
            )
            super().end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self._write_json(
                    {
                        "status": "ok",
                        "decision_log_exists": decision_log_path.is_file(),
                    }
                )
                return
            if parsed.path == "/api/decisions":
                query = parse_qs(parsed.query)
                try:
                    limit = int(query.get("limit", ["300"])[0])
                except ValueError:
                    limit = 300
                records = read_decisions(
                    decision_log_path,
                    max(1, min(limit, 1000)),
                    images_dir,
                )
                self._write_json(
                    {
                        "source": "CT108 · Live",
                        "count": len(records),
                        "records": records,
                    }
                )
                return
            if parsed.path.startswith("/api/image/"):
                selection_id = unquote(parsed.path[len("/api/image/") :])
                image_path = resolve_image_for_selection(
                    decision_log_path,
                    selection_id,
                    images_dir,
                )
                if image_path is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                    return
                content_type = _image_content_type(image_path)
                payload = image_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

        def _write_json(
            self,
            payload: dict[str, Any],
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def list_directory(self, path: str):
            self.send_error(HTTPStatus.FORBIDDEN, "Directory listing disabled")
            return None

        def log_message(self, message: str, *args: object) -> None:
            logging.info("Dashboard %s - %s", self.client_address[0], message % args)

    return partial(DashboardHandler, directory=str(static_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default=os.getenv("DASHBOARD_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DASHBOARD_PORT", "8090")),
    )
    parser.add_argument(
        "--static-root",
        type=Path,
        default=Path(
            os.getenv(
                "DASHBOARD_STATIC_DIR",
                str(Path(__file__).resolve().parent.parent / "dashboard" / "dist"),
            )
        ),
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    static_root = args.static_root.resolve()
    if not (static_root / "index.html").is_file():
        raise SystemExit(f"Dashboard build not found: {static_root}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    handler = create_handler(
        static_root,
        settings.decision_log_path,
        settings.images_dir,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    logging.info(
        "Mortis dashboard listening on http://%s:%s",
        args.host,
        args.port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
