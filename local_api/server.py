"""Local HTTP server for the MiniMax-H3 DashScope async video API.

AiVideo's Qwen video client can point its base URL here instead of
``qianwenaiapi.com``. The server validates the same request shape, then runs
the existing ``h3`` binary. No C engine changes are required: duration,
resolution, and image roles already map onto ``--seconds``, ``--width`` /
``--height``, ``--first-frame``, ``--last-frame``, and repeatable
``--ref-image`` / ``--ref-video`` / ``--ref-audio``.

One generation runs at a time. Every invocation passes ``--ssd-streaming``.
``h3`` loads ``h3_shaders.metal`` from its working directory, so the child
process is started in the shader's directory and, when that file is not
already in the server's cwd, a symlink is created there too.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable

from local_api.mapping import (
    APIError,
    JobSpec,
    MediaItem,
    adapt_canvas,
    build_h3_command,
    parse_synthesis_body,
)

SYNTHESIS_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
GENERATE_PATH = "/generate"
TASK_PREFIX = "/api/v1/tasks/"
VIDEO_PREFIX = "/api/v1/videos/"
MAX_BODY_BYTES = 1_000_000
MAX_MEDIA_BYTES = 512 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 180
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


class GenerationError(Exception):
    def __init__(self, message: str, code: str = "InternalError") -> None:
        super().__init__(message)
        self.code = code


@dataclass
class Task:
    task_id: str
    prompt: str
    spec: JobSpec
    status: str
    submit_time: str
    width: int | None = None
    height: int | None = None
    scheduled_time: str | None = None
    end_time: str | None = None
    video_path: Path | None = None
    error_code: str | None = None
    error_message: str | None = None
    image_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"


def publish_shader(shader: Path, work_cwd: Path) -> Path:
    """Make ``h3_shaders.metal`` visible and return the directory h3 should use.

    The returned directory contains the real shader file. When the server cwd
    does not already have that filename, a symlink is added there as well.
    An existing different file in the server cwd is left untouched.
    """
    shader = shader.resolve()
    if not shader.is_file():
        raise FileNotFoundError(f"h3_shaders.metal not found: {shader}")
    dest = work_cwd / "h3_shaders.metal"
    if not dest.exists() and not dest.is_symlink():
        dest.symlink_to(shader)
    return shader.parent


def discover_h3(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"h3 binary not found: {path}")
        return path.resolve()
    for candidate in (Path.cwd() / "h3", REPO_ROOT / "h3"):
        if candidate.is_file():
            return candidate.resolve()
    found = shutil.which("h3")
    if found:
        return Path(found).resolve()
    raise SystemExit("h3 binary not found; build ./h3 or pass --h3")


def discover_shader(explicit: str | None, h3_bin: Path, work_cwd: Path) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            work_cwd / "h3_shaders.metal",
            h3_bin.parent / "h3_shaders.metal",
            REPO_ROOT / "h3_shaders.metal",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "h3_shaders.metal not found; start from the repo directory or pass --shader"
    )


def sniff_media_size(data: bytes) -> tuple[int, int] | None:
    """Read width and height from a short image header, if one is recognized."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24 and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data.startswith((b"GIF87a", b"GIF89a")) and len(data) >= 10:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if data.startswith(b"BM") and len(data) >= 26:
        return int.from_bytes(data[18:22], "little", signed=True), int.from_bytes(
            data[22:26], "little", signed=True
        )
    if data.startswith(b"\xff\xd8"):
        return _jpeg_size(data)
    if len(data) >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp_size(data)
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    index = 2
    limit = len(data)
    while index + 8 < limit:
        if data[index] != 0xFF:
            return None
        while index < limit and data[index] == 0xFF:
            index += 1
        if index >= limit:
            return None
        marker = data[index]
        index += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > limit:
            return None
        segment = int.from_bytes(data[index : index + 2], "big")
        if segment < 2:
            return None
        if marker in (
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        ):
            if index + 7 > limit:
                return None
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            return width, height
        index += segment
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8 " and len(data) >= 30:
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    return None


def media_dimensions(path: Path) -> tuple[int, int]:
    blob = path.read_bytes()[: 2 * 1024 * 1024]
    sniffed = sniff_media_size(blob)
    if sniffed and sniffed[0] > 0 and sniffed[1] > 0:
        return sniffed
    ffprobe = os.environ.get("H3_FFPROBE", "ffprobe")
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GenerationError(
            f"cannot read media dimensions for {path.name}: {exc}",
            "InvalidParameter",
        ) from exc
    line = proc.stdout.strip().splitlines()
    if proc.returncode == 0 and line and "x" in line[0]:
        width_text, height_text = line[0].split("x", 1)
        try:
            width = int(width_text)
            height = int(height_text)
        except ValueError:
            width = height = 0
        if width > 0 and height > 0:
            return width, height
    detail = proc.stderr.strip() or "unrecognized image"
    raise GenerationError(
        f"cannot read media dimensions for {path.name}: {detail}",
        "InvalidParameter",
    )


def local_media_path(url: str, work_cwd: Path) -> Path | None:
    """Return a filesystem path for file/local media, or None for http(s)."""
    if url.startswith(("http://", "https://")):
        return None
    if "://" in url and not url.startswith("file:"):
        scheme = url.split(":", 1)[0]
        raise GenerationError(
            f"unsupported media URL scheme: {scheme}",
            "InvalidParameter",
        )
    if url.startswith("file:"):
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc not in ("", "localhost"):
            raise GenerationError(
                f"unsupported file URL host: {parsed.netloc}",
                "InvalidParameter",
            )
        raw = urllib.request.url2pathname(urllib.parse.unquote(parsed.path))
        path = Path(raw)
    else:
        path = Path(url)
    if not path.is_absolute():
        path = work_cwd / path
    return path


def suffix_for(url: str, role: str) -> str:
    parsed = urllib.parse.urlparse(url) if "://" in url else None
    raw_path = parsed.path if parsed is not None else url
    suffix = Path(raw_path).suffix.lower()
    if suffix and 1 < len(suffix) <= 8 and suffix[1:].isalnum():
        return suffix
    if role in ("reference_video",):
        return ".mp4"
    if role == "reference_audio":
        return ".wav"
    return ".img"


def download_http(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "h3-local-api"})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            remaining = MAX_MEDIA_BYTES
            with dest.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    if remaining < 0:
                        raise GenerationError(
                            f"media exceeds {MAX_MEDIA_BYTES} bytes: {url}",
                            "InvalidParameter",
                        )
                    handle.write(chunk)
    except GenerationError:
        raise
    except urllib.error.HTTPError as exc:
        raise GenerationError(
            f"media download failed: {url} HTTP {exc.code}",
            "InvalidParameter",
        ) from exc
    except urllib.error.URLError as exc:
        raise GenerationError(
            f"media download failed: {url} ({exc.reason})",
            "InvalidParameter",
        ) from exc


def materialize_media(url: str, role: str, dest: Path, work_cwd: Path) -> None:
    source = local_media_path(url, work_cwd)
    partial = dest.with_name(dest.name + ".partial")
    try:
        if source is None:
            download_http(url, partial)
        else:
            if not source.is_file():
                raise GenerationError(
                    f"media file not found: {source}",
                    "InvalidParameter",
                )
            if source.stat().st_size > MAX_MEDIA_BYTES:
                raise GenerationError(
                    f"media exceeds {MAX_MEDIA_BYTES} bytes: {source}",
                    "InvalidParameter",
                )
            shutil.copyfile(source, partial)
        os.replace(partial, dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _first_path(paths: list[tuple[MediaItem, Path]], roles: set[str]) -> Path | None:
    for item, path in paths:
        if item.role in roles:
            return path
    return None


def resolve_canvas(spec: JobSpec, paths: list[tuple[MediaItem, Path]]) -> tuple[int, int]:
    if spec.width is not None and spec.height is not None:
        return spec.width, spec.height
    probe = _first_path(paths, {"first_frame"})
    if probe is None:
        probe = _first_path(paths, {"last_frame"})
    if probe is None:
        probe = _first_path(paths, {"reference_image", "reference_video"})
    if probe is None:
        return adapt_canvas(16, 9)
    width, height = media_dimensions(probe)
    return adapt_canvas(width, height)


class App:
    def __init__(
        self,
        *,
        h3_bin: Path,
        model_dir: Path,
        shader: Path,
        work_cwd: Path,
        output_dir: Path,
        steps: int,
        layers: int,
        reuse: int,
        seed: int | None,
        host: str,
        port: int,
    ) -> None:
        self.h3_bin = h3_bin
        self.model_dir = model_dir.resolve()
        self.shader = shader
        self.work_cwd = work_cwd.resolve()
        self.output_dir = output_dir.resolve()
        self.steps = steps
        self.layers = layers
        self.reuse = reuse
        self.seed = seed
        self.host = host
        self.port = port
        self.bound_port = port
        self.child_cwd = work_cwd
        self.tasks: dict[str, Task] = {}
        self.tasks_lock = threading.Lock()
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.proc_lock = threading.Lock()
        self.current_proc: subprocess.Popen[str] | None = None
        self.httpd: ThreadingHTTPServer | None = None
        self.worker: threading.Thread | None = None

    def prepare(self) -> None:
        if not self.model_dir.is_dir():
            raise SystemExit(f"model directory not found: {self.model_dir}")
        if not (2 <= self.steps <= 1000):
            raise SystemExit("--steps must be in [2, 1000]")
        if not (35 <= self.layers <= 50):
            raise SystemExit("--layers must be in [35, 50]")
        if not (1 <= self.reuse <= 3):
            raise SystemExit("--reuse must be in [1, 3]")
        self.work_cwd.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.child_cwd = publish_shader(self.shader, self.work_cwd)

    def serve(self) -> ThreadingHTTPServer:
        self.prepare()
        handler = handler_factory(self)
        httpd = LocalHTTPServer((self.host, self.port), handler)
        self.httpd = httpd
        self.bound_port = int(httpd.server_address[1])
        self.worker = threading.Thread(
            target=self._worker, name="h3-local-api-worker", daemon=True
        )
        self.worker.start()
        return httpd

    def stop(self) -> None:
        with self.proc_lock:
            proc = self.current_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
        self.queue.put(None)
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.worker is not None:
            self.worker.join(timeout=5)

    def fallback_host(self) -> str:
        return f"{self.host}:{self.bound_port}"

    def submit(self, spec: JobSpec) -> Task:
        task = Task(
            task_id=str(uuid.uuid4()),
            prompt=spec.prompt,
            spec=spec,
            status="PENDING",
            submit_time=timestamp(),
            width=spec.width,
            height=spec.height,
        )
        with self.tasks_lock:
            self.tasks[task.task_id] = task
        self.queue.put(task.task_id)
        return task

    def get_task(self, task_id: str) -> Task | None:
        with self.tasks_lock:
            return self.tasks.get(task_id)

    def _worker(self) -> None:
        while True:
            task_id = self.queue.get()
            try:
                if task_id is None:
                    return
                self._run_task(task_id)
            except Exception as exc:  # noqa: BLE001 - keep the worker alive
                self._fail(task_id, "InternalError", f"internal error: {exc}")
            finally:
                self.queue.task_done()

    def _run_task(self, task_id: str) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        with task.lock:
            task.status = "RUNNING"
            task.scheduled_time = timestamp()
        try:
            self._generate(task)
        except GenerationError as exc:
            self._fail(task_id, exc.code, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._fail(task_id, "InternalError", f"internal error: {exc}")

    def _fail(self, task_id: str | None, code: str, message: str) -> None:
        if not task_id:
            return
        task = self.get_task(task_id)
        if task is None:
            return
        with task.lock:
            task.status = "FAILED"
            task.error_code = code
            task.error_message = message
            task.end_time = timestamp()
        print(f"[h3-local-api] {task_id} FAILED {message}", file=sys.stderr, flush=True)

    def _generate(self, task: Task) -> None:
        task_dir = self.output_dir / task.task_id
        media_dir = task_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        video_path = task_dir / "video.mp4"
        paths: list[tuple[MediaItem, Path]] = []
        for index, item in enumerate(task.spec.media):
            dest = media_dir / f"{index:02d}-{item.role}{suffix_for(item.url, item.role)}"
            materialize_media(item.url, item.role, dest, self.work_cwd)
            paths.append((item, dest))
        width, height = resolve_canvas(task.spec, paths)
        with task.lock:
            task.width = width
            task.height = height
            task.image_count = sum(
                1
                for item, _path in paths
                if item.role in ("first_frame", "last_frame", "reference_image")
            )
        first = _first_path(paths, {"first_frame"})
        last = _first_path(paths, {"last_frame"})
        references = [
            (item.role, path)
            for item, path in paths
            if item.role in ("reference_image", "reference_video", "reference_audio")
        ]
        seed = task.spec.seed if task.spec.seed is not None else self.seed
        command = build_h3_command(
            h3_bin=self.h3_bin,
            model_dir=self.model_dir,
            prompt=task.prompt,
            width=width,
            height=height,
            duration=task.spec.duration,
            output=video_path,
            first_frame=first,
            last_frame=last,
            references=references,
            steps=task.spec.steps if task.spec.steps is not None else self.steps,
            layers=task.spec.layers if task.spec.layers is not None else self.layers,
            reuse=task.spec.reuse if task.spec.reuse is not None else self.reuse,
            seed=seed,
            frames=task.spec.frames,
            render_width=task.spec.render_width,
            render_height=task.spec.render_height,
        )
        self._log_command(task.task_id, command)
        stderr_text, code = self._run_h3(command)
        if code != 0 or not video_path.is_file() or video_path.stat().st_size <= 0:
            detail = stderr_text.strip()
            message = f"h3 exited {code}"
            if detail:
                message = f"{message}: {detail[-2000:]}"
            if code == 0:
                message = "h3 exited 0 without writing a video"
            raise GenerationError(message, "InternalError")
        with task.lock:
            task.status = "SUCCEEDED"
            task.video_path = video_path
            task.end_time = timestamp()
        print(f"[h3-local-api] {task.task_id} SUCCEEDED {video_path}", file=sys.stderr, flush=True)

    def _log_command(self, task_id: str, command: list[str]) -> None:
        logged = list(command)
        try:
            prompt_index = logged.index("-p")
        except ValueError:
            prompt_index = -1
        if prompt_index >= 0 and prompt_index + 1 < len(logged) and len(logged[prompt_index + 1]) > 120:
            logged[prompt_index + 1] = logged[prompt_index + 1][:117] + "..."
        printable = " ".join(shlex.quote(part) for part in logged)
        print(f"[h3-local-api] {task_id} {printable}", file=sys.stderr, flush=True)

    def _run_h3(self, command: list[str]) -> tuple[str, int]:
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(self.child_cwd),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
            )
        except OSError as exc:
            raise GenerationError(f"failed to start h3: {exc}", "InternalError") from exc
        chunks: list[str] = []
        code = 1
        with proc:
            with self.proc_lock:
                self.current_proc = proc
            try:
                assert proc.stderr is not None
                for line in proc.stderr:
                    chunks.append(line)
                    sys.stderr.write(line)
                    sys.stderr.flush()
                code = proc.wait()
            finally:
                if proc.poll() is None:
                    proc.kill()
                    code = proc.wait()
                with self.proc_lock:
                    if self.current_proc is proc:
                        self.current_proc = None
        return "".join(chunks), code


def handler_factory(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write("[h3-local-api] " + (fmt % args) + "\n")

        def do_GET(self) -> None:  # noqa: N802
            self._guard(self._dispatch)

        def do_POST(self) -> None:  # noqa: N802
            self._guard(self._dispatch)

        def _guard(self, method) -> None:
            try:
                method()
            except APIError as exc:
                self._send_json(
                    exc.status,
                    {
                        "request_id": str(uuid.uuid4()),
                        "code": exc.code,
                        "message": exc.message,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    500,
                    {
                        "request_id": str(uuid.uuid4()),
                        "code": "InternalError",
                        "message": str(exc),
                    },
                )
            finally:
                self.close_connection = True

        def _dispatch(self) -> None:
            path = self._path()
            if self.command == "POST" and path in (SYNTHESIS_PATH, GENERATE_PATH):
                self._submit()
                return
            if self.command == "GET" and path.startswith(TASK_PREFIX):
                self._task(path[len(TASK_PREFIX) :])
                return
            if self.command == "GET" and path.startswith(VIDEO_PREFIX):
                self._video(path[len(VIDEO_PREFIX) :])
                return
            if self.command in ("GET", "POST"):
                raise APIError("unknown path", status=404)
            raise APIError("method not allowed", status=405)

        def _path(self) -> str:
            path = urllib.parse.urlsplit(self.path).path
            if len(path) > 1 and path.endswith("/"):
                path = path[:-1]
            return path

        def _submit(self) -> None:
            # Loopback clients may omit Authorization. Any Bearer value is
            # accepted and ignored; this server does not call cloud MiniMax.
            _ = self.headers.get("Authorization")
            _ = self.headers.get("X-DashScope-Async")
            length = self.headers.get("Content-Length")
            if length is None or not length.isdigit():
                raise APIError("Content-Length is required")
            size = int(length)
            if size > MAX_BODY_BYTES:
                raise APIError("request body is too large")
            raw = self.rfile.read(size)
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise APIError(f"request body must be JSON: {exc}") from exc
            spec = parse_synthesis_body(body)
            task = app.submit(spec)
            self._send_json(
                200,
                {
                    "request_id": str(uuid.uuid4()),
                    "output": {
                        "task_id": task.task_id,
                        "task_status": "PENDING",
                    },
                },
            )

        def _task(self, task_id: str) -> None:
            if not UUID_RE.match(task_id):
                raise APIError("task not found", status=404)
            task = app.get_task(task_id)
            if task is None:
                raise APIError("task not found", status=404)
            self._send_json(200, task_view(task, self._public_host()))

        def _video(self, name: str) -> None:
            task_id = name[:-4] if name.endswith(".mp4") else name
            if not UUID_RE.match(task_id):
                raise APIError("video not found", status=404)
            task = app.get_task(task_id)
            path = None
            status = None
            if task is not None:
                with task.lock:
                    status = task.status
                    path = task.video_path
            if task is None or status != "SUCCEEDED" or path is None or not path.is_file():
                raise APIError("video not found", status=404)
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(size))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                with path.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile, length=1024 * 1024)
            except BrokenPipeError:
                return

        def _public_host(self) -> str:
            host = self.headers.get("Host", "")
            host = host.strip()
            if host and not any(char in host for char in "\r\n /\\"):
                return host
            return app.fallback_host()

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                return

    return Handler


def task_view(task: Task, host: str) -> dict:
    with task.lock:
        output: dict[str, object] = {
            "task_id": task.task_id,
            "task_status": task.status,
            "submit_time": task.submit_time,
        }
        if task.scheduled_time:
            output["scheduled_time"] = task.scheduled_time
        if task.end_time:
            output["end_time"] = task.end_time
        if task.prompt:
            output["orig_prompt"] = task.prompt
        payload: dict[str, object] = {
            "request_id": str(uuid.uuid4()),
            "output": output,
        }
        if task.status == "SUCCEEDED" and task.video_path is not None:
            output["video_url"] = f"http://{host}/api/v1/videos/{task.task_id}.mp4"
            payload["usage"] = {
                "SR": "768",
                "duration": task.spec.duration,
                "image_count": task.image_count,
                "input_seconds": 0,
                "output_seconds": task.spec.duration,
                "size": f"{task.width}*{task.height}",
                "total_seconds": task.spec.duration,
                "video_count": 1,
            }
        if task.status == "FAILED":
            output["code"] = task.error_code or "InternalError"
            output["message"] = task.error_message or "generation failed"
        return payload


class LocalHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a local DashScope-compatible MiniMax-H3 video API."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("H3_LOCAL_API_HOST", "127.0.0.1"),
        help="bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("H3_LOCAL_API_PORT", 8765),
        help="bind port (default: 8765)",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("H3_MODEL_DIR"),
        help="MiniMax-H3 checkpoint directory (or H3_MODEL_DIR)",
    )
    parser.add_argument(
        "--h3",
        default=os.environ.get("H3_BIN"),
        help="path to the h3 binary (default: ./h3)",
    )
    parser.add_argument(
        "--shader",
        default=os.environ.get("H3_SHADER"),
        help="path to h3_shaders.metal (default: next to h3 or the repo root)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("H3_LOCAL_API_OUTPUT"),
        help="directory for task videos (default: ./outputs/local-api)",
    )
    parser.add_argument("--steps", type=int, default=_env_int("H3_STEPS", 20))
    parser.add_argument("--layers", type=int, default=_env_int("H3_LAYERS", 50))
    parser.add_argument("--reuse", type=int, default=_env_int("H3_REUSE", 1))
    parser.add_argument(
        "--seed",
        type=int,
        default=None if os.environ.get("H3_SEED") in (None, "") else _env_int("H3_SEED", 0),
        help="optional default h3 seed; omit to use h3's default of 42",
    )
    return parser


def app_from_args(args: argparse.Namespace, work_cwd: Path | None = None) -> App:
    if not args.model_dir:
        raise SystemExit("--model-dir is required (or set H3_MODEL_DIR)")
    cwd = (work_cwd or Path.cwd()).resolve()
    h3_bin = discover_h3(args.h3)
    shader = discover_shader(args.shader, h3_bin, cwd)
    output = Path(args.output_dir) if args.output_dir else cwd / "outputs" / "local-api"
    if not output.is_absolute():
        output = cwd / output
    return App(
        h3_bin=h3_bin,
        model_dir=Path(args.model_dir),
        shader=shader,
        work_cwd=cwd,
        output_dir=output,
        steps=args.steps,
        layers=args.layers,
        reuse=args.reuse,
        seed=args.seed,
        host=args.host,
        port=args.port,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        app = app_from_args(args)
        httpd = app.serve()
    except SystemExit as exc:
        message = exc.code
        if isinstance(message, str):
            print(message, file=sys.stderr)
            return 2
        raise
    host, port = httpd.server_address[0], httpd.server_address[1]
    print(f"h3 local api listening on http://{host}:{port}", flush=True)
    print("POST /api/v1/services/aigc/video-generation/video-synthesis", flush=True)
    print("GET  /api/v1/tasks/<task_id>", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        with app.proc_lock:
            proc = app.current_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
        app.queue.put(None)
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
