"""DashScope-compatible local API tests. They use a fake h3 binary."""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_api.mapping import (  # noqa: E402
    APIError,
    adapt_canvas,
    build_h3_command,
    parse_synthesis_body,
)
from local_api.server import App, publish_shader, sniff_media_size  # noqa: E402

FAKE_H3 = """#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

args = sys.argv[1:]

def values(flag):
    found = []
    index = 0
    while index < len(args):
        if args[index] == flag:
            found.append(args[index + 1])
            index += 2
            continue
        index += 1
    return found

if not Path("h3_shaders.metal").is_file():
    sys.stderr.write("h3_shaders.metal not found in cwd\\n")
    sys.exit(3)

prompt = values("-p")[0]
output = Path(values("-o")[0])
output.parent.mkdir(parents=True, exist_ok=True)
output.with_suffix(".argv.json").write_text(json.dumps(args), encoding="utf-8")
if prompt == "FAIL":
    sys.stderr.write("boom\\n")
    sys.exit(7)
fake_dir = __import__("os").environ.get("H3_FAKE_DIR")
if prompt == "BLOCK" and fake_dir:
    Path(fake_dir, "started").write_text("1", encoding="utf-8")
    while not Path(fake_dir, "go").exists():
        time.sleep(0.02)
output.write_bytes(b"fake-mp4-bytes")
"""


def tiny_png(width: int, height: int, color: bytes = b"\x10\x20\x30") -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + color * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def flag_values(argv: list[str], flag: str) -> list[str]:
    found = []
    index = 0
    while index < len(argv):
        if argv[index] == flag:
            found.append(argv[index + 1])
            index += 2
            continue
        index += 1
    return found


class MappingTests(unittest.TestCase):
    def test_named_canvases_match_h3_768p_limit(self) -> None:
        expected = {
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "1:1": (768, 768),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
            "21:9": (1536, 672),
        }
        for ratio, canvas in expected.items():
            spec = parse_synthesis_body(self._body(ratio=ratio))
            self.assertEqual((spec.width, spec.height), canvas)
            self.assertLessEqual(canvas[0] * canvas[1], 768 * 1344)
            self.assertEqual(canvas[0] % 32, 0)
            self.assertEqual(canvas[1] % 32, 0)

    def test_adaptive_square_image_uses_768(self) -> None:
        self.assertEqual(adapt_canvas(9, 16), (768, 1344))
        self.assertEqual(adapt_canvas(400, 400), (768, 768))

    def test_duration_bounds_and_prompt_truncation(self) -> None:
        with self.assertRaises(APIError):
            parse_synthesis_body(self._body(duration=3))
        with self.assertRaises(APIError):
            parse_synthesis_body(self._body(duration=16))
        with self.assertRaises(APIError):
            parse_synthesis_body(self._body(duration=True))
        with self.assertRaises(APIError):
            parse_synthesis_body(self._body(duration=4.5))
        spec = parse_synthesis_body(self._body(duration=4.0))
        self.assertEqual(spec.duration, 4)
        long_prompt = "字" * 7001
        spec = parse_synthesis_body(self._body(prompt=long_prompt))
        self.assertEqual(len(spec.prompt), 7000)

    def test_rejects_mixed_roles_and_audio_only(self) -> None:
        mixed = self._body()
        mixed["input"]["media"] = [
            {"type": "first_frame", "url": "file:///tmp/a.png"},
            {"type": "reference_image", "url": "file:///tmp/b.png"},
        ]
        with self.assertRaises(APIError) as caught:
            parse_synthesis_body(mixed)
        self.assertIn("cannot mix", caught.exception.message)
        audio_only = self._body()
        audio_only["input"]["media"] = [
            {"type": "reference_audio", "url": "file:///tmp/a.wav"}
        ]
        with self.assertRaises(APIError):
            parse_synthesis_body(audio_only)

    def test_reference_limit_and_aliases(self) -> None:
        body = self._body()
        body["input"]["media"] = [
            {"type": "reference_image", "url": f"file:///tmp/{index}.png"}
            for index in range(10)
        ]
        with self.assertRaises(APIError):
            parse_synthesis_body(body)
        body["input"]["media"] = [
            {"type": "image_url", "url": "file:///tmp/a.png"},
            {"type": "feature", "url": "file:///tmp/a.mp4"},
            {"type": "driving_audio", "url": "file:///tmp/a.wav"},
        ]
        spec = parse_synthesis_body(body)
        self.assertEqual(
            [item.role for item in spec.media],
            ["reference_image", "reference_video", "reference_audio"],
        )

    def test_command_uses_seconds_and_ssd_streaming(self) -> None:
        command = build_h3_command(
            h3_bin=Path("/opt/h3"),
            model_dir=Path("/models/MiniMax-H3"),
            prompt="fox",
            width=1344,
            height=768,
            duration=14,
            output=Path("/tmp/out.mp4"),
            first_frame=Path("/tmp/first.png"),
            last_frame=Path("/tmp/last.png"),
            references=[],
            steps=20,
            layers=50,
            reuse=1,
            seed=None,
        )
        self.assertIn("--ssd-streaming", command)
        self.assertEqual(flag_values(command, "--seconds"), ["14"])
        self.assertNotIn("--frames", command)
        self.assertEqual(flag_values(command, "--first-frame"), ["/tmp/first.png"])
        self.assertEqual(flag_values(command, "--last-frame"), ["/tmp/last.png"])
        self.assertNotIn("--ref-image", command)
        self.assertNotIn("--render-width", command)

    def test_fast_preset_overrides_dashscope_canvas(self) -> None:
        spec = parse_synthesis_body(
            self._body(preset="fast", ratio="16:9", duration=14, watermark=False)
        )
        self.assertEqual(spec.preset, "fast")
        self.assertEqual(spec.duration, 14)
        self.assertEqual((spec.width, spec.height), (512, 512))
        self.assertEqual((spec.render_width, spec.render_height), (320, 320))
        self.assertEqual(spec.frames, 22)
        self.assertEqual(spec.steps, 4)
        self.assertEqual((spec.layers, spec.reuse), (50, 1))
        command = build_h3_command(
            h3_bin=Path("h3"),
            model_dir=Path("MiniMax-H3"),
            prompt=spec.prompt,
            width=spec.width or 0,
            height=spec.height or 0,
            duration=spec.duration,
            output=Path("out.mp4"),
            first_frame=None,
            last_frame=None,
            references=[],
            steps=spec.steps or 0,
            layers=spec.layers or 0,
            reuse=spec.reuse or 0,
            seed=None,
            frames=spec.frames,
            render_width=spec.render_width,
            render_height=spec.render_height,
        )
        self.assertEqual(flag_values(command, "--width"), ["512"])
        self.assertEqual(flag_values(command, "--height"), ["512"])
        self.assertEqual(flag_values(command, "--render-width"), ["320"])
        self.assertEqual(flag_values(command, "--render-height"), ["320"])
        self.assertEqual(flag_values(command, "--frames"), ["22"])
        self.assertEqual(flag_values(command, "--steps"), ["4"])
        self.assertNotIn("--seconds", command)
        self.assertIn("--ssd-streaming", command)

    def test_fast_keeps_explicit_steps_and_size(self) -> None:
        spec = parse_synthesis_body(
            self._body(preset="fast", steps=6, width=768, height=768)
        )
        self.assertEqual(spec.steps, 6)
        self.assertEqual((spec.width, spec.height), (768, 768))
        self.assertIsNone(spec.render_width)
        self.assertEqual(spec.frames, 22)
        plain = parse_synthesis_body(self._body())
        self.assertEqual(plain.preset, "default")
        self.assertEqual((plain.width, plain.height), (1344, 768))
        self.assertIsNone(plain.frames)
        self.assertIsNone(plain.steps)
        self.assertIsNone(plain.render_width)
        stepped = parse_synthesis_body(self._body(steps=6, width=512, height=512))
        self.assertEqual(stepped.preset, "default")
        self.assertEqual(stepped.steps, 6)
        self.assertEqual((stepped.width, stepped.height), (512, 512))
        self.assertIsNone(stepped.frames)

    def test_fast_still_validates_dashscope_fields(self) -> None:
        with self.assertRaises(APIError):
            parse_synthesis_body(self._body(preset="turbo"))
        with self.assertRaises(APIError):
            parse_synthesis_body(self._body(preset="fast", ratio="2:1"))
        with self.assertRaises(APIError):
            parse_synthesis_body(self._body(preset="fast", duration=3))
        with self.assertRaises(APIError):
            parse_synthesis_body(self._body(width=512))
        with self.assertRaises(APIError):
            parse_synthesis_body(
                self._body(width=1344, height=768, render_width=320, render_height=320)
            )

    def test_png_header_size(self) -> None:
        self.assertEqual(sniff_media_size(tiny_png(9, 16)), (9, 16))

    def test_shader_symlink_does_not_replace_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            shader = root / "real" / "h3_shaders.metal"
            shader.parent.mkdir()
            shader.write_text("shader", encoding="utf-8")
            work = root / "work"
            work.mkdir()
            child = publish_shader(shader, work)
            link = work / "h3_shaders.metal"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), shader.resolve())
            self.assertEqual(child, shader.parent)
            other = root / "other"
            other.mkdir()
            existing = other / "h3_shaders.metal"
            existing.write_text("keep", encoding="utf-8")
            publish_shader(shader, other)
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
            self.assertFalse(existing.is_symlink())

    def _body(self, **overrides) -> dict:
        prompt = overrides.pop("prompt", "A red fox walks through snow.")
        parameters = {
            "resolution": "768P",
            "ratio": "16:9",
            "duration": 4,
            "watermark": False,
        }
        parameters.update(overrides)
        return {
            "model": "MiniMax/MiniMax-H3",
            "input": {"prompt": prompt},
            "parameters": parameters,
        }


class ServerTests(unittest.TestCase):
    root: Path
    output: Path
    work: Path
    binary_dir: Path
    model_dir: Path
    fake_dir: Path
    port: int
    app: App
    thread: threading.Thread

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp(prefix="h3-local-api-"))
        cls.output = cls.root / "outputs"
        cls.work = cls.root / "work"
        cls.binary_dir = cls.root / "bin"
        cls.model_dir = cls.root / "MiniMax-H3"
        cls.fake_dir = cls.root / "fake"
        for path in (cls.work, cls.binary_dir, cls.model_dir, cls.fake_dir):
            path.mkdir(parents=True)
        shader = cls.binary_dir / "h3_shaders.metal"
        shader.write_text("shader", encoding="utf-8")
        binary = cls.binary_dir / "h3"
        binary.write_text(FAKE_H3, encoding="utf-8")
        binary.chmod(0o755)
        os.environ["H3_FAKE_DIR"] = str(cls.fake_dir)
        cls.app = App(
            h3_bin=binary,
            model_dir=cls.model_dir,
            shader=shader,
            work_cwd=cls.work,
            output_dir=cls.output,
            steps=20,
            layers=50,
            reuse=1,
            seed=None,
            host="127.0.0.1",
            port=0,
        )
        httpd = cls.app.serve()
        cls.port = int(httpd.server_address[1])
        cls.thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls._wait_until_ready()

    @classmethod
    def tearDownClass(cls) -> None:
        go = cls.fake_dir / "go"
        go.write_text("1", encoding="utf-8")
        cls.app.stop()
        cls.thread.join(timeout=5)
        shutil.rmtree(cls.root, ignore_errors=True)

    @classmethod
    def _wait_until_ready(cls) -> None:
        deadline = time.time() + 3
        url = f"http://127.0.0.1:{cls.port}/api/v1/tasks/not-a-task"
        while time.time() < deadline:
            try:
                urllib.request.urlopen(url, timeout=0.2)
            except urllib.error.HTTPError:
                return
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("local api did not start")

    def setUp(self) -> None:
        started = self.fake_dir / "started"
        go = self.fake_dir / "go"
        started.unlink(missing_ok=True)
        go.unlink(missing_ok=True)

    def tearDown(self) -> None:
        (self.fake_dir / "go").write_text("1", encoding="utf-8")
        deadline = time.time() + 5
        while time.time() < deadline:
            with self.app.tasks_lock:
                busy = any(
                    task.status in ("PENDING", "RUNNING")
                    for task in self.app.tasks.values()
                )
            if not busy:
                return
            time.sleep(0.02)

    def test_fast_preset_overrides_server_quality(self) -> None:
        status, payload = self._post(
            self._body(preset="fast", ratio="16:9", duration=14, watermark=False)
        )
        self.assertEqual(status, 200)
        task_id = payload["output"]["task_id"]
        done = self._wait(task_id, "SUCCEEDED")
        self.assertEqual(done["usage"]["size"], "512*512")
        self.assertEqual(done["usage"]["duration"], 14)
        argv = self._argv(task_id)
        self.assertEqual(flag_values(argv, "--width"), ["512"])
        self.assertEqual(flag_values(argv, "--height"), ["512"])
        self.assertEqual(flag_values(argv, "--render-width"), ["320"])
        self.assertEqual(flag_values(argv, "--render-height"), ["320"])
        self.assertEqual(flag_values(argv, "--frames"), ["22"])
        self.assertNotIn("--seconds", argv)
        self.assertEqual(flag_values(argv, "--steps"), ["4"])
        self.assertEqual(flag_values(argv, "--layers"), ["50"])
        self.assertEqual(flag_values(argv, "--reuse"), ["1"])
        self.assertIn("--ssd-streaming", argv)

    def test_explicit_steps_keep_dashscope_canvas(self) -> None:
        status, payload = self._post(self._body(steps=6))
        self.assertEqual(status, 200)
        task_id = payload["output"]["task_id"]
        done = self._wait(task_id, "SUCCEEDED")
        self.assertEqual(done["usage"]["size"], "1344*768")
        argv = self._argv(task_id)
        self.assertEqual(flag_values(argv, "--steps"), ["6"])
        self.assertEqual(flag_values(argv, "--seconds"), ["4"])
        self.assertEqual(flag_values(argv, "--width"), ["1344"])
        self.assertNotIn("--frames", argv)
        self.assertNotIn("--render-width", argv)

    def test_shader_is_linked_into_server_cwd(self) -> None:
        link = self.work / "h3_shaders.metal"
        self.assertTrue(link.is_symlink())
        self.assertTrue(link.resolve().is_file())

    def test_submit_poll_download_without_auth(self) -> None:
        status, payload = self._post(self._body(prompt="这一镜的投喂正文"))
        self.assertEqual(status, 200)
        self.assertEqual(payload["output"]["task_status"], "PENDING")
        self.assertNotIn("video_url", payload["output"])
        task_id = payload["output"]["task_id"]
        done = self._wait(task_id, "SUCCEEDED")
        video_url = done["output"]["video_url"]
        self.assertTrue(video_url.startswith(f"http://127.0.0.1:{self.port}/api/v1/videos/"))
        self.assertTrue(video_url.endswith(".mp4"))
        self.assertEqual(done["usage"]["size"], "1344*768")
        self.assertEqual(done["usage"]["duration"], 4)
        self.assertEqual(done["output"]["orig_prompt"], "这一镜的投喂正文")
        argv = self._argv(task_id)
        self.assertIn("--ssd-streaming", argv)
        self.assertEqual(flag_values(argv, "--seconds"), ["4"])
        self.assertNotIn("--frames", argv)
        self.assertNotIn("--render-width", argv)
        self.assertEqual(flag_values(argv, "--width"), ["1344"])
        self.assertEqual(flag_values(argv, "--height"), ["768"])
        self.assertEqual(flag_values(argv, "--steps"), ["20"])
        self.assertEqual(flag_values(argv, "--layers"), ["50"])
        self.assertEqual(flag_values(argv, "--reuse"), ["1"])
        self.assertIn("这一镜的投喂正文", argv)
        self.assertEqual(flag_values(argv, "-d"), [str(self.model_dir.resolve())])
        code, headers, body = self._raw("GET", video_url)
        self.assertEqual(code, 200)
        self.assertTrue(headers.get("Content-Type", "").startswith("video/mp4"))
        self.assertEqual(body, b"fake-mp4-bytes")

    def test_bearer_and_async_header_are_optional(self) -> None:
        status, payload = self._post(
            self._body(),
            headers={
                "Authorization": "Bearer any-local-token",
                "X-DashScope-Async": "enable",
            },
        )
        self.assertEqual(status, 200)
        self._wait(payload["output"]["task_id"], "SUCCEEDED")
        status, payload = self._post(
            self._body(),
            headers={"X-DashScope-Async": "disable"},
        )
        self.assertEqual(status, 200)
        self._wait(payload["output"]["task_id"], "SUCCEEDED")

    def test_generate_alias_and_unknown_task(self) -> None:
        status, payload = self._post(self._body(), path="/generate")
        self.assertEqual(status, 200)
        self.assertEqual(payload["output"]["task_status"], "PENDING")
        self._wait(payload["output"]["task_id"], "SUCCEEDED")
        code, error = self._get("/api/v1/tasks/does-not-exist")
        self.assertEqual(code, 404)
        self.assertEqual(error["message"], "task not found")

    def test_first_and_last_frame_files(self) -> None:
        first = self.work / "open.png"
        last = self.work / "end.png"
        first.write_bytes(tiny_png(16, 9))
        last.write_bytes(tiny_png(16, 9, b"\x01\x02\x03"))
        body = self._body(ratio="21:9", duration=14)
        body["input"]["media"] = [
            {"type": "first_frame", "url": first.as_uri()},
            {"type": "last_frame", "url": str(last)},
        ]
        status, payload = self._post(body)
        self.assertEqual(status, 200)
        task_id = payload["output"]["task_id"]
        done = self._wait(task_id, "SUCCEEDED")
        self.assertEqual(done["usage"]["size"], "1536*672")
        self.assertEqual(done["usage"]["image_count"], 2)
        argv = self._argv(task_id)
        self.assertEqual(len(flag_values(argv, "--first-frame")), 1)
        self.assertEqual(len(flag_values(argv, "--last-frame")), 1)
        self.assertNotIn("--ref-image", argv)
        copied = Path(flag_values(argv, "--first-frame")[0])
        self.assertEqual(copied.read_bytes(), first.read_bytes())
        self.assertNotEqual(copied, first)

    def test_adaptive_portrait_and_http_references(self) -> None:
        portrait = tiny_png(9, 16)
        png_server = _PngServer(portrait)
        try:
            body = self._body(ratio="adaptive", duration=6)
            body["input"]["media"] = [
                {"type": "reference_image", "url": first_url(png_server, "/a.png")},
                {"type": "reference_image", "url": self._file_url(portrait, "b.png")},
                {"type": "reference_video", "url": self._file_url(b"video", "clip.mp4")},
                {"type": "reference_audio", "url": self._file_url(b"audio", "tone.wav")},
            ]
            status, payload = self._post(body)
            self.assertEqual(status, 200)
            task_id = payload["output"]["task_id"]
            done = self._wait(task_id, "SUCCEEDED")
            self.assertEqual(done["usage"]["size"], "768*1344")
            argv = self._argv(task_id)
            images = flag_values(argv, "--ref-image")
            videos = flag_values(argv, "--ref-video")
            audios = flag_values(argv, "--ref-audio")
            self.assertEqual(len(images), 2)
            self.assertEqual(Path(images[0]).read_bytes(), portrait)
            self.assertEqual(len(videos), 1)
            self.assertEqual(len(audios), 1)
            image_at = argv.index("--ref-image")
            video_at = argv.index("--ref-video")
            audio_at = argv.index("--ref-audio")
            self.assertLess(image_at, video_at)
            self.assertLess(video_at, audio_at)
            self.assertNotIn("--first-frame", argv)
        finally:
            png_server.stop()

    def test_nine_reference_images(self) -> None:
        url = self._file_url(tiny_png(8, 8), "ref.png")
        body = self._body(ratio="")
        body["input"]["media"] = [
            {"type": "reference_image", "url": url} for _ in range(9)
        ]
        status, payload = self._post(body)
        self.assertEqual(status, 200)
        task_id = payload["output"]["task_id"]
        done = self._wait(task_id, "SUCCEEDED")
        self.assertEqual(done["usage"]["size"], "768*768")
        self.assertEqual(len(flag_values(self._argv(task_id), "--ref-image")), 9)

    def test_validation_errors_do_not_queue(self) -> None:
        before = set(self.output.iterdir())
        mixed = self._body()
        mixed["input"]["media"] = [
            {"type": "first_frame", "url": "file:///tmp/a.png"},
            {"type": "reference_image", "url": "file:///tmp/b.png"},
        ]
        status, payload = self._post(mixed)
        self.assertEqual(status, 400)
        self.assertIn("cannot mix", payload["message"])
        too_many = self._body()
        too_many["input"]["media"] = [
            {"type": "reference_image", "url": f"file:///tmp/{index}.png"}
            for index in range(10)
        ]
        status, _payload = self._post(too_many)
        self.assertEqual(status, 400)
        status, payload = self._post(self._body(duration=3))
        self.assertEqual(status, 400)
        self.assertIn("duration", payload["message"])
        status, payload = self._post(self._body(resolution="2K"))
        self.assertEqual(status, 400)
        self.assertIn("768P", payload["message"])
        bad_model = self._body()
        bad_model["model"] = "wan2.7"
        status, payload = self._post(bad_model)
        self.assertEqual(status, 400)
        self.assertIn("MiniMax/MiniMax-H3", payload["message"])
        self.assertEqual(set(self.output.iterdir()), before)

    def test_h3_failure_maps_to_failed(self) -> None:
        status, payload = self._post(self._body(prompt="FAIL"))
        self.assertEqual(status, 200)
        done = self._wait(payload["output"]["task_id"], "FAILED")
        self.assertEqual(done["output"]["code"], "InternalError")
        self.assertIn("boom", done["output"]["message"])
        code, _error = self._get(f"/api/v1/videos/{payload['output']['task_id']}.mp4")
        self.assertEqual(code, 404)

    def test_one_generation_at_a_time(self) -> None:
        status, first = self._post(self._body(prompt="BLOCK"))
        self.assertEqual(status, 200)
        first_id = first["output"]["task_id"]
        status, second = self._post(self._body(prompt="second clip"))
        second_id = second["output"]["task_id"]
        self._wait_file(self.fake_dir / "started")
        running = self._get_json(f"/api/v1/tasks/{first_id}")
        self.assertEqual(running["output"]["task_status"], "RUNNING")
        self.assertNotIn("video_url", running["output"])
        queued = self._get_json(f"/api/v1/tasks/{second_id}")
        self.assertEqual(queued["output"]["task_status"], "PENDING")
        self.assertFalse((self.output / second_id / "video.argv.json").exists())
        code, _error = self._get(f"/api/v1/videos/{first_id}.mp4")
        self.assertEqual(code, 404)
        (self.fake_dir / "go").write_text("1", encoding="utf-8")
        self._wait(first_id, "SUCCEEDED")
        self._wait(second_id, "SUCCEEDED")

    def test_missing_local_media_fails_the_task(self) -> None:
        body = self._body()
        body["input"]["media"] = [
            {"type": "reference_image", "url": "file:///tmp/h3-missing-reference.png"}
        ]
        status, payload = self._post(body)
        self.assertEqual(status, 200)
        done = self._wait(payload["output"]["task_id"], "FAILED")
        self.assertIn("not found", done["output"]["message"])

    def _body(self, **overrides) -> dict:
        prompt = overrides.pop("prompt", "A red fox walks through snow.")
        parameters = {
            "resolution": "768P",
            "ratio": "16:9",
            "duration": 4,
            "watermark": False,
        }
        parameters.update(overrides)
        return {
            "model": "MiniMax/MiniMax-H3",
            "input": {"prompt": prompt},
            "parameters": parameters,
        }

    def _file_url(self, data: bytes, name: str) -> str:
        path = self.work / name
        path.write_bytes(data)
        return path.as_uri()

    def _post(self, body: dict, headers: dict | None = None, path: str | None = None):
        code, _headers, raw = self._raw(
            "POST",
            path or "/api/v1/services/aigc/video-generation/video-synthesis",
            body,
            headers,
        )
        return code, json.loads(raw.decode("utf-8"))

    def _get(self, path: str):
        code, _headers, raw = self._raw("GET", path)
        return code, json.loads(raw.decode("utf-8"))

    def _get_json(self, path: str) -> dict:
        code, payload = self._get(path)
        self.assertEqual(code, 200, payload)
        return payload

    def _raw(self, method: str, path: str, body: dict | None = None, headers: dict | None = None):
        if path.startswith("http://"):
            url = path
        else:
            url = f"http://127.0.0.1:{self.port}{path}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read()

    def _wait(self, task_id: str, status: str) -> dict:
        deadline = time.time() + 5
        last: dict | None = None
        while time.time() < deadline:
            last = self._get_json(f"/api/v1/tasks/{task_id}")
            current = last["output"]["task_status"]
            if current == status:
                return last
            if current in ("SUCCEEDED", "FAILED"):
                self.fail(f"task ended {current}: {last}")
            time.sleep(0.02)
        self.fail(f"timed out waiting for {status}: {last}")

    def _wait_file(self, path: Path) -> None:
        deadline = time.time() + 5
        while time.time() < deadline:
            if path.exists():
                return
            time.sleep(0.02)
        self.fail(f"timed out waiting for {path}")

    def _argv(self, task_id: str) -> list[str]:
        return json.loads(
            (self.output / task_id / "video.argv.json").read_text(encoding="utf-8")
        )


class _PngServer:
    def __init__(self, payload: bytes) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                body = outer.payload
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                return

        self.payload = payload
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


def first_url(server: _PngServer, path: str) -> str:
    return f"http://127.0.0.1:{server.port}{path}"


if __name__ == "__main__":
    unittest.main()
