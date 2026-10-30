"""Map MiniMax-H3 DashScope requests onto h3 CLI arguments.

Canvas sizing follows ``h3_adapt_canvas`` in ``h3_host.c``: the short side is
768 pixels, the result snaps to a multiple of 32, and the pixel count stays
within ``768 * 1344``. Duration is an integer second count in ``[4, 15]`` and
is passed through as ``--seconds`` so h3 can apply its 24 fps alignment.

``parameters.preset`` ``fast`` is a local extension. It keeps the DashScope
body valid, then replaces an omitted canvas and the quality knobs with the
fast h3 path: 512 square output, 320 square internal canvas, and 4 steps.
An explicit ``width`` / ``height`` pair, including 256 square, is kept, and
the 320 render canvas is added only for that 512 square default. Duration
still comes from ``parameters.duration`` as ``--seconds`` unless
``parameters.frames`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32
MIN_DURATION = 4
MAX_DURATION = 15
MAX_PROMPT_CHARS = 7000
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIO = 3
MAX_REFERENCES = 12

MODEL_NAME = "MiniMax/MiniMax-H3"

NAMED_RATIOS = ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9")

# Lowest-quality h3 path used by AiVideo for local MiniMax-H3.
FAST_WIDTH = 512
FAST_HEIGHT = 512
FAST_RENDER_WIDTH = 320
FAST_RENDER_HEIGHT = 320
FAST_STEPS = 4
FAST_LAYERS = 50
FAST_REUSE = 1

# Official DashScope names plus the AiVideo reference_* names.
ROLE_ALIASES = {
    "first_frame": "first_frame",
    "last_frame": "last_frame",
    "reference_image": "reference_image",
    "image_url": "reference_image",
    "reference_video": "reference_video",
    "feature": "reference_video",
    "reference_audio": "reference_audio",
    "driving_audio": "reference_audio",
}

FRAME_ROLES = {"first_frame", "last_frame"}
REFERENCE_ROLES = {"reference_image", "reference_video", "reference_audio"}


class APIError(Exception):
    """A request that should be rejected before a task is queued."""

    def __init__(
        self,
        message: str,
        code: str = "InvalidParameter",
        status: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


@dataclass(frozen=True)
class MediaItem:
    role: str
    url: str


@dataclass(frozen=True)
class JobSpec:
    prompt: str
    duration: int
    ratio: str
    preset: str
    width: int | None
    height: int | None
    render_width: int | None
    render_height: int | None
    frames: int | None
    steps: int | None
    layers: int | None
    reuse: int | None
    media: tuple[MediaItem, ...]
    seed: int | None


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    """Return a legal 768p canvas for an aspect ratio expressed as width/height."""
    if width < 1 or height < 1:
        raise APIError("media dimensions must be positive")
    ratio = width / height
    if ratio >= 1.0:
        nominal_w = 768.0 * ratio
        nominal_h = 768.0
    else:
        nominal_w = 768.0
        nominal_h = 768.0 / ratio
    pixels = nominal_w * nominal_h
    if pixels > MAX_PIXELS:
        scale = (MAX_PIXELS / pixels) ** 0.5
        nominal_w *= scale
        nominal_h *= scale
    out_w = int(round(nominal_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    out_h = int(round(nominal_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    out_w = max(out_w, CANVAS_MULTIPLE)
    out_h = max(out_h, CANVAS_MULTIPLE)
    while out_w * out_h > MAX_PIXELS and (out_w > CANVAS_MULTIPLE or out_h > CANVAS_MULTIPLE):
        if out_w >= out_h and out_w > CANVAS_MULTIPLE:
            out_w -= CANVAS_MULTIPLE
        elif out_h > CANVAS_MULTIPLE:
            out_h -= CANVAS_MULTIPLE
        else:
            break
    return out_w, out_h


def align_frame_count(requested: int) -> int:
    """Match ``h3_align_frame_count``: round up to ``5 + 17*n``."""
    value = requested if requested >= 5 else 5
    remainder = (value - 5) % 17
    if remainder:
        value += 17 - remainder
    return value


def validate_canvas(
    width: int,
    height: int,
    render_width: int | None,
    render_height: int | None,
) -> None:
    if (
        width < CANVAS_MULTIPLE
        or height < CANVAS_MULTIPLE
        or width % CANVAS_MULTIPLE
        or height % CANVAS_MULTIPLE
    ):
        raise APIError("width and height must be multiples of 32 and at least 32")
    if width * height > MAX_PIXELS:
        raise APIError("canvas exceeds the released 768*1344 pixel limit")
    if (render_width is None) != (render_height is None):
        raise APIError("render_width and render_height must be set together")
    if render_width is None or render_height is None:
        return
    if (
        render_width < CANVAS_MULTIPLE
        or render_height < CANVAS_MULTIPLE
        or render_width % CANVAS_MULTIPLE
        or render_height % CANVAS_MULTIPLE
        or render_width > width
        or render_height > height
        or render_width * height != render_height * width
    ):
        raise APIError(
            "render width and height must be same-aspect multiples of 32 "
            "no larger than the output canvas"
        )


def canvas_for_ratio(ratio: str) -> tuple[int, int]:
    if ratio not in NAMED_RATIOS:
        raise APIError(
            "ratio must be adaptive, " + ", ".join(NAMED_RATIOS)
        )
    left, right = ratio.split(":")
    return adapt_canvas(int(left), int(right))


def _as_object(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise APIError(f"{name} must be an object")
    return value


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise APIError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("+", "-")):
            digits = text[1:]
            sign = text[0]
        else:
            digits = text
            sign = ""
        if digits.isdigit():
            return int(sign + digits)
    raise APIError(f"{name} must be an integer")


def _optional_seed(value: object) -> int | None:
    if value is None:
        return None
    seed = _as_int(value, "parameters.seed")
    if seed < 0:
        raise APIError("parameters.seed must be >= 0")
    return seed


def parse_synthesis_body(body: object) -> JobSpec:
    """Validate a video-synthesis JSON body and return a generation spec."""
    data = _as_object(body, "body")
    model = data.get("model", MODEL_NAME)
    if not isinstance(model, str) or model.strip() not in (MODEL_NAME, "MiniMax-H3"):
        raise APIError(f"model must be {MODEL_NAME}")

    prompt_input = _as_object(data.get("input"), "input")
    prompt = prompt_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise APIError("input.prompt is required")
    if "\x00" in prompt:
        raise APIError("input.prompt must not contain NUL")
    prompt = prompt[:MAX_PROMPT_CHARS]

    raw_media = prompt_input.get("media", [])
    if raw_media is None:
        raw_media = []
    if not isinstance(raw_media, list):
        raise APIError("input.media must be an array")
    media = tuple(_parse_media_item(item, index) for index, item in enumerate(raw_media))
    _validate_media_roles(media)

    parameters = data.get("parameters", {})
    if parameters is None:
        parameters = {}
    parameters = _as_object(parameters, "parameters")

    if "duration" not in parameters:
        raise APIError("parameters.duration is required")
    duration = _as_int(parameters.get("duration"), "parameters.duration")
    if duration < MIN_DURATION or duration > MAX_DURATION:
        raise APIError(
            f"parameters.duration must be an integer from {MIN_DURATION} to {MAX_DURATION}"
        )

    resolution = parameters.get("resolution", "768P")
    if not isinstance(resolution, str) or resolution.strip().upper() != "768P":
        raise APIError("parameters.resolution must be 768P")

    ratio_value = parameters.get("ratio", "adaptive")
    if ratio_value is None:
        ratio_value = "adaptive"
    if not isinstance(ratio_value, str):
        raise APIError("parameters.ratio must be a string")
    ratio = ratio_value.strip().lower()
    if ratio == "":
        ratio = "adaptive"
    named_canvas = None if ratio == "adaptive" else canvas_for_ratio(ratio)

    preset = _preset(parameters.get("preset", "default"))
    steps = _optional_bounded_int(parameters, "steps", 2, 1000)
    frames = _optional_bounded_int(parameters, "frames", 5, 362)
    if frames is not None:
        aligned = align_frame_count(frames)
        if aligned < 22 or aligned > 362:
            raise APIError("parameters.frames must align within the released 22..362 range")
    width, height = _dimension_pair(parameters, "width", "height")
    render_width, render_height = _dimension_pair(
        parameters, "render_width", "render_height"
    )
    layers: int | None = None
    reuse: int | None = None
    if preset == "fast":
        if width is None:
            width, height = FAST_WIDTH, FAST_HEIGHT
        if render_width is None and (width, height) == (FAST_WIDTH, FAST_HEIGHT):
            render_width, render_height = FAST_RENDER_WIDTH, FAST_RENDER_HEIGHT
        # Leave frames unset unless parameters.frames was provided, so
        # build_h3_command passes duration as --seconds.
        if steps is None:
            steps = FAST_STEPS
        layers = FAST_LAYERS
        reuse = FAST_REUSE
    elif width is None and named_canvas is not None:
        width, height = named_canvas
    if render_width is not None and width is None:
        raise APIError("parameters.render_width requires parameters.width and parameters.height")
    if width is not None and height is not None:
        validate_canvas(width, height, render_width, render_height)

    seed = _optional_seed(parameters.get("seed")) if "seed" in parameters else None
    return JobSpec(
        prompt=prompt,
        duration=duration,
        ratio=ratio,
        preset=preset,
        width=width,
        height=height,
        render_width=render_width,
        render_height=render_height,
        frames=frames,
        steps=steps,
        layers=layers,
        reuse=reuse,
        media=media,
        seed=seed,
    )


def _preset(value: object) -> str:
    if value is None:
        return "default"
    if not isinstance(value, str):
        raise APIError("parameters.preset must be fast or default")
    preset = value.strip().lower()
    if preset == "":
        return "default"
    if preset not in ("fast", "default"):
        raise APIError("parameters.preset must be fast or default")
    return preset


def _optional_bounded_int(
    parameters: dict, name: str, low: int, high: int
) -> int | None:
    if name not in parameters or parameters[name] is None:
        return None
    value = _as_int(parameters[name], f"parameters.{name}")
    if value < low or value > high:
        raise APIError(f"parameters.{name} must be in [{low}, {high}]")
    return value


def _dimension_pair(parameters: dict, width_name: str, height_name: str) -> tuple[int | None, int | None]:
    present = [name for name in (width_name, height_name) if name in parameters and parameters[name] is not None]
    if not present:
        return None, None
    if len(present) != 2:
        raise APIError(f"parameters.{width_name} and parameters.{height_name} must be set together")
    width = _as_int(parameters[width_name], f"parameters.{width_name}")
    height = _as_int(parameters[height_name], f"parameters.{height_name}")
    return width, height


def _parse_media_item(item: object, index: int) -> MediaItem:
    entry = _as_object(item, f"input.media[{index}]")
    raw_type = entry.get("type")
    url = entry.get("url")
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise APIError(f"input.media[{index}].type is required")
    role = ROLE_ALIASES.get(raw_type.strip())
    if role is None:
        known = "first_frame, last_frame, reference_image, reference_video, reference_audio"
        raise APIError(
            f"input.media[{index}].type must be one of {known} "
            "(aliases: image_url, feature, driving_audio)"
        )
    if not isinstance(url, str) or not url.strip():
        raise APIError(f"input.media[{index}].url is required")
    if any(char in url for char in "\r\n\x00"):
        raise APIError(f"input.media[{index}].url is invalid")
    return MediaItem(role=role, url=url.strip())


def _validate_media_roles(media: tuple[MediaItem, ...]) -> None:
    counts = {role: 0 for role in (*FRAME_ROLES, *REFERENCE_ROLES)}
    for item in media:
        counts[item.role] += 1
    frames = counts["first_frame"] + counts["last_frame"]
    references = (
        counts["reference_image"] + counts["reference_video"] + counts["reference_audio"]
    )
    if frames and references:
        raise APIError(
            "cannot mix first_frame/last_frame with reference_image, "
            "reference_video, or reference_audio in one request"
        )
    if counts["first_frame"] > 1:
        raise APIError("at most one first_frame is supported")
    if counts["last_frame"] > 1:
        raise APIError("at most one last_frame is supported")
    if counts["reference_image"] > MAX_REFERENCE_IMAGES:
        raise APIError(
            f"at most {MAX_REFERENCE_IMAGES} reference_image inputs are supported"
        )
    if counts["reference_video"] > MAX_REFERENCE_VIDEOS:
        raise APIError(
            f"at most {MAX_REFERENCE_VIDEOS} reference_video inputs are supported"
        )
    if counts["reference_audio"] > MAX_REFERENCE_AUDIO:
        raise APIError(
            f"at most {MAX_REFERENCE_AUDIO} reference_audio inputs are supported"
        )
    if references > MAX_REFERENCES:
        raise APIError(f"at most {MAX_REFERENCES} reference inputs are supported")
    if counts["reference_audio"] and not (
        counts["reference_image"] or counts["reference_video"]
    ):
        raise APIError("reference_audio requires a reference_image or reference_video")


def build_h3_command(
    *,
    h3_bin: Path,
    model_dir: Path,
    prompt: str,
    width: int,
    height: int,
    duration: int,
    output: Path,
    first_frame: Path | None,
    last_frame: Path | None,
    references: list[tuple[str, Path]],
    steps: int,
    layers: int,
    reuse: int,
    seed: int | None,
    frames: int | None = None,
    render_width: int | None = None,
    render_height: int | None = None,
) -> list[str]:
    """Build the argv for one serialized h3 generation.

    ``--ssd-streaming`` is unconditional. A frame count uses ``--frames``.
    Otherwise duration uses ``--seconds`` (24 fps), which h3 rounds up onto
    its legal frame lattice. The two flags are mutually exclusive.
    """
    if first_frame or last_frame:
        if references:
            raise APIError(
                "cannot mix first_frame/last_frame with reference media"
            )
    command = [
        str(h3_bin),
        "-d",
        str(model_dir),
        "-p",
        prompt,
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    if render_width is not None and render_height is not None:
        command.extend(
            [
                "--render-width",
                str(render_width),
                "--render-height",
                str(render_height),
            ]
        )
    if frames is not None:
        command.extend(["--frames", str(frames)])
    else:
        command.extend(["--seconds", str(duration)])
    command.extend(
        [
            "--steps",
            str(steps),
            "--layers",
            str(layers),
            "--reuse",
            str(reuse),
            "--ssd-streaming",
            "-o",
            str(output),
        ]
    )
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if first_frame is not None:
        command.extend(["--first-frame", str(first_frame)])
    if last_frame is not None:
        command.extend(["--last-frame", str(last_frame)])
    flag = {
        "reference_image": "--ref-image",
        "reference_video": "--ref-video",
        "reference_audio": "--ref-audio",
    }
    for role, path in references:
        try:
            command.extend([flag[role], str(path)])
        except KeyError as exc:
            raise APIError(f"unsupported reference role {role}") from exc
    return command
