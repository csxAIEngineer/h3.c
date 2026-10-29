"""Map MiniMax-H3 DashScope requests onto h3 CLI arguments.

Canvas sizing follows ``h3_adapt_canvas`` in ``h3_host.c``: the short side is
768 pixels, the result snaps to a multiple of 32, and the pixel count stays
within ``768 * 1344``. Duration is an integer second count in ``[4, 15]`` and
is passed through as ``--seconds`` so h3 can apply its 24 fps alignment.
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
    width: int | None
    height: int | None
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
    width: int | None
    height: int | None
    if ratio == "adaptive":
        width, height = None, None
    else:
        width, height = canvas_for_ratio(ratio)

    seed = _optional_seed(parameters.get("seed")) if "seed" in parameters else None
    return JobSpec(
        prompt=prompt,
        duration=duration,
        ratio=ratio,
        width=width,
        height=height,
        media=media,
        seed=seed,
    )


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
) -> list[str]:
    """Build the argv for one serialized h3 generation.

    ``--ssd-streaming`` is unconditional. Duration uses ``--seconds`` (24 fps),
    which h3 rounds up onto its legal frame lattice. ``--frames`` is not passed.
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
        "--seconds",
        str(duration),
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
