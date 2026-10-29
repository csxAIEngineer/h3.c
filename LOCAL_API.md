# Local MiniMax-H3 API

`scripts/h3-local-api` is a loopback HTTP server that speaks the DashScope-style
async API AiVideo already uses for `MiniMax/MiniMax-H3`. Point the video client
at this process and leave the request body unchanged. No Qianwen or MiniMax
cloud key is required.

The server checks the request, then runs the existing `h3` binary. Duration,
aspect ratio, and media roles map onto
`--seconds`, `--width` / `--height`, `--first-frame`, `--last-frame`, and
repeatable `--ref-image` / `--ref-video` / `--ref-audio`. Every generation
passes `--ssd-streaming` (the M5 16GB path).

## Start

From the repository root, after `make` has produced `./h3` and with the
checkpoint in `./MiniMax-H3`:

```sh
python3 scripts/h3-local-api --model-dir ./MiniMax-H3
```

The default listen address is `http://127.0.0.1:8765`. Equivalent form:

```sh
python3 -m local_api --model-dir ./MiniMax-H3 --host 127.0.0.1 --port 8765
```

`h3` opens `h3_shaders.metal` from its working directory. The server starts
`h3` in the directory that contains that file (next to the binary, or the repo
root). If the server's own cwd does not already contain `h3_shaders.metal`, it
adds a symlink there at startup.

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--model-dir` | `H3_MODEL_DIR` | Local MiniMax-H3 checkpoint. Required. |
| `--h3` | `./h3`, else `H3_BIN` | h3 binary. |
| `--shader` | beside `h3` or the repo | `h3_shaders.metal`. |
| `--host` / `--port` | `127.0.0.1:8765` | Bind address. |
| `--output-dir` | `./outputs/local-api` | Task working files and mp4s. |
| `--steps` | `20` | Denoising passes passed to h3. |
| `--layers` | `50` | DiT blocks (35–50). |
| `--reuse` | `1` | Whole-denoiser reuse (1–3). |
| `--seed` | h3 default `42` | Used when the request omits `parameters.seed`. |

`Authorization` is optional on this server. A missing header or any
`Bearer` value is accepted and ignored. `X-DashScope-Async: enable` is accepted
and not required. `parameters.watermark` is accepted and ignored; local files
are not watermarked. `parameters.resolution` must be `768P` (`2K` is rejected).
`parameters.seed` is an optional local extension.

One generation runs at a time. Further submits stay `PENDING` until the current
`h3` process exits. Outputs are kept under the output directory until you
delete them.

AiVideo: set the video client's base URL to `http://127.0.0.1:8765` (no path
suffix, no API key). The client already posts to
`/api/v1/services/aigc/video-generation/video-synthesis` and polls
`/api/v1/tasks/{task_id}`.

## Submit

`POST /api/v1/services/aigc/video-generation/video-synthesis`

`POST /generate` is the same async submit with a shorter path.

```sh
curl -sS -X POST http://127.0.0.1:8765/api/v1/services/aigc/video-generation/video-synthesis \
  -H 'Content-Type: application/json' \
  -H 'X-DashScope-Async: enable' \
  -d '{
    "model": "MiniMax/MiniMax-H3",
    "input": {
      "prompt": "A red fox walks through fresh snow in a pine forest."
    },
    "parameters": {
      "resolution": "768P",
      "ratio": "16:9",
      "duration": 4,
      "watermark": false
    }
  }'
```

```json
{
  "request_id": "…",
  "output": {
    "task_id": "…",
    "task_status": "PENDING"
  }
}
```

`duration` is an integer from 4 to 15. It is passed as `h3 --seconds N` at 24
fps. h3 then rounds up to a legal frame count (`5 + 17*n`), so the file can be
slightly longer than the requested second count (4s → 107 frames, about 4.46s;
15s → 362 frames).

`ratio` is `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9`, or `adaptive` (the
default). Empty `ratio` is treated as `adaptive`.

| Ratio | Width | Height |
|---|---:|---:|
| `16:9` | 1344 | 768 |
| `9:16` | 768 | 1344 |
| `1:1` | 768 | 768 |
| `4:3` | 1024 | 768 |
| `3:4` | 768 | 1024 |
| `21:9` | 1536 | 672 |

`adaptive` with no media uses `16:9`. With media, the canvas follows the first
frame, otherwise the last frame, otherwise the first reference image or video,
using the same 768p sizing rule as `h3_adapt_canvas`.

### Media

`input.media` is optional. Each item is `{"type", "url"}`. URLs may be
`http://`, `https://`, `file://`, or a local filesystem path. Remote objects
are downloaded and local files are copied into the task directory, then passed
to h3.

| `type` | h3 flag | Notes |
|---|---|---|
| `first_frame` | `--first-frame` | At most one. |
| `last_frame` | `--last-frame` | At most one. May be used alone or with `first_frame`. |
| `reference_image` | `--ref-image` | Repeatable, at most 9. Alias: `image_url`. |
| `reference_video` | `--ref-video` | Repeatable, at most 3. Alias: `feature`. |
| `reference_audio` | `--ref-audio` | Repeatable, at most 3. Alias: `driving_audio`. Requires an image or video reference. |

Frame roles and reference roles cannot be combined in one request. Reference
order follows the `media` array. At most 12 reference inputs in total.

```sh
curl -sS -X POST http://127.0.0.1:8765/api/v1/services/aigc/video-generation/video-synthesis \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMax/MiniMax-H3",
    "input": {
      "prompt": "The camera moves slowly forward from this opening frame.",
      "media": [
        {"type": "first_frame", "url": "file:///absolute/path/opening.png"}
      ]
    },
    "parameters": {"resolution": "768P", "ratio": "16:9", "duration": 6, "watermark": false}
  }'
```

A mixed request is rejected with HTTP 400 and no task is created:

```json
{
  "code": "InvalidParameter",
  "message": "cannot mix first_frame/last_frame with reference_image, reference_video, or reference_audio in one request",
  "request_id": "…"
}
```

## Poll

`GET /api/v1/tasks/{task_id}`

```sh
curl -sS http://127.0.0.1:8765/api/v1/tasks/TASK_ID
```

`task_status` moves `PENDING` → `RUNNING` → `SUCCEEDED` or `FAILED`. Success
includes an absolute `output.video_url` on this same server:

```json
{
  "request_id": "…",
  "output": {
    "task_id": "TASK_ID",
    "task_status": "SUCCEEDED",
    "submit_time": "2026-10-29 12:00:00.000",
    "scheduled_time": "2026-10-29 12:00:00.100",
    "end_time": "2026-10-29 12:01:00.000",
    "orig_prompt": "A red fox walks through fresh snow in a pine forest.",
    "video_url": "http://127.0.0.1:8765/api/v1/videos/TASK_ID.mp4"
  },
  "usage": {
    "SR": "768",
    "duration": 4,
    "image_count": 0,
    "input_seconds": 0,
    "output_seconds": 4,
    "size": "1344*768",
    "total_seconds": 4,
    "video_count": 1
  }
}
```

Failure stays HTTP 200 so a poller can read `output.code` and `output.message`.
An unknown task id is HTTP 404.

## Download

```sh
curl -L -o clip.mp4 "http://127.0.0.1:8765/api/v1/videos/TASK_ID.mp4"
```

That URL is a plain `GET` and returns `video/mp4`. It is the URL from
`output.video_url`; a client that only changes `base_url` can download it
with that same GET.

## Tests

```sh
python3 -m unittest discover -s tests -p 'test_local_api.py'
```

The tests use a fake `h3` so they run without a GPU or checkpoint. They cover
submit, poll, download, media roles, canvas sizing, and one-at-a-time execution.
