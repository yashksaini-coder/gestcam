# gestcam — plan & findings

## The problem, precisely

On macOS, "Reactions" are not a Discord feature. They're an OS-level camera effect (macOS 14+, Apple silicon) applied inside the camera stack, so every app that reads the camera gets them. Linux has no such layer, and no third-party app ships gesture reactions (Windows Studio Effects, NVIDIA Broadcast: framing/blur/eye-contact only; Snap Camera: dead). So we build the layer: a userspace virtual camera that sits between the webcam and the app.

## Architecture

```
webcam ──V4L2 (MJPG)──▶ Capture thread ──newest frame──▶ main loop ──▶ v4l2loopback (/dev/video10, I420)
                                                            │   ▲                      │
                                          rgb@640w ──▶ MediaPipe thread (LIVE_STREAM)  │
                                                            │                          ▼
                                          Observation ──▶ reaction_for() ──▶ ReactionState ──▶ EffectEngine
                                                                                                   │
                                                                              Discord / Chromium / OBS / Zoom
```

Modules: `streams.py` (capture + sinks) · `recognizer.py` (MediaPipe wrapper + pure gesture rules) · `state.py` (hold/cooldown) · `effects.py` (particles) · `pipeline.py` (loop, demo) · `doctor.py` (preflight) · `cli.py`.

## Decisions and why

| Decision | Reasoning |
|---|---|
| **v4l2loopback, not a PipeWire source node** (yet) | Chromium's PipeWire camera path is still flag-gated (`chrome://flags/#enable-webrtc-pipewire-camera`, default = V4L2). Omarchy's Discord is a Chromium web app → V4L2 is the path it actually uses today. A `pipewiresink` backend is Phase 3; the sink interface is already pluggable. |
| **`exclusive_caps=1`, fixed `video_nr=10`** | Chromium filters devices that report OUTPUT caps; `exclusive_caps` makes the loopback report CAPTURE-only once a producer attaches. Fixed number = stable name in Discord's device list. |
| **Output I420** (pyvirtualcam converts BGR→I420 natively) | The one pixel format every V4L2 consumer accepts. RGB24/BGR24 from a loopback is hit-and-miss in browsers. |
| **MediaPipe Tasks GestureRecognizer, canned model** | Canned classes are `Thumb_Up, Thumb_Down, Victory, ILoveYou, Open_Palm, Closed_Fist, Pointing_Up` — 6 of Apple's 8 reactions fall straight out. Heart and horns are geometric landmark rules (`is_heart`, `is_horns`), no training needed. |
| **LIVE_STREAM mode, newest-result-wins** | Model latency never blocks the frame loop. If inference falls behind, output stays 30 fps and reactions lag ~50 ms, which nobody notices. |
| **Inference at 640 px wide** | Hand landmarker resizes to 224 internally; feeding 1280 just costs a bigger copy. Landmarks are normalized, so effects still render at full resolution. |
| **Time-based hold (0.7 s) + cooldown (2.5 s) + release-to-repeat** | Same feel as Apple's "hold and pause". Frame-count debouncing breaks the moment fps changes. |
| **Opaque OpenCV primitives, ROI alpha only for sprites** | Full-frame float alpha blend ≈ 10–15 ms at 720p. Measured: all 8 effects at once ≈ 10 ms on a 2.1 GHz Xeon; any single effect ≤ 2 ms. |
| ~~**mediapipe ≥ 1.0**~~ → **mediapipe < 1.0**, pinned | **Superseded.** 1.0.1 SIGKILLs the process on its first inference on this machine — reproduced with no camera, no virtual camera and no gestcam code, on a single static frame, in all three running modes. Not OOM (peak RSS 251 MB, 12 GB free, no cgroup limit), no coredump, no seccomp/audit trace, `RLIMIT_RTTIME` already unlimited. 0.10.x runs the same model fine. The ABI-agnostic wheel was only insurance against a bleeding-edge interpreter, which `uv python pin 3.12` retires. |
| **Two loopback devices, not one** | One v4l2loopback device serves exactly **one streaming consumer**: `VIDIOC_REQBUFS` returns `-EBUSY` to the second capture client, while `max_openers=10` gates only `open()` — so the second app opens the device and then shows black. Measured both ways round (native Discord holding it, then Brave holding it). `video_nr`/`card_label`/`exclusive_caps` are per-device arrays, so all three must be comma lists or the second device comes up invisible to Chromium. |
| **`uv` + `pyproject.toml`** | Isolated from system Python; `uv python pin 3.12` is the escape hatch if a wheel is missing for a bleeding-edge interpreter. |

## Risk register (from research + sandbox testing)

| # | Risk | Likelihood on Omarchy | Detection | Mitigation |
|---|---|---|---|---|
| 1 | Discord (Chromium web app) doesn't list the loopback | High if `exclusive_caps=0` | `doctor` → `v4l2loopback.exclusive_caps` | `scripts/loopback.sh`; run gestcam *before* opening Discord's camera picker |
| 2 | DKMS module not rebuilt after kernel update (rolling release!) | Recurring | `doctor` → `dkms.kernel-mismatch` | `sudo dkms autoinstall`; consider a pacman hook |
| 3 | Chromium switches to PipeWire camera → loopback invisible/racy | Low now, rising | `doctor` → `chromium.flags`, `pipewire.video-nodes` | Disable the flag; Phase 3 PipeWire backend |
| 4 | Secure Boot blocks unsigned DKMS module | Unknown (Omarchy uses LUKS; SB depends on firmware) | `doctor` → `secure-boot` | sign module / disable SB |
| 5 | Webcam negotiates YUYV → 5–10 fps at 720p | Medium (USB2 cams) | `doctor --probe` → `camera.probe` fps | MJPG request is in; fall back to 640×480 |
| 6 | `cv2.imshow` preview fails under Wayland | Medium | `doctor` → `session`/`preview` | `QT_QPA_PLATFORM=xcb` auto-set; preview is optional |
| 7 | Two OpenCV wheels in one env | Low with `uv sync` | `doctor` → `dep.opencv-duplicates` | never add `opencv-python` |
| 8 | Model download blocked/offline on first run | Low | log line | `--model path` |
| 9 | False triggers (thumb visible while talking with hands) | Medium | you'll notice | `--min-score`, `--hold`; two-hand gestures are inherently rarer |
| 10 | CPU too slow for model at 30 fps | Low on a dev laptop | stats line `infer p50/p95` | `--infer-every 2`, `--infer-width 480` |
| 11 | Camera stalls when gestcam and Discord both open the *real* webcam | Certain if it happens | "camera stalled" log | only gestcam opens the webcam; Discord uses GestCam |
| 12 | **Two apps fight over one loopback → second shows black** | **Certain; this is what broke Discord-web** | `doctor` → `video.videoN` names the holder; `v4l2loopback.gestcam-devices` counts devices | two devices (`video_nr=10,11`), gestcam fans out to both |

Not risks: Hyprland/Wayland itself (V4L2 doesn't go through the compositor); mirroring (Discord mirrors your self-view locally; output is unmirrored like a real cam).

## Phases

- **Phase 1 — works on your machine (now).** `doctor` → `loopback.sh` → `demo` → `run` → Discord. Bring back: doctor JSON, the stats lines, which gestures misfire.
- **Phase 2 — feel.** Tune hold/cooldown, false-positive rate, effect timing on real video. Per-gesture score thresholds. Config file. Maybe a tray/Waybar indicator. Optional: mirror-aware placement of the thumb bubble.
- **Phase 3 — PipeWire backend.** `pipewiresink` via GStreamer subprocess (`fdsrc` raw frames). No kernel module, works for PipeWire-camera consumers (Firefox, Chromium w/ flag, OBS), and — the real draw — PipeWire multiplexes one source to many consumers natively, which is the problem two loopback devices currently solve by brute force. Still deferred: Chromium's PipeWire camera flag is off by default on this machine for chromium, chrome *and* Brave, so a PipeWire-only source would be invisible to the exact app this project exists for. Revisit when that flag flips.
- **Phase 4 — Windows.** pyvirtualcam already targets OBS Virtual Camera there; `Capture` drops `CAP_V4L2`. The rest is identical. Untested.
- **Phase 5 — custom gestures.** MediaPipe Model Maker to train a real 🫶/🤘 classifier if the geometric rules misfire.

## Data to collect on the first run

1. `uv run gestcam doctor --json > doctor.json`
2. `uv run gestcam run --record first.mp4` for ~1 min: do each gesture twice, talk with your hands for 20 s (false-positive test). Note the `[gestcam] out … fps | infer p50/p95` lines.
3. Whether Discord lists **GestCam** with gestcam running, and what `v4l2-ctl -d /dev/video10 --list-formats-ext` says while it runs.
4. `wpctl status | sed -n '/Video/,$p'` — does the loopback show up as a PipeWire node?

## Verified in the sandbox (Ubuntu 24.04, Python 3.12, no camera, no kernel module)

- `mediapipe 1.0.1` + `pyvirtualcam 0.15.0` install cleanly; Tasks API signature confirmed.
- `pip` OpenCV 5.0 wheel: HighGUI = Qt5, no GTK → Wayland caveat is real.
- 11 unit tests (state machine, gesture rules, effect rendering) pass.
- Demo render at 720p: all eight effects visually checked.

**Not** verified anywhere yet: real gesture recognition quality, v4l2loopback write, Discord device listing. That's the first thing your machine tells us.

## Verified on the target machine (Omarchy, i7-13620H, HP Wide Vision HD)

- Output holds **30.2 fps** at 1280x720 MJPG; loop p50 ~2.4 ms of a 33.3 ms budget.
- v4l2loopback write works; Chromium/Brave list **GestCam** and pick it as the *default*
  `getUserMedia` device, returning live frames. There was never a browser-side bug.
- Per-effect draw cost at 720p: confetti 10.4 ms, rain 2.6, fireworks 2.5, hearts 1.8,
  lasers 1.7, thumbs 1.4/1.3, balloons 0.8. `hold_s + cooldown_s` keeps fires ≥3.2 s
  apart, so at most two overlap: measured worst case 12.5 ms.
- Seven classifier bugs found and fixed by reproduction, the largest being that a held
  gesture could fire **never** under label flicker (10 s of holding, zero triggers).
