# gestcam — plan & findings

## The problem, precisely

On macOS, "Reactions" are not a Discord feature. They're an OS-level camera effect (macOS 14+, Apple silicon) applied inside the camera stack, so every app that reads the camera gets them. Linux has no such layer, and no third-party app ships gesture reactions (Windows Studio Effects, NVIDIA Broadcast: framing/blur/eye-contact only; Snap Camera: dead). So we build the layer: a userspace virtual camera that sits between the webcam and the app.

## Architecture

```
webcam ──V4L2 (MJPG)──▶ Capture thread ──newest frame──▶ main loop ──▶ v4l2loopback (/dev/video10, I420)
                                                            │   ▲                      │
                                          rgb@640w ──▶ MediaPipe threads (LIVE_STREAM) │
                                                       gestures + person mask          ▼
                     new Observation ──▶ decide(): pick() ──▶ ReactionState ──▶ EffectEngine (back | person | front)
                                                                                                   │
                                                                              Discord / Chromium / OBS / Zoom
```

Modules: `streams.py` (capture + sinks) · `recognizer.py` (MediaPipe wrapper + pure gesture rules) · `segment.py` (person mask) · `state.py` (hold/cooldown) · `effects.py` (layered particles) · `pipeline.py` (loop, eval, demo) · `doctor.py` (preflight) · `cli.py`.

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
| ~~**Opaque OpenCV primitives, ROI alpha only for sprites**~~ → **layered compositing in uint8 cv2** | **Superseded.** The ban was right about numpy float (mask blend 18.9 ms at 720p) and wrong about native uint8 cv2: `addWeighted` 0.5 ms, `blendLinear` mask composite ~2.3 ms in a real frame, half-res blurred glow 0.8–1.8 ms. So effects now have back / front / light passes: back is drawn on a copy and blended *under* the person, front over them, and fades are alpha, never darkening to black. `step()` cost with a mask, mean/p95: single effects 0.4–5.3 / ≤6.1 ms, all 8 at once 8.4 / 10.7 ms (idle machine; command and load not recorded). Under load ~9–15 on 16 cores, `scratchpad/build_effects/perf.py` measured masked step() means of 17–25 ms for single effects and 36.2 / 49.4 ms mean/p95 for all 8, against 4–10 ms unmasked. That is why the segmentation auto-off also watches the loop time of masked frames. |
| **Per-hand pose = canned label fused with landmark geometry** | The canned top-1 label missed 4 of 6 clear ✌️ (None:.96 on a textbook V), and MediaPipe 0.10 returns no runner-up. Geometry (V, horns, thumb tip as the hand's extreme point) votes unless the model is ≥0.7 sure of a *different* named class; horns and heart geometry outrank labels (the model has neither class and called horns "Victory", hearts "Thumb_Down"). 22 Commons photos: 4/20 → 11/20 in IMAGE mode, 0 wrong-reaction fires; landmarks are pinned as `tests/fixtures/photo_hands.json`. |
| **Pairing by landmark distance + handedness, not a wrist floor** | "Wrists > 1 palm apart" rejected real thumbs held together and still let a duplicate box through. Now: mean distance over all 21 landmarks < 0.3 palm = the same hand twice; Left+Left / Right+Right rejected only when both handedness scores ≥ 0.8 (`--mirror` flips both, so "they differ" survives it); every pair tried closest-first so a third hand can't hide one. Palms > 6 apart don't pair. `ponytail:` one person's spread arms (4.5 palms) and two people side by side (4.4) overlap, so distance can't split people. |
| **Per-reaction hold accumulators, fed once per observation** | Replaced an incumbent/challenger rule that double-fired 👍 then 👍👍 (thumbs_up, then fireworks 2.5 s later) and let a 50/50 thumbs_up/fireworks flicker always resolve to thumbs_up. Each reaction accrues dt while seen, freezes for `grace_s`, decays at 2·dt. **Promotion**: a seen superset takes max(own, subset's) so fireworks fires 0.7 s after the *first* thumb. **Deferral**: with 2+ hands in frame a subset needs +`defer_s` (0.15 s). **Upgrade**: after a subset fires, its superset may still fire inside the cooldown and the effect engine retires the bubble. Cooldown holds every accumulator at 0, so every fire follows a visible ring. Clocked on `Observation.ts`, once per new observation: per-frame wall-clock feeding made hold length depend on model latency. Fire times identical at 10/30/60 Hz. |
| **Selfie segmenter for "behind the person"** | Google's `selfie_segmenter.tflite` (Apache-2.0, 250 KB) through the same Tasks API, LIVE_STREAM on its own thread, fed the same 640 px RGB as the gesture model. 4.7–5.4 ms p50 alone; both models together keep gesture p95 under 33 ms (it costs the gesture model ~1.5 ms of p95). The mask is dilated 1.2% of height (one frame of lag) and smoothed asymmetrically: growth immediate, shrink EMA, so an effect is never painted over a pixel the person just moved into. Auto-off for the session if either model's p95, or the loop p95 of masked frames, is > 33 ms for two 5 s windows. Runs only while a hold or effect is up (the only mask readers): idle, it cost ~0.5 core. |
| **No Apple code to port: Reactions live in AVFoundation, not the app** | Discord 0.0.419 (Discord.dmg) contains no gesture code. Its Chromium only *logs* `AVCaptureDevice.reactionEffectsEnabled` / `reactionEffectGesturesEnabled` (`media/capture/video/mac/video_capture_metrics_mac.mm`). The recognition and effects are Apple's, closed, inside the camera stack; gestcam copies the public behaviour (support.apple.com/105117, WWDC23 10105), not code or assets. |
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
2. `uv run gestcam run --record-raw first.mp4` for ~2 min: each gesture twice, two-hand ones one hand at a time, and 20 s of talking with your hands (false-positive test). Then `uv run gestcam eval first.mp4`. Note the `[gestcam] out … fps | infer … | seg …` lines.
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
- The full loop (gesture model + segmenter + all 8 effects in rotation + compositing +
  I420 conversion) on a 1280x720 file in place of the webcam: **30 fps** out, loop p50/p95
  10.8/14.2 ms with the mask (5.1/7.9 without), gesture infer 16.9/23.9 ms, segmenter
  7.9/10.8 ms. The live loop and `gestcam eval` fired the identical 4 reactions at the
  same clip times, which is the check that both share one decision path.
- `gestcam eval` on the 22 photos as 1.2 s stills (VIDEO mode, tracking on): 11 fire,
  10 correct + 1 one-hand subset, 0 wrong. One photo the single-frame run got
  (td_videocall) is lost once the tracker re-crops: frame 0 reads thumb_down, frame 1
  Closed_Fist. Live tracking can cut both ways; only a real recording will say which.
