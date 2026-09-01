# gestcam

Apple-style gesture reactions (hearts, thumbs, balloons, rain, confetti, lasers, fireworks) for **any** video-call app on Linux, delivered as a virtual camera. Built for Omarchy (Arch + Hyprland), works on any Linux with `v4l2loopback`.

| Gesture | Reaction | Gesture | Reaction |
|---|---|---|---|
| 👍 | thumbs up | 👍👍 | fireworks |
| 👎 | thumbs down | 👎👎 | rain |
| ✌️ | balloons | ✌️✌️ | confetti |
| 🫶 two-hand heart | hearts | 🤘🤘 | lasers |

Hold the gesture ~0.7 s, away from your face. A white ring shows hold progress.

## Install

```sh
git clone <this repo> && cd gestcam
uv sync                       # or: python -m venv .venv && . .venv/bin/activate && pip install -e .
uv run gestcam doctor         # read this before anything else
```

`doctor` reports kernel headers, module state, video devices, PipeWire, Discord install type, and prints copy-paste fixes. Share `uv run gestcam doctor --json` when asking for help.

## One-time: virtual camera device

```sh
./scripts/loopback.sh         # installs v4l2loopback-dkms + headers, loads with exclusive_caps=1, persists
```

`exclusive_caps=1` is not optional: Chromium-based apps (Omarchy's Discord web app, Chrome, Electron) hide loopback devices without it.

## Run

```sh
uv run gestcam demo --out demo.mp4     # every reaction on a synthetic scene, no camera/model needed
uv run gestcam run                     # webcam → /dev/video10 ("GestCam")
uv run gestcam run --preview           # + local window (XWayland)
uv run gestcam run --width 640 --height 480 --infer-every 2   # slow machine
```

Then in Discord → Voice & Video → Camera → **GestCam**. Check without Discord: `mpv av://v4l2:/dev/video10`.

Flags worth knowing: `--hold 0.7` `--cooldown 2.5` `--min-score 0.5` `--mirror` `--record out.mp4` `--out none` (preview only).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Discord doesn't list GestCam | module loaded without `exclusive_caps=1`; or gestcam isn't running (loopback only advertises capture once a producer is attached). Also try disabling `chrome://flags/#enable-webrtc-pipewire-camera` if it's on. |
| `modprobe: FATAL: Module v4l2loopback not found` | headers missing for the running kernel → `pacman -S linux-headers`, `dkms autoinstall`, reboot. Happens after every kernel update until dkms rebuilds. |
| Module refuses to load, Secure Boot on | unsigned DKMS module. Sign it or disable Secure Boot. |
| Webcam gives 5 fps at 720p | camera stuck in YUYV. gestcam requests MJPG; check `v4l2-ctl -d /dev/video0 --list-formats-ext`. |
| `--preview` window doesn't open | pip OpenCV bundles Qt5 without Wayland support; gestcam sets `QT_QPA_PLATFORM=xcb`. Needs XWayland (`echo $DISPLAY`). |
| Reactions fire on their own | raise `--min-score`, or `--hold 1.0`. |
| First run hangs at "downloading gesture model" | needs network to fetch `gesture_recognizer.task` (~8 MB) into `~/.cache/gestcam/`. Pass `--model path` to use a local copy. |
| `ImportError` around cv2 | two OpenCV packages installed. Keep only `opencv-contrib-python`. |

## Credits

Gesture model: MediaPipe Tasks (Apache-2.0). Thumbs sprites: Noto Color Emoji (Apache-2.0).
