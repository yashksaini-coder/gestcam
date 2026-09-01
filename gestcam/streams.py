"""Frame sources and sinks.

Source: `Capture` reads the webcam on its own thread and keeps only the newest frame.
V4L2 buffers several frames; if the main loop stalls for 100 ms you'd otherwise get
100 ms of stale video on the virtual camera forever. Newest-frame-wins avoids that.

Sinks all implement `send(frame_bgr) -> bool` (False = stop requested) and `close()`.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import cv2
import numpy as np


# ------------------------------------------------------------------ source
class Capture:
    def __init__(self, device: int | str = 0, width: int = 1280, height: int = 720, fps: int = 30,
                 mjpg: bool = True, log=print):
        self.log = log
        if sys.platform.startswith("linux"):
            self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {device!r}")
        # Order matters on V4L2: FOURCC before size, or the size request is ignored.
        # MJPG is the difference between 720p@30 and 720p@5 on most USB2 webcams.
        if mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # NOT CAP_PROP_BUFFERSIZE=1: with a single V4L2 buffer the driver has
        # nothing queued while we decode, and 720p MJPG drops 30 fps -> 20 fps
        # (measured). The newest-frame-wins thread below already keeps latency
        # to ~1 frame, which is what BUFFERSIZE was reaching for.
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera opened but returned no frame")
        self.height, self.width = frame.shape[:2]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.fourcc = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
        self._frame = frame
        self._ts = time.monotonic()
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="capture")
        self._thread.start()

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._ts = time.monotonic()
                self._seq += 1

    def wait_frame(self, last_seq: int, timeout: float = 1.0) -> tuple[np.ndarray, float, int] | None:
        """Block until a frame newer than `last_seq` exists. Returns (frame, ts, seq)."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._lock:
                if self._seq != last_seq:
                    return self._frame, self._ts, self._seq
            time.sleep(0.001)
        return None

    def close(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self.cap.release()


# ------------------------------------------------------------------ sinks
class VirtualCamOutput:
    """pyvirtualcam → v4l2loopback (Linux) / OBS Virtual Camera (Windows, macOS).

    We hand pyvirtualcam BGR frames; its Linux backend converts to I420 (YU12)
    natively, which is the pixel format every consumer (Chromium, Firefox, Zoom,
    OBS) accepts from a loopback device.
    """

    def __init__(self, width: int, height: int, fps: float, device: str | None = None, log=print):
        import pyvirtualcam
        self.cam = pyvirtualcam.Camera(width=width, height=height, fps=fps,
                                       fmt=pyvirtualcam.PixelFormat.BGR, device=device)
        log(f"[gestcam] virtual camera: {self.cam.device} ({self.cam.backend}) native_fmt={self.cam.native_fmt}")

    def send(self, frame: np.ndarray) -> bool:
        self.cam.send(frame)
        return True

    def close(self):
        self.cam.close()


class PreviewOutput:
    """cv2.imshow window. Wayland note: the pip OpenCV wheel bundles Qt5 without a
    Wayland plugin, so we force the xcb (XWayland) platform unless the user set one."""

    def __init__(self, title: str = "gestcam (q to quit)"):
        if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
            os.environ["QT_QPA_PLATFORM"] = "xcb"
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def send(self, frame: np.ndarray) -> bool:
        cv2.imshow(self.title, frame)
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)

    def close(self):
        cv2.destroyAllWindows()


class FileOutput:
    def __init__(self, path: str, width: int, height: int, fps: float):
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open {path} for writing")

    def send(self, frame: np.ndarray) -> bool:
        self.writer.write(frame)
        return True

    def close(self):
        self.writer.release()


class MultiOutput:
    def __init__(self, sinks):
        self.sinks = list(sinks)

    def send(self, frame: np.ndarray) -> bool:
        return all([s.send(frame) for s in self.sinks])  # list(): evaluate every sink

    def close(self):
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                pass
