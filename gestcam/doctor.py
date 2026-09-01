"""`gestcam doctor` — preflight diagnostics.

Collects the facts that decide whether the pipeline will work on THIS machine:
kernel/headers/dkms state, v4l2loopback module + params, video devices, PipeWire,
Chromium's PipeWire-camera flag, how Discord is installed, Python deps, CPU.
Prints a report with concrete fixes, and `--json` for sharing.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

OK, WARN, FAIL, INFO = "OK  ", "WARN", "FAIL", "INFO"


def sh(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def read(p: str) -> str:
    try:
        return Path(p).read_text().strip()
    except Exception:
        return ""


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []
        self.data: dict = {}

    def add(self, level: str, key: str, msg: str, **data):
        self.rows.append((level, key, msg))
        if data:
            self.data[key] = data

    def render(self) -> str:
        out = []
        for level, key, msg in self.rows:
            out.append(f"[{level}] {key}: {msg}")
        return "\n".join(out)


# ----------------------------------------------------------------- checks
def check_system(r: Report):
    osr = {}
    for line in read("/etc/os-release").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            osr[k] = v.strip('"')
    distro = osr.get("PRETTY_NAME", platform.platform())
    kernel = platform.release()
    r.add(INFO, "system", f"{distro} | kernel {kernel} | {platform.machine()}", distro=distro, kernel=kernel, id=osr.get("ID"), id_like=osr.get("ID_LIKE"))
    session = os.environ.get("XDG_SESSION_TYPE", "?")
    comp = "hyprland" if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") else os.environ.get("XDG_CURRENT_DESKTOP", "?")
    xwl = bool(os.environ.get("DISPLAY"))
    r.add(INFO, "session", f"{session} | compositor {comp} | XWayland DISPLAY={'yes' if xwl else 'no'}", session=session, compositor=comp, xwayland=xwl)
    if session == "wayland" and not xwl:
        r.add(WARN, "preview", "No DISPLAY: the OpenCV preview window (Qt/xcb) won't open. Use --no-preview or check the virtual cam in mpv/Discord.")
    cpu = ""
    flags = ""
    for line in read("/proc/cpuinfo").splitlines():
        if line.startswith("model name") and not cpu:
            cpu = line.split(":", 1)[1].strip()
        if line.startswith("flags") and not flags:
            flags = line.split(":", 1)[1]
    cores = os.cpu_count()
    avx2 = " avx2 " in f" {flags} "
    r.add(INFO if avx2 else WARN, "cpu", f"{cpu} | {cores} threads | AVX2={'yes' if avx2 else 'no (MediaPipe will be slow)'}", model=cpu, threads=cores, avx2=avx2)
    mem = read("/proc/meminfo").split("\n")[0] if os.path.exists("/proc/meminfo") else ""
    if mem:
        r.add(INFO, "memory", mem.replace("MemTotal:", "").strip())


def check_python(r: Report):
    r.add(INFO, "python", f"{sys.version.split()[0]} at {sys.executable}", version=sys.version.split()[0], exe=sys.executable)
    for mod, pip_name in [("mediapipe", "mediapipe"), ("cv2", "opencv-contrib-python"), ("numpy", "numpy"), ("pyvirtualcam", "pyvirtualcam")]:
        try:
            m = __import__(mod)
            r.add(OK, f"dep.{pip_name}", getattr(m, "__version__", "?"), version=getattr(m, "__version__", "?"))
        except Exception as e:
            r.add(FAIL, f"dep.{pip_name}", f"import failed: {e}")
    try:
        import importlib.metadata as md
        dup = [d for d in ("opencv-python", "opencv-python-headless", "opencv-contrib-python-headless") if _dist_installed(md, d)]
        if dup:
            r.add(WARN, "dep.opencv-duplicates", f"{dup} also installed — two cv2 packages in one env cause broken imports; keep only opencv-contrib-python")
    except Exception:
        pass


def _dist_installed(md, name: str) -> bool:
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False


def check_kernel_module(r: Report):
    if not sys.platform.startswith("linux"):
        r.add(INFO, "v4l2loopback", f"not applicable on {sys.platform} (pyvirtualcam uses OBS Virtual Camera there)")
        return
    kernel = platform.release()
    headers = os.path.isdir(f"/usr/lib/modules/{kernel}/build") or os.path.isdir(f"/lib/modules/{kernel}/build")
    r.add(OK if headers else WARN, "kernel-headers", f"{'present' if headers else 'MISSING'} for {kernel}" + ("" if headers else " — dkms cannot build v4l2loopback"), present=headers)

    modinfo = sh(["modinfo", "-F", "version", "v4l2loopback"])
    installed = bool(modinfo) or bool(sh(["modinfo", "-n", "v4l2loopback"]))
    loaded = "v4l2loopback" in read("/proc/modules")
    r.add(OK if installed else FAIL, "v4l2loopback.installed", f"{'yes' if installed else 'no'}{(' v' + modinfo) if modinfo else ''}", installed=installed, version=modinfo)
    r.add(OK if loaded else WARN, "v4l2loopback.loaded", "yes" if loaded else "no (modprobe it, see fixes below)", loaded=loaded)

    dkms = sh(["dkms", "status"])
    if dkms:
        lines = [l for l in dkms.splitlines() if "v4l2loopback" in l]
        built_for_this = any(kernel in l and "installed" in l for l in lines)
        r.add(OK if built_for_this or not lines else WARN, "dkms", "; ".join(lines) if lines else "no v4l2loopback entry", entries=lines, built_for_running_kernel=built_for_this)
        if lines and not built_for_this:
            r.add(WARN, "dkms.kernel-mismatch", f"module not built for running kernel {kernel} — after kernel updates run: sudo dkms autoinstall  (then reboot)")

    params = {}
    for p in glob.glob("/sys/module/v4l2loopback/parameters/*"):
        params[os.path.basename(p)] = read(p)
    if params:
        excl = params.get("exclusive_caps", "")
        r.add(OK if excl.startswith("Y") else FAIL, "v4l2loopback.exclusive_caps", f"{excl or '?'}" + ("" if excl.startswith("Y") else " — Chromium-based apps (Discord web app, Chrome, Electron) will NOT list the device without exclusive_caps=1"), **params)
    modconf = "\n".join(read(p) for p in glob.glob("/etc/modprobe.d/*.conf") if "v4l2loopback" in read(p))
    autoload = any("v4l2loopback" in read(p) for p in glob.glob("/etc/modules-load.d/*.conf"))
    r.add(OK if modconf else WARN, "v4l2loopback.persist", (modconf.replace("\n", " | ") if modconf else "no /etc/modprobe.d entry → module params are lost on reboot"), modprobe_conf=modconf, autoload=autoload)

    sb = ""
    for p in glob.glob("/sys/firmware/efi/efivars/SecureBoot-*"):
        try:
            sb = "enabled" if Path(p).read_bytes()[-1] == 1 else "disabled"
        except Exception:
            pass
    if not sb:
        m = sh(["mokutil", "--sb-state"])
        sb = "enabled" if "enabled" in m else ("disabled" if "disabled" in m else "unknown")
    r.add(WARN if sb == "enabled" else INFO, "secure-boot", sb + (" — unsigned DKMS modules won't load; sign the module or disable SB" if sb == "enabled" else ""), state=sb)


def openers(dev: str) -> list[tuple[int, str, str]]:
    """Who currently has `dev` open: [(pid, comm, "producer"|"consumer"), ...].

    This is the answer to the most confusing failure in this project — a second app
    opening a busy loopback goes black rather than erroring visibly — so doctor should
    just name the process holding it. O_WRONLY means a producer (gestcam); anything
    else is a consumer (Discord, a browser, mpv).
    """
    held = []
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd) != dev:
                continue
            pid = int(fd.split("/")[2])
            comm = read(f"/proc/{pid}/comm") or "?"
            flags = 0
            for line in read(f"/proc/{pid}/fdinfo/{os.path.basename(fd)}").splitlines():
                if line.startswith("flags:"):
                    flags = int(line.split()[1], 8)
            held.append((pid, comm, "producer" if flags & 3 == os.O_WRONLY else "consumer"))
        except (OSError, ValueError, IndexError):
            pass          # process exited mid-scan
    return sorted(set(held))


def check_video_devices(r: Report):
    devs = []
    for p in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda s: int(re.sub(r"\D", "", s) or 0)):
        n = os.path.basename(p)
        name = read(f"{p}/name")
        loop = os.path.realpath(p).startswith("/sys/devices/virtual/")
        caps = sh(["v4l2-ctl", "-d", f"/dev/{n}", "--info"]) if shutil.which("v4l2-ctl") else ""
        cap_flags = [l.strip() for l in caps.splitlines() if l.strip().startswith(("Video Capture", "Video Output", "Metadata Capture"))]
        devs.append(dict(dev=f"/dev/{n}", name=name, loopback=loop, caps=cap_flags))
    if not devs:
        r.add(FAIL, "video-devices", "no /dev/video* at all — no webcam detected?")
    for d in devs:
        kind = "LOOPBACK" if d["loopback"] else "camera "
        d["holders"] = [f"{c} pid {n} ({k})" for n, c, k in openers(d["dev"])]
        msg = f"{kind} {d['name']}" + (f" [{', '.join(d['caps'])}]" if d["caps"] else "")
        if d["holders"]:
            msg += " | held by " + ", ".join(d["holders"])
        r.add(INFO, f"video.{os.path.basename(d['dev'])}", msg, **d)
        if d["loopback"]:
            # v4l2loopback's `state` reads "capture" exactly while a producer holds the
            # OUTPUT stream token (attr_show_state, v4l2loopback.c:762). No open needed.
            n = os.path.basename(d["dev"])
            st = read(f"/sys/class/video4linux/{n}/state")
            fmt = read(f"/sys/class/video4linux/{n}/format")
            r.add(INFO, f"loopback.producer[{d['dev']}]",
                  f"{st} ({fmt})" if st == "capture" else
                  f"{st or 'unknown'} — no producer; start `gestcam run` before opening the camera picker")
    real = [d for d in devs if not d["loopback"] and not d["name"].lower().startswith("v4l2 loopback")]
    loops = [d for d in devs if d["loopback"]]
    gest = [d["dev"] for d in devs if d["name"].startswith("GestCam")]
    if gest:
        # One device serves exactly one streaming consumer (vidioc_reqbufs -EBUSY on
        # the second capture client), so the device count is the app count.
        tail = "" if len(gest) >= 2 else \
            " — one device serves one streaming consumer; reload with video_nr=10,11 (see fixes)"
        r.add(INFO, "v4l2loopback.gestcam-devices",
              f"{len(gest)} GestCam device(s): {', '.join(gest)}{tail}", devices=gest)
    r.data["video_devices"] = devs
    if not loops:
        r.add(WARN, "video.loopback", "no v4l2loopback device present (load the module)")
    if not shutil.which("v4l2-ctl"):
        r.add(INFO, "tools.v4l-utils", "v4l2-ctl not installed (optional but handy): pacman -S v4l-utils")
    for t in ("ffmpeg", "mpv"):
        r.add(INFO, f"tools.{t}", "found" if shutil.which(t) else "not found (optional)")
    r.data["real_cameras"] = [d["dev"] for d in real]


def probe_camera(r: Report, device: str | None):
    """Open the first real camera with OpenCV and report what MJPG 720p actually gives."""
    try:
        import cv2
    except Exception:
        return
    cams = [device] if device else r.data.get("real_cameras", [])
    if not cams:
        return
    dev = cams[0]
    # gestcam holds the real camera exclusively while it runs, so probing would fail
    # and (because run_doctor returns 1 on any FAIL) turn the HEALTHY state red — with
    # the loudest row pointing at the wrong device. Say who has it instead.
    held = openers(dev) if isinstance(dev, str) else []
    if held:
        r.add(INFO, "camera.probe",
              f"{dev} held by " + ", ".join(f"{c} pid {n}" for n, c, _ in held) + " — not probing",
              dev=dev, holders=[c for _, c, _ in held])
        return
    idx = int(re.sub(r"\D", "", dev)) if isinstance(dev, str) and dev.startswith("/dev/video") else dev
    try:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, f = cap.read()
        if not ok:
            r.add(FAIL, "camera.probe", f"{dev} opened but returned no frame")
            return
        import time
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < 1.5:
            if cap.read()[0]:
                n += 1
        fps = n / (time.monotonic() - t0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        cc = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
        cap.release()
        r.add(OK if fps >= 20 else WARN, "camera.probe", f"{dev} → {f.shape[1]}x{f.shape[0]} {cc} measured {fps:.1f} fps" + ("" if fps >= 20 else " — low; try --width 640 --height 480 or check v4l2-ctl --list-formats-ext"), dev=dev, width=f.shape[1], height=f.shape[0], fourcc=cc, fps=fps)
    except Exception as e:
        r.add(FAIL, "camera.probe", f"{e}")


def check_pipewire(r: Report):
    ver = sh(["pipewire", "--version"]).splitlines()[-1] if shutil.which("pipewire") else ""
    r.add(INFO, "pipewire", ver or "not found")
    st = sh(["wpctl", "status"]) if shutil.which("wpctl") else ""
    if st:
        vid = st.split("Video")[1] if "Video" in st else ""
        sources = [l.strip(" │*") for l in vid.splitlines() if re.search(r"^\s*[│*\s]*\d+\.", l)]
        r.add(INFO, "pipewire.video-nodes", "; ".join(sources[:8]) or "none listed", nodes=sources)
        loop_in_pw = any("loopback" in s.lower() or "gestcam" in s.lower() for s in sources)
        r.data["loopback_visible_in_pipewire"] = loop_in_pw
    # Chromium flags
    for prof in ("~/.config/chromium/Local State", "~/.config/google-chrome/Local State", "~/.config/BraveSoftware/Brave-Browser/Local State"):
        p = Path(os.path.expanduser(prof))
        if p.exists():
            try:
                labs = json.loads(p.read_text()).get("browser", {}).get("enabled_labs_experiments", [])
            except Exception:
                labs = []
            pw = [l for l in labs if "pipewire" in l.lower()]
            r.add(WARN if any("camera" in l and "@1" in l for l in pw) else OK, f"chromium.flags[{p.parent.name}]",
                  (", ".join(pw) if pw else "PipeWire camera flag at default (V4L2 path) — good for v4l2loopback"), flags=pw)


def check_discord(r: Report):
    found = []
    if shutil.which("discord") or os.path.isdir("/opt/discord"):
        found.append("native (pacman/AUR/tarball)")
    if shutil.which("flatpak") and "com.discordapp.Discord" in sh(["flatpak", "list", "--app", "--columns=application"]):
        found.append("flatpak com.discordapp.Discord (sandbox: needs --device=all; loopback must exist before launch)")
    for d in glob.glob(os.path.expanduser("~/.local/share/applications/*.desktop")) + glob.glob("/usr/share/applications/*.desktop"):
        txt = read(d)
        if "discord" in txt.lower() and "--app=" in txt:
            found.append(f"Chromium web app ({os.path.basename(d)}) → exclusive_caps=1 required")
            break
    r.add(INFO, "discord", " | ".join(found) if found else "no Discord install detected (web in browser?)", installs=found)


# ----------------------------------------------------------------- fixes
def fixes(r: Report) -> str:
    distro = (r.data.get("system", {}).get("id") or "") + " " + (r.data.get("system", {}).get("id_like") or "")
    arch = "arch" in distro
    deb = any(x in distro for x in ("debian", "ubuntu"))
    fed = "fedora" in distro
    lines = ["", "Fixes / setup commands (copy-paste, review first):"]
    if not r.data.get("v4l2loopback.installed", {}).get("installed"):
        if arch:
            lines.append("  sudo pacman -S --needed linux-headers v4l2loopback-dkms v4l-utils   # use linux-zen-headers etc. if you run another kernel")
        elif deb:
            lines.append("  sudo apt install v4l2loopback-dkms v4l-utils")
        elif fed:
            lines.append("  sudo dnf install v4l2loopback v4l-utils   # (rpmfusion) or akmod-v4l2loopback")
        else:
            lines.append("  install v4l2loopback-dkms + kernel headers for your distro")
    lines += [
        "  # one-shot (lost on reboot):",
        "  sudo modprobe -r v4l2loopback 2>/dev/null; sudo modprobe v4l2loopback video_nr=10,11 card_label=GestCam,GestCam2 exclusive_caps=1,1",
        "  # persistent:",
        "  echo 'options v4l2loopback video_nr=10,11 card_label=GestCam,GestCam2 exclusive_caps=1,1' | sudo tee /etc/modprobe.d/v4l2loopback.conf",
        "  echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf",
        "  # verify the device and (while gestcam runs) what it advertises:",
        "  v4l2-ctl --list-devices; v4l2-ctl -d /dev/video10 --list-formats-ext; v4l2-ctl -d /dev/video11 --list-formats-ext",
        "  # one loopback device serves exactly ONE streaming consumer: the second app to",
        "  # start gets 'Device or resource busy' (VIDIOC_REQBUFS -EBUSY) and shows black.",
        "  # Two devices, one per app — point native Discord at GestCam and the browser at GestCam2.",
        "  # watch the virtual cam without Discord:  mpv av://v4l2:/dev/video10",
    ]
    return "\n".join(lines)


def run_doctor(as_json: bool = False, probe: bool = True, device: str | None = None) -> int:
    r = Report()
    check_system(r)
    check_python(r)
    check_kernel_module(r)
    check_video_devices(r)
    if probe:
        probe_camera(r, device)
    check_pipewire(r)
    check_discord(r)
    fails = sum(1 for lvl, _, _ in r.rows if lvl == FAIL)
    if as_json:
        print(json.dumps({"rows": r.rows, "data": r.data}, indent=2, default=str))
    else:
        print(r.render())
        print(fixes(r))
        print(f"\n{fails} FAIL, {sum(1 for lvl, _, _ in r.rows if lvl == WARN)} WARN")
    return 1 if fails else 0
