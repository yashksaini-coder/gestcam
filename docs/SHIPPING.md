# Shipping gestcam

Goal: a stranger on a Linux laptop goes from nothing to reactions in Discord with three
commands, and never has to read PLAN.md.

```bash
uv tool install gestcam        # or: pipx install gestcam
gestcam setup                  # one sudo prompt: loads v4l2loopback, makes it persistent
gestcam run                    # pick "GestCam" as the camera in Discord / Zoom / Meet
```

## What constrains the shape

| Fact (checked 2026-10-23) | Consequence |
|---|---|
| The virtual camera is a **kernel module** (v4l2loopback). pip can't install it, and Flatpak/Snap/AppImage sandboxes can't load it. | Ship the app through PyPI. The module comes from the distro (`v4l2loopback-dkms`), and `gestcam setup` drives that step. |
| mediapipe 0.10.x ships Linux wheels for **x86_64 only** (manylinux_2_28, so glibc 2.28 or newer). There is no aarch64 Linux wheel. | Supported platform: x86_64 Linux, glibc ≥ 2.28 (Ubuntu 20.04+, Fedora 29+, Debian 10+, Arch). Say so in the README and doctor, rather than failing at import. |
| 0.10.21 wheels are cp39–cp312. 0.10.30+ are `py3` (any Python). We pin `<1.0` because 1.0.1 SIGKILLs. | Leave the pin as is. The lock resolves 0.10.35, so 3.11–3.14 all work (tested 2026-10-23: full suite green on each). CI tests all four. |
| The models (gesture 8 MB, segmenter 250 KB, Apache-2.0) are downloaded on first run to `~/.cache/gestcam`. | Don't bundle them: the wheel stays small and they aren't redistributed. `gestcam setup` pre-fetches them so the first `run` doesn't stall. |
| The wheel is pure Python and includes the Noto emoji assets. CI already builds it and smoke-tests `gestcam demo` from a clean venv. | Nothing to compile per platform: one wheel and one sdist per release. |
| The name `gestcam` is free on PyPI (404 today). | Claim it with the first release. |

## Steps

1. **`gestcam setup` subcommand** (done: runs the packaged `gestcam/loopback.sh`, fetches
   both models, runs doctor; `--show` prints the script). Original intent: move what `scripts/loopback.sh` does into the
   package, because a PyPI install doesn't include `scripts/`. The command should:
   - detect the distro
   - print or run the install line (`pacman -S v4l2loopback-dkms`, `apt install v4l2loopback-dkms`, `dnf install v4l2loopback`)
   - write `/etc/modprobe.d/gestcam.conf` and `/etc/modules-load.d/gestcam.conf` (devices 10 and 11, `exclusive_caps=1`, label "GestCam") so the devices survive a reboot
   - `modprobe` the module now
   - pre-fetch both models
   - finish by running `doctor`

   It should show every root command before running it, and `--dry-run` should only print them.
2. **Release workflow** (done: `.github/workflows/release.yml`). On tag `v*` it:
   - fails unless the tag matches `version` in `pyproject.toml` and `gestcam/__init__.py`
   - runs the whole CI workflow (lint, tests on 3.11–3.14, build + wheel smoke test)
   - publishes that exact `dist` artifact to PyPI with **trusted publishing** (OIDC, no stored token)
   - creates a GitHub Release with the wheel, the sdist and the demo GIF attached

   One-time setup on PyPI, before the first tag: *Your projects → Publishing → Add a pending
   publisher*, with owner `yashksaini-coder`, repo `gestcam`, workflow `release.yml`, environment `pypi`.
   To release: bump both version strings, commit, then `git tag v0.1.0 && git push origin v0.1.0`.
   Switch to hatch-vcs once releases get frequent.
3. **README install section.** Add the three commands above, plus a "supported" line and
   a short troubleshooting table that `doctor` already backs.
4. **Autostart (optional).** `gestcam setup --autostart` installs a systemd *user* unit
   (`gestcam run`, `Restart=on-failure`) so the camera is always available. Make it opt-in,
   because it keeps the webcam open, and the webcam LED with it.
5. **AUR package (optional, later).** mediapipe isn't packaged for Arch, so a native
   PKGBUILD would have to vendor it. A `gestcam` AUR package that depends on
   `v4l2loopback-dkms` and installs the PyPI wheel into `/opt/gestcam` is the realistic
   option. Do this only if people ask for it.

## Out of scope for now

- **macOS** already has Reactions built in, which is the reason this project exists.
- **Windows** is possible later: pyvirtualcam supports the OBS virtual camera there. It
  needs a sink choice in `cli.make_sink`, plus a Windows CI job.
- **Docker** doesn't fit: the container would need `/dev/video*`, the host module and a
  display for `--preview`, which adds steps compared to a plain install.
