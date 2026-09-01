#!/usr/bin/env bash
# Install + load v4l2loopback the way Chromium-based apps (Discord web app, Chrome,
# Electron) need it: exclusive_caps=1, fixed device numbers, persistent across reboots.
# Loads TWO devices so two apps can stream GestCam at the same time.
# Arch/Omarchy first; Debian/Ubuntu and Fedora handled best-effort.
set -euo pipefail

# One v4l2loopback device serves exactly ONE streaming consumer: 0.15.4's
# vidioc_reqbufs returns -EBUSY to the second capture client, and max_openers=10
# gates open() only — which is why the second app opens the device and then just
# goes black. Two devices = native Discord on one, Discord-in-a-browser on the other.
#
# video_nr / card_label / exclusive_caps are per-device ARRAYS: a bare
# `exclusive_caps=1` with two devices leaves the second at N and Chromium then
# refuses to list it. `devices=` is inferred from video_nr, so it is not passed.
# No space in a label: the kernel splits the module cmdline on whitespace before
# the charp array splits on commas, so "GestCam 2" would persist as label
# "GestCam,GestCam" plus an unknown parameter "2".
#
# Tuning knob: MODOPTS — add a third video_nr/label/1 to serve a third app.
read -ra OPTS <<<"${MODOPTS:-video_nr=10,11 card_label=GestCam,GestCam2 exclusive_caps=1,1}"

. /etc/os-release
case "${ID} ${ID_LIKE:-}" in
  *arch*)
    KERNEL_PKG=$(pacman -Qq linux linux-zen linux-lts linux-hardened 2>/dev/null | head -n1 || true)
    HEADERS="${KERNEL_PKG:-linux}-headers"
    sudo pacman -S --needed "${HEADERS}" v4l2loopback-dkms v4l-utils ;;
  *debian*|*ubuntu*)
    sudo apt install -y v4l2loopback-dkms v4l-utils ;;
  *fedora*)
    sudo dnf install -y v4l2loopback v4l-utils ;;
  *) echo "unknown distro: install v4l2loopback-dkms + kernel headers manually"; ;;
esac

echo "options v4l2loopback ${OPTS[*]}" | sudo tee /etc/modprobe.d/v4l2loopback.conf >/dev/null
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf >/dev/null
sudo modprobe -r v4l2loopback 2>/dev/null || true
sudo modprobe v4l2loopback "${OPTS[@]}"

# `modprobe -r` fails silently while any app holds a device, leaving the OLD params
# loaded. Count what actually came up rather than trusting the exit status.
mapfile -t DEVS < <(grep -l '^GestCam' /sys/class/video4linux/*/name 2>/dev/null \
                    | sed 's|/sys/class/video4linux/|/dev/|; s|/name$||')
if [ "${#DEVS[@]}" -lt 2 ]; then
  echo "WARNING: only ${#DEVS[@]} GestCam device(s) came up — close every app using"
  echo "         GestCam (and stop gestcam itself), then run this again."
fi
echo
echo "exclusive_caps: $(cat /sys/module/v4l2loopback/parameters/exclusive_caps 2>/dev/null)"
for d in "${DEVS[@]}"; do
  echo "device: ${d}   (watch it with: mpv av://v4l2:${d})"
done
echo
echo "gestcam publishes the same frames to every one of these; point a different app"
echo "at each. They only appear in a camera picker while gestcam is running."
