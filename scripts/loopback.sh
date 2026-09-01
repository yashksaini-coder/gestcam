#!/usr/bin/env bash
# Install + load v4l2loopback the way Chromium-based apps (Discord web app, Chrome,
# Electron) need it: exclusive_caps=1, fixed /dev/video10, persistent across reboots.
# Arch/Omarchy first; Debian/Ubuntu and Fedora handled best-effort.
set -euo pipefail

NR="${VIDEO_NR:-10}"
LABEL="${CARD_LABEL:-GestCam}"
OPTS="devices=1 video_nr=${NR} card_label=\"${LABEL}\" exclusive_caps=1"

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

echo "options v4l2loopback ${OPTS}" | sudo tee /etc/modprobe.d/v4l2loopback.conf >/dev/null
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf >/dev/null
sudo modprobe -r v4l2loopback 2>/dev/null || true
eval sudo modprobe v4l2loopback ${OPTS}

echo
echo "loaded: $(cat /sys/module/v4l2loopback/parameters/exclusive_caps 2>/dev/null | sed 's/^/exclusive_caps=/')"
v4l2-ctl --list-devices 2>/dev/null | grep -A2 -i "${LABEL}" || ls -l /dev/video${NR}
echo "device: /dev/video${NR}   (test later with: mpv av://v4l2:/dev/video${NR})"
