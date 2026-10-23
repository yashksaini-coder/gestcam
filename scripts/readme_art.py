"""Regenerate the README images in docs/img/ from the real effect engine.

    uv run --with pillow python scripts/readme_art.py

Every effect pixel comes from gestcam's own EffectEngine on the synthetic demo scene,
so the pictures can't drift from what the app draws. Pillow is only needed here (text
and emoji), which is why it's `--with`, not a dependency.
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from gestcam.effects import EffectEngine
from gestcam.pipeline import SHEET_AT, synthetic_frame
from gestcam.sprites import load_sprites

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "img"
W, H = 1280, 720
FONT = "/usr/share/fonts/noto/NotoSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/noto/NotoSans-Medium.ttf"
EMOJI = "/usr/share/fonts/noto/NotoColorEmoji.ttf"
BG = (13, 15, 22)
# Moments that read best as a still; others use the demo sheet's SHEET_AT.
STILL_AT = {"fireworks": 1.4}
ACCENT = (255, 196, 92)

# reaction → (gesture glyphs, caption); README order: one-hand first, then two-hand
GESTURES = {
    "thumbs_up": ("👍", "thumbs up"),
    "thumbs_down": ("👎", "thumbs down"),
    "balloons": ("✌️", "balloons"),
    "hearts": ("🫶", "hearts"),
    "fireworks": ("👍👍", "fireworks"),
    "rain": ("👎👎", "rain"),
    "confetti": ("✌️✌️", "confetti"),
    "lasers": ("🤘🤘", "lasers"),
}


def render(reaction: str, at: float | None = None, seed: int = 7) -> np.ndarray:
    """One BGR frame of `reaction`, `at` seconds after it fired, on the demo scene."""
    at = STILL_AT.get(reaction, SHEET_AT[reaction]) if at is None else at
    engine = EffectEngine(W, H, sprites=load_sprites(log=lambda *a: None), seed=seed)
    rng = np.random.default_rng(seed)
    engine.trigger(reaction, [(W * 0.42, H * 0.6), (W * 0.58, H * 0.6)])
    t, dt = 0.0, 1 / 30
    while True:
        frame, mask = synthetic_frame(W, H, t, rng)
        engine.step(frame, dt, mask=mask)
        t += dt
        if t >= at:
            return frame


def pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def emoji(text: str, size: int) -> Image.Image:
    """Noto Color Emoji only renders at 109 px; draw there, then scale."""
    f = ImageFont.truetype(EMOJI, 109)
    box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=f, embedded_color=True)
    im = Image.new("RGBA", (box[2] - box[0] + 8, box[3] - box[1] + 8))
    ImageDraw.Draw(im).text((4 - box[0], 4 - box[1]), text, font=f, embedded_color=True)
    return im.resize((round(im.width * size / 109), round(im.height * size / 109)), Image.LANCZOS)


def rounded(im: Image.Image, r: int) -> Image.Image:
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, *im.size), r, fill=255)
    out = im.convert("RGBA")
    out.putalpha(m)
    return out


def shadowed(canvas: Image.Image, im: Image.Image, xy: tuple[int, int], blur: int = 18) -> None:
    sh = Image.new("RGBA", (im.width + 4 * blur, im.height + 4 * blur), (0, 0, 0, 0))
    sh.paste((0, 0, 0, 170), (2 * blur, 2 * blur + 8), im.split()[3])
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(sh, (xy[0] - 2 * blur, xy[1] - 2 * blur))
    canvas.alpha_composite(im, xy)


def gradient(w: int, h: int) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    glow = np.exp(-(((xx - w * 0.78) / (w * 0.45)) ** 2 + ((yy - h * 0.4) / (h * 0.9)) ** 2))
    base = np.array(BG, np.float32)
    tint = np.array((70, 52, 110), np.float32)  # violet bloom behind the tiles
    img = base + glow[..., None] * tint * 0.55
    return Image.fromarray(img.clip(0, 255).astype(np.uint8)).convert("RGBA")


def banner() -> None:
    bw, bh = 1600, 560
    c = gradient(bw, bh)
    d = ImageDraw.Draw(c)
    d.text((80, 150), "gestcam", font=ImageFont.truetype(FONT, 118), fill=(245, 246, 250))
    d.text((84, 300), "Apple-style gesture reactions", font=ImageFont.truetype(FONT_REG, 38), fill=(205, 208, 220))
    d.text((84, 348), "for every video call on Linux", font=ImageFont.truetype(FONT_REG, 38), fill=(205, 208, 220))
    x = 84
    for g in ("👍", "✌️", "🫶", "🤘"):
        e = emoji(g, 46)
        d.rounded_rectangle((x, 420, x + 74, 494), 18, fill=(32, 35, 48))
        c.alpha_composite(e, (x + (74 - e.width) // 2, 420 + (74 - e.height) // 2))
        x += 88
    tw, th = 400, 225
    picks = [("fireworks", 900, 50), ("hearts", 1150, 150), ("lasers", 860, 300), ("balloons", 1120, 330)]
    for r, px, py in picks:
        tile = rounded(pil(cv2.resize(render(r), (tw, th), interpolation=cv2.INTER_AREA)), 22)
        shadowed(c, tile, (px, py))
    c.convert("RGB").save(OUT / "banner.png", optimize=True)


def gestures() -> None:
    tw, th, pad, head = 420, 236, 22, 64
    cols, rows = 4, 2
    gw = cols * tw + (cols + 1) * pad
    gh = rows * (th + head) + (rows + 1) * pad
    c = Image.new("RGBA", (gw, gh), BG + (255,))
    d = ImageDraw.Draw(c)
    f = ImageFont.truetype(FONT, 28)
    for i, (r, (g, cap)) in enumerate(GESTURES.items()):
        x = pad + (i % cols) * (tw + pad)
        y = pad + (i // cols) * (th + head + pad)
        d.rounded_rectangle((x, y, x + tw, y + th + head), 20, fill=(26, 28, 38))
        e = emoji(g, 40)
        c.alpha_composite(e, (x + 18, y + (head - e.height) // 2))
        d.text((x + 30 + e.width, y + head // 2), cap, font=f, fill=(235, 236, 242), anchor="lm")
        tile = rounded(pil(cv2.resize(render(r), (tw - 16, th - 8), interpolation=cv2.INTER_AREA)), 14)
        c.alpha_composite(tile, (x + 8, y + head))
    c.convert("RGB").save(OUT / "gestures.png", optimize=True)


def demo_gif() -> None:
    """gestcam demo → mp4 → palette-optimised gif (ffmpeg), small enough for a README."""
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = pathlib.Path(tmp) / "demo.mp4"
        subprocess.run(["gestcam", "demo", "--out", str(mp4), "--seconds", "12"], check=True)
        vf = "fps=10,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-vf", vf, str(OUT / "demo.gif")], check=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    banner()
    gestures()
    demo_gif()
    for p in sorted(OUT.iterdir()):
        print(f"{p}  {p.stat().st_size / 1e3:.0f} kB")
