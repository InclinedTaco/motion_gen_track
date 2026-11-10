"""Stitch two GIFs horizontally into one side-by-side GIF with labels."""
import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_frames(path):
    rdr = imageio.get_reader(path)
    return [np.asarray(Image.fromarray(f).convert("RGB")) for f in rdr]


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label(frame, text, bar_h=30, font_size=16):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, bar_h], fill=(255, 255, 255))
    font = _font(font_size)
    tw = d.textlength(text, font=font)
    x = max(4, (img.width - tw) // 2)
    y = (bar_h - font_size) // 2
    d.text((x, y), text, fill=(0, 0, 0), font=font)
    return np.asarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()

    fa, fb = load_frames(args.a), load_frames(args.b)
    # Hold the last frame of the shorter clip so both play full-length and
    # neither truncates; the shorter one freezes on its final pose at the end.
    n = max(len(fa), len(fb))
    fa += [fa[-1]] * (n - len(fa))
    fb += [fb[-1]] * (n - len(fb))
    h = min(fa[0].shape[0], fb[0].shape[0])
    out = []
    for i in range(n):
        la = label(np.asarray(Image.fromarray(fa[i]).resize(
            (int(fa[i].shape[1] * h / fa[i].shape[0]), h))), args.label_a)
        lb = label(np.asarray(Image.fromarray(fb[i]).resize(
            (int(fb[i].shape[1] * h / fb[i].shape[0]), h))), args.label_b)
        out.append(np.concatenate([la, lb], axis=1))
    from PIL import Image as PILImage
    pil = [PILImage.fromarray(f) for f in out]
    duration_ms = int(1000.0 / args.fps)
    pil[0].save(
        args.out,
        save_all=True,
        append_images=pil[1:],
        duration=duration_ms,
        loop=0,  # loop forever
    )
    print(f"Saved -> {args.out} ({n} frames)")


if __name__ == "__main__":
    main()
