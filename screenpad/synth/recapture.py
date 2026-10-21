"""
Physical screen-recapture simulator.
------------------------------------

Turns a bona fide image into what a phone photo of it shown on a screen would look like, by simulating each stage of the real process rather than painting artefacts on:

1. Display: the image is resized to the screen's resolution, passed through the panel's tone response, gamut and white point, and emitted through a sub-pixel layout (RGB/BGR stripes or an RGBG diamond, as on OLED phones), with dark gaps between sub-pixels and a backlight black level.
2. Environment: optional specular glare, a uniform reflected-light veil and dust specks on the glass.
3. Geometry and optics: the phone views the screen through a mild perspective tilt and in-plane rotation. Each camera pixel integrates the sub-pixel pattern over its footprint (supersampled, then box-averaged), then lens blur. Moiré is not drawn: it appears by itself where the sub-pixel grid and the camera grid interfere.
4. Camera: auto-exposure, white-balance error, Bayer mosaic with signal-dependent noise, demosaicing, denoising, tone curve, sharpening and JPEG compression.

Ranges in `configs/synth.yaml` were set from the 6 dev twin pairs (notebooks/Recapture.ipynb): screen-pixel periods of 3-7 camera pixels, 2-3 camera pixels per source pixel, lifted blacks and compressed whites, per-pair colour casts in both directions, lower fine detail than the source, and sometimes dust. They are deliberately wider than those six pairs.

`camera_only` runs stage 4 alone on a bona fide image (no screen). Training on such images as bona fide stops "has been through a phone camera" from being a cue for "attack": then only the screen stages separate the classes.

All parameters for one image come from one `np.random.Generator`, so a (seed, image, variant) triple always reproduces the same output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

def _u(rng: np.random.Generator, lo_hi) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def _logu(rng: np.random.Generator, lo_hi) -> float:
    lo, hi = lo_hi
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


@dataclass
class CameraParams:
    exposure: float  # target 98th-percentile level after auto-exposure
    wb: tuple[float, float, float]  # white-balance error gains (G fixed at 1)
    shot: float  # signal-dependent noise variance per unit signal
    read: float  # signal-independent noise variance
    demosaic: int  # cv2 demosaicing code (OpenCV names Bayer patterns from the second row: BayerBG means RGGB)
    denoise: float  # Gaussian sigma (px), 0 = off
    contrast: float  # S-curve strength in the encoded domain
    sharpen_amount: float
    sharpen_radius: float
    jpeg_quality: int

    @classmethod
    def sample(cls, rng: np.random.Generator, r: dict) -> "CameraParams":
        wb = (_u(rng, r["wb"]), 1.0, _u(rng, r["wb"]))
        return cls(
            exposure=_u(rng, r["exposure"]),
            wb=wb,
            shot=_logu(rng, r["shot"]),
            read=_logu(rng, r["read"]),
            demosaic=int(rng.choice([cv2.COLOR_BayerBG2RGB, cv2.COLOR_BayerBG2RGB_EA])),
            denoise=_u(rng, r["denoise"]) if rng.random() < r["p_denoise"] else 0.0,
            contrast=_u(rng, r["contrast"]),
            sharpen_amount=_u(rng, r["sharpen_amount"]),
            sharpen_radius=_u(rng, r["sharpen_radius"]),
            jpeg_quality=int(rng.integers(r["jpeg_quality"][0], r["jpeg_quality"][1] + 1)),
        )


@dataclass
class RecaptureParams:
    layout: str
    screen_per_src: float  # screen pixels per source pixel
    cam_per_src: float  # camera pixels per source pixel (sets the screen-pixel period on the sensor)
    fill: float  # fraction of each sub-pixel slot that emits
    display_gamma: float  # panel gamma relative to sRGB's ~2.2
    saturation: float
    white_point: tuple[float, float, float]
    black_level: float  # backlight leak, linear, relative to peak white
    veil: float  # uniform reflected ambient light, linear
    glare: float  # peak of a soft specular highlight, linear (0 = none)
    glare_xy: tuple[float, float]
    glare_sigma: float  # fraction of the frame width
    dust: int  # number of dust specks
    tilt: float  # perspective: max corner displacement as a fraction of the frame
    rotation: float  # in-plane camera rotation, degrees
    blur: float  # lens blur sigma, camera px
    vignette: float
    camera: CameraParams

    @classmethod
    def sample(cls, rng: np.random.Generator, r: dict) -> "RecaptureParams":
        return cls(
            layout=str(rng.choice(r["layouts"])),
            screen_per_src=_logu(rng, r["screen_per_src"]),
            cam_per_src=_logu(rng, r["cam_per_src"]),
            fill=_u(rng, r["fill"]),
            display_gamma=_u(rng, r["display_gamma"]),
            saturation=_u(rng, r["saturation"]),
            white_point=(_u(rng, r["white_point"]), 1.0, _u(rng, r["white_point"])),
            black_level=_logu(rng, r["black_level"]),
            veil=_u(rng, r["veil"]),
            glare=_u(rng, r["glare"]) if rng.random() < r["p_glare"] else 0.0,
            glare_xy=(float(rng.uniform(0, 1)), float(rng.uniform(0, 1))),
            glare_sigma=_u(rng, r["glare_sigma"]),
            dust=int(rng.integers(1, r["max_dust"] + 1)) if rng.random() < r["p_dust"] else 0,
            tilt=_u(rng, r["tilt"]),
            rotation=_u(rng, r["rotation"]),
            blur=_u(rng, r["blur"]),
            vignette=_u(rng, r["vignette"]),
            camera=CameraParams.sample(rng, r["camera"]),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        cam = d.pop("camera")
        return {**d, **{f"camera_{k}": v for k, v in cam.items()}}


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def _display(src: np.ndarray, p: RecaptureParams) -> np.ndarray:
    """Source uint8 RGB -> linear emitted framebuffer at screen resolution (before sub-pixel structure)."""
    h, w = src.shape[:2]
    dw, dh = max(8, round(w * p.screen_per_src)), max(8, round(h * p.screen_per_src))
    interp = cv2.INTER_AREA if p.screen_per_src < 1 else cv2.INTER_CUBIC
    fb = cv2.resize(src, (dw, dh), interpolation=interp).astype(np.float32) / 255
    lin = srgb_to_linear(fb) ** (p.display_gamma / 2.2)
    lum = lin @ np.float32([0.2126, 0.7152, 0.0722])
    lin = lum[..., None] + p.saturation * (lin - lum[..., None])
    lin = np.clip(lin * np.float32(p.white_point), 0, None)
    return p.black_level + (1 - p.black_level) * lin


def _subpixel_masks(u: np.ndarray, v: np.ndarray, iu: np.ndarray, iv: np.ndarray, p: RecaptureParams) -> np.ndarray:
    """Emission weight of each channel at screen coordinates (u, v), normalised so a uniform field keeps its mean."""
    fu, fv = u - iu, v - iv
    gap = (1 - p.fill) / 2
    out = np.zeros(u.shape + (3,), np.float32)
    if p.layout in ("rgb_stripe", "bgr_stripe"):
        order = (0, 1, 2) if p.layout == "rgb_stripe" else (2, 1, 0)
        row = (fv > gap * 0.5) & (fv < 1 - gap * 0.5)  # thinner horizontal gaps than vertical
        for slot, c in enumerate(order):
            lo = slot / 3 + gap / 3
            out[..., c] = (fu >= lo) & (fu < lo + p.fill / 3) & row
        out *= 3 / (p.fill * (1 - gap * 0.5 * 2))  # each channel emits from fill/3 of the width and (1 - gap) of the height
    else:
        # RGBG diamond: a small green dot in every pixel; red and blue dots alternate on a checkerboard
        rg = 0.18 + 0.1 * p.fill
        rb = 0.28 + 0.12 * p.fill
        green = (fu - 0.5) ** 2 + (fv - 0.5) ** 2 < rg**2
        corner = np.minimum(fu, 1 - fu) ** 2 + np.minimum(fv, 1 - fv) ** 2 < rb**2
        checker = ((iu + iv) % 2).astype(bool)
        out[..., 1] = green
        out[..., 0] = corner & ~checker
        out[..., 2] = corner & checker
        out[..., 1] /= np.pi * rg**2
        out[..., 0] /= np.pi * rb**2 / 2  # each corner dot is shared by four pixels, half of them red
        out[..., 2] /= np.pi * rb**2 / 2
    return out


def _camera_homography(cw: int, ch: int, sw: int, sh: int, p: RecaptureParams, rng: np.random.Generator) -> np.ndarray:
    """Map camera pixel coordinates to screen pixel coordinates: scale, small rotation and perspective.

    The frame is not zoomed in to stay on the screen: like the real attack files (cropped by hand to about the displayed image), a tilted frame shows thin slivers of the dark screen border at some corners. Zooming in instead would shrink the margin around the face and make synthetic loose crops more tightly framed than any real attack.
    """
    corners = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    d = rng.uniform(-1, 1, (4, 2)).astype(np.float32) * p.tilt * np.float32([sw, sh])
    c = corners.mean(0)
    t = np.radians(p.rotation)
    R = np.float32([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    quad = (corners - c) @ R.T + c + d
    cam = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    return cv2.getPerspectiveTransform(cam, quad)


def _environment(lin: np.ndarray, p: RecaptureParams, rng: np.random.Generator) -> np.ndarray:
    h, w = lin.shape[:2]
    out = lin + p.veil
    if p.glare > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gx, gy, s = p.glare_xy[0] * w, p.glare_xy[1] * h, p.glare_sigma * w
        out = out + p.glare * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * s * s))[..., None]
    for _ in range(p.dust):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = max(1, int(rng.uniform(0.5, 2.5) * w / 1000))
        cv2.circle(out, (x, y), r, (float(rng.uniform(0.3, 1.0)),) * 3, -1, lineType=cv2.LINE_AA)
    if p.vignette > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        out = out * (1 - p.vignette * r2)[..., None]
    return out


def render_screen(src: np.ndarray, p: RecaptureParams, rng: np.random.Generator) -> np.ndarray:
    """Linear light reaching the sensor (before the camera pipeline), at camera resolution."""
    fb = _display(src, p)
    sh, sw = fb.shape[:2]
    h, w = src.shape[:2]
    cw, ch = max(16, round(w * p.cam_per_src)), max(16, round(h * p.cam_per_src))
    cam_per_screen = p.cam_per_src / p.screen_per_src
    ss = int(np.clip(np.ceil(6 / cam_per_screen), 1, 4))  # resolve sub-pixels: >= ~2 samples per stripe
    H = _camera_homography(cw, ch, sw, sh, p, rng)
    # supersampled camera sample centres -> screen coordinates
    xs = (np.arange(cw * ss, dtype=np.float32) + 0.5) / ss
    ys = (np.arange(ch * ss, dtype=np.float32) + 0.5) / ss
    out = np.empty((ch, cw, 3), np.float32)
    rows = max(1, (4_000_000 // (cw * ss * ss)))  # process in bands to bound memory
    for y0 in range(0, ch, rows):
        y1 = min(ch, y0 + rows)
        X, Y = np.meshgrid(xs, ys[y0 * ss : y1 * ss])
        den = H[2, 0] * X + H[2, 1] * Y + H[2, 2]
        u = (H[0, 0] * X + H[0, 1] * Y + H[0, 2]) / den
        v = (H[1, 0] * X + H[1, 1] * Y + H[1, 2]) / den
        iu, iv = np.floor(u), np.floor(v)
        px = cv2.remap(fb, iu.astype(np.float32), iv.astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))  # off-screen: black border
        em = px * _subpixel_masks(u, v, iu, iv, p)
        out[y0:y1] = cv2.resize(em, (cw, y1 - y0), interpolation=cv2.INTER_AREA) if ss > 1 else em
    out = _environment(out, p, rng)
    if p.blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), p.blur)
    return out


def camera(lin: np.ndarray, c: CameraParams, rng: np.random.Generator) -> np.ndarray:
    """Phone camera pipeline: linear scene light -> processed uint8 RGB (after JPEG)."""
    x = lin * np.float32(c.wb)
    x = x * (c.exposure / max(float(np.percentile(x, 98)), 1e-6))
    h, w = x.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    x = np.clip(x[:h2, :w2], 0, 1)
    bayer = np.empty((h2, w2), np.float32)  # RGGB
    bayer[0::2, 0::2] = x[0::2, 0::2, 0]
    bayer[0::2, 1::2] = x[0::2, 1::2, 1]
    bayer[1::2, 0::2] = x[1::2, 0::2, 1]
    bayer[1::2, 1::2] = x[1::2, 1::2, 2]
    bayer += rng.standard_normal(bayer.shape).astype(np.float32) * np.sqrt(c.shot * bayer + c.read)
    raw = (np.clip(bayer, 0, 1) * 65535).astype(np.uint16)
    rgb = cv2.demosaicing(raw, c.demosaic).astype(np.float32) / 65535
    if c.denoise > 0:
        rgb = cv2.GaussianBlur(rgb, (0, 0), c.denoise)
    enc = linear_to_srgb(rgb)
    if c.contrast > 0:  # smoothstep blend: steeper mid-tones, rolled-off ends
        enc = (1 - c.contrast) * enc + c.contrast * enc * enc * (3 - 2 * enc)
    if c.sharpen_amount > 0:
        enc = enc + c.sharpen_amount * (enc - cv2.GaussianBlur(enc, (0, 0), c.sharpen_radius))
    out = (np.clip(enc, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, c.jpeg_quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy()


def recapture(src: np.ndarray, p: RecaptureParams, rng: np.random.Generator) -> np.ndarray:
    """Simulated phone photo of `src` (uint8 RGB) shown on a screen. Output is uint8 RGB at camera resolution."""
    return camera(render_screen(src, p, rng), p.camera, rng)


def camera_only(src: np.ndarray, c: CameraParams, cam_per_src: float, rng: np.random.Generator) -> np.ndarray:
    """Bona fide control: `src` resampled to camera resolution and passed through the same camera pipeline, with no screen."""
    h, w = src.shape[:2]
    size = (max(16, round(w * cam_per_src)), max(16, round(h * cam_per_src)))
    up = cv2.resize(src, size, interpolation=cv2.INTER_AREA if cam_per_src < 1 else cv2.INTER_CUBIC)
    return camera(srgb_to_linear(up.astype(np.float32) / 255), c, rng)
