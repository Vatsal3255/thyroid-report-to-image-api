from PIL import Image, ImageDraw, ImageFont
import re
import numpy as np
import cv2
from typing import Optional, Tuple
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_IMAGE_PATH = Path(__file__).resolve().parent / "static" / "Base Image.png"
OUTPUT_IMAGE_PATH = "thyroid_report_output_final.png"

NODULE_SCALE = 1.5

# -----------------------------
# Canvas padding
# -----------------------------
PAD_LEFT   = 180
PAD_RIGHT  = 180
PAD_TOP    = 120
PAD_BOTTOM = 160

# Distance from lobe edge to the callout rail
RAIL_GAP = 70
MIN_RAIL_MARGIN = 40

# -----------------------------
# Sizes / styles
# -----------------------------
PX_PER_MM = 3.0  # ~30 px per cm

# Final palette per your spec (RGB)
TR_RGB = {
    1: (114, 252, 146),  # green
    2: (176, 255, 196),  # light green
    3: (255, 190, 102),  # light orange / yellow
    4: (255, 232, 28),  # warning yellow
    5: (250, 5, 5),  # dangerous red
}

CALLOUT_FILL   = (255, 255, 255, 255)
CALLOUT_BORDER = (0, 0, 0, 255)
CALLOUT_SHADOW = (0, 0, 0, 55)
LEADER         = (0, 0, 0, 255)

LABEL_FILL     = (255,255,255,255)
LABEL_BORDER   = (0,0,0,255)

EDGE_FEATHER_PX = 2.0
SUPERSAMPLE     = 2

_TIRADS_STYLE = {
    1: dict(aspect=1.00, jaggedness=0.06, lobes=0),
    2: dict(aspect=1.00, jaggedness=0.09, lobes=1),
    3: dict(aspect=1.05, jaggedness=0.12, lobes=2),
    4: dict(aspect=0.85, jaggedness=0.18, lobes=3),
    5: dict(aspect=0.75, jaggedness=0.24, lobes=4),
}

def parse_coords(s: str):
    pts = list(map(int, [p.strip() for p in s.split(",") if p.strip() != ""]))
    return [(pts[i], pts[i+1]) for i in range(0, len(pts), 2)]

AREAS_ORIG = {
    "left upper": parse_coords("671,372,675,365,678,356,681,346,685,335,689,322,695,307,702,293,711,277,718,260,728,248,734,240,738,236,743,234,749,230,756,228,764,230,772,232,780,238,789,243,797,251,802,262,808,273,816,288,825,310,830,328,835,343,841,362,843,373"),
    "left upper middle": parse_coords("670,374,665,382,657,389,650,401,639,415,630,423,625,430,857,454,855,438,854,429,852,413,848,401,847,392,846,375"),
    "left middle": parse_coords("623,432,612,439,606,445,597,451,590,454,583,609,841,651,847,619,851,590,855,564,856,545,856,524,858,501,857,483,856,468,856,457"),
    "left lower middle": parse_coords("584,612,583,679,809,734,818,719,824,702,830,687,836,672,840,654"),
    "left lower": parse_coords("584,682,583,709,594,720,601,731,609,743,617,753,624,766,630,774,642,790,651,800,663,809,678,818,697,824,713,824,726,818,746,811,770,792,784,774,797,757,809,736"),
    "Isthmus": parse_coords("587,454,571,460,560,464,552,467,545,471,540,479,532,486,526,492,518,495,509,499,500,495,491,490,483,482,471,476,456,470,428,456,435,711,456,697,469,688,477,683,484,674,491,668,497,661,502,656,508,651,513,654,517,659,523,666,530,674,538,682,547,689,556,695,564,698,570,701,581,707"),
    "right upper": parse_coords("175,394,184,357,190,338,197,317,204,296,218,267,224,255,237,241,250,233,262,231,274,231,285,234,293,242,298,250,304,258,308,270,320,294,327,316,335,329,345,354,351,369,358,380,363,389"),
    "right upper middle": parse_coords("175,397,171,410,169,426,170,448,166,472,166,483,425,459,416,445,406,440,396,430,389,422,380,414,372,402,364,392"),
    "right middle": parse_coords("167,486,166,508,167,531,169,552,174,593,178,620,182,638,432,620,430,592,426,462"),
    "right lower middle": parse_coords("182,639,192,673,201,698,207,716,432,700,432,621"),
    "right lower": parse_coords("209,718,215,733,225,750,234,766,240,774,250,787,258,796,270,804,285,815,299,820,316,822,335,822,350,816,367,807,379,796,386,788,393,777,402,761,413,747,418,735,424,726,434,714,432,701"),
}
LEFT_KEYS  = [k for k in AREAS_ORIG if k.startswith("left")]
RIGHT_KEYS = [k for k in AREAS_ORIG if k.startswith("right")]

def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except:
        return ImageFont.load_default()

# def font(size: int):
#     try:
#         return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
#     except:
#         try:
#             return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
#         except:
#             return ImageFont.load_default()

# -----------------------------
# Geometry helpers
# -----------------------------

def shift_polygon(poly, dx, dy):
    return [(x+dx, y+dy) for (x,y) in poly]

def centroid(poly):
    x = [p[0] for p in poly]; y = [p[1] for p in poly]; n=len(poly)
    a=cx=cy=0.0
    for i in range(n):
        j=(i+1)%n
        cross=x[i]*y[j]-x[j]*y[i]
        a+=cross
        cx+=(x[i]+x[j])*cross
        cy+=(y[i]+y[j])*cross
    a*=0.5
    if abs(a)<1e-6: return (sum(x)/n, sum(y)/n)
    return (cx/(6*a), cy/(6*a))

def bbox(poly):
    xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)

def parse_size_to_mm(s: str):
    s=(s or "").lower().strip()
    nums=[float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if "cm" in s: nums=[n*10 for n in nums]
    if not nums: return (12.0,12.0)
    return (nums[0], nums[0] if len(nums)==1 else nums[1])

def union_bbox(polys):
    xs=[]; ys=[]
    for p in polys:
        xs += [pt[0] for pt in p]
        ys += [pt[1] for pt in p]
    return min(xs), min(ys), max(xs), max(ys)

# -----------------------------
# Region normalization / inference (KEY FIX)
# -----------------------------
def _normalize_region(region_text: Optional[str]) -> str:
    """
    Normalize free-text region into one of:
    'upper', 'upper middle', 'middle', 'lower middle', 'lower', 'isthmus', ''.
    Handles inputs like 'upper lobe', 'mid lobe', 'lower-middle', etc.
    """
    t = (region_text or "").lower().strip()
    t = re.sub(r"\blobe\b", "", t)       # strip 'lobe'
    t = t.replace("-", " ")              # hyphens -> spaces
    t = re.sub(r"\s+", " ", t).strip()

    # --- Compound phrases FIRST (order matters) ---
    if t in ("lower middle", "lower mid", "lowermiddle", "lowermid", "mid lower", "middle lower"):
        return "lower middle"
    if t in ("upper middle", "upper mid", "uppermiddle", "uppermid", "mid upper", "middle upper"):
        return "upper middle"

    if "isthmus" in t:
        return "isthmus"

    # --- Singles / synonyms ---
    if t in ("upper", "superior"):
        return "upper"
    if t in ("mid", "middle"):
        return "middle"
    if t in ("lower", "inferior"):
        return "lower"

    # --- Fallback contains checks (keep compound handled above) ---
    if "lower middle" in t or "middle lower" in t:
        return "lower middle"
    if "upper middle" in t or "middle upper" in t:
        return "upper middle"
    if "upper" in t or "superior" in t:
        return "upper"
    if "middle" in t or "mid" in t:
        return "middle"
    if "lower" in t or "inferior" in t:
        return "lower"

    return ""

def _infer_region_from_text(s: Optional[str]) -> str:
    """Fallback: look inside the lobe string, e.g. 'Right lower lobe'."""
    return _normalize_region(s)

def make_lowfreq_texture(shape_hw, k=15, sigma=3.0, seed=None):
    h, w = shape_hw
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    k = max(3, int(k) | 1)
    n = cv2.GaussianBlur(n, (k, k), sigma)
    n -= n.min()
    n /= (n.max() + 1e-6)
    return n

def _smooth_noise_1d(n_points: int, smooth_sigma: float = 6.0, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n_points).astype(np.float32)
    noise = np.concatenate([noise, noise, noise])
    noise = cv2.GaussianBlur(noise[:, None], (0, 0), smooth_sigma,
                             borderType=cv2.BORDER_REFLECT101)[:, 0]
    noise = noise[n_points:2*n_points]
    noise /= (np.max(np.abs(noise)) + 1e-6)
    return noise

def _irregular_contour(center: Tuple[int, int], size_px: float, *,
                       aspect: float, jaggedness: float,
                       seed: Optional[int], n_points: int = 256, lobes: int = 0):
    cx, cy = center
    R_major = size_px / 2.0
    R_minor = max(4.0, R_major * aspect)
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    r_perturb = _smooth_noise_1d(n_points, 5.0, seed) * jaggedness
    if lobes > 0:
        rng = np.random.default_rng(None if seed is None else seed + 913)
        for _ in range(lobes):
            angle = rng.uniform(0, 2*np.pi)
            width = rng.uniform(np.pi/32, np.pi/12)
            amp = rng.uniform(0.08, 0.18)
            ang_diff = np.angle(np.exp(1j*(theta - angle)))
            r_perturb += amp * np.exp(-0.5 * (ang_diff / width)**2)
    base_r = (R_major * R_minor) / np.sqrt((R_minor*np.cos(theta))**2 + (R_major*np.sin(theta))**2 + 1e-6)
    r = base_r * (1.0 + r_perturb)
    xs = cx + r * np.cos(theta)
    ys = cy + r * np.sin(theta)
    return np.stack([xs, ys], axis=1).astype(np.int32)

def _soft_mask_from_contour(shape_hw, contour_xy, feather_px=3.0):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour_xy.reshape(-1,1,2)], 255)
    if feather_px > 0:
        dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dist_inside = np.clip(dist_inside / float(feather_px), 0, 1)
        mask = (dist_inside * 255).astype(np.uint8)
    return mask

def _parse_tr_category(score: Optional[str]) -> int:
    if not score:
        return 3
    s = score.lower()
    s = s.replace("ti-rads","").replace("tirads","").replace("tr","")
    s = re.sub(r"[^0-9]", "", s)
    try:
        n = int(s or "3")
    except:
        n = 3
    return max(1, min(5, n))

def _render_nodule_rgba(category: int, size_major: float, *,
                        aspect: float, jaggedness: float, lobes: int,
                        seed: int, color_rgb: Tuple[int,int,int],
                        add_dots: bool) -> Image.Image:
    pad = int(size_major * 0.35)
    H = W = int((size_major + 2*pad) * SUPERSAMPLE)
    cx = cy = W // 2

    contour_xy = _irregular_contour((cx, cy), size_major*SUPERSAMPLE,
                                    aspect=aspect, jaggedness=jaggedness,
                                    seed=seed, n_points=256, lobes=lobes)
    mask = _soft_mask_from_contour((H, W), contour_xy, feather_px=EDGE_FEATHER_PX*SUPERSAMPLE)
    m = (mask.astype(np.float32) / 255.0)[..., None]

    tex_low  = make_lowfreq_texture((H, W), k=19, sigma=2.5, seed=seed)
    tex_high = make_lowfreq_texture((H, W), k=5,  sigma=0.9, seed=seed+31)
    texture  = 0.80 + 0.30*(tex_low - 0.5) + 0.15*(tex_high - 0.5)
    texture  = np.clip(texture, 0.0, 1.2)

    yy, xx = np.indices((H, W), dtype=np.float32)
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    dist /= dist.max() + 1e-6
    shading = 1.1 - 0.3*dist
    texture *= shading

    col = np.array(color_rgb, dtype=np.float32)[None, None, :] / 255.0
    gray = np.mean(col)
    col = col * 0.7 + gray * 0.3
    lesion = np.clip(col * texture[..., None] * 255.0, 0, 255).astype(np.float32)

    rng = np.random.default_rng(seed+999)
    grain = rng.normal(0, 4, (H, W, 1)).astype(np.float32)
    lesion = np.clip(lesion + grain, 0, 255)

    out_alpha = mask.copy()

    if add_dots:
        hard = (mask > 0).astype(np.uint8)*255
        ring_in  = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        ring_out = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)))
        ring = cv2.subtract(ring_out, ring_in)
        yy2, xx2 = np.where(ring > 0)
        if len(yy2) > 0:
            rng = np.random.default_rng(seed+456)
            n_dots = 70 if category==5 else 45
            idx = rng.choice(len(yy2), size=min(n_dots, len(yy2)), replace=False)
            dots_alpha = np.zeros_like(out_alpha, dtype=np.uint8)
            for i in idx:
                x, y = int(xx2[i]), int(yy2[i])
                cv2.circle(lesion, (x, y), 2 if category==5 else 1, (20,20,20), -1, lineType=cv2.LINE_AA)
                cv2.circle(dots_alpha, (x, y), 2 if category==5 else 1, 230, -1, lineType=cv2.LINE_AA)
            out_alpha = np.maximum(out_alpha, dots_alpha)

    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[..., :3] = (lesion * m).astype(np.uint8)
    out[..., 3]  = out_alpha

    pil_img = Image.fromarray(out, mode="RGBA")
    pil_img = pil_img.resize((W//SUPERSAMPLE, H//SUPERSAMPLE), Image.LANCZOS)
    return pil_img

def _paste_center(canvas_pil: Image.Image, sprite_rgba: Image.Image, center_xy: Tuple[int,int]):
    x, y = center_xy
    w, h = sprite_rgba.size
    canvas_pil.alpha_composite(sprite_rgba, (int(x - w/2), int(y - h/2)))

def region_to_key(lobe, region):
    """
    Map (lobe, region) to one of the polygon keys in AREAS.
    Handles 'upper lobe' / 'mid lobe' / 'lower lobe' and 'lower middle'.
    """
    lobe   = (lobe or "").lower()
    region = _normalize_region(region) or _infer_region_from_text(lobe)

    if "isthmus" in lobe or region == "isthmus":
        return "Isthmus"

    if "left" in lobe:
        if region == "upper":         return "left upper"
        if region == "upper middle":  return "left upper middle"
        if region == "middle":        return "left middle"
        if region == "lower middle":  return "left lower middle"
        if region == "lower":         return "left lower"
        return "left middle"

    if "right" in lobe:
        if region == "upper":         return "right upper"
        if region == "upper middle":  return "right upper middle"
        if region == "middle":        return "right middle"
        if region == "lower middle":  return "right lower middle"
        if region == "lower":         return "right lower"
        return "right middle"

    return None

def setup_callout_rails(W, H, left_bbox, right_bbox):
    left_rail_x  = max(MIN_RAIL_MARGIN, right_bbox[0] - RAIL_GAP)
    right_rail_x = min(W - MIN_RAIL_MARGIN, left_bbox[2] + RAIL_GAP)
    return {"left": {"x": left_rail_x, "y": PAD_TOP + 40},
            "right":{"x": right_rail_x,"y": PAD_TOP + 40}}

def draw_sorted_callouts(draw, rails, items):
    left_items  = sorted([it for it in items if it["side"]=="left"],  key=lambda d: d["cy"])
    right_items = sorted([it for it in items if it["side"]=="right"], key=lambda d: d["cy"])
    for side_items in (left_items, right_items):
        if not side_items: continue
        side = side_items[0]["side"]
        rail = rails[side]
        for it in side_items:
            ft = font(20); fb = font(18); pad = 14
            title, lines = it["title"], it["lines"]
            title_w = draw.textlength(title, font=ft)
            body_w = max([draw.textlength(l, font=fb) for l in lines]) if lines else 0
            w = int(max(title_w, body_w) + 2*pad)
            h = int((ft.size+4) + (len(lines) * (fb.size+4)) + 2*pad)

            x1 = rail["x"] - (w if side=="left" else 0); x2 = x1 + w
            y1 = rail["y"]; y2 = y1 + h

            draw.rounded_rectangle([x1+3, y1+3, x2+3, y2+3], radius=10, fill=CALLOUT_SHADOW)
            draw.rounded_rectangle([x1, y1, x2, y2], radius=10, fill=CALLOUT_FILL, outline=CALLOUT_BORDER, width=3)
            draw.text((x1+pad, y1+pad), title, fill=(30,30,30,255), font=ft)
            ty = y1 + pad + ft.size + 6
            for l in lines:
                draw.text((x1+pad, ty), l, fill=(30,30,30,255), font=fb)
                ty += fb.size + 4

            box_edge = x2 if side=="left" else x1
            mid_y = (y1 + y2)//2
            draw.line([it["anchor"], (box_edge, mid_y)], fill=LEADER, width=4)

            rail["y"] = y2 + 18

def generate_image(json_extract):
    if not BASE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Base image not found at: {BASE_IMAGE_PATH}")

    base = Image.open(BASE_IMAGE_PATH).convert("RGBA")
    bw, bh = base.size
    W = bw + PAD_LEFT + PAD_RIGHT
    H = bh + PAD_TOP + PAD_BOTTOM

    canvas = Image.new("RGBA", (W, H), (255,255,255,0))
    OFFSET_X, OFFSET_Y = PAD_LEFT, PAD_TOP
    canvas.paste(base, (OFFSET_X, OFFSET_Y))

    AREAS = {k: shift_polygon(v, OFFSET_X, OFFSET_Y) for k, v in AREAS_ORIG.items()}
    LEFT_KEYS  = [k for k in AREAS if k.startswith("left")]
    RIGHT_KEYS = [k for k in AREAS if k.startswith("right")]

    draw = ImageDraw.Draw(canvas, "RGBA")

    # Compute lobe bboxes for rails and labels
    left_bbox  = union_bbox([AREAS[k] for k in LEFT_KEYS])
    right_bbox = union_bbox([AREAS[k] for k in RIGHT_KEYS])

    # RIGHT badge near the right lobe (canvas-left)
    badge_font = font(20)
    rb_text = "RIGHT"
    rb_w = draw.textlength(rb_text, font=badge_font) + 18
    rbx = right_bbox[0] - rb_w - 12
    rby = (right_bbox[1] + right_bbox[3])//2 - 18
    draw.rounded_rectangle([rbx, rby, rbx+rb_w, rby+36], 10, fill=LABEL_FILL, outline=LABEL_BORDER, width=2)
    draw.text((rbx+9, rby+7), rb_text, fill=(0,0,0,255), font=badge_font)

    # LEFT badge near the left lobe
    lb_text = "LEFT"
    lb_w = draw.textlength(lb_text, font=badge_font) + 18
    lbx = left_bbox[2] + 12
    lby = (left_bbox[1] + left_bbox[3])//2 - 18
    draw.rounded_rectangle([lbx, lby, lbx+lb_w, lby+36], 10, fill=LABEL_FILL, outline=LABEL_BORDER, width=2)
    draw.text((lbx+9, lby+7), lb_text, fill=(0,0,0,255), font=badge_font)

    # Echotexture badge (top-left)
    fe = font(17)
    etxt = f"Echotexture: {json_extract.get('overall_echotexture','N/A')}"
    etw = draw.textlength(etxt, font=fe) + 24
    draw.rounded_rectangle([24, 24, 24+etw, 24+fe.size+18], radius=10, fill=CALLOUT_FILL, outline=CALLOUT_BORDER, width=2)
    draw.text((34, 31), etxt, fill=(10,10,10,255), font=fe)

    rails = setup_callout_rails(W, H, left_bbox, right_bbox)

    # ---------- Group nodules by region key and spread side-by-side ----------
    groups = {}
    for idx, n in enumerate(json_extract.get("nodules", []), start=1):
        lobe   = n.get("lobe", "")
        region = _normalize_region(n.get("region", "")) or _infer_region_from_text(lobe)

        key = region_to_key(lobe, region)
        if not key or key not in AREAS:
            # Last-resort fallbacks per side
            if "right" in lobe.lower():   key = "right middle"
            elif "left" in lobe.lower():  key = "left middle"
            elif "isthmus" in lobe.lower(): key = "Isthmus"
            else: continue

        reg = AREAS[key]
        bx0, by0, bx1, by1 = bbox(reg)
        max_w = (bx1 - bx0) * 0.88
        max_h = (by1 - by0) * 0.88

        mm_w, mm_h = parse_size_to_mm(n.get("size_mm",""))
        px_w = min(max(14, mm_w * PX_PER_MM * NODULE_SCALE), max_w)
        px_h = min(max(14, mm_h * PX_PER_MM * NODULE_SCALE), max_h)

        item = {"idx": idx, "data": n, "key": key, "px_w": px_w, "px_h": px_h}
        groups.setdefault(key, []).append(item)

    callouts = []

    for key, items in groups.items():
        reg = AREAS[key]
        cx, cy = centroid(reg)
        bx0, by0, bx1, by1 = bbox(reg)

        k = len(items)
        max_w_in_group = max(i["px_w"] for i in items)
        spacing = max(22.0, max_w_in_group) + 10.0

        centers = []
        for i in range(k):
            rel = i - (k - 1) / 2.0
            x_center = cx + rel * spacing
            y_center = cy
            half_w = items[i]["px_w"] / 2.0
            half_h = items[i]["px_h"] / 2.0
            x_center = max(bx0 + half_w, min(bx1 - half_w, x_center))
            y_center = max(by0 + half_h, min(by1 - half_h, y_center))
            centers.append((x_center, y_center))

        for item, (ccx, ccy) in zip(items, centers):
            n = item["data"]
            tr = _parse_tr_category(n.get("ti_rads_score", ""))
            style = _TIRADS_STYLE.get(tr, _TIRADS_STYLE[3])
            size_major = 0.92 * max(item["px_w"], item["px_h"]) if tr >= 4 else max(item["px_w"], item["px_h"])
            color_rgb = TR_RGB[tr]
            add_dots = tr in (4, 5)

            sprite = _render_nodule_rgba(
                tr, size_major,
                aspect=float(style["aspect"]),
                jaggedness=float(style["jaggedness"]),
                lobes=int(style["lobes"]),
                seed=101 + item["idx"],
                color_rgb=color_rgb,
                add_dots=add_dots
            )
            _paste_center(canvas, sprite, (int(ccx), int(ccy)))

            side = (
                "left" if key.startswith("right")
                else ("right" if key.startswith("left")
                      else ("left" if ccx < (OFFSET_X + bw/2) else "right"))
            )

            pretty_region = _normalize_region(n.get("region","")).title() or _infer_region_from_text(n.get("lobe","")).title()
            title = f"Nodule #{item['idx']} · {n.get('ti_rads_score','')}"
            lines = [
                f"{n.get('lobe','')}, {pretty_region}",
                f"Size: {n.get('size_mm','')}",
                f"Echo: {n.get('echogenicity','N/A')}",
                f"Margins: {n.get('margins','N/A')}",
            ]
            callouts.append({
                "side": side,
                "cy": ccy,
                "anchor": (int(ccx), int(ccy)),
                "title": title,
                "lines": lines
            })

    # ---------- End grouping/drawing ----------
    draw_sorted_callouts(draw, rails, callouts)

    # Impression footer
    imp = (json_extract.get("impression_or_conclusion") or "").strip()
    if imp:
        fi = font(14)
        text = "Impression: " + imp
        maxw = int(W*0.9)
        words=text.split(); lines=[]; cur=""
        for w in words:
            trial=(cur+" "+w).strip()
            if draw.textlength(trial, font=fi) > maxw:
                lines.append(cur); cur=w
            else:
                cur=trial
        if cur: lines.append(cur)
        lh = fi.size + 6
        box_h = 14 + len(lines)*lh + 14
        y0 = H - box_h - 20
        draw.rounded_rectangle([int(W*0.05)+3, y0+3, int(W*0.95)+3, y0+box_h+3], 12, fill=CALLOUT_SHADOW)
        draw.rounded_rectangle([int(W*0.05), y0, int(W*0.95), y0+box_h], 12, fill=CALLOUT_FILL, outline=CALLOUT_BORDER, width=2)
        ty = y0 + 14
        for ln in lines:
            draw.text((int(W*0.06), ty), ln, fill=(10,10,10,255), font=fi); ty += lh

    return canvas
