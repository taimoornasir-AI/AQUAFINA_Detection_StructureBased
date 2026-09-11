"""Aquafina Bottle Detector — Precision 3D Shape + Color + RL Vision System.

Key features:
  1. Universal Bottle Localizer: Automatically locks onto and focuses any bottle in view.
  2. Strict Color Discrimination: Pure Dark Royal Blue (Aquafina) vs Light Blue/Cyan (Dasani) vs Red (Coke).
  3. Clear Classification: If a Coke, Dasani, or generic bottle is shown, it puts a steady RED bounding box
     on it and explicitly indicates 'NOT AQUAFINA' with the detected brand/color scheme.
  4. Unwrapped Aquafina Detection: Detects Aquafina without labels based on 3D OBJ profile + dark blue cap.
  5. Box Smoother: Eliminates bounding box jitter, keeping the camera focused smoothly on the bottle.
  6. Live Reinforcement Learning: Dynamic thresholding + positive/negative shape prototype memory in SQLite.
  7. Premium Dark UI: Real-time telemetry, 4-channel breakdown, and hands-free keyboard shortcuts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

# ── UI Colour Palette (Dark Glassmorphism) ────────────────────────────────────
BG_DARK   = "#070b16"
BG_PANEL  = "#0d1527"
BG_CARD   = "#121c33"
ACCENT    = "#00d4ff"
ACCENT2   = "#0070f3"
GREEN     = "#00e5a0"
RED       = "#ff4d6a"
WARN      = "#ffb830"
TEXT_PRI  = "#edf5ff"
TEXT_SEC  = "#738eb5"
TEXT_DIM  = "#3b4f6e"

DB_NAME   = "aquafina_embeddings.sqlite3"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Color Classification (Aquafina Dark Blue vs Dasani Light Blue vs Coke Red)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_cap_and_scheme(roi: np.ndarray) -> tuple[str, float, dict[str, float]]:
    """Strictly classify the dominant color scheme in a candidate region.

    Returns:
      (scheme_name, confidence, all_ratios)
      scheme_name is one of:
        - 'AQUAFINA_DARK_BLUE': Deep royal blue (H: 106-128, S >= 70, V >= 40)
        - 'DASANI_LIGHT_BLUE': Light blue / cyan (H: 80-106, S >= 50, V >= 45)
        - 'COKE_RED': Bright or dark red (H: 0-12 or 165-180, S >= 60, V >= 40)
        - 'SPRITE_GREEN': Green (H: 38-82, S >= 50, V >= 40)
        - 'WHITE_NEUTRAL': Very low saturation / white
        - 'UNKNOWN': No dominant bottle color
    """
    if roi.size == 0:
        return "UNKNOWN", 0.0, {}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total_px = max(1, roi.shape[0] * roi.shape[1])

    # Aquafina Deep Royal Blue: H 106..128, S >= 70, V >= 40
    # Much tighter than Dasani (which uses H 80-106 light blue/cyan)
    m_aqua = cv2.inRange(hsv, np.array([106, 70, 40]), np.array([128, 255, 255]))
    aqua_ratio = float(np.count_nonzero(m_aqua)) / total_px

    # Dasani Light Blue / Cyan: H 80..106, S >= 50, V >= 45
    m_dasani = cv2.inRange(hsv, np.array([80, 50, 45]), np.array([106, 255, 255]))
    dasani_ratio = float(np.count_nonzero(m_dasani)) / total_px

    # Red (Coca-Cola): H 0..12 or 165..180, S >= 60, V >= 40
    m_red1 = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([12, 255, 255]))
    m_red2 = cv2.inRange(hsv, np.array([165, 60, 40]), np.array([180, 255, 255]))
    red_ratio = float(np.count_nonzero(m_red1) + np.count_nonzero(m_red2)) / total_px

    # Green (Sprite): H 38..82, S >= 50, V >= 40
    m_green = cv2.inRange(hsv, np.array([38, 50, 40]), np.array([82, 255, 255]))
    green_ratio = float(np.count_nonzero(m_green)) / total_px

    # White / Neutral: S < 35, V > 120
    m_white = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 35, 255]))
    white_ratio = float(np.count_nonzero(m_white)) / total_px

    ratios = {
        "aquafina": aqua_ratio,
        "dasani": dasani_ratio,
        "coke": red_ratio,
        "sprite": green_ratio,
        "white": white_ratio
    }

    # Strict discrimination — require aqua_ratio to clearly dominate Dasani ratio
    if aqua_ratio >= 0.10 and aqua_ratio > dasani_ratio * 1.30:
        return "AQUAFINA_DARK_BLUE", min(1.0, aqua_ratio / 0.18), ratios
    elif dasani_ratio >= 0.08 and dasani_ratio > aqua_ratio:
        return "DASANI_LIGHT_BLUE", min(1.0, dasani_ratio / 0.18), ratios
    elif red_ratio >= 0.08:
        return "COKE_RED", min(1.0, red_ratio / 0.18), ratios
    elif green_ratio >= 0.08:
        return "SPRITE_GREEN", min(1.0, green_ratio / 0.18), ratios
    elif white_ratio >= 0.35:
        return "WHITE_NEUTRAL", min(1.0, white_ratio / 0.50), ratios

    return "UNKNOWN", 0.0, ratios


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 3D OBJ & MTL Model: Geometry, Cross-Section & Material Signatures
# ═══════════════════════════════════════════════════════════════════════════════

class Aquafina3DModel:
    """Parses aquafina_bottle_3d.obj and aquafina_bottle_3d.mtl to extract precise
    3D geometry, cross-sectional width profiles, and material color signatures.
    """

    def __init__(self, obj_path: Path, mtl_path: Path, n_slices: int = 20) -> None:
        self.obj_path = obj_path
        self.mtl_path = mtl_path
        self.n_slices = n_slices
        self.materials = self._parse_mtl(mtl_path)
        self.raw_profile, self.meta = self._parse_obj(obj_path)
        self.uniform_w = self._get_uniform_profile(self.raw_profile, n_slices)
        self.silhouette = profile_to_silhouette_image(self.raw_profile)
        self.hu_moments = compute_hu_moments(self.silhouette)
        self.aspect_ratio_3d = self.meta.get("aspect_ratio", 1.96)
        self.cap_diameter_ratio = self.meta.get("cap_to_body_ratio", 0.434)

    def _parse_mtl(self, path: Path) -> dict[str, dict]:
        props: dict[str, dict] = {}
        if not path.exists():
            return props
        cur = None
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                p = line.strip().split()
                if not p:
                    continue
                if p[0] == "newmtl":
                    cur = p[1]
                    props[cur] = {}
                elif cur and len(p) >= 2:
                    try:
                        props[cur][p[0]] = [float(x) for x in p[1:]]
                    except ValueError:
                        pass
        return props

    def _parse_obj(self, path: Path) -> tuple[np.ndarray, dict]:
        if not path.exists():
            return np.empty((0, 2), dtype=np.float32), {}

        radii_by_z: dict[float, list[float]] = {}
        all_v = []
        mtl_v: dict[str, set[int]] = {}
        cur_mtl = "Bottle"

        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            r = math.sqrt(x * x + y * y)
                            all_v.append((x, y, z, r))
                            radii_by_z.setdefault(round(z, 4), []).append(r)
                        except ValueError:
                            continue
                elif line.startswith("usemtl "):
                    parts = line.split()
                    if len(parts) >= 2:
                        cur_mtl = parts[1]
                elif line.startswith("f "):
                    for tok in line.split()[1:]:
                        try:
                            v_idx = int(tok.split("/")[0]) - 1
                            mtl_v.setdefault(cur_mtl, set()).add(v_idx)
                        except (ValueError, IndexError):
                            pass

        if not radii_by_z or not all_v:
            return np.empty((0, 2), dtype=np.float32), {}

        z_vals = [v[2] for v in all_v]
        r_vals = [v[3] for v in all_v]
        z_min, z_max = min(z_vals), max(z_vals)
        r_max = max(r_vals) if max(r_vals) > 0 else 1.0
        z_rng = z_max - z_min if z_max != z_min else 1.0

        zvals = sorted(radii_by_z.keys())
        points = np.array([(np.mean(radii_by_z[z]) / r_max, (z - z_min) / z_rng) for z in zvals], dtype=np.float32)

        cap_indices = list(mtl_v.get("Cap", []))
        if cap_indices:
            cap_radii = [all_v[i][3] for i in cap_indices if i < len(all_v)]
            cap_r = max(cap_radii) if cap_radii else 0.49
            cap_ratio = cap_r / r_max
        else:
            cap_ratio = 0.434

        meta = {
            "aspect_ratio": z_rng / (2.0 * r_max),
            "cap_to_body_ratio": cap_ratio,
            "total_height": z_rng,
            "max_radius": r_max,
        }
        return points, meta

    def _get_uniform_profile(self, profile: np.ndarray, n_slices: int = 20) -> np.ndarray:
        if profile.shape[0] < 2:
            return np.ones(n_slices, dtype=np.float32)
        z_top_to_bottom = np.linspace(0.98, 0.05, n_slices)
        w = np.interp(z_top_to_bottom, profile[:, 1], profile[:, 0])
        max_w = w.max() if w.max() > 0 else 1.0
        return (w / max_w).astype(np.float32)

    def compute_shape_similarity(self, cand_w: np.ndarray, sym: float, scheme: str = "UNKNOWN") -> tuple[float, int, dict]:
        """Strictly compares candidate width profile and geometry against the 3D OBJ model.
        Also validates material color compatibility against aquafina_bottle_3d.mtl.

        IMPORTANT: Only block DASANI/COKE/SPRITE — do NOT block WHITE or UNKNOWN.
        Unwrapped / transparent Aquafina bottles appear as WHITE_NEUTRAL or UNKNOWN
        and MUST pass through shape scoring to be detected without their label.
        """
        # MTL Material Contradiction: Only known competing brands are blocked.
        # WHITE_NEUTRAL and UNKNOWN may be unwrapped Aquafina — allow shape scoring.
        if scheme in ("DASANI_LIGHT_BLUE", "COKE_RED", "SPRITE_GREEN"):
            return 0.0, 0, {"reason": "MTL_MATERIAL_CONTRADICTION", "scheme": scheme}

        if cand_w.shape[0] != self.uniform_w.shape[0] or np.all(cand_w == 0):
            return 0.0, 0, {}

        # 1. Profile L1 Error against 3D OBJ uniform slices
        l1_err = float(np.mean(np.abs(cand_w - self.uniform_w)))
        l1_match = max(0.0, 1.0 - (l1_err / 0.20))

        # 2. Pearson & Cosine profile correlation across body slices
        c_sub = cand_w[1:19]
        r_sub = self.uniform_w[1:19]
        if np.std(c_sub) > 1e-4 and np.std(r_sub) > 1e-4:
            pearson = max(0.0, float(np.corrcoef(c_sub, r_sub)[0, 1]))
        else:
            pearson = 0.0

        denom = float(np.linalg.norm(cand_w) * np.linalg.norm(self.uniform_w))
        cosine = max(0.0, float(np.dot(cand_w, self.uniform_w) / denom)) if denom > 1e-6 else 0.0
        corr_sim = 0.60 * pearson + 0.40 * cosine

        # 3. Cap-to-Body diameter ratio match (OBJ cap is ~43% of max bottle body)
        cand_top_ratio = float(np.mean(cand_w[0:2]))
        cap_diff = abs(cand_top_ratio - self.cap_diameter_ratio)
        cap_match = max(0.0, 1.0 - (cap_diff / 0.28))

        # 4. Shoulder curvature match (slices 1 to 4 widen from neck to body)
        shoulder_diff = float(cand_w[4] - cand_w[1])
        shoulder_match = max(0.0, min(1.0, shoulder_diff / 0.18)) if shoulder_diff > 0 else 0.0

        # 5. Waist indentation match (slices 6-12 waist proportion is ~0.885 in Aquafina OBJ)
        waist_ratio = float(np.mean(cand_w[6:12]))
        waist_match = max(0.0, 1.0 - abs(waist_ratio - 0.885) / 0.24)

        # 6. Neck taper check — Aquafina neck (slices 15-19) is narrower than body
        neck_ratio = float(np.mean(cand_w[15:19]))
        body_avg = float(np.mean(cand_w[5:14]))
        taper_diff = body_avg - neck_ratio
        neck_match = max(0.0, min(1.0, taper_diff / 0.18)) if taper_diff > 0 else 0.0

        # Combined 3D shape score (0-100) scaled by symmetry
        raw_shape = (0.28 * corr_sim + 0.27 * l1_match + 0.15 * cap_match +
                     0.15 * waist_match + 0.08 * shoulder_match + 0.07 * neck_match)
        shape_sc = int(100 * raw_shape * min(1.0, sym * 1.12))

        breakdown = {
            "corr": corr_sim,
            "l1_match": l1_match,
            "l1_err": l1_err,
            "cap_match": cap_match,
            "shoulder_match": shoulder_match,
            "waist_match": waist_match,
            "neck_match": neck_match,
            "sym": sym,
        }
        return corr_sim, shape_sc, breakdown


def extract_obj_profile(path: Path) -> np.ndarray:
    return Aquafina3DModel(path, path.with_suffix(".mtl")).raw_profile


def get_obj_uniform_profile(profile: np.ndarray, n_slices: int = 20) -> np.ndarray:
    if profile.shape[0] < 2:
        return np.ones(n_slices, dtype=np.float32)
    z_top_to_bottom = np.linspace(0.98, 0.05, n_slices)
    w = np.interp(z_top_to_bottom, profile[:, 1], profile[:, 0])
    max_w = w.max() if w.max() > 0 else 1.0
    return (w / max_w).astype(np.float32)


def profile_to_silhouette_image(profile: np.ndarray, w: int = 64, h: int = 192) -> np.ndarray:
    """Render the 3D profile as a solid filled 2D silhouette polygon."""
    canvas = np.zeros((h, w), dtype=np.uint8)
    if profile.shape[0] < 2:
        return canvas
    pts_left = []
    pts_right = []
    for r_norm, z_norm in profile:
        cy = int((1.0 - z_norm) * (h - 1))
        cx = w // 2
        hw = max(1, int(r_norm * (w // 2 - 2)))
        pts_left.append([cx - hw, cy])
        pts_right.append([cx + hw, cy])
    poly = np.array(pts_left + pts_right[::-1], dtype=np.int32)
    cv2.fillPoly(canvas, [poly], 255)
    return canvas


def compute_hu_moments(silhouette: np.ndarray) -> np.ndarray:
    moments = cv2.moments(silhouette)
    hu = cv2.HuMoments(moments).flatten()
    with np.errstate(divide="ignore", invalid="ignore"):
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Shape Analysis & Full Bottle Localization
# ═══════════════════════════════════════════════════════════════════════════════

def locate_full_bottle(frame: np.ndarray, cap_bbox: tuple[int, int, int, int], est_body_ratio: float = 2.30) -> tuple[int, int, int, int]:
    """Expands from cap anchor downward to locate the FULL COMPLETE BOTTLE bounding box (cap top to base)."""
    fh, fw = frame.shape[:2]
    cx, cy, cw, ch = cap_bbox
    mid_x = cx + cw / 2.0
    est_w = int(cw * est_body_ratio)

    col_left = max(0, int(mid_x - est_w * 0.70))
    col_right = min(fw, int(mid_x + est_w * 0.70))

    # Inspect column directly below cap for bottle base and lateral boundaries
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 85)
    col_crop = edges[cy:fh, col_left:col_right]
    y_idx, x_idx = np.where(col_crop > 0)

    # Typical bottle aspect ratio in 2D camera view is ~2.0 to 4.0
    min_h = int(est_w * 1.8)
    max_h = int(est_w * 4.2)

    if len(y_idx) > 20:
        # Base is near the bottom edges in the bottle column
        bottom_cand = cy + int(np.percentile(y_idx, 95))
        h_cand = bottom_cand - cy
        if min_h <= h_cand <= max_h:
            full_h = min(fh - cy, h_cand + int(ch * 0.5))
        else:
            full_h = min(fh - cy, max(min_h, min(max_h, int(est_w * 3.2))))

        left_cand = col_left + int(np.percentile(x_idx, 5))
        right_cand = col_left + int(np.percentile(x_idx, 95))
        full_w = max(est_w, right_cand - left_cand)
        bx = max(0, int(mid_x - full_w / 2.0))
        bw = min(fw - bx, full_w)
    else:
        full_h = min(fh - cy, int(est_w * 3.2))
        bx = max(0, int(mid_x - est_w / 2.0))
        bw = min(fw - bx, est_w)

    return (bx, cy, bw, full_h)


def extract_width_profile(crop: np.ndarray, n_slices: int = 20) -> tuple[np.ndarray, float]:
    """Extracts normalized widths across vertical band slices with gap interpolation and bilateral symmetry."""
    if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 15:
        return np.zeros(n_slices, dtype=np.float32), 0.0

    ch, cw = crop.shape[:2]
    mid_x = cw / 2.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 7, 45, 45)
    edges = cv2.Canny(blurred, 25, 85)

    band_h = ch / n_slices
    widths = []
    sym_diffs = []
    valid_pairs = 0

    for i in range(n_slices):
        y1 = int(i * band_h)
        y2 = max(y1 + 1, int((i + 1) * band_h))
        band = edges[y1:y2, :]
        col_has = np.any(band > 0, axis=0)
        nz = np.where(col_has)[0]
        if len(nz) >= 2:
            left, right = nz[0], nz[-1]
            widths.append(right - left)
            left_d = abs(mid_x - left)
            right_d = abs(right - mid_x)
            sym_diffs.append(abs(left_d - right_d) / max(cw, 1))
            valid_pairs += 1
        elif len(nz) == 1:
            widths.append(cw * 0.45)
            sym_diffs.append(0.35)
        else:
            widths.append(0.0)
            sym_diffs.append(0.8)

    w_arr = np.array(widths, dtype=np.float32)

    # Clean gaps/reflections on transparent plastic via 1D neighbor interpolation
    for i in range(len(w_arr)):
        if w_arr[i] < cw * 0.20:
            prev_v = [w_arr[j] for j in range(i - 1, -1, -1) if w_arr[j] >= cw * 0.20]
            next_v = [w_arr[j] for j in range(i + 1, len(w_arr)) if w_arr[j] >= cw * 0.20]
            if prev_v and next_v:
                w_arr[i] = (prev_v[0] + next_v[0]) / 2.0
            elif prev_v:
                w_arr[i] = prev_v[0]
            elif next_v:
                w_arr[i] = next_v[0]

    max_w = w_arr.max() if w_arr.max() > 0 else 1.0
    norm_w = w_arr / max_w
    raw_sym = max(0.0, 1.0 - float(np.mean(sym_diffs)))
    sym = raw_sym if valid_pairs >= 7 else raw_sym * (valid_pairs / 7.0)
    return norm_w, sym


def compare_profiles(cand_w: np.ndarray, ref_w: np.ndarray) -> float:
    """Compare candidate width profile with reference using Pearson + Cosine blend."""
    if cand_w.shape[0] != ref_w.shape[0] or np.all(cand_w == 0):
        return 0.0

    c_sub = cand_w[1:19]
    r_sub = ref_w[1:19]

    if np.std(c_sub) > 1e-4 and np.std(r_sub) > 1e-4:
        pearson = float(np.corrcoef(c_sub, r_sub)[0, 1])
        pearson = max(0.0, pearson)
    else:
        pearson = 0.0

    denom = float(np.linalg.norm(cand_w) * np.linalg.norm(ref_w))
    cosine = float(np.dot(cand_w, ref_w) / denom) if denom > 1e-6 else 0.0
    cosine = max(0.0, cosine)

    return 0.65 * pearson + 0.35 * cosine


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Temporal Bounding Box Smoother (Autofocus Tracker)
# ═══════════════════════════════════════════════════════════════════════════════

class BoxSmoother:
    """Maintains a stable, smoothed bounding box across video frames to prevent jitter."""

    def __init__(self, alpha: float = 0.32) -> None:
        # alpha=0.32 gives faster response; lost_frames threshold = 4 for quicker drop
        self.alpha = alpha
        self.current_box: tuple[int, int, int, int] | None = None
        self.lost_frames = 0
        self.last_bottle_type: str = "NONE"
        self.last_detected: bool = False
        self.last_trigger: str = "SCANNING..."
        self.last_confidence: int = 0

    def update(
        self,
        new_box: tuple[int, int, int, int] | None,
        detected: bool = False,
        bottle_present: bool = False,
        bottle_type: str = "NONE",
        confidence: int = 0,
        trigger: str = "SCANNING...",
    ) -> tuple[tuple[int, int, int, int] | None, bool, bool, str, int, str]:
        """Returns smoothed (bbox, detected, bottle_present, bottle_type, confidence, trigger)."""
        if new_box is None or not bottle_present:
            self.lost_frames += 1
            # Drop after 4 missed frames (was 6) for faster clearing of ghost detections
            if self.lost_frames > 4 or self.current_box is None:
                self.current_box = None
                self.last_bottle_type = "NONE"
                self.last_detected = False
                self.last_trigger = "SCANNING..."
                self.last_confidence = 0
                return None, False, False, "NONE", 0, "SCANNING..."
            # Decay confidence faster during brief drops (was 0.90, now 0.80)
            decayed_conf = max(0, int(self.last_confidence * 0.80))
            return self.current_box, self.last_detected, True, self.last_bottle_type, decayed_conf, self.last_trigger

        self.lost_frames = 0
        self.last_bottle_type = bottle_type
        self.last_detected = detected
        self.last_trigger = trigger
        self.last_confidence = confidence

        if self.current_box is None:
            self.current_box = new_box
            return self.current_box, detected, bottle_present, bottle_type, confidence, trigger

        cx, cy, cw, ch = self.current_box
        nx, ny, nw, nh = new_box

        # Check center distance — if too far, jump directly (new object in view)
        center_dist = math.hypot((cx + cw / 2.0) - (nx + nw / 2.0), (cy + ch / 2.0) - (ny + nh / 2.0))
        if center_dist < max(cw, nw) * 1.2:
            sx = int(cx * (1.0 - self.alpha) + nx * self.alpha)
            sy = int(cy * (1.0 - self.alpha) + ny * self.alpha)
            sw = int(cw * (1.0 - self.alpha) + nw * self.alpha)
            sh = int(ch * (1.0 - self.alpha) + nh * self.alpha)
            self.current_box = (sx, sy, sw, sh)
        else:
            # Big jump — treat as fresh object
            self.current_box = new_box

        return self.current_box, detected, bottle_present, bottle_type, confidence, trigger


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SQLite Store (ORB Embeddings, Prototypes, RL Stats)
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingStore:
    """Manages visual embeddings, 3D shape prototypes, and dynamic RL stats."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        source TEXT NOT NULL UNIQUE,
        keypoints BLOB NOT NULL,
        descriptors BLOB NOT NULL,
        shape BLOB NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS shape_prototypes (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        profile BLOB NOT NULL,
        hu BLOB NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS rl_stats (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        true_pos INTEGER NOT NULL DEFAULT 0,
        false_pos INTEGER NOT NULL DEFAULT 0,
        false_neg INTEGER NOT NULL DEFAULT 0,
        threshold INTEGER NOT NULL DEFAULT 60
    );
    INSERT OR IGNORE INTO rl_stats(id) VALUES(1);
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        with sqlite3.connect(path) as con:
            cols = [r[1] for r in con.execute("PRAGMA table_info(embeddings)").fetchall()]
            if cols and "source" not in cols:
                con.execute("ALTER TABLE embeddings RENAME TO embeddings_legacy")
            con.executescript(self.SCHEMA)
            con.commit()

    def add_embedding(self, label: str, source: str, keypoints: np.ndarray,
                      descriptors: np.ndarray, shape: np.ndarray, size: tuple) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute(
                "INSERT OR REPLACE INTO embeddings"
                "(label, source, keypoints, descriptors, shape, width, height, created_at)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (label, source,
                 keypoints.astype(np.float32).tobytes(),
                 descriptors.astype(np.uint8).tobytes(),
                 shape.astype(np.float32).tobytes(),
                 size[0], size[1],
                 dt.datetime.now(dt.timezone.utc).isoformat()))
            con.commit()

    def get_embeddings(self) -> list[dict]:
        with sqlite3.connect(self.path) as con:
            rows = con.execute(
                "SELECT label, source, keypoints, descriptors, shape, width, height"
                " FROM embeddings ORDER BY id").fetchall()
        return [
            {"label": lab, "source": src,
             "keypoints":   np.frombuffer(kp,  dtype=np.float32).reshape(-1, 2),
             "descriptors": np.frombuffer(des, dtype=np.uint8).reshape(-1, 32),
             "shape":       np.frombuffer(sh,  dtype=np.float32).reshape(96, 32),
             "size": (width, height)}
            for lab, src, kp, des, sh, width, height in rows
        ]

    def add_shape_prototype(self, label: str, profile: np.ndarray, hu: np.ndarray) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute(
                "INSERT INTO shape_prototypes (label, profile, hu, created_at) VALUES (?,?,?,?)",
                (label, profile.astype(np.float32).tobytes(),
                 hu.astype(np.float32).tobytes(),
                 dt.datetime.now(dt.timezone.utc).isoformat()))
            con.commit()

    def get_shape_prototypes(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        with sqlite3.connect(self.path) as con:
            rows = con.execute("SELECT label, profile FROM shape_prototypes").fetchall()
        positives, negatives = [], []
        for lab, prof_b in rows:
            arr = np.frombuffer(prof_b, dtype=np.float32)
            if lab == "aquafina":
                positives.append(arr)
            else:
                negatives.append(arr)
        return positives, negatives

    def count_prototypes(self) -> tuple[int, int]:
        with sqlite3.connect(self.path) as con:
            pos = con.execute("SELECT COUNT(*) FROM shape_prototypes WHERE label='aquafina'").fetchone()[0]
            neg = con.execute("SELECT COUNT(*) FROM shape_prototypes WHERE label!='aquafina'").fetchone()[0]
        return pos, neg

    def get_rl_stats(self) -> dict:
        with sqlite3.connect(self.path) as con:
            row = con.execute(
                "SELECT true_pos, false_pos, false_neg, threshold FROM rl_stats WHERE id=1"
            ).fetchone()
        return {"true_pos": row[0], "false_pos": row[1],
                "false_neg": row[2], "threshold": row[3]}

    def record_true_positive(self) -> dict:
        with sqlite3.connect(self.path) as con:
            con.execute("UPDATE rl_stats SET true_pos=true_pos+1, threshold=MAX(38, threshold-1) WHERE id=1")
            con.commit()
        return self.get_rl_stats()

    def record_false_positive(self) -> dict:
        with sqlite3.connect(self.path) as con:
            con.execute("UPDATE rl_stats SET false_pos=false_pos+1, threshold=MIN(88, threshold+3) WHERE id=1")
            con.commit()
        return self.get_rl_stats()

    def record_false_negative(self) -> dict:
        with sqlite3.connect(self.path) as con:
            con.execute("UPDATE rl_stats SET false_neg=false_neg+1, threshold=MAX(38, threshold-3) WHERE id=1")
            con.commit()
        return self.get_rl_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ORB Helper & Seed Reference
# ═══════════════════════════════════════════════════════════════════════════════

def shape_signature(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(cv2.Canny(gray, 45, 125), (32, 96),
                      interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def crop_product(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points = cv2.findNonZero((gray < 245).astype(np.uint8))
    if points is None:
        raise ValueError("No detectable product in image")
    x, y, w, h = cv2.boundingRect(points)
    pad = max(8, int(max(w, h) * 0.03))
    return image[max(0, y - pad): min(image.shape[0], y + h + pad),
                 max(0, x - pad): min(image.shape[1], x + w + pad)]


def make_embedding(image: np.ndarray, crop: bool = False) -> tuple:
    product = crop_product(image) if crop else image
    gray    = cv2.cvtColor(product, cv2.COLOR_BGR2GRAY)
    kp, des = cv2.ORB_create(nfeatures=1200).detectAndCompute(gray, None)
    if des is None or len(kp) < 12:
        raise ValueError("Not enough visual features")
    pts = np.array([k.pt for k in kp], dtype=np.float32)
    return pts, des, shape_signature(product), (product.shape[1], product.shape[0])


def seed_store(store: EmbeddingStore, ref_path: Path, obj_w: np.ndarray, ref_hu: np.ndarray) -> None:
    source_full = f"reference:{ref_path.resolve()}"
    source_logo = f"reference_logo:{ref_path.resolve()}"
    existing = [item["source"] for item in store.get_embeddings()]
    if ref_path.exists():
        img = cv2.imread(str(ref_path))
        if img is not None:
            if source_full not in existing:
                try:
                    pts, des, sh, sz = make_embedding(img, crop=True)
                    store.add_embedding("aquafina", source_full, pts, des, sh, sz)
                except Exception:
                    pass
            if source_logo not in existing:
                try:
                    h, w = img.shape[:2]
                    logo = img[int(h * 0.33):int(h * 0.67), int(w * 0.35):int(w * 0.65)]
                    pts, des, sh, sz = make_embedding(logo, crop=False)
                    store.add_embedding("aquafina", source_logo, pts, des, sh, sz)
                except Exception:
                    pass

    pos, _ = store.count_prototypes()
    if pos == 0 and len(obj_w) > 0:
        store.add_shape_prototype("aquafina", obj_w, ref_hu)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Detection Engine (Localization, Classification, RL Penalty)
# ═══════════════════════════════════════════════════════════════════════════════

class DetectionResult(NamedTuple):
    detected: bool
    bottle_present: bool
    bottle_type: str
    bbox: tuple[int, int, int, int] | None
    confidence: int
    cap_score: int
    shape_score: int
    sym_score: int
    orb_score: int
    trigger: str
    candidate_profile: np.ndarray | None
    candidate_hu: np.ndarray | None


class AquafinaDetector:
    """Precision bottle detector with active classification of non-Aquafina bottles
    based on 3D OBJ geometry, MTL material signatures, and color schemes.
    """

    def __init__(self, mtl_path: Path, obj_path: Path, store: EmbeddingStore) -> None:
        self.store = store
        self.model_3d = Aquafina3DModel(obj_path, mtl_path, n_slices=20)
        self.raw_obj_profile = self.model_3d.raw_profile
        self.obj_uniform_w   = self.model_3d.uniform_w
        self.obj_silhouette  = self.model_3d.silhouette
        self.obj_hu          = self.model_3d.hu_moments

        pos_cnt, _ = self.store.count_prototypes()
        if pos_cnt == 0:
            self.store.add_shape_prototype("aquafina", self.obj_uniform_w, self.obj_hu)

        self.orb = cv2.ORB_create(nfeatures=1400)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.box_smoother = BoxSmoother(alpha=0.28)

    def find_bottle_candidates(self, frame: np.ndarray) -> list[tuple[int, int, int, int, str]]:
        """Locates bottle candidates in the scene.

        Design principle: LOCALIZE GENEROUSLY, CLASSIFY STRICTLY.
        This function finds anything shaped like a bottle. The detect_frame
        classifier then decides if it is actually Aquafina.
        """
        fh, fw = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates: list[tuple[int, int, int, int, str]] = []

        # ── 1. Cap Colour Anchor (finds bottle from its cap) ─────────────────
        # Broad colour ranges — we want to catch the cap even in difficult lighting.
        m_aqua   = cv2.inRange(hsv, np.array([105, 70, 35]), np.array([130, 255, 255]))
        m_dasani = cv2.inRange(hsv, np.array([84,  55, 35]), np.array([107, 255, 255]))
        m_red    = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   65, 40]), np.array([12,  255, 255])),
            cv2.inRange(hsv, np.array([165, 65, 40]), np.array([180, 255, 255]))
        )
        m_green  = cv2.inRange(hsv, np.array([38,  65, 40]), np.array([82,  255, 255]))
        all_caps = cv2.bitwise_or(
            cv2.bitwise_or(m_aqua, m_dasani),
            cv2.bitwise_or(m_red,  m_green)
        )
        k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        all_caps = cv2.morphologyEx(all_caps, cv2.MORPH_CLOSE, k5)
        cap_cnts, _ = cv2.findContours(all_caps, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cap_cnts:
            area = cv2.contourArea(c)
            if 100 < area < 6000:   # wide range — catches small or large caps
                cx, cy, cw, ch = cv2.boundingRect(c)
                asp = cw / max(ch, 1)
                if 0.4 <= asp <= 4.0:   # caps can be wide if viewed at angle
                    bx, by, bw, bh = locate_full_bottle(
                        frame, (cx, cy, cw, ch),
                        est_body_ratio=1.0 / self.model_3d.cap_diameter_ratio
                    )
                    if bw >= 40 and bh >= 100 and (bh / max(bw, 1)) >= 1.25:
                        candidates.append((bx, by, bw, bh, "cap_anchor"))

        # ── 2. Vertical Edge / Silhouette Detector (works for transparent bottles) ─
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 20, 75)
        k_vert  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 22))
        closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_vert)
        edge_cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in edge_cnts:
            area = cv2.contourArea(c)
            if 2500 < area < (fh * fw * 0.80):
                bx, by, bw, bh = cv2.boundingRect(c)
                asp = bh / max(bw, 1)
                # Must be noticeably taller than wide (bottle shape)
                if 1.30 <= asp <= 6.0 and bw >= 40 and bh >= 100 and bw <= int(fw * 0.65):
                    hull      = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    solidity  = float(area) / max(hull_area, 1.0)
                    # 0.42 rejects very irregular shapes (crumpled clothes, hands)
                    # but accepts bottles even if partially transparent/occluded
                    if solidity >= 0.42:
                        candidates.append((bx, by, bw, bh, "edge_contour"))

        # ── NMS: remove heavily overlapping duplicates ─────────────────────────
        if len(candidates) > 1:
            kept: list[tuple[int, int, int, int, str]] = []
            for cand in candidates:
                bx, by, bw, bh, src = cand
                ok = True
                for kx, ky, kw, kh, _ in kept:
                    ix = max(0, min(bx + bw, kx + kw) - max(bx, kx))
                    iy = max(0, min(by + bh, ky + kh) - max(by, ky))
                    if min(bw * bh, kw * kh) > 0 and (ix * iy) / min(bw * bh, kw * kh) > 0.50:
                        ok = False
                        break
                if ok:
                    kept.append(cand)
            candidates = kept

        return candidates

    def detect_frame(self, frame: np.ndarray, references: list[dict],
                     threshold: int) -> DetectionResult:
        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates = self.find_bottle_candidates(frame)

        # ── 1. Evaluate Bottle Candidates ─────────────────────────────────────
        pos_protos, neg_protos = self.store.get_shape_prototypes()
        
        best_candidate = None
        best_cand_score = 0
        best_profile = None
        best_hu = None
        best_scheme = "UNKNOWN"
        best_cap_score = 0
        best_shape_score = 0
        best_sym_score = 0
        best_is_aquafina = False
        best_trigger = "SCANNING..."
        rejection_reason = "NO BOTTLE IN VIEW"
        identified_non_aqua_type = "OTHER BOTTLE"

        for bx, by, bw, bh, source in candidates:
            if bw < 40 or bh < 100 or (bh / max(bw, 1)) < 1.25:
                continue
            crop = frame[by: by + bh, bx: bx + bw]
            if crop.size == 0 or crop.shape[0] < 100 or crop.shape[1] < 40:
                continue

            # Analyze top 20% of crop for cap color
            top_h = max(1, int(bh * 0.20))
            cap_roi = crop[0:top_h, :]
            scheme, scheme_conf, ratios = classify_cap_and_scheme(cap_roi)

            # Also check full body color to catch Dasani's light blue body
            body_scheme, _, body_ratios = classify_cap_and_scheme(crop)

            # Width profile + symmetry from 3D OBJ geometry
            cand_w, sym = extract_width_profile(crop, n_slices=20)

            # Symmetry gate: genuine bottles are roughly bilaterally symmetric.
            # 0.42 is permissive enough to handle slightly off-center captures.
            if sym < 0.42:
                continue

            corr_sim, shape_sc, breakdown = self.model_3d.compute_shape_similarity(
                cand_w, sym, scheme=scheme
            )

            sim_pos = max([compare_profiles(cand_w, p) for p in pos_protos] + [corr_sim])
            sim_neg = max([compare_profiles(cand_w, n) for n in neg_protos]) if neg_protos else 0.0

            sym_sc = int(sym * 100)
            cap_sc = min(100, int(ratios.get("aquafina", 0.0) * 400))

            # Negative prototype penalty
            if sim_neg > 0.70 and sim_neg >= sim_pos * 0.88:
                shape_sc = int(shape_sc * 0.45)

            # ── Classification ────────────────────────────────────────────────
            # If either cap OR body is a known competing brand, reject as non-Aquafina.
            is_non_aqua_brand = (
                scheme      in ("DASANI_LIGHT_BLUE", "COKE_RED", "SPRITE_GREEN") or
                body_scheme in ("DASANI_LIGHT_BLUE", "COKE_RED", "SPRITE_GREEN")
            )

            # Aquafina's dark royal blue cap — looser gate (>=14) so it fires in dim light
            has_dark_blue_cap = (scheme == "AQUAFINA_DARK_BLUE" and cap_sc >= 14)

            # PATH A — Wrapped Aquafina: dark blue cap + shape match
            # PATH B — Unwrapped Aquafina: very strong shape match alone
            is_candidate_aquafina = False
            total_conf  = 0
            trigger_text = ""

            if not is_non_aqua_brand:
                if has_dark_blue_cap and shape_sc >= 42:
                    is_candidate_aquafina = True
                    total_conf  = int(0.52 * shape_sc + 0.33 * cap_sc + 0.15 * sym_sc)
                    trigger_text = "3D OBJ SHAPE + DARK BLUE CAP"
                elif shape_sc >= 62 and sym >= 0.50:
                    # Unwrapped / transparent — very high shape confidence required
                    is_candidate_aquafina = True
                    total_conf  = int(0.72 * shape_sc + 0.28 * sym_sc)
                    trigger_text = "3D OBJ SHAPE (UNWRAPPED)"

            if is_candidate_aquafina:
                if total_conf > best_cand_score and total_conf >= 40:
                    best_cand_score  = total_conf
                    best_candidate   = (bx, by, bw, bh)
                    best_scheme      = scheme
                    best_cap_score   = cap_sc
                    best_shape_score = shape_sc
                    best_sym_score   = sym_sc
                    best_profile     = cand_w
                    best_is_aquafina = True
                    best_trigger     = trigger_text
            else:
                # Confirmed non-Aquafina bottle in view — OBJ bar stays at 0
                bottle_saliency = int(0.50 * sym_sc + 0.50 * min(100, int(bh / max(bw, 1) * 25)))
                if (best_candidate is None or not best_is_aquafina) and bottle_saliency > best_cand_score:
                    best_cand_score  = bottle_saliency
                    best_candidate   = (bx, by, bw, bh)
                    best_scheme      = scheme
                    best_cap_score   = 0
                    best_shape_score = 0
                    best_sym_score   = sym_sc
                    best_profile     = cand_w
                    best_is_aquafina = False

                    if scheme == "DASANI_LIGHT_BLUE" or body_scheme == "DASANI_LIGHT_BLUE":
                        identified_non_aqua_type = "DASANI (LIGHT BLUE)"
                        rejection_reason = "REJECTED: DASANI / CYAN SCHEME"
                    elif scheme == "COKE_RED" or body_scheme == "COKE_RED":
                        identified_non_aqua_type = "COCA-COLA (RED)"
                        rejection_reason = "REJECTED: COCA-COLA / RED SCHEME"
                    elif scheme == "SPRITE_GREEN" or body_scheme == "SPRITE_GREEN":
                        identified_non_aqua_type = "SPRITE (GREEN)"
                        rejection_reason = "REJECTED: SPRITE / GREEN SCHEME"
                    else:
                        identified_non_aqua_type = "OTHER BOTTLE"
                        rejection_reason = "REJECTED: NON-AQUAFINA SHAPE"
                    best_trigger = rejection_reason

        # ── 2. ORB Matching for Aquafina Label (Dynamic bar + strict gating) ──
        orb_score = 0
        orb_bbox = None
        best_inliers = 0
        fkp, fdes = self.orb.detectAndCompute(gray, None)

        if fdes is not None and references:
            best_good_count = 0
            best_orb_conf = 0

            for ref in references:
                if ref["label"] != "aquafina":
                    continue
                pairs = self.bf.knnMatch(ref["descriptors"], fdes, k=2)
                good = [m for m, n in pairs if len(pairs) >= 2 and m.distance < 0.75 * n.distance]
                if len(good) > best_good_count:
                    best_good_count = len(good)

                inliers = 0
                if len(good) >= 6:
                    src = ref["keypoints"][[m.queryIdx for m in good]].reshape(-1, 1, 2)
                    dst = np.float32([fkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                    if mask is not None and H is not None:
                        inliers = int(mask.ravel().sum())
                        rw, rh = ref["size"]
                        corners = np.float32([[[0, 0], [rw, 0], [rw, rh], [0, rh]]])
                        proj = cv2.perspectiveTransform(corners, H)[0]
                        proj_area = cv2.contourArea(proj)
                        is_convex = cv2.isContourConvex(proj.astype(np.int32))
                        if not is_convex or proj_area < 800:
                            inliers = 0
                        else:
                            obx, oby, obw, obh = cv2.boundingRect(proj.astype(np.float32))
                            aspect = obh / max(obw, 1)
                            if 0.8 <= aspect <= 4.5:
                                obx = max(0, min(obx, fw - 1))
                                oby = max(0, min(oby, fh - 1))
                                obw = min(obw, fw - obx)
                                obh = min(obh, fh - oby)
                                orb_bbox = (obx, oby, obw, obh)

                if inliers > best_inliers:
                    best_inliers = inliers

            # Dynamic continuous ORB score calculation so the bar moves smoothly:
            if best_inliers >= 8:
                orb_score = min(100, int(45 + best_inliers * 3.5))
            elif best_good_count >= 5:
                orb_score = min(40, int(best_good_count * 3.0))
            else:
                orb_score = min(12, int(best_good_count * 1.5))

            # If the bottle is NOT Aquafina (Dasani, Coke, or generic), zero ORB score entirely.
            # This prevents the ORB bar from moving on any non-Aquafina bottle.
            if not best_is_aquafina and best_candidate is not None:
                orb_score = 0

        # ── 3. Decision Synthesis ─────────────────────────────────────────────
        raw_box = None
        if orb_score >= threshold and best_inliers >= 8:
            # Genuine Aquafina logo confirmed
            raw_box = best_candidate if best_candidate is not None else orb_bbox
            detected = True
            bottle_present = True
            bottle_type = "AQUAFINA (WRAPPED)"
            final_conf = orb_score
            trigger = "AQUAFINA LOGO / WRAP"

        elif best_is_aquafina and best_candidate is not None and best_cand_score >= threshold:
            # Genuine Aquafina (unwrapped or dark blue cap + 3D OBJ shape match)
            raw_box = best_candidate
            detected = True
            bottle_present = True
            bottle_type = "AQUAFINA (UNWRAPPED)"
            final_conf = best_cand_score
            trigger = best_trigger

        elif not best_is_aquafina and best_candidate is not None and best_cand_score >= 35:
            # Verified non-Aquafina bottle is in view (Dasani, Coke, Sprite, or similar/other)
            raw_box = best_candidate
            detected = False
            bottle_present = True
            final_conf = 0
            trigger = rejection_reason
            bottle_type = f"NOT AQUAFINA: {identified_non_aqua_type}"

        else:
            raw_box = None
            detected = False
            bottle_present = False
            bottle_type = "NONE"
            final_conf = 0
            trigger = "SCANNING..."

        # Apply temporal box smoother to prevent jitter
        smoothed_box, s_det, s_pres, s_type, s_conf, s_trig = self.box_smoother.update(
            raw_box,
            detected=detected,
            bottle_present=bottle_present,
            bottle_type=bottle_type,
            confidence=final_conf,
            trigger=trigger,
        )

        return DetectionResult(
            detected          = s_det,
            bottle_present    = s_pres,
            bottle_type       = s_type,
            bbox              = smoothed_box,
            confidence        = s_conf,
            cap_score         = best_cap_score,
            shape_score       = best_shape_score,
            sym_score         = best_sym_score,
            orb_score         = orb_score,
            trigger           = s_trig,
            candidate_profile = best_profile,
            candidate_hu      = best_hu,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# 7.5  Vote Buffer — Temporal Stability Filter
# ═══════════════════════════════════════════════════════════════════════════════

class VoteBuffer:
    """Accumulates detection votes over N frames to prevent flickering.
    A state change is only accepted after it wins a majority of votes.
    This makes the detection feel solid and professional, like a real scanner.
    """
    WINDOW = 5   # frames to vote over

    def __init__(self) -> None:
        self._votes: list[str] = []   # "AQUAFINA" | "NOT_AQUAFINA" | "NONE"
        self._stable: str = "NONE"

    def push(self, result: "DetectionResult") -> str:
        """Push a new raw detection result and return the stable verdict."""
        if result.detected:
            vote = "AQUAFINA"
        elif result.bottle_present:
            vote = "NOT_AQUAFINA"
        else:
            vote = "NONE"

        self._votes.append(vote)
        if len(self._votes) > self.WINDOW:
            self._votes.pop(0)

        # Count votes for each state
        counts = {"AQUAFINA": 0, "NOT_AQUAFINA": 0, "NONE": 0}
        for v in self._votes:
            counts[v] += 1

        # Must win a strict majority (> half) in the window to change state
        majority = len(self._votes) / 2.0
        if counts["AQUAFINA"] > majority:
            self._stable = "AQUAFINA"
        elif counts["NOT_AQUAFINA"] > majority:
            self._stable = "NOT_AQUAFINA"
        elif counts["NONE"] > majority:
            self._stable = "NONE"
        # else: keep current stable state (ambiguous)

        return self._stable

    @property
    def stable(self) -> str:
        return self._stable



# ═══════════════════════════════════════════════════════════════════════════════
# 8. Modern Tkinter Dark UI with RL Controls
# ═══════════════════════════════════════════════════════════════════════════════

class DetectorUI:
    HISTORY_MAX = 6

    def __init__(self, args: argparse.Namespace) -> None:
        self.args       = args
        self.store      = EmbeddingStore(Path.cwd() / DB_NAME)
        self.detector   = AquafinaDetector(args.mtl, args.obj, self.store)
        seed_store(self.store, args.reference, self.detector.obj_uniform_w, self.detector.obj_hu)
        self.references = self.store.get_embeddings()
        self.rl_stats   = self.store.get_rl_stats()

        self.latest_frame: np.ndarray | None = None
        self.latest_result: DetectionResult | None = None
        self.result_queue: queue.Queue = queue.Queue(maxsize=2)
        self.stop_event  = threading.Event()
        self.vote_buf    = VoteBuffer()   # temporal stability filter

        self.history: list[dict] = []
        self._fps_times: list[float] = []
        self._fps: float = 0.0

        # Camera Initialization with Fallback
        self.camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(args.camera)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera index {args.camera}")
        # Set camera resolution for smooth feed
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        self._build_window()
        self._build_ui()
        self._bind_keys()

        threading.Thread(target=self._camera_worker, daemon=True).start()
        self._poll()

    def _build_window(self) -> None:
        self.root = tk.Tk()
        self.root.title("Aquafina Vision System — 3D Shape + Color + RL")
        self.root.geometry("1240x760")
        self.root.minsize(980, 660)
        self.root.configure(bg=BG_DARK)
        try:
            import ctypes
            hwnd = int(self.root.frame(), 16)
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int))
        except Exception:
            pass

    def _build_ui(self) -> None:
        root = self.root

        # ── Header ───────────────────────────────────────────────────────────
        hdr = tk.Frame(root, bg="#0a1226", height=60)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        logo_f = tk.Frame(hdr, bg="#0a1226")
        logo_f.pack(side="left", padx=18)
        tk.Label(logo_f, text="◆", font=("Segoe UI", 18, "bold"), fg=ACCENT, bg="#0a1226").pack(side="left", padx=(0, 6))
        tk.Label(logo_f, text="AQUAFINA", font=("Segoe UI", 16, "bold"), fg=TEXT_PRI, bg="#0a1226").pack(side="left")
        tk.Label(logo_f, text="  Bottle Vision System (Autofocus & Precision)", font=("Segoe UI", 10), fg=TEXT_SEC, bg="#0a1226").pack(side="left")

        self.live_badge = tk.Label(hdr, text="● LIVE", font=("Segoe UI", 9, "bold"),
                                   fg=GREEN, bg="#0d241c", padx=12, pady=5)
        self.live_badge.pack(side="right", padx=(0, 20))

        self.fps_lbl = tk.Label(hdr, text="0 fps", font=("Segoe UI", 10), fg=TEXT_SEC, bg="#0a1226")
        self.fps_lbl.pack(side="right", padx=(0, 14))

        # ── Main Content Area ─────────────────────────────────────────────────
        body = tk.Frame(root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=14, pady=(8, 14))

        # Left: Video Feed
        vid_frame = tk.Frame(body, bg="#0a1020", bd=0)
        vid_frame.pack(side="left", fill="both", expand=True)

        self.video_lbl = tk.Label(vid_frame, bg="#0a1020")
        self.video_lbl.pack(fill="both", expand=True, padx=2, pady=2)

        # Right: Telemetry & RL Controls Panel
        panel = tk.Frame(body, bg=BG_PANEL, width=340)
        panel.pack(side="right", fill="y", padx=(12, 0))
        panel.pack_propagate(False)

        self._build_panel(panel)

    def _build_panel(self, panel: tk.Frame) -> None:
        pad = 16

        # ── Detection Status ──────────────────────────────────────────────────
        tk.Label(panel, text="DETECTION STATUS", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", padx=pad, pady=(16, 2))

        self.result_lbl = tk.Label(panel, text="INITIALIZING",
                                   font=("Segoe UI", 18, "bold"),
                                   fg=ACCENT, bg=BG_PANEL, anchor="w")
        self.result_lbl.pack(anchor="w", padx=pad, pady=(0, 2))

        self.trigger_lbl = tk.Label(panel, text="[ STANDBY ]",
                                    font=("Segoe UI", 9, "bold"),
                                    fg=ACCENT2, bg=BG_PANEL)
        self.trigger_lbl.pack(anchor="w", padx=pad)

        # ── Status Indicator (replaces arc gauge — no percentages) ────────────
        self.status_canvas = tk.Canvas(panel, bg=BG_PANEL, width=308, height=110,
                                       highlightthickness=0)
        self.status_canvas.pack(padx=pad, pady=(10, 4))
        self._draw_status("READY")

        # ── Multi-Channel Signal Bars (no % numbers) ──────────────────────────
        sep1 = tk.Frame(panel, bg=TEXT_DIM, height=1)
        sep1.pack(fill="x", padx=pad, pady=(8, 6))

        tk.Label(panel, text="DETECTION SIGNALS", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", padx=pad, pady=(0, 4))

        scores_frame = tk.Frame(panel, bg=BG_PANEL)
        scores_frame.pack(fill="x", padx=pad)
        self.bar_cap = self._score_row(scores_frame, "Dark Blue Cap", ACCENT)
        self.bar_shp = self._score_row(scores_frame, "3D Shape Match", GREEN)
        self.bar_sym = self._score_row(scores_frame, "Symmetry", WARN)
        self.bar_orb = self._score_row(scores_frame, "Logo / Label", ACCENT2)

        # ── Reinforcement Learning Section ────────────────────────────────────
        sep2 = tk.Frame(panel, bg=TEXT_DIM, height=1)
        sep2.pack(fill="x", padx=pad, pady=(8, 6))

        tk.Label(panel, text="REINFORCEMENT LEARNING", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", padx=pad, pady=(0, 2))

        self.rl_threshold_lbl = tk.Label(
            panel, text=f"Threshold: {self.rl_stats['threshold']}",
            font=("Segoe UI", 9, "bold"), fg=TEXT_SEC, bg=BG_PANEL)
        self.rl_threshold_lbl.pack(anchor="w", padx=pad)

        self.rl_stats_lbl = tk.Label(
            panel, text=self._rl_text(),
            font=("Segoe UI", 8), fg=TEXT_SEC, bg=BG_PANEL, justify="left")
        self.rl_stats_lbl.pack(anchor="w", padx=pad, pady=(2, 4))

        # RL Progress Bar
        pb_f = tk.Frame(panel, bg=BG_PANEL)
        pb_f.pack(fill="x", padx=pad, pady=(0, 8))
        self.rl_bar_bg = tk.Canvas(pb_f, bg="#18233c", height=6, highlightthickness=0)
        self.rl_bar_bg.pack(fill="x")
        self.rl_bar_fill = self.rl_bar_bg.create_rectangle(0, 0, 0, 6, fill=GREEN, outline="")

        # ── Interactive Feedback Buttons ──────────────────────────────────────
        btn_frame = tk.Frame(panel, bg=BG_PANEL)
        btn_frame.pack(fill="x", padx=pad, pady=(0, 6))

        self.btn_confirm = self._btn(btn_frame, "✔  Confirm Aquafina  (C)", GREEN, self._confirm_aquafina)
        self.btn_confirm.pack(fill="x", pady=(0, 6))

        self.btn_reject = self._btn(btn_frame, "✖  Not Aquafina  (X)", RED, self._mark_not_aquafina)
        self.btn_reject.pack(fill="x")

        # ── Event Log ─────────────────────────────────────────────────────────
        sep3 = tk.Frame(panel, bg=TEXT_DIM, height=1)
        sep3.pack(fill="x", padx=pad, pady=(8, 6))

        tk.Label(panel, text="FEEDBACK LOG", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", padx=pad, pady=(0, 2))

        self.history_frame = tk.Frame(panel, bg=BG_PANEL)
        self.history_frame.pack(fill="x", padx=pad)

        pos_cnt, neg_cnt = self.store.count_prototypes()
        self.store_lbl = tk.Label(
            panel, text=f"Prototypes: {pos_cnt}+ / {neg_cnt}-",
            font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_PANEL)
        self.store_lbl.pack(anchor="w", padx=pad, pady=(6, 0))

    def _score_row(self, parent: tk.Frame, label: str, colour: str) -> tuple:
        row = tk.Frame(parent, bg=BG_PANEL)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, font=("Segoe UI", 8), fg=TEXT_SEC,
                 bg=BG_PANEL, width=15, anchor="w").pack(side="left")
        bar = tk.Canvas(row, bg="#18233c", height=6, highlightthickness=0)
        bar.pack(side="left", fill="x", expand=True, padx=(4, 0))
        fill = bar.create_rectangle(0, 0, 0, 6, fill=colour, outline="")
        # Dot indicator on far right (no % text)
        dot = tk.Label(row, text="●", font=("Segoe UI", 7), fg=TEXT_DIM, bg=BG_PANEL, width=2)
        dot.pack(side="left", padx=(4, 0))
        return bar, fill, dot

    def _update_score_bar(self, bar_tuple: tuple, value: int) -> None:
        bar, fill, dot = bar_tuple
        bar.update_idletasks()
        w = bar.winfo_width()
        if w < 2:
            w = 160
        bar.coords(fill, 0, 0, int(w * value / 100), 6)
        # Dot lights up green when signal is strong (>= 50), dim otherwise
        dot.configure(fg=GREEN if value >= 50 else (WARN if value >= 25 else TEXT_DIM))

    def _draw_status(self, state: str) -> None:
        """Draw a clean large status badge: AQUAFINA / NOT AQUAFINA / SCANNING."""
        c = self.status_canvas
        c.delete("all")
        w, h = 308, 110

        if state == "AQUAFINA":
            bg_col = "#062b1a"
            border  = GREEN
            icon    = "✔"
            text    = "AQUAFINA"
            sub     = "Bottle Confirmed"
        elif state == "NOT_AQUAFINA":
            bg_col = "#2a0a10"
            border  = RED
            icon    = "✖"
            text    = "NOT AQUAFINA"
            sub     = "Different Brand"
        else:  # READY / SCANNING
            bg_col  = BG_CARD
            border  = TEXT_DIM
            icon    = "◉"
            text    = "SCANNING"
            sub     = "Align bottle in view"

        # Background rectangle
        c.create_rectangle(4, 4, w - 4, h - 4, fill=bg_col, outline=border, width=2)
        # Icon
        c.create_text(w // 2, 30, text=icon, font=("Segoe UI", 22, "bold"), fill=border)
        # Main text
        c.create_text(w // 2, 62, text=text, font=("Segoe UI", 16, "bold"), fill=border)
        # Sub text
        c.create_text(w // 2, 88, text=sub, font=("Segoe UI", 8), fill=TEXT_SEC)

    def _draw_arc(self, value: int, colour: str) -> None:
        """Legacy stub — kept for compatibility, routes to _draw_status."""
        pass  # no-op: arc gauge removed

    def _btn(self, parent: tk.Widget, text: str, colour: str, command) -> tk.Button:
        b = tk.Button(parent, text=text, command=command,
                      bg=colour, fg="#060912",
                      activebackground=colour, activeforeground="#060912",
                      relief="flat", bd=0, font=("Segoe UI", 9, "bold"),
                      padx=10, pady=8, cursor="hand2")
        b.bind("<Enter>", lambda e, b=b, c=colour: b.configure(bg=self._lighten(c)))
        b.bind("<Leave>", lambda e, b=b, c=colour: b.configure(bg=c))
        return b

    @staticmethod
    def _lighten(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = min(255, r + 25); g = min(255, g + 25); b = min(255, b + 25)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _bind_keys(self) -> None:
        self.root.bind("<c>", lambda e: self._confirm_aquafina())
        self.root.bind("<C>", lambda e: self._confirm_aquafina())
        self.root.bind("<space>", lambda e: self._confirm_aquafina())
        self.root.bind("<x>", lambda e: self._mark_not_aquafina())
        self.root.bind("<X>", lambda e: self._mark_not_aquafina())
        self.root.bind("<q>", lambda e: self._close())
        self.root.bind("<Escape>", lambda e: self._close())

    def _push_history(self, label: str, conf: int) -> None:
        ts = dt.datetime.now().strftime("%H:%M:%S")
        self.history.insert(0, {"ts": ts, "label": label, "conf": conf})
        if len(self.history) > self.HISTORY_MAX:
            self.history.pop()
        self._redraw_history()

    def _redraw_history(self) -> None:
        for w in self.history_frame.winfo_children():
            w.destroy()
        for item in self.history:
            col = GREEN if item["label"] == "aquafina" else RED
            sym = "✔" if item["label"] == "aquafina" else "✖"
            text = f"{sym} {item['ts']}  {item['conf']}%  {item['label'].upper()}"
            tk.Label(self.history_frame, text=text, font=("Segoe UI", 8),
                     fg=col, bg=BG_PANEL, anchor="w").pack(fill="x", pady=1)

    def _rl_text(self) -> str:
        s = self.rl_stats
        total = s["true_pos"] + s["false_pos"] + s["false_neg"]
        acc = (s["true_pos"] / total * 100) if total > 0 else 0
        return (f"TP: {s['true_pos']}   FP: {s['false_pos']}   FN: {s['false_neg']}\n"
                f"Accuracy: {acc:.0f}%  ({total} feedback samples)")

    def _update_rl_bar(self) -> None:
        s = self.rl_stats
        total = s["true_pos"] + s["false_pos"] + s["false_neg"]
        acc = (s["true_pos"] / total) if total > 0 else 0
        self.rl_bar_bg.update_idletasks()
        w = self.rl_bar_bg.winfo_width()
        if w < 2:
            w = 280
        self.rl_bar_bg.coords(self.rl_bar_fill, 0, 0, int(w * acc), 6)

    def _refresh_rl_ui(self) -> None:
        self.rl_stats = self.store.get_rl_stats()
        self.rl_threshold_lbl.configure(text=f"Threshold: {self.rl_stats['threshold']}")
        self.rl_stats_lbl.configure(text=self._rl_text())
        self._update_rl_bar()
        pos_cnt, neg_cnt = self.store.count_prototypes()
        self.store_lbl.configure(text=f"Prototypes: {pos_cnt}+ / {neg_cnt}-")

    # ── Camera Worker Thread ──────────────────────────────────────────────────

    def _draw_viewfinder_brackets(self, frame: np.ndarray, x: int, y: int, w: int, h: int, color: tuple, length: int = 18) -> None:
        """Draw camera autofocus corner brackets on the bounding box."""
        t = 2
        # Top-Left
        cv2.line(frame, (x, y), (x + length, y), color, t)
        cv2.line(frame, (x, y), (x, y + length), color, t)
        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - length, y), color, t)
        cv2.line(frame, (x + w, y), (x + w, y + length), color, t)
        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + length, y + h), color, t)
        cv2.line(frame, (x, y + h), (x, y + h - length), color, t)
        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, t)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, t)

    def _camera_worker(self) -> None:
        while not self.stop_event.is_set():
            ok, frame = self.camera.read()
            if not ok:
                time.sleep(0.01)
                continue

            threshold = self.rl_stats["threshold"]
            result = self.detector.detect_frame(frame, self.references, threshold)

            # Push into vote buffer — stabilises result over 5 frames
            stable = self.vote_buf.push(result)

            # Draw visual overlay
            if result.bbox is not None and result.bottle_present:
                x, y, w, h = result.bbox

                if stable == "AQUAFINA":
                    b_col = (0, 220, 140)   # emerald green
                    tag   = "AQUAFINA"
                elif stable == "NOT_AQUAFINA":
                    b_col = (70, 70, 245)   # blue-ish red
                    brand = result.bottle_type.replace("NOT AQUAFINA: ", "").strip()
                    tag   = f"NOT AQUAFINA  {brand}" if brand else "NOT AQUAFINA"
                else:
                    b_col = (110, 110, 110)
                    tag   = "SCANNING..."

                cv2.rectangle(frame, (x, y), (x + w, y + h), b_col, 1)
                self._draw_viewfinder_brackets(frame, x, y, w, h, b_col)

                (tw, _th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                pill_w = max(w, tw + 18)
                cv2.rectangle(frame, (x, max(0, y - 26)), (x + pill_w, y), b_col, -1)
                cv2.putText(frame, tag, (x + 8, max(14, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (8, 12, 20), 2)

            now = time.time()
            self._fps_times.append(now)
            self._fps_times = [t for t in self._fps_times if now - t < 1.5]
            self._fps = len(self._fps_times) / 1.5

            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.result_queue.put_nowait((frame, result))
            except queue.Full:
                pass

    # ── UI Main-Thread Poll Loop ──────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            frame, result = self.result_queue.get_nowait()
            self.latest_frame  = frame.copy()
            self.latest_result = result

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.thumbnail((840, 660), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.video_lbl.configure(image=photo)
            self.video_lbl.image = photo

            # Use the stable vote buffer state for the panel — no flicker
            stable = self.vote_buf.stable
            if stable == "AQUAFINA":
                self.result_lbl.configure(text="AQUAFINA", fg=GREEN)
                self.trigger_lbl.configure(text="[ BOTTLE CONFIRMED ]", fg=GREEN)
                self._draw_status("AQUAFINA")
            elif stable == "NOT_AQUAFINA":
                self.result_lbl.configure(text="NOT AQUAFINA", fg=RED)
                self.trigger_lbl.configure(text=f"[ {result.trigger} ]", fg=RED)
                self._draw_status("NOT_AQUAFINA")
            else:
                self.result_lbl.configure(text="SCANNING", fg=TEXT_SEC)
                self.trigger_lbl.configure(text="[ ALIGN BOTTLE IN VIEW ]", fg=TEXT_DIM)
                self._draw_status("READY")

            self._update_score_bar(self.bar_cap, result.cap_score)
            self._update_score_bar(self.bar_shp, result.shape_score)
            self._update_score_bar(self.bar_sym, result.sym_score)
            self._update_score_bar(self.bar_orb, result.orb_score)

            self.fps_lbl.configure(text=f"{self._fps:.0f} fps")

        except queue.Empty:
            pass

        if not self.stop_event.is_set():
            self.root.after(40, self._poll)

    # ── User Feedback (Reinforcement Learning) ────────────────────────────────

    def _save_feedback_data(self, label: str) -> None:
        if self.latest_frame is None:
            return

        folder = Path.cwd() / "feedback" / label
        folder.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = folder / f"{stamp}.jpg"
        cv2.imwrite(str(img_path), self.latest_frame)

        with (Path.cwd() / "feedback" / "labels.csv").open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([img_path.as_posix(), label])

        # Store candidate shape profile into RL memory
        if self.latest_result and self.latest_result.candidate_profile is not None:
            hu = self.latest_result.candidate_hu if self.latest_result.candidate_hu is not None else np.zeros(7, dtype=np.float32)
            self.store.add_shape_prototype(label, self.latest_result.candidate_profile, hu)

        # Store ORB embedding if positive label
        if label == "aquafina":
            try:
                pts, des, sh, sz = make_embedding(self.latest_frame)
                self.store.add_embedding(label, f"feedback:{img_path.resolve()}", pts, des, sh, sz)
                self.references = self.store.get_embeddings()
            except ValueError:
                pass

    def _confirm_aquafina(self) -> None:
        self._save_feedback_data("aquafina")
        result = self.latest_result
        if result is not None and result.detected:
            self.rl_stats = self.store.record_true_positive()
        elif result is not None and not result.detected:
            self.rl_stats = self.store.record_false_negative()

        self._push_history("aquafina", result.confidence if result else 0)
        self._refresh_rl_ui()
        self.live_badge.configure(text="✔ CONFIRMED", fg=GREEN, bg="#0d241c")
        self.root.after(1400, lambda: self.live_badge.configure(text="● LIVE", fg=GREEN, bg="#0d241c"))

    def _mark_not_aquafina(self) -> None:
        self._save_feedback_data("other")
        result = self.latest_result
        if result is not None and result.detected:
            self.rl_stats = self.store.record_false_positive()

        self._push_history("other", result.confidence if result else 0)
        self._refresh_rl_ui()
        self.live_badge.configure(text="✖ REJECTED", fg=RED, bg="#2b1016")
        self.root.after(1400, lambda: self.live_badge.configure(text="● LIVE", fg=GREEN, bg="#0d241c"))

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.mainloop()

    def _close(self) -> None:
        self.stop_event.set()
        self.camera.release()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aquafina Bottle Detector — 3D Shape + Color + RL")
    p.add_argument("--reference", type=Path, default=Path("orgsize_5861771927856.jpg"))
    p.add_argument("--obj", type=Path, default=Path("aquafina_bottle_3d.obj"))
    p.add_argument("--mtl", type=Path, default=Path("aquafina_bottle_3d.mtl"))
    p.add_argument("--camera", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    DetectorUI(parse_args()).run()
