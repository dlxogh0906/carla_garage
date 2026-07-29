#!/usr/bin/env python3
"""
Object detection benchmark at 3 resolution levels using YOLOv8n.
Compares detection count, confidence, and IoU vs. reference (full-res) on 20 CARLA frames.

Run with:
  /home/kwy00/anaconda3/envs/ch/bin/python tools/detection_benchmark.py
"""

import os, glob, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "/home/kwy00/cheolho/yolov8n.pt"
FRAME_DIR   = "/mnt/2/carla_metric_result/carla_viz/은수이미지보정시각화_0417"
OUT_DIR     = "/mnt/2/carla_metric_result/resolution_benchmark"
NUM_SAMPLES = 20
CONF_THR    = 0.25   # detection confidence threshold
IOU_THR     = 0.45   # NMS iou threshold

# Driving-relevant COCO classes only
DRIVE_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                 5: "bus", 7: "truck", 9: "traffic light", 11: "stop sign"}

RESOLUTIONS = {
    "Large\n(163K px)":  163840,
    "Medium\n(82K px)":   81920,
    "Small\n(41K px)":    40960,
}
REF_LABEL = "Reference\n(native 525K px)"

BG   = "#0d1117"
GRID = "#2a2a3a"
COLS_RES = ["#00d4ff", "#ffd700", "#ff6b6b"]
COL_REF  = "#aaaaaa"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model ──────────────────────────────────────────────────────────────
print("Loading YOLOv8n...")
model = YOLO(MODEL_PATH)
model.overrides["verbose"] = False

# ── Collect frames ──────────────────────────────────────────────────────────
print("Collecting frames...")
raw_frames = []
for png in sorted(glob.glob(f"{FRAME_DIR}/**/enhance_compare/00004.png", recursive=True))[:NUM_SAMPLES]:
    img = Image.open(png).convert("RGB")
    W, H = img.size
    cam = img.crop((0, 0, W // 2, H))   # left half = camera image
    raw_frames.append((cam, Path(png).parts[-3]))

print(f"  {len(raw_frames)} frames  (native {raw_frames[0][0].size})")

# ── Helpers ─────────────────────────────────────────────────────────────────
def resize_to_pixels(img: Image.Image, target_px: int) -> Image.Image:
    w, h = img.size
    scale = math.sqrt(target_px / (w * h))
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

def run_yolo(img: Image.Image):
    """Run YOLO on a PIL image; return boxes in original image coords as numpy."""
    results = model.predict(img, conf=CONF_THR, iou=IOU_THR, verbose=False, device=0)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 6))   # (x1,y1,x2,y2,conf,cls)
    boxes = r.boxes.xyxyn.cpu().numpy()   # normalized [0,1]
    conf  = r.boxes.conf.cpu().numpy()
    cls   = r.boxes.cls.cpu().numpy()
    return np.column_stack([boxes, conf, cls])

def compute_iou(box_a, box_b):
    """IoU between two boxes (x1,y1,x2,y2) normalized."""
    ix1 = max(box_a[0], box_b[0]); iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2]); iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def match_detections(ref_boxes, det_boxes, iou_thr=0.4):
    """Compute recall and precision of det_boxes vs ref_boxes (reference = GT)."""
    if len(ref_boxes) == 0:
        return 1.0, 1.0 if len(det_boxes) == 0 else 0.0, len(det_boxes)
    if len(det_boxes) == 0:
        return 0.0, 0.0, 0
    matched_ref = set()
    matched_det = set()
    for ri, rb in enumerate(ref_boxes):
        best_iou, best_di = 0, -1
        for di, db in enumerate(det_boxes):
            if di in matched_det:
                continue
            iou = compute_iou(rb[:4], db[:4])
            if iou > best_iou:
                best_iou, best_di = iou, di
        if best_iou >= iou_thr and best_di >= 0:
            matched_ref.add(ri)
            matched_det.add(best_di)
    recall    = len(matched_ref) / len(ref_boxes)
    precision = len(matched_det) / len(det_boxes) if len(det_boxes) > 0 else 0.0
    false_pos = len(det_boxes) - len(matched_det)
    return recall, precision, false_pos

def drive_only(detections):
    """Filter to driving-relevant classes only."""
    if len(detections) == 0:
        return detections
    mask = np.array([int(c) in DRIVE_CLASSES for c in detections[:, 5]])
    return detections[mask]

# ── Benchmark loop ───────────────────────────────────────────────────────────
print(f"\nRunning detection on {NUM_SAMPLES} frames...")

# Per-sample per-label accumulator
stats = {label: {"n_det": [], "conf": [], "recall": [], "precision": [], "fp": []}
         for label in list(RESOLUTIONS.keys()) + [REF_LABEL]}

for i, (orig, route) in enumerate(raw_frames):
    print(f"  [{i+1:2d}/{NUM_SAMPLES}] {route}", end="", flush=True)

    # Reference: native resolution
    ref_dets = drive_only(run_yolo(orig))
    stats[REF_LABEL]["n_det"].append(len(ref_dets))
    stats[REF_LABEL]["conf"].append(ref_dets[:, 4].mean() if len(ref_dets) > 0 else 0.0)
    stats[REF_LABEL]["recall"].append(1.0)
    stats[REF_LABEL]["precision"].append(1.0)
    stats[REF_LABEL]["fp"].append(0)

    for label, min_px in RESOLUTIONS.items():
        resized = resize_to_pixels(orig, min_px)
        dets    = drive_only(run_yolo(resized))
        recall, precision, fp = match_detections(ref_dets, dets)
        stats[label]["n_det"].append(len(dets))
        stats[label]["conf"].append(dets[:, 4].mean() if len(dets) > 0 else 0.0)
        stats[label]["recall"].append(recall)
        stats[label]["precision"].append(precision)
        stats[label]["fp"].append(fp)

    print(" ✓")

# ── Print table ──────────────────────────────────────────────────────────────
all_labels = [REF_LABEL] + list(RESOLUTIONS.keys())
print("\n── Detection Results ─────────────────────────────────────────────────")
print(f"{'Label':<28} {'N_det':>6} {'Conf':>6} {'Recall':>8} {'Prec':>7} {'FP':>5}")
print("-"*62)
for lbl in all_labels:
    s = stats[lbl]
    n  = np.mean(s["n_det"])
    c  = np.mean(s["conf"])
    rc = np.mean(s["recall"])
    pr = np.mean(s["precision"])
    fp = np.mean(s["fp"])
    print(f"{lbl.replace(chr(10),' '):<28} {n:>6.1f} {c:>6.3f} {rc:>7.1%} {pr:>6.1%} {fp:>5.1f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
def styled_ax(ax, title, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, color="#aaa", fontsize=8)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.7, linestyle="--")
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
fig.suptitle(
    "YOLOv8n Object Detection: Performance vs. Resolution  |  20 CARLA Frames",
    color="white", fontsize=14, fontweight="bold"
)

x_pos   = [0, 1, 2, 3]
x_lbls  = [l.replace("\n","\n") for l in [REF_LABEL] + list(RESOLUTIONS.keys())]
all_cols = [COL_REF] + COLS_RES

# 1. Objects detected per frame
ax = axes[0, 0]; styled_ax(ax, "Avg Objects Detected / Frame\n(driving-relevant classes only)", "count")
n_means = [np.mean(stats[l]["n_det"]) for l in all_labels]
n_stds  = [np.std(stats[l]["n_det"])  for l in all_labels]
bars = ax.bar(x_pos, n_means, color=all_cols, alpha=0.85, width=0.6,
              yerr=n_stds, error_kw=dict(ecolor="#555", capsize=4))
for bar, v in zip(bars, n_means):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.05, f"{v:.1f}",
            ha="center", va="bottom", color="white", fontsize=9)
ax.set_xticks(x_pos); ax.set_xticklabels(x_lbls, color="#aaa", fontsize=8)

# 2. Average confidence
ax = axes[0, 1]; styled_ax(ax, "Avg Detection Confidence", "confidence score")
c_means = [np.mean(stats[l]["conf"]) for l in all_labels]
c_stds  = [np.std(stats[l]["conf"])  for l in all_labels]
bars = ax.bar(x_pos, c_means, color=all_cols, alpha=0.85, width=0.6,
              yerr=c_stds, error_kw=dict(ecolor="#555", capsize=4))
ax.set_ylim(0, 1.0)
for bar, v in zip(bars, c_means):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}",
            ha="center", va="bottom", color="white", fontsize=9)
ax.set_xticks(x_pos); ax.set_xticklabels(x_lbls, color="#aaa", fontsize=8)
ax.axhline(0.5, color="#555", linestyle=":", linewidth=1)

# 3. Recall vs reference
ax = axes[0, 2]; styled_ax(ax, "Recall vs Reference\n(matched detections / ref count)", "%")
rc_means = [np.mean(stats[l]["recall"]) * 100 for l in all_labels]
rc_stds  = [np.std(stats[l]["recall"])  * 100 for l in all_labels]
bars = ax.bar(x_pos, rc_means, color=all_cols, alpha=0.85, width=0.6,
              yerr=rc_stds, error_kw=dict(ecolor="#555", capsize=4))
ax.set_ylim(0, 115)
for bar, v in zip(bars, rc_means):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.0f}%",
            ha="center", va="bottom", color="white", fontsize=9)
ax.set_xticks(x_pos); ax.set_xticklabels(x_lbls, color="#aaa", fontsize=8)
ax.axhline(90, color="#ff6b6b", linestyle="--", linewidth=1.2)
ax.text(3.6, 90.5, "90%", color="#ff6b6b", fontsize=8)

# 4. Box plot: confidence distribution
ax = axes[1, 0]; styled_ax(ax, "Confidence Distribution\n(box plot per resolution)", "confidence")
data_conf = [stats[l]["conf"] for l in all_labels]
bp = ax.boxplot(data_conf, patch_artist=True, widths=0.5,
                medianprops=dict(color="white", linewidth=2))
for patch, c in zip(bp["boxes"], all_cols):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for el in bp["whiskers"] + bp["caps"] + bp["fliers"]:
    el.set_color("#666")
ax.set_xticks(range(1, len(all_labels)+1))
ax.set_xticklabels(x_lbls, color="#aaa", fontsize=8)
ax.set_ylim(0, 1.0)

# 5. Box plot: recall distribution
ax = axes[1, 1]; styled_ax(ax, "Recall Distribution\n(box plot per resolution)", "recall")
data_recall = [stats[l]["recall"] for l in all_labels]
bp = ax.boxplot(data_recall, patch_artist=True, widths=0.5,
                medianprops=dict(color="white", linewidth=2))
for patch, c in zip(bp["boxes"], all_cols):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for el in bp["whiskers"] + bp["caps"] + bp["fliers"]:
    el.set_color("#666")
ax.set_xticks(range(1, len(all_labels)+1))
ax.set_xticklabels(x_lbls, color="#aaa", fontsize=8)
ax.set_ylim(-0.05, 1.1)

# 6. Sample visualization: bounding boxes at each resolution
ax = axes[1, 2]; styled_ax(ax, "Detection Sample", "")
ax.axis("off")

# Pick a frame that has detections
sample_idx = next((i for i, (img, _) in enumerate(raw_frames)
                   if len(drive_only(run_yolo(img))) > 0), 0)
sample_img, sample_route = raw_frames[sample_idx]

vis_labels = [REF_LABEL] + list(RESOLUTIONS.keys())
vis_cols   = [COL_REF] + COLS_RES
vis_sizes  = [sample_img.size[0] * sample_img.size[1]] + list(RESOLUTIONS.values())

inset_positions = [(0.02, 0.5), (0.52, 0.5), (0.02, 0.0), (0.52, 0.0)]
parent_pos = axes[1, 2].get_position()

for idx, (vlbl, vcol, vpx, (xi, yi)) in enumerate(
        zip(vis_labels, vis_cols, vis_sizes, inset_positions)):
    if idx == 0:
        vis_img = sample_img.copy()
    else:
        vis_img = resize_to_pixels(sample_img, vpx).resize(sample_img.size, Image.LANCZOS)

    dets = drive_only(run_yolo(vis_img if idx == 0 else resize_to_pixels(sample_img, vpx)))

    inax = fig.add_axes([
        parent_pos.x0 + xi * parent_pos.width,
        parent_pos.y0 + yi * parent_pos.height,
        0.46 * parent_pos.width,
        0.46 * parent_pos.height,
    ])
    inax.imshow(np.array(vis_img))
    W, H = vis_img.size
    for det in dets:
        x1, y1, x2, y2 = det[0]*W, det[1]*H, det[2]*W, det[3]*H
        cls_id = int(det[5])
        cls_name = DRIVE_CLASSES.get(cls_id, str(cls_id))
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=1.5, edgecolor=vcol, facecolor="none")
        inax.add_patch(rect)
        inax.text(x1, y1-2, f"{cls_name} {det[4]:.2f}",
                  color=vcol, fontsize=5, fontweight="bold")
    short = vlbl.replace("Reference\n(native 525K px)", "Ref (525K)").replace("\n", " ")
    inax.set_title(f"{short} | {len(dets)} obj", color=vcol, fontsize=7, fontweight="bold")
    for sp in inax.spines.values():
        sp.set_edgecolor(vcol); sp.set_linewidth(1.5)
    inax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = f"{OUT_DIR}/detection_benchmark.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
print(f"\nSaved: {out_path}")

# ── Final summary ────────────────────────────────────────────────────────────
print("\n" + "="*60)
lbls = list(RESOLUTIONS.keys())
lnames = [l.replace("\n", " ") for l in lbls]
print("Detection drop vs Reference:")
ref_n  = np.mean(stats[REF_LABEL]["n_det"])
ref_rc = np.mean(stats[REF_LABEL]["recall"])
for lbl, lname in zip(lbls, lnames):
    n  = np.mean(stats[lbl]["n_det"])
    rc = np.mean(stats[lbl]["recall"])
    print(f"  {lname}: recall={rc:.1%}  n_det={n:.1f} (ref={ref_n:.1f})")
