#!/usr/bin/env python3
"""
Resolution benchmark for Alpamayo camera input.
Measures tokenization time, vision token count, and image quality (PSNR/SSIM)
across 3 min_pixels settings on real CARLA camera frames.

Run with:
  /mnt/2/alpamayo1.5/a1_5_venv/bin/python tools/resolution_benchmark.py
"""

import sys, os, time, math, io, glob
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, "/mnt/2/alpamayo1.5/src")
from transformers import AutoProcessor

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSOR_BASE = "Qwen/Qwen3-VL-2B-Instruct"
FRAME_DIR      = "/mnt/2/carla_metric_result/carla_viz/은수이미지보정시각화_0417"
OUT_DIR        = "/mnt/2/carla_metric_result/resolution_benchmark"
NUM_SAMPLES    = 20
NUM_FRAMES     = 4   # temporal frames per camera (matches Alpamayo training)
NUM_REPEATS    = 3   # repeat tokenization to get stable timing

RESOLUTIONS = {
    "Large\n(163,840 px)":  163840,
    "Medium\n(81,920 px)":   81920,
    "Small\n(40,960 px)":    40960,
}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Collect sample frames ─────────────────────────────────────────────────────
print("Collecting sample frames...")
raw_frames = []
for png in sorted(glob.glob(f"{FRAME_DIR}/**/enhance_compare/00004.png", recursive=True)):
    img = Image.open(png).convert("RGB")
    W, H = img.size
    cam_frame = img.crop((0, 0, W // 2, H))   # left half = camera image
    raw_frames.append((cam_frame, Path(png).parts[-3]))  # (image, route_name)
    if len(raw_frames) >= NUM_SAMPLES:
        break

print(f"Loaded {len(raw_frames)} frames  (native size {raw_frames[0][0].size})")

# ── PSNR helper ───────────────────────────────────────────────────────────────
def psnr(orig: Image.Image, comp: Image.Image) -> float:
    a = np.array(orig, dtype=np.float32)
    b = np.array(comp.resize(orig.size, Image.LANCZOS), dtype=np.float32)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))

def resize_to_pixels(img: Image.Image, target_px: int) -> Image.Image:
    w, h = img.size
    aspect = w / h
    new_h = int(math.sqrt(target_px / aspect))
    new_w = int(new_h * aspect)
    return img.resize((new_w, new_h), Image.LANCZOS)

# ── Build processor variants ──────────────────────────────────────────────────
print("Loading processors (3 variants)...")
processors = {}
for label, min_px in RESOLUTIONS.items():
    processors[label] = AutoProcessor.from_pretrained(
        PROCESSOR_BASE,
        min_pixels=min_px,
        max_pixels=min_px + 32768,   # tight range to force target size
    )
    print(f"  {label.replace(chr(10),' ')} : ok")

# ── Benchmark loop ────────────────────────────────────────────────────────────
def make_chat_message(frames_list):
    content = []
    for i, frm in enumerate(frames_list):
        content.append({"type": "text", "text": f"frame {i} "})
        content.append({"type": "image", "image": frm})
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a driving assistant."}]},
        {"role": "user",   "content": content + [{"type": "text", "text": "Describe the scene."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<|answer_start|>"}]},
    ]

results = {label: {"tok_time": [], "n_tokens": [], "n_img_tokens": [], "psnr_db": []}
           for label in RESOLUTIONS}

print(f"\nRunning benchmark on {len(raw_frames)} samples × 3 resolutions × {NUM_REPEATS} repeats...")
for sample_i, (orig_frame, route) in enumerate(raw_frames):
    print(f"  [{sample_i+1:2d}/{NUM_SAMPLES}] {route}", end="", flush=True)

    # 4 temporal frames from same image (simulate history)
    frames_4 = [orig_frame] * NUM_FRAMES

    for label, min_px in RESOLUTIONS.items():
        proc = processors[label]
        # Downsize to target pixels first (processor will also apply min_pixels clamp)
        resized_4 = [resize_to_pixels(f, min_px) for f in frames_4]

        # PSNR vs original
        p = psnr(orig_frame, resized_4[0])
        results[label]["psnr_db"].append(p)

        # Tokenization timing
        msg = make_chat_message(resized_4)
        times = []
        n_tok = n_img = 0
        for _ in range(NUM_REPEATS):
            t0 = time.perf_counter()
            out = proc.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            times.append(time.perf_counter() - t0)
            ids = out["input_ids"][0]
            n_tok = len(ids)
            # Qwen3-VL image token id = 151655 (visual_start) context
            # Count all image-patch tokens (pixel_values > 0 means image)
            if "pixel_values" in out:
                pv = out["pixel_values"]
                n_img = pv.shape[0] if pv.ndim >= 1 else 0
            else:
                n_img = 0

        results[label]["tok_time"].append(min(times))
        results[label]["n_tokens"].append(n_tok)
        results[label]["n_img_tokens"].append(n_img)

    print(" ✓")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n── Results ────────────────────────────────────────────────────────")
print(f"{'Label':<30} {'TokTime(ms)':>12} {'±':>6} {'Tokens':>8} {'ImgToks':>8} {'PSNR(dB)':>10}")
print("-"*78)
summary = {}
for label, r in results.items():
    tt = np.array(r["tok_time"]) * 1000
    nt = np.array(r["n_tokens"])
    ni = np.array(r["n_img_tokens"])
    ps = np.array(r["psnr_db"])
    summary[label] = dict(
        tt_mean=tt.mean(), tt_std=tt.std(),
        nt_mean=nt.mean(),
        ni_mean=ni.mean(),
        ps_mean=ps.mean(), ps_std=ps.std(),
        tt_all=tt, nt_all=nt, ni_all=ni, ps_all=ps,
    )
    print(f"{label.replace(chr(10),' '):<30} {tt.mean():>10.1f}ms {tt.std():>5.1f}  "
          f"{nt.mean():>8.0f} {ni.mean():>8.0f} {ps.mean():>9.1f}dB")

# ── Plot ───────────────────────────────────────────────────────────────────────
BG   = "#0d1117"
GRID = "#2a2a3a"
COLS = ["#00d4ff", "#ffd700", "#ff6b6b"]
labels = list(RESOLUTIONS.keys())

fig = plt.figure(figsize=(22, 14), facecolor=BG)
fig.suptitle(
    "Alpamayo Resolution Benchmark  |  20 CARLA Frames × 3 Resolutions × 3 Repeats",
    color="white", fontsize=15, fontweight="bold", y=0.98,
)

def styled_ax(ax, title, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="--")
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    return ax

axes = [fig.add_subplot(2, 4, i+1) for i in range(8)]

# 1. Tokenization time box plot
ax = styled_ax(axes[0], "Tokenization Time (ms)", "ms")
bp = ax.boxplot(
    [summary[l]["tt_all"] for l in labels],
    patch_artist=True, widths=0.5,
    medianprops=dict(color="white", linewidth=2),
)
for patch, c in zip(bp["boxes"], COLS):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for el in bp["whiskers"] + bp["caps"] + bp["fliers"]:
    el.set_color("#888")
ax.set_xticks([1,2,3])
ax.set_xticklabels([l.replace("\n", "\n") for l in labels], color="#aaa", fontsize=8)

# 2. Total input tokens
ax = styled_ax(axes[1], "Total Input Tokens", "# tokens")
means_nt = [summary[l]["nt_mean"] for l in labels]
bars = ax.bar([1,2,3], means_nt, color=COLS, alpha=0.85, width=0.5)
ax.set_xticks([1,2,3])
ax.set_xticklabels([l.replace("\n","\n") for l in labels], color="#aaa", fontsize=8)
for bar, v in zip(bars, means_nt):
    ax.text(bar.get_x()+bar.get_width()/2, v+5, f"{v:.0f}", ha="center", color="white", fontsize=9)

# 3. Image (pixel_values) token count — if available
ax = styled_ax(axes[2], "Image Patch Tokens\n(vision encoder input)", "# image tokens")
means_ni = [summary[l]["ni_mean"] for l in labels]
bars = ax.bar([1,2,3], means_ni, color=COLS, alpha=0.85, width=0.5)
ax.set_xticks([1,2,3])
ax.set_xticklabels([l.replace("\n","\n") for l in labels], color="#aaa", fontsize=8)
for bar, v in zip(bars, means_ni):
    ax.text(bar.get_x()+bar.get_width()/2, v+2, f"{v:.0f}", ha="center", color="white", fontsize=9)

# 4. PSNR box plot
ax = styled_ax(axes[3], "Image Quality vs Original (PSNR)", "dB  (higher = better)")
bp2 = ax.boxplot(
    [summary[l]["ps_all"] for l in labels],
    patch_artist=True, widths=0.5,
    medianprops=dict(color="white", linewidth=2),
)
for patch, c in zip(bp2["boxes"], COLS):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for el in bp2["whiskers"] + bp2["caps"] + bp2["fliers"]:
    el.set_color("#888")
ax.set_xticks([1,2,3])
ax.set_xticklabels([l.replace("\n","\n") for l in labels], color="#aaa", fontsize=8)
# Annotation: PSNR reference zones
for thresh, txt in [(30, "Good"), (25, "Acceptable"), (20, "Degraded")]:
    ax.axhline(thresh, color="#555", linestyle=":", linewidth=1)
    ax.text(3.4, thresh+0.3, txt, color="#888", fontsize=7)

# 5. Token count per image (bars, per-resolution)
ax = styled_ax(axes[4], "Estimated Vision Tokens\nper Camera Frame", "tokens / frame")
per_img = [min_px / (14*14) for min_px in RESOLUTIONS.values()]
bars = ax.bar([1,2,3], per_img, color=COLS, alpha=0.85, width=0.5)
ax.set_xticks([1,2,3])
ax.set_xticklabels([l.replace("\n","\n") for l in labels], color="#aaa", fontsize=8)
for bar, v in zip(bars, per_img):
    ax.text(bar.get_x()+bar.get_width()/2, v+2, f"{v:.0f}", ha="center", color="white", fontsize=9)
ax.set_ylabel("tokens/frame (theoretical)", color="#aaa", fontsize=9)

# 6. Total vision token budget (1 cam × 4 frames)
ax = styled_ax(axes[5], "Total Vision Token Budget\n(1 cam × 4 frames)", "total vision tokens")
total_1cam = [v * 4 for v in per_img]
bars = ax.bar([1,2,3], total_1cam, color=COLS, alpha=0.85, width=0.5)
ax.set_xticks([1,2,3])
ax.set_xticklabels([l.replace("\n","\n") for l in labels], color="#aaa", fontsize=8)
for bar, v in zip(bars, total_1cam):
    ax.text(bar.get_x()+bar.get_width()/2, v+5, f"{v:.0f}", ha="center", color="white", fontsize=9)

# 7. 4-cam total token budget comparison
ax = styled_ax(axes[6], "4-Camera Total Token Budget\n(4 cam × 4 frames)", "total vision tokens")
total_4cam = [v * 4 * 4 for v in per_img]
bars = ax.bar([1,2,3], total_4cam, color=COLS, alpha=0.85, width=0.5)
for bar, v in zip(bars, total_4cam):
    ax.text(bar.get_x()+bar.get_width()/2, v+10, f"{v:.0f}", ha="center", color="white", fontsize=9)
ax.set_xticks([1,2,3])
ax.set_xticklabels([l.replace("\n","\n") for l in labels], color="#aaa", fontsize=8)
# Horizontal line at current 1-cam budget
ax.axhline(total_1cam[0], color=COLS[0], linestyle="--", linewidth=1.5)
ax.text(3.55, total_1cam[0]+5, "≡ Current\n1-cam budget", color=COLS[0], fontsize=7)

# 8. Sample images at 3 resolutions (one row)
ax8 = styled_ax(axes[7], "Visual Quality Sample", "")
ax8.axis("off")

sample_img, _ = raw_frames[5]
inset_w = 0.28; inset_h = 0.20
y0 = 0.04
positions = [(0.02, y0), (0.35, y0), (0.68, y0)]
for (x0, y_), (label, min_px), col in zip(positions, RESOLUTIONS.items(), COLS):
    resized = resize_to_pixels(sample_img, min_px)
    display = resized.resize(sample_img.size, Image.NEAREST)
    inset = fig.add_axes([
        axes[7].get_position().x0 + x0 * axes[7].get_position().width,
        axes[7].get_position().y0 + y_ * axes[7].get_position().height,
        inset_w * axes[7].get_position().width,
        inset_h * axes[7].get_position().height * 4,
    ])
    inset.imshow(np.array(display))
    inset.set_title(f"{label}\n{resized.size[0]}x{resized.size[1]}",
                    color=col, fontsize=7, fontweight="bold")
    for sp in inset.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(2)
    inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = f"{OUT_DIR}/resolution_benchmark.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
print(f"\nSaved: {out_path}")

# ── Text summary ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
lnames = [l.replace("\n"," ") for l in labels]
tt_means = [summary[l]["tt_mean"] for l in labels]
ps_means = [summary[l]["ps_mean"] for l in labels]

print(f"\nTokenization time speedup vs Large:")
for l, tt in zip(lnames, tt_means):
    print(f"  {l:<30}: {tt*1000:.1f}ms  ({tt_means[0]/tt:.2f}x faster vs Large)")

print(f"\nPSNR vs original image:")
for l, ps in zip(lnames, ps_means):
    print(f"  {l:<30}: {ps:.1f} dB")

print(f"\n4-cam inference token budget vs current (1-cam Large):")
for l, t4 in zip(lnames, total_4cam):
    ratio = t4 / total_1cam[0]
    print(f"  4cam × {l:<22}: {t4:.0f} tokens  ({ratio:.1f}x vs current)")

print(f"\nConclusion:")
print(f"  Small (40K) 4-cam: {total_4cam[2]:.0f} tokens = {total_4cam[2]/total_1cam[0]:.1f}x current budget")
print(f"  PSNR at Small: {ps_means[2]:.1f} dB  (>25dB = visually acceptable for semantic tasks)")
