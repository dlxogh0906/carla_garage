"""Unified VLA dashboard renderer.

This matches the dark per-frame dashboard used by the Qwen snapshot GIFs:
large front camera, right-side VLA state table, TTC history, and speed-scale
history with meta-action colors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


BG = "#101027"
PANEL = "#1a1a2d"
GRID = "#40405f"
TEXT = "#f4f4f8"
MUTED = "#a5a5b7"
BLUE = "#7ea7d8"
GREEN = "#2ee881"
YELLOW = "#e5c100"
RED = "#ff6464"

ACTION_NAMES = [
    "proceed",
    "cautious",
    "slow_down",
    "yield",
    "crawl",
    "hard_brake",
    "stop",
    "emerg_stop",
]
ACTION_COLORS = [
    "#33d17a",
    "#f1c232",
    "#e06c75",
    "#e67e22",
    "#56b6c2",
    "#61afef",
    "#c678dd",
    "#7f849c",
]
_HISTORY: Dict[str, Dict[str, list]] = {}


def render_qwen_dashboard(
    frame_data: Dict[str, Any],
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Render one Unified VLA dashboard frame and optionally save it."""
    history_key = str(frame_data.get("history_key", "default"))
    history = _get_history(history_key, reset=bool(frame_data.get("reset_history", False)))

    step = int(frame_data.get("step", 0))
    ttc = _float(frame_data.get("ttc"), 999.0)
    speed_scale = _clip01(frame_data.get("speed_scale", 1.0))
    action_id = _action_id(frame_data)

    history["step"].append(step)
    history["ttc"].append(min(ttc, 10.0) if np.isfinite(ttc) and ttc < 999.0 else np.nan)
    history["scale"].append(speed_scale)
    history["action"].append(action_id)
    history["rule_hold"].append(1.0 if bool(frame_data.get("rule_hold_active", False)) else 0.0)

    fig = plt.figure(figsize=(17.02, 10.90), dpi=100, facecolor=BG)
    title = f"Unified VLA - step {step} | ego {_float(frame_data.get('ego_speed'), 0.0):.1f} m/s"
    fig.text(0.5, 0.979, title, color=TEXT, ha="center", va="top", fontsize=14, weight="bold")

    rear_image = frame_data.get("rear_image")
    if rear_image is None:
        camera_ax = fig.add_axes([0.032, 0.545, 0.422, 0.330], facecolor=BG)
        rear_ax = None
    else:
        camera_ax = fig.add_axes([0.032, 0.610, 0.422, 0.265], facecolor=BG)
        rear_ax = fig.add_axes([0.032, 0.380, 0.422, 0.165], facecolor=BG)
    status_ax = fig.add_axes([0.571, 0.345, 0.422, 0.600], facecolor=PANEL)
    ttc_ax = fig.add_axes([0.032, 0.047, 0.422, 0.288], facecolor=PANEL)
    scale_ax = fig.add_axes([0.571, 0.047, 0.422, 0.288], facecolor=PANEL)

    _draw_camera(camera_ax, frame_data, image_key="image", title_prefix="Front", state_overlay=True)
    if rear_ax is not None:
        _draw_camera(rear_ax, frame_data, image_key="rear_image", title_prefix="Rear", state_overlay=False)
    _draw_status(status_ax, frame_data)
    _draw_ttc_history(ttc_ax, history, _float(frame_data.get("ttc_threshold"), 3.0))
    _draw_scale_history(scale_ax, history)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rendered = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

    return rendered


def _get_history(key: str, reset: bool = False) -> Dict[str, list]:
    if reset or key not in _HISTORY:
        _HISTORY[key] = {"step": [], "ttc": [], "scale": [], "action": [], "rule_hold": []}
    return _HISTORY[key]


def _draw_camera(
    ax,
    fd: Dict[str, Any],
    image_key: str = "image",
    title_prefix: str = "Camera",
    state_overlay: bool = True,
) -> None:
    image = fd.get(image_key)
    if image is None:
        img = np.zeros((512, 1024, 3), dtype=np.uint8)
    else:
        img = np.asarray(image)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

    if state_overlay:
        speed_scale = _clip01(fd.get("speed_scale", 1.0))
        rule_hold = bool(fd.get("rule_hold_active", False))
        risky = bool(fd.get("risky", False))
        if rule_hold or speed_scale < 0.05:
            state_label, color = "Rule Hold" if rule_hold else "Stop", RED
        elif risky or speed_scale < 0.95:
            state_label, color = "Intervening", YELLOW
        else:
            state_label, color = "Normal", GREEN
        label = f"{title_prefix} | {state_label}"
    else:
        label, color = f"{title_prefix} View", BLUE

    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(1.4)
    ax.text(0.5, 1.005, label, color=color, ha="center", va="bottom",
            transform=ax.transAxes, fontsize=10)


def _draw_status(ax, fd: Dict[str, Any]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    y = 0.965

    def section(title: str) -> None:
        nonlocal y
        ax.axhline(y + 0.012, color="#343454", linewidth=0.9)
        ax.text(0.04, y, f"- {title} -", color=BLUE, fontsize=8.8,
                va="top", weight="bold")
        y -= 0.032

    def row(label: str, value: str, color: str = TEXT) -> None:
        nonlocal y
        ax.text(0.04, y, label, color=MUTED, fontsize=9.3, va="top")
        ax.text(0.48, y, value, color=color, fontsize=9.3, va="top", weight="bold")
        y -= 0.035

    action_id = _action_id(fd)
    action_name = _action_name(fd, action_id)
    multiplier = _clip01(fd.get("speed_scale", fd.get("action_multiplier", 1.0)))
    action_color = RED if multiplier <= 0.05 else (YELLOW if multiplier < 0.95 else GREEN)

    section("Speed Critic (Discrete Meta-Action)")
    row("Action", f"[{action_id}] {action_name}", action_color)
    row("Multiplier", f"x{multiplier:.1f}", action_color)
    y -= 0.005

    hold_active = bool(fd.get("rule_hold_active", False))
    tl_prestop_active = bool(fd.get("tl_prestop_active", False))
    gap_active = bool(fd.get("gap_active", False))
    recovery_active = bool(fd.get("recovery_active", False))
    turn_caution_active = bool(fd.get("turn_caution_active", False))
    escape_reverse_active = bool(fd.get("escape_reverse_active", False))
    gap_decision = str(fd.get("gap_decision", "unknown"))
    gap_conf = _float(fd.get("gap_confidence"), 0.0)
    gap_count = int(_float(fd.get("gap_candidate_count"), 0.0))
    rule_type = str(fd.get("rule_type", "none"))
    rule_conf = _float(fd.get("rule_confidence"), 0.0)
    tl_state = str(fd.get("traffic_light_state", "unknown"))
    section("State Machine (Rule Hold)")
    row("Hold", "active" if hold_active else "inactive", RED if hold_active else GREEN)
    row("Rule Critic", f"{rule_type} conf={rule_conf:.2f}", BLUE if rule_type != "none" else MUTED)
    row("TL State", tl_state, GREEN if tl_state == "green" else (RED if tl_state in ("red", "yellow") else MUTED))
    if tl_prestop_active:
        row("TL Pre-stop", "active", YELLOW)
    gap_color = RED if gap_decision in ("stop", "yield") else (YELLOW if gap_active else MUTED)
    row("Gap Critic", f"{gap_decision} conf={gap_conf:.2f}", gap_color)
    if gap_count > 0:
        gx = _float(fd.get("gap_nearest_x"), np.nan)
        gy = _float(fd.get("gap_nearest_y"), np.nan)
        row("Gap Cand", f"{gap_count} nearest {_fmt_num(gx)},{_fmt_num(gy)}", YELLOW if gap_active else MUTED)
    if turn_caution_active:
        row("Turn Caution", "speed cap", YELLOW)
    if recovery_active:
        row("Recovery", "reverse escape" if escape_reverse_active else "crawl override", YELLOW)
    y -= 0.005

    ttc = _float(fd.get("ttc"), 999.0)
    ttc_source = str(fd.get("ttc_source", "none"))
    front_dist = _float(fd.get("front_distance"), 999.0)
    section("TTC")
    ttc_text = f"{ttc:.2f}s [{ttc_source}]" if np.isfinite(ttc) and ttc < 999.0 else "n/a"
    row("TTC", ttc_text,
        RED if np.isfinite(ttc) and ttc < _float(fd.get("ttc_threshold"), 3.0) else GREEN)
    row("Front dist", f"{front_dist:.1f}m" if front_dist < 999.0 else "n/a")
    y -= 0.005

    ego_speed = _float(fd.get("ego_speed"), 0.0)
    tfpp_speed = _float(fd.get("tfpp_target_speed"), np.nan)
    final_speed = _float(fd.get("final_target_speed"), np.nan)
    guard_scale = _float(fd.get("guard_scale"), 1.0)
    section("Speed (m/s)")
    row("Ego speed", f"{ego_speed:.2f}")
    row("TF++ target", _fmt_num(tfpp_speed))
    row("Final target", _fmt_num(final_speed))
    row("Scale applied", f"{multiplier:.3f}  (guard {guard_scale:.2f})", YELLOW if multiplier < 0.95 else GREEN)
    y -= 0.005

    section("VLM")
    row("Called", "yes" if bool(fd.get("vlm_called", False)) else "no",
        GREEN if bool(fd.get("vlm_called", False)) else MUTED)
    row("Trigger", str(fd.get("vlm_trigger", "none")), BLUE if str(fd.get("vlm_trigger", "none")) != "none" else MUTED)
    row("Ready", "yes" if bool(fd.get("vlm_ready", True)) else "no",
        GREEN if bool(fd.get("vlm_ready", True)) else RED)
    y -= 0.005

    emergency_active = bool(fd.get("emergency_yield_active", False))
    emergency_action = str(fd.get("emergency_action", "none"))
    emergency_conf = _float(fd.get("emergency_confidence"), 0.0)
    section("Emergency Yield")
    row(
        "Enabled",
        f"{'yes' if bool(fd.get('emergency_enabled', False)) else 'no'} / rear {'yes' if bool(fd.get('emergency_rear_available', False)) else 'no'}",
        GREEN if bool(fd.get("emergency_enabled", False)) and bool(fd.get("emergency_rear_available", False)) else RED,
    )
    row("State", "active" if emergency_active else "inactive", YELLOW if emergency_active else MUTED)
    row("Rear critic", f"{emergency_action} conf={emergency_conf:.2f}",
        YELLOW if emergency_active else (BLUE if emergency_conf >= 0.4 else MUTED))
    row("Direction", str(fd.get("emergency_yield_direction", "unknown")),
        YELLOW if emergency_active else MUTED)
    row("Override", str(fd.get("emergency_control_phase", "inactive")),
        GREEN if bool(fd.get("emergency_control_override", False)) else MUTED)
    row("Offset", f"{_float(fd.get('emergency_offset_m'), 0.0):.2f}m",
        YELLOW if abs(_float(fd.get("emergency_offset_m"), 0.0)) >= 0.01 else MUTED)
    y -= 0.005

    section("Control")
    row("Throttle", f"{_float(fd.get('control_throttle'), 0.0):.3f}")
    row("Brake", f"{_float(fd.get('control_brake'), 0.0):.3f}")
    row("Steer", f"{_float(fd.get('control_steer'), 0.0):.3f}")


def _draw_ttc_history(ax, history: Dict[str, list], threshold: float) -> None:
    _style_plot(ax, "TTC History (red = rule hold)", "TTC (s)")
    steps = np.asarray(history["step"], dtype=float)
    ttcs = np.asarray(history["ttc"], dtype=float)
    holds = np.asarray(history["rule_hold"], dtype=float)

    if len(steps) == 0:
        return
    for step, hold in zip(steps, holds):
        if hold > 0:
            ax.axvspan(step - 0.5, step + 0.5, color=RED, alpha=0.24, linewidth=0)

    if len(steps) >= 2 and np.isfinite(ttcs).any():
        ax.plot(steps, ttcs, color="#9fbde5", linewidth=1.4, label="TTC (s)")
    ax.axhline(threshold, color=YELLOW, linestyle="--", linewidth=1.2, label=f"Threshold {threshold:.1f}s")
    ax.set_ylim(0, 10)
    _auto_xlim(ax, steps)
    ax.legend(loc="upper left", fontsize=7.5, facecolor=PANEL, edgecolor="#343454",
              labelcolor=TEXT)


def _draw_scale_history(ax, history: Dict[str, list]) -> None:
    _style_plot(ax, "Speed Scale History (color = meta-action)", "Scale")
    steps = np.asarray(history["step"], dtype=float)
    scales = np.asarray(history["scale"], dtype=float)
    actions = np.asarray(history["action"], dtype=int)

    if len(steps) > 0:
        for step, action in zip(steps, actions):
            ax.axvspan(step - 0.5, step + 0.5, color=ACTION_COLORS[action % len(ACTION_COLORS)],
                       alpha=0.34, linewidth=0)
    if len(steps) >= 2:
        ax.plot(steps, scales, color="#d7d7e8", linewidth=1.1)
    ax.axhline(1.0, color="#7f849c", linestyle=":", linewidth=0.8)
    ax.axhline(0.0, color="#7f849c", linestyle=":", linewidth=0.8)
    ax.set_ylim(-0.05, 1.10)
    _auto_xlim(ax, steps)

    handles = [
        mpatches.Patch(color=ACTION_COLORS[i], label=f"{i}:{name[:8]}")
        for i, name in enumerate(ACTION_NAMES)
    ]
    ax.legend(handles=handles, loc="lower left", ncol=2, fontsize=7.0,
              facecolor=PANEL, edgecolor="#343454", labelcolor=TEXT)


def _style_plot(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10, pad=8)
    ax.set_xlabel("Step", color=MUTED, fontsize=8.5)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8.5)
    ax.tick_params(colors="#8b8ba2", labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color("#3c3c62")
    ax.grid(True, color=GRID, linestyle=":", alpha=0.55)


def _auto_xlim(ax, steps: np.ndarray) -> None:
    if len(steps) == 0:
        ax.set_xlim(-0.05, 0.05)
        return
    if len(steps) == 1:
        ax.set_xlim(steps[0] - 0.05, steps[0] + 0.05)
        return
    ax.set_xlim(float(np.nanmin(steps)) - 0.5, float(np.nanmax(steps)) + 0.5)


def _action_id(fd: Dict[str, Any]) -> int:
    for key in ("action_id", "qwen_action_id", "meta_action_id"):
        try:
            value = int(fd.get(key))
            if 0 <= value < len(ACTION_NAMES):
                return value
        except (TypeError, ValueError):
            pass
    name = str(fd.get("action", fd.get("qwen_action", ""))).lower()
    for idx, action_name in enumerate(ACTION_NAMES):
        if name == action_name:
            return idx
    scale = _clip01(fd.get("speed_scale", 1.0))
    if scale <= 0.05:
        return 6
    if scale < 0.25:
        return 5
    if scale < 0.5:
        return 3
    if scale < 0.75:
        return 2
    if scale < 0.95:
        return 1
    return 0


def _action_name(fd: Dict[str, Any], action_id: int) -> str:
    for key in ("action", "qwen_action", "meta_action"):
        value = fd.get(key)
        if value:
            return str(value)
    return ACTION_NAMES[action_id]


def _fmt_num(value: float) -> str:
    return f"{value:.2f}" if np.isfinite(value) else "n/a"


def _float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _clip01(value: Any) -> float:
    return max(0.0, min(1.0, _float(value, 1.0)))
