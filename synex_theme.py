"""
SynEx brand and data-visualization color palette.

Single source of truth for UI branding and warm/cold percentile colors.
Import in app.py and backend_waves.py.

Usage:
    from synex_theme import SYnex_BRAND, SYnex_WARM, warm_rgba, cold_rgba
"""

from __future__ import annotations


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --- UI / Brand ---
SYnex_BRAND = {
    "primary": "#0056B3",
    "nav_bg": "#E6F2FF",
    "nav_hover": "#CCE5FF",
    "nav_active": "#FFFFFF",
    "text_on_primary": "#FFFFFF",
    "text_on_light": "#000000",
}

# --- Warm (heat) percentiles ---
SYnex_WARM = {
    "p75": "#FFE699",   # Moderate
    "p90": "#FF9933",   # Strong
    "p95": "#CC0000",   # Extreme
    "rec": "#E91E8C",   # All-time record — vivid pink, distinct from red Extreme (cf. Cold rec indigo vs blue)
}

# --- Cold percentiles ---
SYnex_COLD = {
    "p25": "#CCF2FF",   # Moderate
    "p10": "#3399FF",   # Strong
    "p5": "#0056B3",    # Extreme — aligned with brand primary
    "rec": "#4B0082",   # All-time record (indigo)
}

# --- Overlays & auxiliary series ---
SYnex_OVERLAY = {
    "mslp_contour": "#2E7D32",
    "z500_contour": "#0056B3",
    "apparent_temp": "#388E3C",
    "border": "#000000",
    "grid": "rgba(200,200,200,0.3)",
    "annotation": "gray",
}

# --- Meteogram fill opacities (hex from palette above) ---
_METEO_ALPHA = {"moderate": 0.50, "strong": 0.60, "extreme": 0.70, "record": 0.85}


def warm_rgba(level: str) -> str:
    """Return rgba string for meteogram warm fills. level: moderate|strong|extreme|record."""
    key = {"moderate": "p75", "strong": "p90", "extreme": "p95", "record": "rec"}[level]
    return _hex_to_rgba(SYnex_WARM[key], _METEO_ALPHA[level])


def cold_rgba(level: str) -> str:
    """Return rgba string for meteogram cold fills. level: moderate|strong|extreme|record."""
    key = {"moderate": "p25", "strong": "p10", "extreme": "p5", "record": "rec"}[level]
    return _hex_to_rgba(SYnex_COLD[key], _METEO_ALPHA[level])


def map_extremes_colorscale() -> list[list]:
    """Plotly colorscale for synoptic map discrete extremes (cold → warm)."""
    return [
        [0.0, SYnex_COLD["rec"]],
        [0.125, SYnex_COLD["rec"]],
        [0.125, SYnex_COLD["p5"]],
        [0.25, SYnex_COLD["p5"]],
        [0.25, SYnex_COLD["p10"]],
        [0.375, SYnex_COLD["p10"]],
        [0.375, SYnex_COLD["p25"]],
        [0.5, SYnex_COLD["p25"]],
        [0.5, SYnex_WARM["p75"]],
        [0.625, SYnex_WARM["p75"]],
        [0.625, SYnex_WARM["p90"]],
        [0.75, SYnex_WARM["p90"]],
        [0.75, SYnex_WARM["p95"]],
        [0.875, SYnex_WARM["p95"]],
        [0.875, SYnex_WARM["rec"]],
        [1.0, SYnex_WARM["rec"]],
    ]


def diverging_persistence_colorscale() -> list[list]:
    """Diverging cold (blue) ← 0 → warm (red) palette for combined persistence maps."""
    return [
        [0.0, SYnex_COLD["rec"]],
        [0.15, SYnex_COLD["p5"]],
        [0.30, SYnex_COLD["p10"]],
        [0.42, SYnex_COLD["p25"]],
        [0.5, "#FFFFFF"],
        [0.58, SYnex_WARM["p75"]],
        [0.70, SYnex_WARM["p90"]],
        [0.85, SYnex_WARM["p95"]],
        [1.0, SYnex_WARM["rec"]],
    ]


def warm_persistence_colorscale() -> list[list]:
    """Sequential warm palette for persistence heatmaps."""
    return [
        [0.0, SYnex_WARM["p75"]],
        [0.33, SYnex_WARM["p90"]],
        [0.66, SYnex_WARM["p95"]],
        [1.0, SYnex_WARM["rec"]],
    ]


def cold_persistence_colorscale() -> list[list]:
    """Sequential cold palette for persistence heatmaps."""
    return [
        [0.0, SYnex_COLD["p25"]],
        [0.33, SYnex_COLD["p10"]],
        [0.66, SYnex_COLD["p5"]],
        [1.0, SYnex_COLD["rec"]],
    ]


def legend_badge_style(warm_or_cold: str, level: str, *, highlight: bool = False) -> str:
    """Inline CSS for HTML legend chips in Streamlit."""
    palette = SYnex_WARM if warm_or_cold == "warm" else SYnex_COLD
    key = {"moderate": "p75" if warm_or_cold == "warm" else "p25",
           "strong": "p90" if warm_or_cold == "warm" else "p10",
           "extreme": "p95" if warm_or_cold == "warm" else "p5",
           "record": "rec"}[level]
    bg = palette[key]
    # Light fills → black text; dark fills → white text
    light_keys = {"p75", "p90", "p25", "p10"}
    fg = SYnex_BRAND["text_on_light"] if key in light_keys else SYnex_BRAND["text_on_primary"]
    style = f"background-color:{bg}; color:{fg}; padding: 2px 6px; border-radius: 3px;"
    if highlight:
        style += " font-weight: bold; border: 2px solid black; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);"
    return style


# Legacy hex values still in app.py — for migration reference only
LEGACY_COLORS = {
    "warm_rec": "#FF1493",
    "cold_p5": "#0000FF",
    "mslp_contour": "#006400",
}
