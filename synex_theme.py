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


# --- Typography (Google Fonts CDN) ---
SYnex_FONTS = {
    "outfit": "Outfit",
    "sora": "Sora",
    "logo_weight": 750,
    "ui_weight": 500,
    "outfit_css": "'Outfit', sans-serif",
    "sora_css": "'Sora', sans-serif",
}

GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Outfit:wght@100..900&family=Sora:wght@100..800&display=swap"
)


def plotly_hoverlabel(*, font_size: int = 13) -> dict:
    return dict(font_size=font_size, font_family=SYnex_FONTS["sora_css"])


def plotly_typography(*, font_size: int = 12) -> dict:
    """Default Plotly layout fragment: Sora for chart text, axes, legends, hovers."""
    return {
        "font": dict(family=SYnex_FONTS["sora_css"], size=font_size),
        "hoverlabel": plotly_hoverlabel(),
    }


def plotly_title_font(*, size: int = 20) -> dict:
    return dict(size=size, family=SYnex_FONTS["sora_css"])


def map_contour_label_font(*, size: int = 10) -> dict:
    """Small contour labels — readable at 100% browser zoom."""
    return dict(size=size, family=SYnex_FONTS["sora_css"])


def synex_streamlit_css(brand: dict) -> str:
    """Inject Google Fonts + Outfit/Sora rules for Streamlit UI chrome."""
    o = SYnex_FONTS["outfit_css"]
    s = SYnex_FONTS["sora_css"]
    lw = SYnex_FONTS["logo_weight"]
    uw = SYnex_FONTS["ui_weight"]
    return f"""
@import url('{GOOGLE_FONTS_URL}');

/* Nav bar */
.st-key-synex_nav_bar {{
    background-color: {brand['nav_bg']} !important; border-radius: 12px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
    padding: 8px 16px !important; margin-bottom: 20px !important;
    display: flex !important; align-items: center !important;
    min-height: 72px !important; height: auto !important; overflow: visible !important;
}}
.st-key-synex_nav_bar > div,
.st-key-synex_nav_bar [data-testid="stHorizontalBlock"],
.st-key-synex_nav_bar [data-testid="stVerticalBlockBorderWrapper"] {{
    align-items: center !important;
}}
.st-key-synex_nav_bar div[data-testid="stElementContainer"],
.st-key-synex_nav_bar .st-key-synex_top_nav,
.st-key-synex_nav_bar div[data-testid="stRadio"],
.st-key-synex_nav_bar div[data-testid="stMarkdownContainer"] {{
    margin: 0 !important; padding: 0 !important;
    align-self: center !important;
}}
.st-key-synex_nav_bar .st-key-synex_top_nav {{ flex: 1 !important; min-width: 0 !important; }}
.synex-nav-logo {{
    display: flex !important; align-items: center !important; gap: 6px !important;
    font-family: {o} !important; font-size: clamp(34px, 3.2vw, 48px) !important;
    font-weight: {lw} !important; color: {brand['primary']} !important; line-height: 1 !important;
    margin: 0 !important; padding-left: 4px !important; white-space: nowrap !important;
    flex-shrink: 0 !important; align-self: center !important;
}}
.synex-nav-logo span {{ display: inline-flex !important; align-items: center !important; line-height: 1 !important; }}
.st-key-synex_nav_bar div[data-testid="stMarkdownContainer"] p {{ margin: 0 !important; padding: 0 !important; }}

/* Top navigation tabs — single row, scales down before wrapping */
.st-key-synex_top_nav div[data-testid="stRadio"] > div[role="radiogroup"],
.st-key-synex_nav_bar div[data-testid="stRadio"] > div[role="radiogroup"] {{
    display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important;
    align-items: center !important; gap: clamp(4px, 0.6vw, 10px) !important;
    margin: 0 !important; padding: 0 !important; width: 100% !important; overflow: hidden !important;
}}
.st-key-synex_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label,
.st-key-synex_nav_bar div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
    background-color: transparent; padding: clamp(4px, 0.5vw, 8px) clamp(8px, 1vw, 16px) !important;
    border-radius: 8px; cursor: pointer; transition: all 0.2s; margin: 0 !important;
    display: flex !important; align-items: center !important; flex-shrink: 1 !important; min-width: 0 !important;
}}
.st-key-synex_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover,
.st-key-synex_nav_bar div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{ background-color: {brand['nav_hover']}; }}
.st-key-synex_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
.st-key-synex_nav_bar div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {{
    background-color: {brand['nav_active']}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}}
.st-key-synex_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label p,
.st-key-synex_nav_bar div[data-testid="stRadio"] > div[role="radiogroup"] > label p {{
    font-family: {o} !important; font-size: clamp(13px, 1.05vw, 19px) !important;
    font-weight: {uw} !important; color: {brand['primary']} !important;
    line-height: 1.1 !important; margin: 0 !important; white-space: nowrap !important;
}}
.st-key-synex_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child,
.st-key-synex_top_nav [data-testid="stRadioOption"] svg,
.st-key-synex_nav_bar div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {{ display: none; }}

/* Sidebar: filters & controls */
section[data-testid="stSidebar"] {{
    font-family: {o} !important; font-weight: {uw} !important;
}}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {{
    font-family: {o} !important; font-weight: {uw} !important;
}}
section[data-testid="stSidebar"] .stRadio > div {{ gap: 0rem; }}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {{
    font-size: 14px !important; font-weight: {uw} !important; color: inherit !important;
}}
section[data-testid="stSidebar"] .stCheckbox {{ margin-top: -12px; }}
section[data-testid="stSidebar"] button, section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] {{
    font-family: {o} !important; font-weight: {uw} !important;
}}

/* Main content: UI labels & headings (not Plotly canvases) */
.main .block-container, .main .block-container h1, .main .block-container h2,
.main .block-container h3, .main .block-container p, .main .block-container label,
.main .block-container [data-testid="stMarkdownContainer"] {{
    font-family: {o} !important; font-weight: {uw} !important;
}}
.main div[data-testid="stRadio"] label p,
.main div[data-testid="stSelectbox"] label,
.main .stSlider label,
.main [data-testid="stAlert"] {{
    font-family: {o} !important; font-weight: {uw} !important;
}}

/* Shared subsection labels: map legends + Top-10 headers */
.synex-subsection-label,
.synex-map-legend {{
    font-family: {o} !important;
    font-size: 13px !important;
    font-weight: {uw} !important;
    line-height: 1.35 !important;
}}
.synex-map-legend span {{
    font-family: {o} !important;
    font-size: 13px !important;
    font-weight: {uw} !important;
}}
.main div[data-testid="stDataFrame"],
.main div[data-testid="stDataFrame"] th,
.main div[data-testid="stDataFrame"] td {{
    font-family: {o} !important; font-weight: {uw} !important;
}}
"""


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
    style = (
        f"background-color:{bg}; color:{fg}; padding: 1px 5px; border-radius: 3px;"
        f" font-family:{SYnex_FONTS['outfit_css']}; font-size:13px; font-weight:{SYnex_FONTS['ui_weight']};"
    )
    if highlight:
        style += " font-weight: bold; border: 1.5px solid black; box-shadow: 1px 1px 3px rgba(0,0,0,0.25);"
    return style


# Legacy hex values still in app.py — for migration reference only
LEGACY_COLORS = {
    "warm_rec": "#FF1493",
    "cold_p5": "#0000FF",
    "mslp_contour": "#006400",
}
