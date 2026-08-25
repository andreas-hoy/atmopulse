"""
AtmoPulse Central Theme & Visualization Configuration Module (atmopulse_theme.py)

This module acts as the authoritative 'Single Source of Truth' for brand styling,
typography, CSS injection, and color palette management across the AtmoPulse 
digital climate service architecture.

Core functionalities:
- Manages Google Fonts webfont ingestion (Outfit, Sora) and Plotly typography profiles.
- Injects customized Streamlit CSS to style the responsive top navigation bar, 
  sidebar controls, dataframes, and dynamic UI badges.
- Defines ETCCDI-compliant color sequences for thermal anomaly thresholds 
  (Moderate/P75/P25, Strong/P90/P10, Extreme/P95/P5, and All-Time Records).
- Exports discrete, sequential, and diverging color palettes for Plotly maps, 
  persistence heatmaps, and meteogram anomaly fill layers.
"""

from __future__ import annotations


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color string (#RRGGBB) to a standard CSS/Plotly rgba() string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --- Typography (Google Fonts CDN) ---
ATMOPULSE_FONTS = {
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
    """Return typography dictionary for Plotly hover labels."""
    return dict(font_size=font_size, font_family=ATMOPULSE_FONTS["sora_css"])


def plotly_typography(*, font_size: int = 12) -> dict:
    """Default Plotly layout fragment: Sora for chart text, axes, legends, and hovers."""
    return {
        "font": dict(family=ATMOPULSE_FONTS["sora_css"], size=font_size),
        "hoverlabel": plotly_hoverlabel(),
    }


def plotly_title_font(*, size: int = 20) -> dict:
    """Return typography dictionary for primary Plotly figure titles."""
    return dict(size=size, family=ATMOPULSE_FONTS["sora_css"])


def map_contour_label_font(*, size: int = 10) -> dict:
    """Return small font dictionary for synoptic contour labels (MSLP/Z500 isolines)."""
    return dict(size=size, family=ATMOPULSE_FONTS["sora_css"])


def atmopulse_streamlit_css(brand: dict) -> str:
    """Inject Google Fonts and Outfit/Sora layout rules for the Streamlit UI."""
    o = ATMOPULSE_FONTS["outfit_css"]
    s = ATMOPULSE_FONTS["sora_css"]
    lw = ATMOPULSE_FONTS["logo_weight"]
    uw = ATMOPULSE_FONTS["ui_weight"]
    return f"""
@import url('{GOOGLE_FONTS_URL}');

/* Navigation bar container */
.st-key-atmopulse_nav_bar {{
    background-color: {brand['nav_bg']} !important; 
    border-radius: 12px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
    padding: 8px 16px !important; 
    margin-bottom: 20px !important;
    display: flex !important; 
    align-items: center !important;
    min-height: 72px !important; 
    height: auto !important; 
    overflow: visible !important;
}}
.st-key-atmopulse_nav_bar > div,
.st-key-atmopulse_nav_bar [data-testid="stHorizontalBlock"],
.st-key-atmopulse_nav_bar [data-testid="stVerticalBlockBorderWrapper"] {{
    align-items: center !important;
}}
.st-key-atmopulse_nav_bar div[data-testid="stElementContainer"],
.st-key-atmopulse_nav_bar .st-key-atmopulse_top_nav,
.st-key-atmopulse_nav_bar div[data-testid="stRadio"],
.st-key-atmopulse_nav_bar div[data-testid="stMarkdownContainer"] {{
    margin: 0 !important; 
    padding: 0 !important;
    align-self: center !important;
}}
.st-key-atmopulse_nav_bar .st-key-atmopulse_top_nav {{ 
    flex: 1 !important; 
    min-width: 0 !important; 
}}
.atmopulse-nav-logo {{
    display: flex !important; 
    align-items: center !important; 
    gap: 6px !important;
    font-family: {o} !important; 
    font-size: clamp(34px, 3.2vw, 48px) !important;
    font-weight: {lw} !important; 
    color: {brand['primary']} !important; 
    line-height: 1 !important;
    margin: 0 !important; 
    padding-left: 4px !important; 
    white-space: nowrap !important;
    flex-shrink: 0 !important; 
    align-self: center !important;
}}
.atmopulse-nav-logo span {{ 
    display: inline-flex !important; 
    align-items: center !important; 
    line-height: 1 !important; 
}}
.st-key-atmopulse_nav_bar div[data-testid="stMarkdownContainer"] p {{ 
    margin: 0 !important; 
    padding: 0 !important; 
}}

/* Top navigation tabs — single row, scales down responsively */
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"],
.st-key-atmopulse_nav_bar div[data-testid="stRadio"] > div[role="radiogroup"] {{
    display: flex !important; 
    flex-direction: row !important; 
    flex-wrap: nowrap !important;
    align-items: center !important; 
    gap: clamp(4px, 0.6vw, 10px) !important;
    margin: 0 !important; 
    padding: 0 !important; 
    width: 100% !important; 
    overflow: hidden !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label,
.st-key-atmopulse_nav_bar div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
    background-color: transparent; 
    padding: clamp(4px, 0.5vw, 8px) clamp(8px, 1vw, 16px) !important;
    border-radius: 8px; 
    cursor: pointer; 
    transition: all 0.2s; 
    margin: 0 !important;
    display: flex !important; 
    align-items: center !important; 
    flex-shrink: 1 !important; 
    min-width: 0 !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover,
.st-key-atmopulse_nav_bar div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{ 
    background-color: {brand['nav_hover']}; 
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
.st-key-atmopulse_nav_bar div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {{
    background-color: {brand['nav_active']}; 
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label p,
.st-key-atmopulse_nav_bar div[data-testid="stRadio"] > div[role="radiogroup"] > label p {{
    font-family: {o} !important; 
    font-size: clamp(13px, 1.05vw, 19px) !important;
    font-weight: {uw} !important; 
    color: {brand['primary']} !important;
    line-height: 1.1 !important; 
    margin: 0 !important; 
    white-space: nowrap !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child,
.st-key-atmopulse_top_nav [data-testid="stRadioOption"] svg,
.st-key-atmopulse_nav_bar div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {{ 
    display: none; 
}}

/* Sidebar: Filters & Controls */
section[data-testid="stSidebar"] {{
    font-family: {o} !important; 
    font-weight: {uw} !important;
}}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label, 
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span, 
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {{
    font-family: {o} !important; 
    font-weight: {uw} !important;
}}
section[data-testid="stSidebar"] .stRadio > div {{ 
    gap: 0rem; 
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {{
    font-size: 14px !important; 
    font-weight: {uw} !important; 
    color: inherit !important;
}}
section[data-testid="stSidebar"] .stCheckbox {{ 
    margin-top: -12px; 
}}
section[data-testid="stSidebar"] button, 
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] {{
    font-family: {o} !important; 
    font-weight: {uw} !important;
}}

/* Main content: UI labels & Headings */
.main .block-container, 
.main .block-container h1, 
.main .block-container h2,
.main .block-container h3, 
.main .block-container p, 
.main .block-container label,
.main .block-container [data-testid="stMarkdownContainer"] {{
    font-family: {o} !important; 
    font-weight: {uw} !important;
}}
.main div[data-testid="stRadio"] label p,
.main div[data-testid="stSelectbox"] label,
.main .stSlider label,
.main [data-testid="stAlert"] {{
    font-family: {o} !important; 
    font-weight: {uw} !important;
}}

/* Shared subsection labels: Map legends & Top-10 headers */
.atmopulse-subsection-label,
.atmopulse-map-legend {{
    font-family: {o} !important;
    font-size: 13px !important;
    font-weight: {uw} !important;
    line-height: 1.35 !important;
}}
.atmopulse-map-legend span {{
    font-family: {o} !important;
    font-size: 13px !important;
    font-weight: {uw} !important;
}}
.main div[data-testid="stDataFrame"],
.main div[data-testid="stDataFrame"] th,
.main div[data-testid="stDataFrame"] td {{
    font-family: {o} !important; 
    font-weight: {uw} !important;
}}
"""


# --- UI / Brand ---
ATMOPULSE_BRAND = {
    "primary": "#0056B3",
    "nav_bg": "#E6F2FF",
    "nav_hover": "#CCE5FF",
    "nav_active": "#FFFFFF",
    "text_on_primary": "#FFFFFF",
    "text_on_light": "#000000",
}

# --- Warm (heat) percentiles ---
ATMOPULSE_WARM = {
    "p75": "#FFE699",   # Moderate (75th percentile)
    "p90": "#FF9933",   # Strong (90th percentile)
    "p95": "#CC0000",   # Extreme (95th percentile)
    "rec": "#E91E8C",   # All-time record (distinct vivid pink)
}

# --- Cold percentiles ---
ATMOPULSE_COLD = {
    "p25": "#CCF2FF",   # Moderate (25th percentile)
    "p10": "#3399FF",   # Strong (10th percentile)
    "p5": "#0056B3",    # Extreme (5th percentile, aligned with primary brand)
    "rec": "#4B0082",   # All-time record (indigo)
}

# --- Overlays & auxiliary series ---
ATMOPULSE_OVERLAY = {
    "mslp_contour": "#2E7D32",
    "z500_contour": "#0056B3",
    "apparent_temp": "#388E3C",
    "border": "#000000",
    "grid": "rgba(200,200,200,0.3)",
    "annotation": "gray",
}

# --- Meteogram fill opacities ---
_METEO_ALPHA = {"moderate": 0.50, "strong": 0.60, "extreme": 0.70, "record": 0.85}


def warm_rgba(level: str) -> str:
    """Return rgba string for meteogram warm anomaly fills (moderate, strong, extreme, record)."""
    key = {"moderate": "p75", "strong": "p90", "extreme": "p95", "record": "rec"}[level]
    return _hex_to_rgba(ATMOPULSE_WARM[key], _METEO_ALPHA[level])


def cold_rgba(level: str) -> str:
    """Return rgba string for meteogram cold anomaly fills (moderate, strong, extreme, record)."""
    key = {"moderate": "p25", "strong": "p10", "extreme": "p5", "record": "rec"}[level]
    return _hex_to_rgba(ATMOPULSE_COLD[key], _METEO_ALPHA[level])


def map_extremes_colorscale() -> list[list]:
    """Plotly colorscale for synoptic map discrete thermal extremes (Cold Record -> Warm Record)."""
    return [
        [0.0, ATMOPULSE_COLD["rec"]],
        [0.125, ATMOPULSE_COLD["rec"]],
        [0.125, ATMOPULSE_COLD["p5"]],
        [0.25, ATMOPULSE_COLD["p5"]],
        [0.25, ATMOPULSE_COLD["p10"]],
        [0.375, ATMOPULSE_COLD["p10"]],
        [0.375, ATMOPULSE_COLD["p25"]],
        [0.5, ATMOPULSE_COLD["p25"]],
        [0.5, ATMOPULSE_WARM["p75"]],
        [0.625, ATMOPULSE_WARM["p75"]],
        [0.625, ATMOPULSE_WARM["p90"]],
        [0.75, ATMOPULSE_WARM["p90"]],
        [0.75, ATMOPULSE_WARM["p95"]],
        [0.875, ATMOPULSE_WARM["p95"]],
        [0.875, ATMOPULSE_WARM["rec"]],
        [1.0, ATMOPULSE_WARM["rec"]],
    ]


def diverging_persistence_colorscale() -> list[list]:
    """Diverging cold (blue) -> neutral (white) -> warm (red) palette for persistence heatmaps."""
    return [
        [0.0, ATMOPULSE_COLD["rec"]],
        [0.15, ATMOPULSE_COLD["p5"]],
        [0.30, ATMOPULSE_COLD["p10"]],
        [0.42, ATMOPULSE_COLD["p25"]],
        [0.5, "#FFFFFF"],
        [0.58, ATMOPULSE_WARM["p75"]],
        [0.70, ATMOPULSE_WARM["p90"]],
        [0.85, ATMOPULSE_WARM["p95"]],
        [1.0, ATMOPULSE_WARM["rec"]],
    ]


def warm_persistence_colorscale() -> list[list]:
    """Sequential warm palette for positive thermal persistence heatmaps."""
    return [
        [0.0, ATMOPULSE_WARM["p75"]],
        [0.33, ATMOPULSE_WARM["p90"]],
        [0.66, ATMOPULSE_WARM["p95"]],
        [1.0, ATMOPULSE_WARM["rec"]],
    ]


def cold_persistence_colorscale() -> list[list]:
    """Sequential cold palette for negative thermal persistence heatmaps."""
    return [
        [0.0, ATMOPULSE_COLD["p25"]],
        [0.33, ATMOPULSE_COLD["p10"]],
        [0.66, ATMOPULSE_COLD["p5"]],
        [1.0, ATMOPULSE_COLD["rec"]],
    ]


def legend_badge_style(warm_or_cold: str, level: str, *, highlight: bool = False) -> str:
    """Inline CSS for HTML legend badges rendered inside Streamlit UI containers."""
    palette = ATMOPULSE_WARM if warm_or_cold == "warm" else ATMOPULSE_COLD
    key = {
        "moderate": "p75" if warm_or_cold == "warm" else "p25",
        "strong": "p90" if warm_or_cold == "warm" else "p10",
        "extreme": "p95" if warm_or_cold == "warm" else "p5",
        "record": "rec",
    }[level]
    bg = palette[key]
    # Light fills -> dark text; dark fills -> white text
    light_keys = {"p75", "p90", "p25", "p10"}
    fg = ATMOPULSE_BRAND["text_on_light"] if key in light_keys else ATMOPULSE_BRAND["text_on_primary"]
    style = (
        f"background-color:{bg}; color:{fg}; padding: 1px 5px; border-radius: 3px;"
        f" font-family:{ATMOPULSE_FONTS['outfit_css']}; font-size:13px; font-weight:{ATMOPULSE_FONTS['ui_weight']};"
    )
    if highlight:
        style += " font-weight: bold; border: 1.5px solid black; box-shadow: 1px 1px 3px rgba(0,0,0,0.25);"
    return style