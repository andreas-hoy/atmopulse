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

from urllib.parse import quote


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color string (#RRGGBB) to a standard CSS/Plotly rgba() string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _svg_data_uri(svg: str) -> str:
    """Encode an inline SVG as a CSS data URI."""
    return "data:image/svg+xml," + quote(" ".join(svg.split()))


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
    "family=Outfit:ital,wght@0,100..900;1,100..900&family=Sora:ital,wght@0,100..800;1,100..800&display=swap"
)


def atmopulse_wordmark_html() -> str:
    """Brand lockup: Atmo upright, Pulse italic. Use with unsafe_allow_html."""
    return '<span class="atmopulse-brand">Atmo<i>Pulse</i></span>'


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
    map_uri = _svg_data_uri(MAP_TRACKER_SVG)
    meteo_uri = _svg_data_uri(METEOGRAM_SVG)
    wave_uri = _svg_data_uri(WAVOGRAM_SVG)
    return f"""
@import url('{GOOGLE_FONTS_URL}');

/* Navigation bar container — single-row flex. The logo now lives at the top
   of the sidebar, so this bar is dedicated entirely to the nav tabs and
   normally fits at 100% desktop zoom without scrolling. */
.st-key-atmopulse_nav_bar {{
    background-color: {brand['nav_bg']} !important; 
    border-radius: 12px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
    padding: 8px 16px !important; 
    margin-bottom: 20px !important;
    display: flex !important; 
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    min-height: 72px !important; 
    height: auto !important; 
    overflow-x: auto !important;
    overflow-y: hidden !important;
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
.atmopulse-nav-logo,
.atmopulse-sidebar-logo {{
    display: flex !important; 
    align-items: center !important; 
    gap: 6px !important;
    font-family: {o} !important; 
    font-weight: {lw} !important; 
    font-style: normal !important;
    color: {brand['primary']} !important; 
    line-height: 1 !important;
    margin: 0 !important; 
    white-space: nowrap !important;
    flex-shrink: 0 !important; 
    align-self: center !important;
}}
.atmopulse-nav-logo {{
    font-size: clamp(34px, 3.2vw, 48px) !important;
    padding-left: 4px !important; 
}}
.atmopulse-sidebar-logo {{
    font-size: 34px !important;
    justify-content: center !important;
    width: 100% !important;
    margin-top: -12px !important;
    margin-bottom: 26px !important;
}}
/* Pull the sidebar content closer to the top so the (now larger) logo sits
   higher on the page instead of leaving a big gap above it. */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
    padding-top: 0.75rem !important;
}}
.atmopulse-nav-logo span,
.atmopulse-sidebar-logo span {{ 
    display: inline-flex !important; 
    align-items: center !important; 
    line-height: 1 !important; 
}}
.atmopulse-nav-logo svg,
.atmopulse-sidebar-logo svg {{
    width: 1em !important;
    height: 1em !important;
    display: block !important;
    flex-shrink: 0 !important;
}}
.atmopulse-brand {{
    font-family: {o} !important;
    font-style: normal !important;
    font-synthesis: style !important;
}}
.atmopulse-brand i,
.atmopulse-nav-logo i,
.atmopulse-sidebar-logo i {{
    font-style: italic !important;
    font-synthesis: style !important;
    font-weight: inherit !important;
}}
.st-key-atmopulse_nav_bar div[data-testid="stMarkdownContainer"] p {{ 
    margin: 0 !important; 
    padding: 0 !important; 
}}

/* Top navigation tabs — single row. A slim, visible scrollbar acts as a
   "slider" to pan through the tabs on narrow screens / high zoom instead of
   shrinking or overlapping them. */
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] {{
    display: flex !important; 
    flex-direction: row !important; 
    flex-wrap: nowrap !important;
    align-items: center !important; 
    gap: clamp(4px, 0.6vw, 10px) !important;
    margin: 0 !important; 
    padding: 0 0 6px 0 !important; 
    width: 100% !important; 
    overflow-x: auto !important;
    overflow-y: hidden !important;
    scrollbar-width: thin !important;
    scrollbar-color: {brand['primary']} transparent !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"]::-webkit-scrollbar {{
    height: 5px !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"]::-webkit-scrollbar-track {{
    background: transparent !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"]::-webkit-scrollbar-thumb {{
    background-color: {brand['primary']} !important;
    border-radius: 999px !important;
    opacity: 0.6 !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
    background-color: transparent; 
    padding: 6px 14px !important;
    border-radius: 8px; 
    cursor: pointer; 
    transition: all 0.2s; 
    margin: 0 !important;
    display: flex !important; 
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    flex-shrink: 0 !important; 
    min-width: 0 !important;
    line-height: 1 !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:not(:first-child),
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] {{
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 8px !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{ 
    background-color: {brand['nav_hover']}; 
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {{
    background-color: {brand['nav_active']}; 
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label p {{
    font-family: {o} !important; 
    font-size: clamp(15px, 1.05vw, 16px) !important;
    font-weight: {uw} !important; 
    color: {brand['primary']} !important;
    line-height: 1 !important; 
    margin: 0 !important; 
    padding: 0 !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 8px !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label svg {{
    width: 22px !important;
    height: 22px !important;
    flex-shrink: 0 !important;
    display: block !important;
    vertical-align: middle !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-of-type(2) p::before,
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-of-type(3) p::before,
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-of-type(4) p::before {{
    content: "" !important;
    display: inline-block !important;
    width: 22px !important;
    height: 22px !important;
    flex-shrink: 0 !important;
    vertical-align: middle !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: contain !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-of-type(2) p::before {{
    background-image: url("{map_uri}") !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-of-type(3) p::before {{
    background-image: url("{meteo_uri}") !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-of-type(4) p::before {{
    background-image: url("{wave_uri}") !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child,
.st-key-atmopulse_top_nav [data-testid="stRadioOption"] svg,
.st-key-atmopulse_top_nav [data-testid="stRadioIcon"],
.st-key-atmopulse_top_nav span[data-baseweb="radio"],
.st-key-atmopulse_top_nav input[type="radio"] {{ 
    display: none !important;
    appearance: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}}

/* Audience mode switch (Standard | Expert) — segmented control, now docked
   at the very top of the sidebar above Control Panel */
.st-key-atmopulse_ui_mode,
.st-key-atmopulse_ui_mode [data-testid="stRadio"],
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div {{
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
}}
section[data-testid="stSidebar"] .st-key-atmopulse_ui_mode {{
    margin-bottom: 12px !important;
}}
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] {{
    display: inline-flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 2px !important;
    margin: 0 !important;
    padding: 3px !important;
    width: auto !important;
    background: {brand['mode_track']} !important;
    border: none !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    overflow: visible !important;
}}
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label {{
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    margin: 0 !important;
    padding: 8px 18px !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 999px !important;
    cursor: pointer !important;
    background: {brand['mode_inactive']} !important;
    transition: background-color 0.22s ease, color 0.22s ease, box-shadow 0.22s ease !important;
}}
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {{
    display: none !important;
    width: 0 !important;
    margin: 0 !important;
}}
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label:hover:not([data-checked="true"]):not(:has(input:checked)) {{
    background: {brand['nav_hover']} !important;
}}
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {{
    background: {brand['primary']} !important;
    box-shadow: 0 1px 3px rgba(0, 86, 179, 0.28) !important;
}}
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label p,
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label span {{
    font-family: {o} !important;
    font-size: 15px !important;
    font-weight: {uw} !important;
    color: {brand['text_on_light']} !important;
    line-height: 1.2 !important;
    margin: 0 !important;
    text-align: center !important;
    white-space: nowrap !important;
    transition: color 0.22s ease !important;
}}
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] p,
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] span,
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p,
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) span {{
    color: {brand['text_on_primary']} !important;
}}
.st-key-atmopulse_ui_mode [role="radiogroup"] * {{
    border-color: transparent !important;
    outline-color: transparent !important;
}}
.st-key-atmopulse_ui_mode [data-testid="stRadioIcon"],
.st-key-atmopulse_ui_mode input[type="radio"],
.st-key-atmopulse_ui_mode label svg,
.st-key-atmopulse_ui_mode [data-testid="stRadio"] svg,
.st-key-atmopulse_ui_mode [data-testid="stRadioOption"] svg,
.st-key-atmopulse_ui_mode span[data-baseweb="radio"],
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child,
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child * {{
    display: none !important;
    width: 0 !important;
    appearance: none !important;
    -webkit-appearance: none !important;
    visibility: hidden !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    overflow: hidden !important;
}}
/* Belt-and-braces: hide any direct child of the label that is NOT the text
   itself and does NOT wrap the text (":has" lets this survive an extra
   nesting level around the label text without also nuking the indicator's
   own wrapper, which was the bug in an earlier, cruder attempt). */
.st-key-atmopulse_ui_mode [data-testid="stRadio"] > div[role="radiogroup"] > label > *:not([data-testid="stMarkdownContainer"]):not(:has([data-testid="stMarkdownContainer"])) {{
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}}
.st-key-atmopulse_ui_mode label::before,
.st-key-atmopulse_ui_mode label::after,
.st-key-atmopulse_ui_mode input[type="radio"]::before,
.st-key-atmopulse_ui_mode input[type="radio"]::after {{
    content: none !important;
    display: none !important;
}}
.st-key-atmopulse_ui_mode *:focus,
.st-key-atmopulse_ui_mode *:focus-visible,
.st-key-atmopulse_ui_mode [data-testid="stRadio"] label:focus-within {{
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}}

/* Forecast Offset slider — force the native -7 / +3 tick labels to stay
   permanently visible (not just on hover), and make our injected "0" label
   share the exact same size/weight/color so all three anchors match. */
.st-key-offset_slider [data-testid="stTickBar"],
.st-key-offset_slider [data-testid="stTickBarMin"],
.st-key-offset_slider [data-testid="stTickBarMax"] {{
    opacity: 1 !important;
    visibility: visible !important;
}}
.st-key-offset_slider [data-testid="stTickBarMin"],
.st-key-offset_slider [data-testid="stTickBarMax"],
.st-key-offset_slider .atmopulse-slider-zero {{
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #31333F !important;
    line-height: 1 !important;
}}

/* Map Tracker: seamless side-by-side Plotly maps, pulled together with a
   thin vertical divider instead of a wide gutter. */
.st-key-atmopulse_map_columns [data-testid="stHorizontalBlock"] {{
    gap: 0 !important;
}}
.st-key-atmopulse_map_columns [data-testid="column"] {{
    padding: 0 !important;
    margin: 0 !important;
}}
.st-key-atmopulse_map_columns [data-testid="column"]:nth-of-type(1) {{
    border-right: 2px solid #D3D3D3 !important;
    padding-right: 10px !important;
}}

/* Synoptic maps: the keyed frame is locked to EUROPE_BBOX (70° lon : 42° lat).
   Plotly fills that box (absolute inset) so width and height scale together
   at any browser zoom. Titles sit outside the frame (see .atmopulse-map-title). */
.atmopulse-map-title {{
    text-align: center !important;
    font-family: {s} !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: {brand['primary']} !important;
    margin: 0 0 6px 0 !important;
    line-height: 1.2 !important;
}}
.st-key-map_a,
.st-key-map_b,
.st-key-map_flicker {{
    position: relative !important;
    aspect-ratio: 70 / 42 !important;
    width: 100% !important;
    height: auto !important;
    max-width: 100% !important;
    overflow: hidden !important;
}}
.st-key-map_a > div,
.st-key-map_b > div,
.st-key-map_flicker > div,
.st-key-map_a [data-testid="stPlotlyChart"],
.st-key-map_b [data-testid="stPlotlyChart"],
.st-key-map_flicker [data-testid="stPlotlyChart"],
.st-key-map_a .js-plotly-plot,
.st-key-map_b .js-plotly-plot,
.st-key-map_flicker .js-plotly-plot,
.st-key-map_a .plot-container,
.st-key-map_b .plot-container,
.st-key-map_flicker .plot-container,
.st-key-map_a .svg-container,
.st-key-map_b .svg-container,
.st-key-map_flicker .svg-container {{
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    max-height: 100% !important;
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
section[data-testid="stSidebar"] span:not([data-testid="stIconMaterial"]), 
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {{
    font-family: {o} !important; 
    font-weight: {uw} !important;
}}
/* Never override the Material icon font — doing so on the sidebar
   collapse/expand control made it render its literal icon name
   ("keyboard_double_arrow_left") instead of the arrow glyph. */
[data-testid="stIconMaterial"] {{
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
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
    font-size: 12px !important;
    font-weight: {uw} !important;
    line-height: 1.25 !important;
    white-space: nowrap;
}}
.atmopulse-map-legend span {{
    font-family: {o} !important;
    font-size: 12px !important;
    font-weight: {uw} !important;
    white-space: nowrap;
    display: inline-block;
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
    "mode_track": "#E8EEF4",
    "mode_inactive": "transparent",
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
        f"background-color:{bg}; color:{fg}; padding: 1px 6px; border-radius: 3px;"
        f" font-family:{ATMOPULSE_FONTS['outfit_css']}; font-size:12px; font-weight:{ATMOPULSE_FONTS['ui_weight']};"
    )
    if highlight:
        style += " font-weight: bold; border: 1.5px solid black; box-shadow: 1px 1px 3px rgba(0,0,0,0.25);"
    return style


# --- UI Icons (SVG Inline) ---

LOGO_SVG = """<svg
xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
width="28" height="28" fill="none">
  <path
d="M1,12 Q3,6 5,12 Q7,18 9,12 L10.5,12 L12,3 L13.5,21 L15,12 L16.5,12
Q18,9.5 19.5,12 L23,12" stroke="#0056B3"
stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

MAP_TRACKER_SVG = """<svg
xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
width="24" height="24" fill="none">
  <path
d="M2,22 L3,18 L2,15 L4,13 L6,14 L7,11 L9,9 L8,7 L10,6 L11,8 L13,7 L15,9
L17,8 L19,10 L20,13 L18,15 L19,18 L16,19 L14,17 L12,19 L9,18 L7,20 L5,21
Z" stroke="#2F4F4F" stroke-width="0.75"
stroke-linejoin="round"/>
  <ellipse
cx="7" cy="9" rx="3" ry="2.2"
stroke="#0056B3" stroke-width="1"/>
  <ellipse
cx="7" cy="9" rx="1.6" ry="1.1"
stroke="#0056B3" stroke-width="1"/>
  <ellipse
cx="15.5" cy="15" rx="3.2" ry="2.3"
stroke="#CC0000" stroke-width="1"/>
  <ellipse
cx="15.5" cy="15" rx="1.7" ry="1.2"
stroke="#CC0000" stroke-width="1"/>
</svg>"""

METEOGRAM_SVG = """<svg
xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
width="24" height="24" fill="none">
  <defs>
    <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#0056B3"/>
      <stop offset="100%" stop-color="#CC0000"/>
    </linearGradient>
  </defs>
  <rect
x="3" y="11" width="18" height="6"
fill="#808080" opacity="0.25"/>
  <line
x1="3" y1="2" x2="3" y2="21"
stroke="#808080" stroke-width="1"
stroke-linecap="round"/>
  <line
x1="3" y1="21" x2="22" y2="21"
stroke="#808080" stroke-width="1"
stroke-linecap="round"/>
  <polyline
points="4,17 7,15 9,16 11,14 13,15 15,13 17,8 19,4 21,3"
stroke="url(#lineGrad)" stroke-width="2"
stroke-linecap="round" stroke-linejoin="miter"/>
</svg>"""

WAVOGRAM_SVG = """<svg
xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
width="24" height="24" fill="none">
  <line
x1="1" y1="12" x2="23" y2="12"
stroke="#808080" stroke-width="1" stroke-dasharray="2
2" stroke-linecap="round"/>
  <path
d="M2,12 L4,7 L6,10 L8,3 L10,8 L12,5 L13,12 Z" fill="#CC0000"/>
  <path
d="M10,12 L12,17 L14,14 L16,21 L18,15 L20,18 L22,12 Z" fill="#0056B3"/>
</svg>"""