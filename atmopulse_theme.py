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


def inject_monday_weekstart() -> None:
    """Force Streamlit date pickers to Monday–Sunday.

    ``st.date_input`` has no first-day parameter: it reads
    ``Intl.Locale.getWeekInfo()`` from the browser language, so an en-US
    Cursor/Chrome session shows Sunday first. Patch the parent window before
    any calendar opens. ISO weekend is Saturday–Sunday.
    """
    import streamlit.components.v1 as components

    components.html(
        """
<script>
(function () {
  var w = window.parent || window;
  try {
    if (w.__atmopulseMondayWeek) return;
    var proto = w.Intl && w.Intl.Locale && w.Intl.Locale.prototype;
    if (!proto) return;
    var iso = {firstDay: 1, weekend: [6, 7], minimalDays: 4};
    var orig = proto.getWeekInfo;
    proto.getWeekInfo = function () {
      if (typeof orig === "function") {
        try {
          return Object.assign({}, orig.call(this), {firstDay: 1, weekend: [6, 7]});
        } catch (e) {}
      }
      return iso;
    };
    w.__atmopulseMondayWeek = true;
  } catch (e) {}
})();
</script>
        """,
        height=0,
        width=0,
    )


def map_contour_label_font(*, size: int = 9, color: str | None = None) -> dict:
    """Return small font dictionary for synoptic contour labels (MSLP/Z500 isolines)."""
    font = dict(size=size, family=ATMOPULSE_FONTS["sora_css"])
    if color:
        font["color"] = color
    return font


def atmopulse_streamlit_css(brand: dict) -> str:
    """Inject Google Fonts and Outfit/Sora layout rules for the Streamlit UI."""
    o = ATMOPULSE_FONTS["outfit_css"]
    s = ATMOPULSE_FONTS["sora_css"]
    lw = ATMOPULSE_FONTS["logo_weight"]
    uw = ATMOPULSE_FONTS["ui_weight"]
    from config import FORECAST_OFFSET_MAX, FORECAST_OFFSET_MIN
    _zero_pct = (0 - FORECAST_OFFSET_MIN) / (FORECAST_OFFSET_MAX - FORECAST_OFFSET_MIN) * 100
    map_uri = _svg_data_uri(MAP_TRACKER_SVG)
    meteo_uri = _svg_data_uri(METEOGRAM_SVG)
    wave_uri = _svg_data_uri(WAVOGRAM_SVG)
    return f"""
@import url('{GOOGLE_FONTS_URL}');

/* Hide the 0-height iframe that patches date-picker week start. */
div[data-testid="stIFrame"]:has(iframe[height="0"]),
.stElementContainer:has(iframe[height="0"]) {{
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}}

/* Navigation bar container — single-row flex. The logo now lives at the top
   of the sidebar, so this bar is dedicated entirely to the nav tabs and
   normally fits at 100% desktop zoom without scrolling. */
.st-key-atmopulse_nav_bar {{
    background-color: {brand['nav_bg']} !important; 
    border-radius: 12px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
    padding: 8px 16px !important; 
    margin-bottom: 0.35rem !important;
    display: flex !important; 
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    min-height: 72px !important; 
    height: auto !important; 
    overflow-x: hidden !important;
    overflow-y: hidden !important;
    scrollbar-width: none !important;
}}
.st-key-atmopulse_nav_bar::-webkit-scrollbar {{
    display: none !important;
    height: 0 !important;
    width: 0 !important;
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
.st-key-atmopulse_nav_bar .st-key-atmopulse_top_nav,
.st-key-atmopulse_nav_bar div[data-testid="stRadio"] {{
    overflow-x: hidden !important;
    overflow-y: hidden !important;
    scrollbar-width: none !important;
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
    padding-top: 0.4rem !important;
    padding-bottom: 0.35rem !important;
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

/* Top navigation tabs — single row. Only the radiogroup itself scrolls;
   the blue thumb is the sole scrollbar (the grey OS/track bar is hidden). */
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] {{
    display: flex !important; 
    flex-direction: row !important; 
    flex-wrap: nowrap !important;
    align-items: center !important; 
    gap: clamp(4px, 0.6vw, 10px) !important;
    margin: 0 !important; 
    padding: 0 0 8px 0 !important; 
    width: 100% !important; 
    overflow-x: auto !important;
    overflow-y: hidden !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"]::-webkit-scrollbar {{
    height: 6px !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"]::-webkit-scrollbar-track {{
    background: transparent !important;
}}
.st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"]::-webkit-scrollbar-thumb {{
    background-color: {brand['primary']} !important;
    border-radius: 999px !important;
}}
@supports not selector(::-webkit-scrollbar) {{
    .st-key-atmopulse_top_nav div[data-testid="stRadio"] > div[role="radiogroup"] {{
        scrollbar-width: thin !important;
        scrollbar-color: {brand['primary']} transparent !important;
    }}
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
    margin-bottom: 8px !important;
    margin-left: -10px !important;
}}
.st-key-map_archive_mode + [data-testid="stElementContainer"] {{
    margin-top: 0.95rem !important;
    margin-bottom: 0.95rem !important;
}}
.st-key-map_archive_mode + [data-testid="stElementContainer"] hr {{
    margin: 0 !important;
}}
.st-key-forecast_model {{
    margin-top: 0.2rem !important;
    margin-bottom: 0.55rem !important;
}}
.st-key-offset_slider {{
    margin-top: 0.45rem !important;
    margin-bottom: 0.35rem !important;
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
    padding: 5px 12px !important;
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
    font-size: 13px !important;
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

/* Forecast Offset: keep the end ticks visible, and mirror the value "0"
   directly under the thumb's "0" (same face, colour, and size). */
.st-key-offset_slider [data-testid="stSliderTickBar"] {{
    opacity: 1 !important;
    visibility: visible !important;
}}
.st-key-offset_slider [data-testid="stSliderTickBar"]::after {{
    content: "0";
    position: absolute;
    left: {_zero_pct}%;
    top: 0;
    transform: translateX(-50%);
    font-family: {o} !important;
    font-size: 0.875rem !important;
    font-weight: {uw} !important;
    line-height: 1.6 !important;
    color: {brand['primary']} !important;
    pointer-events: none;
}}

/* Side-by-side A/B compare: one vertical brand-blue divider. Used by Map
   Tracker, Meteogram, and Wavogram (`st.container(key="atmopulse_split_…")`).
   min-width:0 lets nested dataframes/charts shrink. */
[class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] {{
    gap: 0 !important;
    align-items: stretch !important;
}}
[class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
[class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
    flex: 1 1 0 !important;
    width: 50% !important;
    max-width: 50% !important;
    box-sizing: border-box !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
    overflow: visible !important;
    position: relative !important;
}}
[class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
[class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {{
    padding-right: 10px !important;
    box-shadow: inset -2px 0 0 0 {_hex_to_rgba(brand['primary'], 0.55)} !important;
}}
[class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child,
[class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child {{
    padding-left: 10px !important;
}}
/* Same small inset from the centre line on the table row as on the maps. */
.st-key-atmopulse_split_map_tables [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
.st-key-atmopulse_split_map_tables [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child,
.st-key-atmopulse_split_map_tables_overlay [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
.st-key-atmopulse_split_map_tables_overlay [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {{
    padding-right: 10px !important;
}}
/* Press buttons are a nested row. Keep them content-sized; the 50% column
   rule and the table gap must not spread SVG / PDF / CSV apart. */
[class*="st-key-atmopulse_split"] div[class*="st-key-press-row"],
[class*="st-key-atmopulse_split"] div[class*="st-key-press-row"][data-testid="stHorizontalBlock"],
[class*="st-key-atmopulse_split"] div[class*="st-key-press-row"] [data-testid="stHorizontalBlock"] {{
    gap: 0.35rem !important;
    flex-wrap: nowrap !important;
    justify-content: flex-start !important;
    align-items: center !important;
    width: fit-content !important;
    max-width: 100% !important;
}}
[class*="st-key-atmopulse_split"] div[class*="st-key-press-row"] [data-testid="stHorizontalBlock"] > *,
[class*="st-key-atmopulse_split"] div[class*="st-key-press-row"][data-testid="stHorizontalBlock"] > * {{
    flex: 0 0 auto !important;
    width: auto !important;
    max-width: max-content !important;
    min-width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}}
/* Warm/Cold sit in a nested row. The divider belongs between the two maps, not between those tables. */
[class*="st-key-atmopulse_split"] [data-testid="stColumn"] [data-testid="stHorizontalBlock"] {{
    gap: 22px !important;
}}
[class*="st-key-atmopulse_split"] [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
[class*="st-key-atmopulse_split"] [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {{
    box-shadow: none !important;
    padding-right: 10px !important;
}}
[class*="st-key-atmopulse_split"] [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child,
[class*="st-key-atmopulse_split"] [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child {{
    padding-left: 0 !important;
}}
[class*="st-key-atmopulse_split"] [data-testid="stDataFrame"],
[class*="st-key-atmopulse_split"] [data-testid="stDataFrame"] > div {{
    width: max-content !important;
    max-width: 100% !important;
}}
@media (max-width: 900px) {{
    [class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
    }}
    [class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: 100% !important;
    }}
    [class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
    [class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {{
        border-right: none !important;
        box-shadow: none !important;
        border-bottom: 2px solid {_hex_to_rgba(brand['primary'], 0.55)} !important;
        padding-right: 10px !important;
        padding-bottom: 12px !important;
        margin-bottom: 12px !important;
        width: 100% !important;
        max-width: 100% !important;
    }}
    [class*="st-key-atmopulse_split"] [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
    [class*="st-key-atmopulse_split"] [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {{
        border-bottom: none !important;
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }}
    [class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child,
    [class*="st-key-atmopulse_split"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child {{
        padding-left: 10px !important;
        width: 100% !important;
        max-width: 100% !important;
    }}
}}

.atmopulse-europe-share-gap {{
    height: 2.1rem;
}}
.atmopulse-map-table-gap {{
    height: 0.85rem;
}}
.atmopulse-panel-credit {{
    font-family: {o} !important;
    font-size: 11px !important;
    font-weight: {uw} !important;
    color: rgba(49, 51, 63, 0.62) !important;
    line-height: 1.35 !important;
    margin: 1.15rem 0 0.35rem 0 !important;
}}
.atmopulse-sev-pair {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    gap: 28px !important;
    width: max-content !important;
    max-width: 100% !important;
    margin: 0 0 0.85rem 0 !important;
}}
.atmopulse-sev-block {{
    width: max-content !important;
}}
.atmopulse-sev-heading {{
    font-family: {o} !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    margin: 0 0 6px 0 !important;
    line-height: 1.2 !important;
}}
.atmopulse-sev-heading-warm {{
    color: {ATMOPULSE_WARM['p95']} !important;
}}
.atmopulse-sev-heading-cold {{
    color: {ATMOPULSE_COLD['p5']} !important;
}}
.atmopulse-sev-table {{
    width: auto !important;
    min-width: 0 !important;
    table-layout: fixed !important;
    border-collapse: collapse;
    font-family: {o};
    font-size: 13px;
    font-weight: {uw};
    color: {brand['text_on_light']};
    line-height: 1.35;
}}
.atmopulse-sev-table th,
.atmopulse-sev-table td {{
    width: 7.5rem !important;
    min-width: 7.5rem !important;
    max-width: 7.5rem !important;
    box-sizing: border-box !important;
    padding: 4px 8px !important;
    text-align: right !important;
    border-bottom: 1px solid {_hex_to_rgba(brand['primary'], 0.10)};
    white-space: nowrap;
}}
.atmopulse-sev-table thead th {{
    font-weight: 600;
    border-bottom: 1px solid {_hex_to_rgba(brand['primary'], 0.35)};
    padding-bottom: 6px !important;
}}
.atmopulse-sev-table tbody th[scope="row"] {{
    text-align: left !important;
    font-weight: {uw};
}}
.atmopulse-sev-table tbody tr.atmopulse-sev-row-active th,
.atmopulse-sev-table tbody tr.atmopulse-sev-row-active td {{
    font-weight: 600;
    background: {_hex_to_rgba(brand['primary'], 0.07)};
}}

/* Synoptic maps: the keyed frame is locked to EUROPE_BBOX (70° lon : 42° lat).
   Plotly fills that box (absolute inset) so width and height scale together
   at any browser zoom. Titles sit outside the frame (see .atmopulse-map-title). */
.atmopulse-map-title {{
    text-align: center !important;
    font-family: {s} !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    color: {brand['primary']} !important;
    margin: 8px 0 4px 0 !important;
    line-height: 1.25 !important;
    white-space: nowrap !important;
}}
.atmopulse-map-credit,
.atmopulse-chart-credit {{
    font-family: {o} !important;
    font-size: 14px !important;
    font-weight: {uw} !important;
    color: rgba(49, 51, 63, 0.6) !important;
    line-height: 1.3 !important;
}}
.atmopulse-map-credit {{
    height: 0 !important;
    margin: 0 10px 0 0 !important;
    padding: 0 !important;
    overflow: visible !important;
    text-align: right !important;
    font-size: 11px !important;
    line-height: 1.15 !important;
    position: relative !important;
    z-index: 6 !important;
    top: -18px !important;
    pointer-events: none !important;
    background: rgba(255,255,255,0.78);
    width: fit-content !important;
    margin-left: auto !important;
}}
.atmopulse-chart-credit {{
    height: 0 !important;
    margin: 0 0 0 48px !important;
    padding: 0 !important;
    overflow: visible !important;
    text-align: left !important;
    font-size: 11px !important;
    line-height: 1.15 !important;
    position: relative !important;
    z-index: 6 !important;
    top: -22px !important;
    pointer-events: none !important;
    background: rgba(255,255,255,0.78);
    width: fit-content !important;
}}
[data-testid="stElementContainer"]:has(.atmopulse-map-legend) + [data-testid="stElementContainer"] .atmopulse-chart-credit,
[data-testid="stElementContainer"]:has(.atmopulse-map-legend) + [data-testid="stElementContainer"] .atmopulse-map-credit,
[data-testid="stElementContainer"]:has(.atmopulse-narrative-banner.is-under-figure) + [data-testid="stElementContainer"] .atmopulse-chart-credit,
[data-testid="stElementContainer"]:has(.atmopulse-narrative-banner.is-under-figure) + [data-testid="stElementContainer"] .atmopulse-map-credit {{
    margin-top: 4px !important;
}}
/* Markdown help icons are pinned to the right of a full-width element.
   Shrink that element to the title so the icon sits directly after the text. */
.st-key-swipe_map_title [data-testid="stElementContainer"],
.st-key-map_opacity_title [data-testid="stElementContainer"] {{
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    width: fit-content !important;
    max-width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
    gap: 0.35rem !important;
    position: relative !important;
}}
.st-key-swipe_map_title [data-testid="stTooltipIcon"],
.st-key-map_opacity_title [data-testid="stTooltipIcon"] {{
    position: static !important;
    top: auto !important;
    right: auto !important;
    margin: 0 !important;
    transform: none !important;
}}
.st-key-swipe_map_title .atmopulse-map-title,
.st-key-map_opacity_title .atmopulse-map-title {{
    margin: 0 !important;
}}
.st-key-map_a,
.st-key-map_b,
.st-key-map_flicker,
.st-key-map_opacity {{
    position: relative !important;
    aspect-ratio: 70 / 42 !important;
    width: 100% !important;
    height: auto !important;
    flex: 0 0 auto !important;
    max-width: 100% !important;
    overflow: hidden !important;
}}
.ap-opacity-ctrl {{
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    margin: 12px 2px 0 2px !important;
    font-family: {o} !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    color: #31333F !important;
}}
.ap-opacity-ctrl span {{
    white-space: nowrap !important;
}}
.ap-opacity-ctrl input[type="range"] {{
    flex: 1 1 auto !important;
    accent-color: {brand['primary']} !important;
}}
.st-key-map_a > div,
.st-key-map_b > div,
.st-key-map_flicker > div,
.st-key-map_opacity > div,
.st-key-map_a [data-testid="stPlotlyChart"],
.st-key-map_b [data-testid="stPlotlyChart"],
.st-key-map_flicker [data-testid="stPlotlyChart"],
.st-key-map_opacity [data-testid="stPlotlyChart"],
.st-key-map_a .js-plotly-plot,
.st-key-map_b .js-plotly-plot,
.st-key-map_flicker .js-plotly-plot,
.st-key-map_opacity .js-plotly-plot,
.st-key-map_a .plot-container,
.st-key-map_b .plot-container,
.st-key-map_flicker .plot-container,
.st-key-map_opacity .plot-container,
.st-key-map_a .svg-container,
.st-key-map_b .svg-container,
.st-key-map_flicker .svg-container,
.st-key-map_opacity .svg-container {{
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    max-height: 100% !important;
}}
/* Plotly keeps the drawing at the size of its first layout. Stretch that
   SVG to the frame so the field fills the map instead of sitting inside it.
   The opacity slider is HTML under the frame, so this stretch stays on the map. */
.st-key-map_a svg.main-svg,
.st-key-map_b svg.main-svg,
.st-key-map_flicker svg.main-svg,
.st-key-map_opacity svg.main-svg {{
    width: 100% !important;
    height: 100% !important;
    max-width: none !important;
    max-height: none !important;
}}
.st-key-swipe_map_frame + [data-testid="stElementContainer"] .atmopulse-map-legend,
[data-testid="stElementContainer"]:has(.ap-opacity-ctrl) + [data-testid="stElementContainer"] .atmopulse-map-legend {{
    margin-top: 8px !important;
}}
[data-testid="stHtml"]:has(.ap-map-fit) {{
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}}

/* Compact left-aligned press-export buttons; SVG, PDF, and CSV sit together. */
div[class*="st-key-press-row"],
div[class*="st-key-press-row"][data-testid="stHorizontalBlock"],
div[class*="st-key-press-row"] [data-testid="stHorizontalBlock"] {{
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 0.35rem !important;
    width: fit-content !important;
    max-width: 100% !important;
}}
div[class*="st-key-press-row"] [data-testid="stHorizontalBlock"] > *,
div[class*="st-key-press-row"][data-testid="stHorizontalBlock"] > * {{
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: max-content !important;
    padding: 0 !important;
}}
div[class*="st-key-press-row"] [data-testid="stDownloadButton"] button,
div[class*="st-key-press-row"] [data-testid="stButton"] button,
div[class*="st-key-press-row"] button {{
    width: auto !important;
    min-height: 1.65rem !important;
    padding: 0.12rem 0.7rem !important;
    font-size: 12px !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
}}
div[class*="st-key-wave_event_foot_"] {{
    width: max-content !important;
    max-width: 100% !important;
    gap: 0.35rem !important;
}}
div[class*="st-key-wave_event_foot_"] > [data-testid="stElementContainer"] {{
    margin-bottom: 0 !important;
}}
div[class*="st-key-wave_event_foot_"] > [data-testid="stElementContainer"]:last-child {{
    width: 0 !important;
    min-width: 100% !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}}
div[class*="st-key-wave_event_foot_"] [data-testid="stButton"] button {{
    width: 100% !important;
    min-width: 0 !important;
    min-height: 1.65rem !important;
    height: 1.65rem !important;
    padding: 0.12rem 0.7rem !important;
    font-size: 12px !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
}}
div[class*="st-key-press-row-map_"] {{
    margin-bottom: 0 !important;
}}
.atmopulse-meteo-yearly-gap {{
    height: 1.75rem !important;
}}
.atmopulse-z500-gap {{
    height: 0.95rem;
}}
.st-key-press-row-meteogram_A,
.st-key-press-row-meteogram_B,
.st-key-press-row-meteogram_live,
.st-key-press-row-meteogram_year {{
    margin-top: 0.45rem !important;
}}
.st-key-atmopulse_split_meteo_foot {{
    margin-top: 0.45rem !important;
}}

/* Swipe iframe is given height=900 so the map can lay out. The used box
   follows the map (70×42) plus the slider, which drops the empty band
   without shrinking the drawing. */
.st-key-swipe_map_frame,
.st-key-swipe_map_frame [data-testid="stElementContainer"],
.st-key-swipe_map_frame .stElementContainer {{
    height: auto !important;
    flex: 0 0 auto !important;
}}
.st-key-swipe_map_frame iframe {{
    width: 100% !important;
    aspect-ratio: 70 / 45.2 !important;
    display: block !important;
    border: 0 !important;
    overflow: hidden !important;
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
/* Live and Archive need a visible gap; the rule above collapses every sidebar radio. */
.st-key-map_archive_mode [data-testid="stRadio"] div[role="radiogroup"] {{
    gap: 1.35rem !important;
}}
/* Compare, Map Layout, and the period choice sit on one row with equal gaps,
   so Map Layout is centred between the other two. They stack on a phone. */
.st-key-map_top_controls [data-testid="stHorizontalBlock"] {{
    justify-content: space-between !important;
    align-items: flex-start !important;
}}
.st-key-map_top_controls [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
.st-key-map_top_controls [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
    flex: 0 0 auto !important;
    width: auto !important;
    max-width: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}}
@media (max-width: 820px) {{
    .st-key-map_top_controls [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
        justify-content: flex-start !important;
        row-gap: 0.6rem !important;
    }}
    .st-key-map_top_controls [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-map_top_controls [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: 100% !important;
    }}
}}
/* Meteogram: Compare left, Layout centred, reference period on the right.
   Wavogram: Layout left, reference period centred. */
.st-key-met_top_controls [data-testid="stHorizontalBlock"],
.st-key-wave_top_controls [data-testid="stHorizontalBlock"] {{
    position: relative !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    width: 100% !important;
}}
.st-key-met_top_controls [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
.st-key-met_top_controls [data-testid="stHorizontalBlock"] > [data-testid="column"],
.st-key-wave_top_controls [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
.st-key-wave_top_controls [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
    flex: 0 0 auto !important;
    width: auto !important;
    max-width: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}}
.st-key-wave_top_controls [data-testid="stHorizontalBlock"] > :nth-child(2) {{
    position: absolute !important;
    left: 50% !important;
    top: 0 !important;
    transform: translateX(-50%) !important;
}}
.st-key-met_top_controls [data-testid="stHorizontalBlock"] {{
    justify-content: space-between !important;
}}
.st-key-met_top_controls [data-testid="stHorizontalBlock"] > :nth-child(2),
.st-key-met_top_controls [data-testid="stHorizontalBlock"] > :nth-child(3),
.st-key-met_top_controls [data-testid="stHorizontalBlock"] > :nth-child(4) {{
    position: static !important;
    left: auto !important;
    top: auto !important;
    transform: none !important;
    margin-left: 0 !important;
}}
.st-key-met_top_controls .st-key-met_archive_year [data-testid="stSelectbox"] {{
    width: 8.5rem !important;
}}
@media (max-width: 820px) {{
    .st-key-met_top_controls [data-testid="stHorizontalBlock"],
    .st-key-wave_top_controls [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
        justify-content: flex-start !important;
        row-gap: 0.6rem !important;
    }}
    .st-key-met_top_controls [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-met_top_controls [data-testid="stHorizontalBlock"] > [data-testid="column"],
    .st-key-wave_top_controls [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-wave_top_controls [data-testid="stHorizontalBlock"] > [data-testid="column"],
    .st-key-met_top_controls [data-testid="stHorizontalBlock"] > :nth-child(2),
    .st-key-met_top_controls [data-testid="stHorizontalBlock"] > :nth-child(3),
    .st-key-met_top_controls [data-testid="stHorizontalBlock"] > :nth-child(4),
    .st-key-wave_top_controls [data-testid="stHorizontalBlock"] > :nth-child(2) {{
        position: static !important;
        left: auto !important;
        top: auto !important;
        transform: none !important;
        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: 100% !important;
        margin-left: 0 !important;
    }}
    .st-key-met_top_controls .st-key-met_archive_year [data-testid="stSelectbox"] {{
        width: 100% !important;
    }}
}}
/* Help icons sit on the label text, not at the far end of the date field. */
.st-key-map_archive_date [data-testid="stWidgetLabel"],
.st-key-map_compare_date [data-testid="stWidgetLabel"],
.st-key-met_archive_year [data-testid="stWidgetLabel"],
.st-key-met_compare_year [data-testid="stWidgetLabel"],
.st-key-wave_drill_ms [data-testid="stWidgetLabel"] {{
    width: fit-content !important;
    max-width: 100% !important;
    justify-content: flex-start !important;
    gap: 0.35rem !important;
}}
.st-key-met_archive_year [data-testid="stTooltipIcon"],
.st-key-met_compare_year [data-testid="stTooltipIcon"],
.st-key-wave_drill_ms [data-testid="stTooltipIcon"] {{
    position: static !important;
    top: auto !important;
    right: auto !important;
}}
.st-key-met_year_row {{
    position: relative !important;
}}
.st-key-met_year_row [data-testid="stHorizontalBlock"] > :nth-child(1),
.st-key-met_year_row [data-testid="stHorizontalBlock"] > :nth-child(2) {{
    flex: 0 0 8.4rem !important;
    width: 8.4rem !important;
    max-width: 8.4rem !important;
}}
.st-key-met_year_row .st-key-met_archive_year [data-testid="stSelectbox"],
.st-key-met_year_row .st-key-met_compare_year [data-testid="stSelectbox"] {{
    width: 7.6rem !important;
}}
.st-key-met_top_controls [data-testid="stHorizontalBlock"] > :nth-child(2) [data-testid="stRadio"] + [data-testid="stElementContainer"] {{
    margin-top: 0.15rem !important;
}}
.atmopulse-date-title-gap {{
    height: 18px;
}}
[data-testid="stElementContainer"]:has(.atmopulse-date-title-gap) {{
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}}
/* Archive / compare clocks: the date field is 75% of the row, and
   Day before / Day after sit in the remaining space, vertically centred. */
.st-key-map_archive_clock,
.st-key-map_compare_clock {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 0.75rem !important;
}}
.st-key-map_archive_clock > [data-testid="stElementContainer"]:first-child,
.st-key-map_compare_clock > [data-testid="stElementContainer"]:first-child {{
    flex: 0 0 75% !important;
    width: 75% !important;
    max-width: 75% !important;
    min-width: 0 !important;
}}
.st-key-map_archive_clock > [data-testid="stElementContainer"]:last-child,
.st-key-map_compare_clock > [data-testid="stElementContainer"]:last-child {{
    flex: 1 1 auto !important;
    width: auto !important;
    max-width: none !important;
    min-width: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
.st-key-map_archive_clock [data-testid="stHorizontalBlock"],
.st-key-map_compare_clock [data-testid="stHorizontalBlock"] {{
    width: 100% !important;
    gap: 0.4rem !important;
    align-items: center !important;
    justify-content: center !important;
}}
.st-key-map_archive_clock button,
.st-key-map_compare_clock button {{
    text-align: center !important;
    line-height: 1.15 !important;
    padding-left: 0.35rem !important;
    padding-right: 0.35rem !important;
}}
.st-key-map_archive_clock button p,
.st-key-map_compare_clock button p {{
    font-size: 1rem !important;
    white-space: nowrap !important;
    text-align: center !important;
    line-height: 1.15 !important;
    overflow: visible !important;
    text-overflow: clip !important;
}}
.st-key-map_archive_clock [data-baseweb="input"] input,
.st-key-map_compare_clock [data-baseweb="input"] input {{
    font-size: 1rem !important;
}}
/* Two dates share one row. Each date field is half of its column so
   Day before / Day after keep the other half and the label stays inside. */
.st-key-map_date_pair .st-key-map_archive_clock,
.st-key-map_date_pair .st-key-map_compare_clock {{
    align-items: flex-end !important;
    gap: 0.4rem !important;
}}
.st-key-map_date_pair .st-key-map_archive_clock > :first-child {{
    flex: 0 0 50% !important;
    width: 50% !important;
    max-width: 50% !important;
    min-width: 0 !important;
}}
.st-key-map_date_pair .st-key-map_compare_clock > :first-child {{
    flex: 0 0 35% !important;
    width: 35% !important;
    max-width: 35% !important;
    min-width: 0 !important;
}}
.st-key-map_date_pair .st-key-map_archive_clock > :last-child,
.st-key-map_date_pair .st-key-map_compare_clock > :last-child {{
    flex: 0 0 35% !important;
    width: 35% !important;
    max-width: 35% !important;
    min-width: 0 !important;
    margin-right: auto !important;
    padding-bottom: 0.15rem !important;
}}
.st-key-map_date_pair .st-key-map_archive_clock [data-testid="stHorizontalBlock"] > div,
.st-key-map_date_pair .st-key-map_compare_clock [data-testid="stHorizontalBlock"] > div {{
    min-width: 0 !important;
    flex: 1 1 0 !important;
    width: auto !important;
}}
.st-key-map_date_pair .st-key-map_archive_clock button,
.st-key-map_date_pair .st-key-map_compare_clock button {{
    min-width: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
    overflow: visible !important;
    padding: 0.3rem 0.35rem !important;
}}
.st-key-map_date_pair .st-key-map_archive_clock button p,
.st-key-map_date_pair .st-key-map_compare_clock button p {{
    font-size: 1rem !important;
    line-height: 1.15 !important;
    white-space: nowrap !important;
    text-align: center !important;
    overflow: visible !important;
    text-overflow: clip !important;
    margin: 0 !important;
}}
.st-key-map_synoptic_overlays [data-testid="stCaptionContainer"] p {{
    font-size: 0.75rem !important;
    line-height: 1.35 !important;
}}
.st-key-map_live_day_buttons [data-testid="stHorizontalBlock"] {{
    gap: 0.3rem !important;
}}
.st-key-map_live_day_buttons button {{
    padding: 0.4rem 0.2rem !important;
    overflow: visible !important;
}}
.st-key-map_live_day_buttons button,
.st-key-map_live_day_buttons button p {{
    white-space: nowrap !important;
    text-align: center !important;
    line-height: 1.15 !important;
    font-size: 0.78rem !important;
}}
.st-key-map_target_date [data-testid="stAlert"] {{
    flex-wrap: nowrap !important;
    align-items: center !important;
}}
.st-key-map_target_date [data-testid="stAlert"],
.st-key-map_target_date [data-testid="stAlert"] p,
.st-key-map_target_date [data-testid="stAlert"] span,
.st-key-map_target_date [data-testid="stAlert"] strong {{
    font-size: 14px !important;
    line-height: 1.15 !important;
    white-space: nowrap !important;
}}
.st-key-wave_z500_outline {{
    margin-top: 0.85rem !important;
    width: fit-content !important;
    max-width: 100% !important;
}}
.st-key-wave_z500_outline [data-testid="stCheckbox"] {{
    width: fit-content !important;
    max-width: 100% !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-start !important;
    gap: 0.2rem !important;
}}
.st-key-wave_z500_outline label {{
    width: auto !important;
    max-width: 100% !important;
}}
.st-key-wave_z500_outline [data-testid="stTooltipIcon"] {{
    position: static !important;
    inset: auto !important;
    margin: 0.12rem 0 0 0 !important;
    transform: none !important;
    flex: 0 0 auto !important;
}}
div:has(> .st-key-atmopulse_nav_bar) + [data-testid="stElementContainer"] {{
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}}
.st-key-atmopulse_nav_bar + div hr,
.st-key-atmopulse_nav_bar + div [data-testid="stDivider"],
div:has(> .st-key-atmopulse_nav_bar) + [data-testid="stElementContainer"] hr,
div:has(> .st-key-atmopulse_nav_bar) + [data-testid="stElementContainer"] [data-testid="stDivider"] {{
    margin-top: 0.15rem !important;
    margin-bottom: 0.35rem !important;
}}
@media (max-width: 820px) {{
    .st-key-map_archive_clock,
    .st-key-map_compare_clock {{
        flex-wrap: wrap !important;
    }}
    .st-key-map_archive_clock > [data-testid="stElementContainer"]:first-child,
    .st-key-map_compare_clock > [data-testid="stElementContainer"]:first-child,
    .st-key-map_archive_clock > [data-testid="stElementContainer"]:last-child,
    .st-key-map_compare_clock > [data-testid="stElementContainer"]:last-child {{
        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: 100% !important;
    }}
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label p,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label span,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] li {{
    font-size: 14px !important; 
    font-weight: {uw} !important; 
    color: inherit !important;
    line-height: 1.35 !important;
}}
section[data-testid="stSidebar"] .stCheckbox {{ 
    margin-top: -12px; 
}}
/* Overlay checkboxes share one size. Labels that start with digits
   (500 hPa) can be parsed as markdown lists and otherwise render larger. */
.st-key-map_synoptic_overlays [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"],
.st-key-map_synoptic_overlays [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] *,
.st-key-map_overlay_mslp [data-testid="stMarkdownContainer"],
.st-key-map_overlay_mslp [data-testid="stMarkdownContainer"] *,
.st-key-map_overlay_z500 [data-testid="stMarkdownContainer"],
.st-key-map_overlay_z500 [data-testid="stMarkdownContainer"] *,
.st-key-map_overlay_mslp_anom [data-testid="stMarkdownContainer"],
.st-key-map_overlay_mslp_anom [data-testid="stMarkdownContainer"] *,
.st-key-map_overlay_z500_anom [data-testid="stMarkdownContainer"],
.st-key-map_overlay_z500_anom [data-testid="stMarkdownContainer"] * {{
    font-family: {o} !important;
    font-size: 14px !important;
    font-weight: {uw} !important;
    line-height: 1.35 !important;
    font-variant-numeric: lining-nums !important;
}}
.st-key-map_synoptic_overlays [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] ol,
.st-key-map_overlay_z500 [data-testid="stMarkdownContainer"] ol,
.st-key-map_overlay_z500_anom [data-testid="stMarkdownContainer"] ol {{
    list-style: none !important;
    padding-left: 0 !important;
    margin: 0 !important;
}}
section[data-testid="stSidebar"] button, 
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] {{
    font-family: {o} !important; 
    font-weight: {uw} !important;
}}

/* Main column uses the width beside the sidebar. Side padding stays tight
   so the map frames, not the margins, take the screen. */
[data-testid="stMainBlockContainer"].block-container {{
    max-width: none !important;
    width: 100% !important;
    padding-top: 0.75rem !important;
    padding-left: 1rem !important;
    padding-right: 0.2rem !important;
    padding-bottom: 2rem !important;
}}
section[data-testid="stSidebar"][aria-expanded="true"] {{
    width: 15.5rem !important;
    min-width: 15.5rem !important;
    max-width: 15.5rem !important;
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

.atmopulse-data-vintage {{
    font-size: 12px !important;
    color: #555 !important;
    margin-top: 0.7rem !important;
    margin-bottom: 0.85rem !important;
}}
.atmopulse-data-vintage.is-disabled {{
    color: rgba(0, 0, 0, 0.4) !important;
}}
.st-key-target_location_row {{
    margin-top: 0.7rem !important;
    margin-bottom: 0.35rem !important;
}}
.st-key-target_location_row [data-testid="stHorizontalBlock"] {{
    align-items: center !important;
    flex-wrap: nowrap !important;
    gap: 0.45rem !important;
}}
.st-key-target_location_row [data-testid="stElementContainer"],
.st-key-target_location_row [data-testid="stMarkdownContainer"] {{
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}}
.atmopulse-target-location-label {{
    margin: 0 !important;
    font-family: {o} !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    line-height: 1.15 !important;
    white-space: nowrap !important;
}}
.st-key-target_location_row [data-testid="stHorizontalBlock"] > div {{
    display: flex !important;
    align-items: center !important;
    align-self: center !important;
}}
.st-key-target_location_row [data-testid="stHorizontalBlock"] > div:first-child {{
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: max-content !important;
}}
.st-key-target_location_row [data-testid="stHorizontalBlock"] > div:last-child {{
    flex: 1 1 auto !important;
    width: auto !important;
    min-width: 12rem !important;
}}
.st-key-loc_field [data-testid="stWidgetLabel"] {{
    display: none !important;
}}
.st-key-loc_field [data-testid="stSelectbox"] {{
    width: 100% !important;
    margin-bottom: 0 !important;
}}
.st-key-loc_field input {{
    font-family: {o} !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    height: 2.35rem !important;
    line-height: 1.15 !important;
    padding: 0.54rem 8px 0 8px !important;
    transform: translateY(-2px) !important;
}}
.st-key-loc_field input::placeholder {{
    font-weight: 500 !important;
}}
@media (max-width: 820px) {{
    .st-key-target_location_row [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
    }}
    .st-key-target_location_row [data-testid="stHorizontalBlock"] > div:first-child,
    .st-key-target_location_row [data-testid="stHorizontalBlock"] > div:last-child {{
        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }}
}}
.atmopulse-narrative-banner {{
    background-color: {brand['nav_bg']} !important;
    color: {brand['text_on_light']};
    padding: 0.75rem 1rem !important;
    border-radius: 0.5rem !important;
    font-family: {o} !important;
    font-weight: 400 !important;
    font-size: 15px !important;
    margin: 0 0 1.15rem 0 !important;
    line-height: 1.45 !important;
}}
.atmopulse-narrative-banner.is-under-figure {{
    margin: 0 !important;
    padding: 0.55rem 1rem !important;
}}
[data-testid="stElementContainer"]:has(.atmopulse-narrative-banner.is-under-figure),
[data-testid="stElementContainer"]:has(.atmopulse-narrative-banner.is-map),
[data-testid="stElementContainer"]:has(.atmopulse-narrative-banner.is-wave) {{
    margin-top: 0.35rem !important;
    margin-bottom: 0.45rem !important;
}}
.st-key-map_opacity .slider-group,
.st-key-map_opacity .slider-container,
.st-key-map_opacity .slider-rail,
.st-key-map_opacity .slider-handle,
.st-key-map_opacity .slider-ticks,
.st-key-map_opacity text.slider-label {{
    display: none !important;
}}
.atmopulse-narrative-chip {{
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 0 !important;
    background: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
    font-family: inherit !important;
    font-size: inherit !important;
    font-weight: 400 !important;
    white-space: nowrap !important;
    display: inline !important;
    line-height: inherit !important;
    vertical-align: baseline !important;
}}
.atmopulse-narrative-chip.atmopulse-sev-normal,
.atmopulse-narrative-chip.atmopulse-sev-warm-moderate,
.atmopulse-narrative-chip.atmopulse-sev-warm-strong,
.atmopulse-narrative-chip.atmopulse-sev-warm-extreme,
.atmopulse-narrative-chip.atmopulse-sev-warm-record,
.atmopulse-narrative-chip.atmopulse-sev-cold-moderate,
.atmopulse-narrative-chip.atmopulse-sev-cold-strong,
.atmopulse-narrative-chip.atmopulse-sev-cold-extreme,
.atmopulse-narrative-chip.atmopulse-sev-cold-record {{
    background: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    border-radius: 0 !important;
}}
.atmopulse-narrative-chip.atmopulse-sev-normal {{
    color: {brand['text_on_light']} !important;
}}
.atmopulse-narrative-chip.atmopulse-sev-warm-moderate {{ color: #C47D00 !important; }}
.atmopulse-narrative-chip.atmopulse-sev-warm-strong {{ color: #D86A00 !important; }}
.atmopulse-narrative-chip.atmopulse-sev-warm-extreme {{ color: {ATMOPULSE_WARM['p95']} !important; }}
.atmopulse-narrative-chip.atmopulse-sev-warm-record {{ color: {ATMOPULSE_WARM['rec']} !important; }}
.atmopulse-narrative-chip.atmopulse-sev-cold-moderate {{ color: #1A7AAD !important; }}
.atmopulse-narrative-chip.atmopulse-sev-cold-strong {{ color: {ATMOPULSE_COLD['p10']} !important; }}
.atmopulse-narrative-chip.atmopulse-sev-cold-extreme {{ color: {ATMOPULSE_COLD['p5']} !important; }}
.atmopulse-narrative-chip.atmopulse-sev-cold-record {{ color: {ATMOPULSE_COLD['rec']} !important; }}
.atmopulse-sev-normal {{
    background-color: {brand['mode_track']} !important;
    color: {brand['text_on_light']} !important;
}}
.atmopulse-sev-warm-moderate {{
    background-color: {ATMOPULSE_WARM['p75']} !important;
    color: {brand['text_on_light']} !important;
}}
.atmopulse-sev-warm-strong {{
    background-color: {ATMOPULSE_WARM['p90']} !important;
    color: {brand['text_on_light']} !important;
}}
.atmopulse-sev-warm-extreme {{
    background-color: {ATMOPULSE_WARM['p95']} !important;
    color: {brand['text_on_primary']} !important;
}}
.atmopulse-sev-warm-record {{
    background-color: {ATMOPULSE_WARM['rec']} !important;
    color: {brand['text_on_primary']} !important;
}}
.atmopulse-sev-cold-moderate {{
    background-color: {ATMOPULSE_COLD['p25']} !important;
    color: {brand['text_on_light']} !important;
}}
.atmopulse-sev-cold-strong {{
    background-color: {ATMOPULSE_COLD['p10']} !important;
    color: {brand['text_on_light']} !important;
}}
.atmopulse-sev-cold-extreme {{
    background-color: {ATMOPULSE_COLD['p5']} !important;
    color: {brand['text_on_primary']} !important;
}}
.atmopulse-sev-cold-record {{
    background-color: {ATMOPULSE_COLD['rec']} !important;
    color: {brand['text_on_primary']} !important;
}}


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
/* Wavogram ridge hover: Plotly gives every <br> a full-height tspan, so a
   half-line gap before Z500 is done here — 5th line is a spacer (Duration,
   Length, Severity, Rank, spacer, Z500-…). */
[class*="st-key-wave_ridge_click"] .hoverlayer .hovertext tspan:nth-of-type(5) {{
    font-size: 6px !important;
    fill: transparent !important;
    stroke: none !important;
}}
[class*="st-key-wave_ridge_click"] .hoverlayer .hovertext br:nth-of-type(4) {{
    display: block;
    content: "";
    line-height: 0.5em;
    margin: 0.15em 0;
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
# Shared Map Tracker + Meteogram ladder. "above" is meteogram-only (very pale
# above-average fill); Moderate/Strong/Extreme/Record mean the same colours
# in every view.
ATMOPULSE_WARM = {
    "above": "#FFF4CC",  # very pale gold — meteogram above-average only
    "p75": "#FFD166",    # Moderate (slightly stronger than the old cream gold)
    "p90": "#FF9933",    # Strong (90th percentile)
    "p95": "#CC0000",    # Extreme (95th percentile)
    "rec": "#E91E8C",    # All-time record (distinct vivid pink)
}

# --- Cold percentiles ---
ATMOPULSE_COLD = {
    "below": "#EAF6FF",  # very pale cyan — meteogram below-average only
    "p25": "#B3E8FF",    # Moderate (slightly stronger than the old ice blue)
    "p10": "#3399FF",    # Strong (10th percentile)
    "p5": "#0056B3",     # Extreme (5th percentile, aligned with primary brand)
    "rec": "#4B0082",    # All-time record (indigo)
}

# --- Overlays & auxiliary series ---
ATMOPULSE_OVERLAY = {
    "mslp_contour": "#2E7D32",
    "mslp_hl": "#2E7D32",
    "mslp_anom_contour": "#6B4F1D",
    "z500_contour": "#0056B3",
    "z500_anom_contour": "#7B1FA2",
    "coast": "#5A5A5A",
    "coast_width": 1.25,
    "border": "#8E9499",
    "border_width": 0.65,
    "land": "#FAF8F4",
    "sea": "#F3F6F8",
    "grid": "rgba(200,200,200,0.3)",
    "annotation": "gray",
}

# --- Meteogram fill opacities ---
_METEO_ALPHA = {
    "above": 0.50,
    "below": 0.50,
    "moderate": 0.50,
    "strong": 0.60,
    "extreme": 0.70,
    "record": 0.85,
}
_WARM_FILL_KEY = {"above": "above", "moderate": "p75", "strong": "p90", "extreme": "p95", "record": "rec"}
_COLD_FILL_KEY = {"below": "below", "moderate": "p25", "strong": "p10", "extreme": "p5", "record": "rec"}


def warm_rgba(level: str) -> str:
    """Return rgba string for meteogram warm fills (above, moderate, strong, extreme, record)."""
    return _hex_to_rgba(ATMOPULSE_WARM[_WARM_FILL_KEY[level]], _METEO_ALPHA[level])


def cold_rgba(level: str) -> str:
    """Return rgba string for meteogram cold fills (below, moderate, strong, extreme, record)."""
    return _hex_to_rgba(ATMOPULSE_COLD[_COLD_FILL_KEY[level]], _METEO_ALPHA[level])


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
    """Inline CSS for HTML legend badges. One palette everywhere: Moderate/Strong/
    Extreme/Record match Map Tracker; ``above``/``below`` are meteogram-only pales.
    """
    palette = ATMOPULSE_WARM if warm_or_cold == "warm" else ATMOPULSE_COLD
    key = (_WARM_FILL_KEY if warm_or_cold == "warm" else _COLD_FILL_KEY)[level]
    bg = palette[key]
    light_keys = {"above", "below", "p75", "p90", "p25", "p10"}
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