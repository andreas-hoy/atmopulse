"""
AtmoPulse Press Export & Output Credit (frontend_export.py)

Generic, chart-agnostic Plotly export plumbing extracted from
frontend_plots.py (Phase 0 split): the "AtmoPulse (atmopulse.eu) · CC BY
4.0" output-credit annotation baked onto every exported figure, the
Kaleido-backed on-demand SVG/PDF renderer (with an in-memory cache keyed on
a cheap figure fingerprint), the compact SVG/PDF/CSV export row
(`render_press_export`), and the `st_plotly_press` wrapper that pairs
`st.plotly_chart` with that row.

Pure Streamlit/Plotly plumbing — no map/meteogram/wavogram-specific drawing
code lives here, and this module never imports frontend_maps.py,
frontend_meteogram.py, or frontend_wavogram.py (they import FROM here
instead), so there is no import cycle.
"""

from __future__ import annotations

import base64
import hashlib
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from atmopulse_theme import (
    ATMOPULSE_BRAND,
    ATMOPULSE_FONTS,
    plotly_title_font,
)
from config import is_aifs_model

# Every chart except the synoptic maps (which use SYNOPTIC_MAP_CONFIG in
# frontend_maps.py, sized to the fixed 70:42 map frame) keeps zoom and pan.
# The on-screen camera is the PNG; SVG/PDF under the figure are the press
# files, so the modebar's own "toImage" PNG button is redundant and removed.
PLOTLY_UI_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["toImage"],
}


def _output_credit_text(*, html_break: bool = False) -> str:
    """CC BY 4.0 credit required by LICENSE.md / Legal — not a © claim."""
    src = "AIFS" if is_aifs_model() else "IFS"
    data = f"Data: ERA5 / ECMWF {src}"
    sep = "<br>" if html_break else " \u00b7 "
    return f"AtmoPulse (atmopulse.eu) \u00b7 CC BY 4.0{sep}{data}"


def _credit_meta(fig) -> dict:
    meta = fig.layout.meta
    return dict(meta) if isinstance(meta, dict) else {}


def _has_output_credit(fig) -> bool:
    meta = fig.layout.meta
    return isinstance(meta, dict) and bool(meta.get("output_credit"))


def _mark_output_credit(fig) -> None:
    payload = _credit_meta(fig)
    payload["output_credit"] = True
    fig.update_layout(meta=payload)


def _strip_credit_annotations(fig) -> None:
    anns = list(fig.layout.annotations or ())
    kept = tuple(
        a for a in anns
        if "atmopulse.eu" not in str(getattr(a, "text", "") or "").lower()
    )
    fig.layout.annotations = kept


def _axes_draw_ticks(fig) -> bool:
    xa = fig.layout.xaxis
    return not (xa is not None and xa.visible is False)


def _ensure_output_credit(fig) -> None:
    """Two-line credit in the bottom margin, below ticks — never on the data.

    Paper y=0 is the bottom of the plotting domain (the x-axis). yanchor='top'
    plus a pixel yshift hangs the text into the extra bottom margin. Maps have
    no ticks, so they need only a small gap; charts need room for tick labels.
    """
    if fig is None:
        return
    if _has_output_credit(fig):
        return
    margin = fig.layout.margin
    b = int(margin.b) if margin is not None and margin.b is not None else 40
    tick_gap = 44 if _axes_draw_ticks(fig) else 10
    fig.update_layout(margin=dict(b=max(b, tick_gap + 52)))
    fig.add_annotation(
        text=_output_credit_text(html_break=True),
        xref="paper", yref="paper",
        x=1.0, y=0.0,
        xanchor="right", yanchor="top",
        yshift=-tick_gap,
        showarrow=False,
        align="right",
        font=dict(
            size=9,
            color=ATMOPULSE_BRAND["text_on_light"],
            family=ATMOPULSE_FONTS["sora_css"],
        ),
    )
    _mark_output_credit(fig)


def _fig_for_press(fig: go.Figure, export_title: str | None = None) -> go.Figure:
    """Clone used only for SVG/PDF: optional heading + credit in extra margins."""
    export = go.Figure(fig)
    margin = export.layout.margin
    t0 = int(margin.t) if margin is not None and margin.t is not None else 40
    b0 = int(margin.b) if margin is not None and margin.b is not None else 40
    l0 = int(margin.l) if margin is not None and margin.l is not None else 40
    r0 = int(margin.r) if margin is not None and margin.r is not None else 20
    paper = export.layout.paper_bgcolor
    if not paper or str(paper).lower() in ("rgba(0,0,0,0)", "transparent"):
        export.update_layout(paper_bgcolor="white")
    export.update_layout(
        margin=dict(
            t=max(t0, 72) if export_title else max(t0, 16),
            b=max(b0, 96),
            l=max(l0, 16),
            r=max(r0, 16),
        ),
    )
    if export_title:
        export.update_layout(
            title=dict(
                text=export_title,
                font=plotly_title_font(size=14),
                x=0.5, xanchor="center",
                y=0.98, yanchor="top",
            ),
        )
    payload = _credit_meta(export)
    payload.pop("output_credit", None)
    export.update_layout(meta=payload)
    _strip_credit_annotations(export)
    _ensure_output_credit(export)
    return export


def _browser_download(data: bytes, filename: str, mime: str) -> None:
    """Trigger a file download in the same click that rendered the vector.

    Streamlit's component iframe often blocks ``a.download`` on itself, so
    the click is issued on ``window.parent.document``. A nonce keeps
    Streamlit from skipping a repeated identical HTML block.
    """
    b64 = base64.b64encode(data).decode("ascii")
    safe_name = (
        filename.replace("\\", "_").replace("'", "").replace('"', "").replace("`", "")
    )
    nonce = int(st.session_state.get("_press_dl_n", 0)) + 1
    st.session_state["_press_dl_n"] = nonce
    components.html(
        f"""<script>
(function() {{
  const n = {nonce};
  const mime = {mime!r};
  const name = {safe_name!r};
  const b64 = {b64!r};
  function blobFromB64() {{
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], {{type: mime}});
  }}
  const url = URL.createObjectURL(blobFromB64());
  function clickOn(doc) {{
    const a = doc.createElement('a');
    a.href = url;
    a.download = name;
    a.rel = 'noopener';
    doc.body.appendChild(a);
    a.click();
    a.remove();
  }}
  try {{ clickOn(window.parent.document); }}
  catch (e) {{ clickOn(document); }}
  setTimeout(function() {{ URL.revokeObjectURL(url); }}, 4000);
}})();
</script>""",
        height=1,
        width=1,
    )


def _attach_press_csv(fig: go.Figure, csv_text: str | None) -> None:
    if not csv_text:
        return
    meta = fig.layout.meta
    payload = dict(meta) if isinstance(meta, dict) else {"press_csv": csv_text}
    if isinstance(meta, dict):
        payload["press_csv"] = csv_text
    fig.update_layout(meta=payload)


def _press_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "chart"


def _fig_fingerprint(fig: go.Figure) -> str:
    parts = [
        str(fig.layout.title.text if fig.layout.title else ""),
        str(fig.layout.height),
        str(len(fig.data)),
    ]
    for t in fig.data:
        parts.append(str(getattr(t, "type", "")))
        for attr in ("z", "y", "x"):
            val = getattr(t, attr, None)
            if val is None:
                continue
            arr = np.asarray(val)
            parts.append(attr + str(arr.shape))
            if arr.size:
                parts.append(str(arr.flat[0]))
                parts.append(str(arr.flat[-1]))
            break
    parts.append(_output_credit_text())
    return hashlib.md5("|".join(parts).encode("utf-8", errors="ignore")).hexdigest()


@st.cache_resource(show_spinner=False)
def _press_image_store() -> dict:
    return {}


def _kaleido_image(fig: go.Figure, fmt: str) -> bytes:
    store = _press_image_store()
    key = (_fig_fingerprint(fig), fmt)
    cached = store.get(key)
    if cached is not None:
        return cached
    height = int(fig.layout.height) if fig.layout.height else 840
    width = 1400 if fig.layout.height is None else 1200
    blob = fig.to_image(format=fmt, width=width, height=height)
    store[key] = blob
    return blob


def _press_vector_slot(fig: go.Figure, stem: str, fmt: str, label: str, mime: str) -> None:
    """One click renders the vector and starts the browser download."""
    if st.button(label, key=f"press_go_{fmt}_{stem}", width="content"):
        try:
            with st.spinner(f"Rendering {label}…"):
                blob = _kaleido_image(fig, fmt)
            _browser_download(blob, f"AtmoPulse_{stem}.{fmt}", mime)
        except Exception as exc:
            st.caption(f"{label} unavailable")
            st.caption(f"Vector export needs the kaleido package ({exc})")


def render_press_export(
    fig: go.Figure, stem: str, csv_text: str | None = None, *, heavy: bool = False,
    export_title: str | None = None,
) -> None:
    """Compact SVG / PDF / CSV row. Vector files are built only on request.

    ``heavy`` is kept so existing map call-sites do not break; it is no longer
    a separate code path (maps used to bundle SVG+PDF behind one Prepare click).
    ``export_title`` is drawn only on the downloaded SVG/PDF, not on the live chart.
    """
    del heavy
    if csv_text is None:
        meta = fig.layout.meta
        if isinstance(meta, dict):
            csv_text = meta.get("press_csv")
    press_fig = _fig_for_press(fig, export_title=export_title)
    stem = _press_stem(stem)
    with st.container(
        key=f"press-row-{stem}",
        horizontal=True,
        horizontal_alignment="left",
        vertical_alignment="center",
        gap="xsmall",
        width="content",
    ):
        _press_vector_slot(press_fig, stem, "svg", "SVG", "image/svg+xml")
        _press_vector_slot(press_fig, stem, "pdf", "PDF", "application/pdf")
        if csv_text:
            st.download_button(
                "CSV", data=csv_text.encode("utf-8"),
                file_name=f"AtmoPulse_{stem}.csv", mime="text/csv",
                key=f"press_{stem}_csv",
                width="content",
            )


def st_plotly_press(
    fig: go.Figure, stem: str, csv_text: str | None = None,
    *, on_select: str | None = None, selection_mode=("points",), key: str | None = None,
    export_title: str | None = None,
    legend_html: str | None = None,
    note_html: str | None = None,
    show_chart: bool = True,
    show_footer: bool = True,
    **chart_kw,
):
    """`st.plotly_chart` + the compact SVG/PDF/CSV row.

    `on_select`/`selection_mode`/`key` are optional and default to the old,
    unset behaviour (no selection wiring, no return value used) — existing
    call sites (maps, meteogram, wavogram) are unaffected. Pass
    `on_select="rerun"` to get the click-selection event back for a
    click-to-drill-down UI; the return value is Streamlit's selection dict
    (or None when selection isn't enabled/nothing is selected).
    """
    plot_kwargs = dict(use_container_width=True, **chart_kw)
    plot_kwargs.setdefault("config", PLOTLY_UI_CONFIG)
    if on_select is not None:
        plot_kwargs["on_select"] = on_select
        plot_kwargs["selection_mode"] = selection_mode
    if key is not None:
        plot_kwargs["key"] = key
    if note_html and show_chart:
        st.markdown(note_html, unsafe_allow_html=True)
    event = st.plotly_chart(fig, **plot_kwargs) if show_chart else None
    if legend_html:
        st.markdown(legend_html, unsafe_allow_html=True)
    if show_footer:
        render_press_export(fig, stem, csv_text=csv_text, export_title=export_title)
    return event


