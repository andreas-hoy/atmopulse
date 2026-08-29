\# AtmoPulse Architecture Schema



This document contains the Mermaid.js code for the AtmoPulse architecture, based on the provided technical dossier and scripts.



\## Mermaid.js Code





```mermaid

graph TD

&#x20;   %% Define Styles

&#x20;   classDef base fill:#f9f2f4,stroke:#333,stroke-width:2px;

&#x20;   classDef ops fill:#e6f2ff,stroke:#333,stroke-width:2px;

&#x20;   classDef prep fill:#eef9e6,stroke:#333,stroke-width:2px;

&#x20;   classDef front fill:#fff2e6,stroke:#333,stroke-width:2px;

&#x20;   classDef crit fill:#ffcccc,stroke:#cc0000,stroke-width:2px,stroke-dasharray: 5 5;



&#x20;   subgraph Data\_Ingestion\_and\_Climatological\_Baselining \["1. Data Ingestion \& Climatological Baselining (The Secular Anchor)"]

&#x20;       A1\[ERA5 Archive <br> 1940-Present] -->|NetCDF / Zarr <br> 00Z-23:59Z \& 12Z| B1(Calculate Percentiles <br> 5-day moving window)

&#x20;       A1 --> B2(Extract Absolute Extremes)

&#x20;       A1 --> B3(Extract Seasonal Extremes)

&#x20;       B1 --> C1\[Reference Climatology <br> 1961-1990 \& 1996-2025]

&#x20;       B2 --> C1

&#x20;       B3 --> C1

&#x20;       

&#x20;       %% Note on Leap Years

&#x20;       C1 -.-> N1>ETCCDI 365-day Calendar <br> Feb 29 excised for homoscedasticity]

&#x20;   end



&#x20;   subgraph Operational\_Forecasting \["2. Operational Forecasting \& Spatial Harmonization (The Live Engine)"]

&#x20;       D1\[IFS Deterministic <br> Live Forecast] --> E1(Temporal \& Spatial Alignment <br> CDO conservative / SciPy Regridding)

&#x20;       D2\[AIFS Deterministic <br> Live Forecast] --> E1

&#x20;       D3\[IFS Hindcasts <br> 20-year climatology] --> E2(Calculate QDM Bias)

&#x20;       

&#x20;       E1 --> F1{Model Type \& <br> Variable}

&#x20;       F1 -->|IFS Surface (TG, TN, TX)| F2(Apply QDM Bias Correction <br> CDF-Matching)

&#x20;       F1 -->|AIFS / Synoptics| F3(Apply Static Spatial Matrices <br> Sparse Matrix Mult.)

&#x20;       E2 --> F2

&#x20;   end



&#x20;   subgraph Pre\_Computation\_and\_Storage \["3. High-Performance Pre-computation (The Latency Bypass)"]

&#x20;       C1 --> G1(Scheduled Python Background Tasks)

&#x20;       F2 --> G1

&#x20;       F3 --> G1

&#x20;       G1 --> H1\[(Ultra-lightweight Binaries <br> Parquet format)]

&#x20;   end



&#x20;   subgraph Frontend\_Delivery \["4. Frontend Delivery \& UI (Streamlit)"]

&#x20;       H1 --> I1{Audience Mode Toggle}

&#x20;       I1 -->|Standard Mode| J1\[Map Tracker <br> TG anomalies, Waves, Top 10 Impact]

&#x20;       I1 -->|Expert Mode| J2\[Expert Tracker <br> T850, Z500, U300, MSLP, UTCI]

&#x20;       

&#x20;       H1 --> K1\[Point Meteogram <br> 365d history + 3d forecast]

&#x20;       H1 --> K2\[Point Wavogram <br> 1940-present heat/coldwaves]

&#x20;   end

&#x20;   

&#x20;   %% Apply Styles

&#x20;   class Data\_Ingestion\_and\_Climatological\_Baselining base;

&#x20;   class Operational\_Forecasting ops;

&#x20;   class Pre\_Computation\_and\_Storage prep;

&#x20;   class Frontend\_Delivery front;

&#x20;   class F2 crit;

```





\## How to Render and Use this Schema



This Mermaid code generates a dynamic, scalable vector graphic of your architecture. Here is how you can use it:



\### 1. GitHub (Native Support)

GitHub natively supports Mermaid in all Markdown files (e.g., `README.md`, issues, wikis).

\*   \*\*How:\*\* Simply copy the code block above (including the ` ```mermaid ` and ` ``` ` backticks) and paste it into any Markdown file on GitHub.

\*   \*\*Result:\*\* GitHub will automatically render it as a diagram when you view the file. This is ideal for your repository documentation, as the diagram will stay version-controlled alongside your code.



\### 2. Notion (Native Support)

Notion also has built-in support for Mermaid.

\*   \*\*How:\*\* In a Notion page, type `/mermaid` and select the "Mermaid" block. Paste the code (excluding the backticks) into the code section of the block.

\*   \*\*Result:\*\* Notion renders the diagram immediately.



\### 3. Dedicated Mermaid Editors (For Exporting)

If you need to export the schema as a high-quality image (PNG, SVG) for a presentation or a static document:

\*   \*\*Mermaid Live Editor:\*\* Go to \[https://mermaid.live/](https://mermaid.live/). Paste the code into the "Code" panel on the left. You can tweak the layout and then use the "Actions" menu to download it as a PNG or SVG.

\*   \*\*Draw.io / diagrams.net:\*\* Open Draw.io, go to `Arrange` -> `Insert` -> `Advanced` -> `Mermaid...`, paste the code, and click insert.



\## Analytical Breakdown of the Schema



This schema strictly reflects your current Python backend and operational constraints:



1\.  \*\*The Secular Anchor:\*\* The architecture clearly delineates the heavy, secular ERA5 baselining (handled via Zarr/NetCDF) from the agile, daily operational ingestion (IFS/AIFS via Azure mirrors). The ETCCDI 365-day calendar mapping is explicitly noted, as it's a critical methodological step.

2\.  \*\*The Live Engine \& The QDM Fork:\*\* The diagram captures the critical divergence in your methodology. This is the core engine of your physical consistency.

&#x20;   \*   \*\*IFS Surface variables\*\* require Quantile Delta Mapping (QDM) against hindcasts to neutralize physical model biases and orographic lapse-rate discrepancies. (Highlighted as a critical step).

&#x20;   \*   \*\*AIFS (natively trained on ERA5) and large-scale synoptics\*\* only require spatial regridding via CDO/SciPy conservative matrices.

3\.  \*\*The Latency Bypass:\*\* The schema visualizes how you avoid frontend latency: the complex math (CDO remapping, QDM, area-weighted masking) happens in scheduled background tasks, writing out lightweight Parquet binaries. Streamlit only ever reads these pre-computed states, bypassing NetCDF I/O overhead.

4\.  \*\*Frontend Delivery:\*\* The frontend block separates standard meteorological tracking from the expert synoptic tracking (Z500, U300, UTCI).



This dynamic mapping ensures that as you evolve AtmoPulse (e.g., adding ensemble forecasting), you simply add a node to the text code, rather than redrawing a static image.

