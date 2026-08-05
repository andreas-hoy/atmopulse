import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.ticker import FixedLocator
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
file_path = 'P90-HWs 9 stations_3.xlsm'

color_first = '#C0392B'   # Red
color_last = '#E67E22'    # Orange
color_inter = '#D5DBDB'   # Light Grey
color_2026 = '#9100EA'    # Purple for 2026

ticks_doy = [121, 152, 182, 213, 244, 274]
labels_doy = ['May 1', 'June 1', 'July 1', 'Aug 1', 'Sept 1', 'Oct 1']

definition_text = (
    "Methodological Definition:\n"
    "• Triggered: Max. temperature > local JJA P90 threshold (Ref. 1961–1990) for 3 consecutive days.\n"
    "• Termination: Average max. temperature drops below the P90 threshold OR single day below the P75 treshold.\n"
    "• Plotting: Earliest onset (Red), Latest termination (Orange), Midpoint of intermediate events (Grey). Bubble Size: Proportional to cumulative intensity (K × days)."
)

stations_reversed = ['Madrid', 'Geneva', 'Paris', 'Vienna', 'Prague', 'Frankfurt-Main', 'Oxford', 'Stockholm', 'Helsinki']
transect_reversed = ['Madrid', 'Frankfurt-Main', 'Helsinki']

plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. RIGOROUS DATA EXTRACTION ---
xls = pd.ExcelFile(file_path)
compiled_data = {}

for station in xls.sheet_names:
    df_st = pd.read_excel(file_path, sheet_name=station, header=None)
    
    valid_cols = []
    for c in range(df_st.shape[1]):
        val = df_st.iloc[0, c]
        try:
            year = int(float(val))
            if 1850 <= year <= 2026:
                valid_cols.append((year, c))
        except: continue
                
    all_hws = []
    for year, col_idx in valid_cols:
        year_events = []
        # Ranks 0-9 map to lines 375-384, etc. (0-indexed: 374-383)
        for rank in range(10):
            t_sum_val = df_st.iloc[374 + rank, col_idx]
            start_val = df_st.iloc[397 + rank, col_idx]
            end_val = df_st.iloc[408 + rank, col_idx]
            
            if pd.notna(start_val) and pd.notna(end_val) and pd.notna(t_sum_val) and t_sum_val > 0:
                s_doy = start_val.timetuple().tm_yday if hasattr(start_val, 'timetuple') else np.nan
                e_doy = end_val.timetuple().tm_yday if hasattr(end_val, 'timetuple') else np.nan
                
                if pd.notna(s_doy) and pd.notna(e_doy):
                    year_events.append({
                        'year': year, 's_doy': s_doy, 'e_doy': e_doy,
                        'mid_doy': s_doy + (e_doy - s_doy) / 2, 
                        't_sum': float(t_sum_val), 'id': rank,
                        'start_date_str': start_val.strftime('%d.%m.'),
                        'end_date_str': end_val.strftime('%d.%m.')
                    })
        
        if year_events:
            earliest = min(year_events, key=lambda x: x['s_doy'])
            latest = max(year_events, key=lambda x: x['e_doy'])
            
            for ev in year_events:
                is_2026 = (ev['year'] == 2026)
                
                # Assign categories
                if ev['id'] == earliest['id']:
                    cat = 'first'
                    plot_doy = ev['s_doy']
                elif ev['id'] == latest['id'] and earliest['id'] != latest['id']:
                    cat = 'last'
                    # SPECIAL 2026 RULE: All non-first 2026 events centered on midpoint
                    plot_doy = ev['mid_doy'] if is_2026 else ev['e_doy']
                else:
                    cat = 'inter'
                    plot_doy = ev['mid_doy']
                
                all_hws.append({**ev, 'cat': cat, 'plot_doy': plot_doy, 'is_2026': is_2026})
            
    compiled_data[station] = pd.DataFrame(all_hws)

# --- HELPER FUNCTIONS ---
def apply_styling(ax, title):
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlim(1845, 2032) # Extended x-axis so 2026 bubbles aren't clipped
    ax.set_ylim(121, 274)
    ax.set_yticks(ticks_doy)
    ax.set_yticklabels(labels_doy, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax.set_xlabel('Observation Year', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Calendar Date', fontsize=12, fontweight='bold', labelpad=10)

def get_legend_handles():
    return [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_inter, markersize=8, alpha=0.5, label='Intermediate HWs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_first, markersize=8, alpha=0.75, label='First HW (Onset)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_last, markersize=8, alpha=0.75, label='Last HW (Termination)'),
        Line2D([0], [0], color=color_first, lw=2.5, label='Trend: Earliest onset'),
        Line2D([0], [0], color=color_last, lw=2.5, label='Trend: Latest termination')
    ]

def add_header_info(ax, df):
    most_intense = df.loc[df['t_sum'].idxmax()]
    info_text = f"Most intense HW: {int(most_intense['year'])} (Intensity: {most_intense['t_sum']:.1f} K)\n"
    
    if 'first' in df['cat'].values:
        first_ever = df[df['cat'] == 'first'].loc[df[df['cat'] == 'first']['s_doy'].idxmin()]
        info_text += f"Earliest HW onset: {first_ever['start_date_str']} ({int(first_ever['year'])})\n"
        
    if 'last' in df['cat'].values:
        last_ever = df[df['cat'] == 'last'].loc[df[df['cat'] == 'last']['e_doy'].idxmax()]
        info_text += f"Latest HW termination: {last_ever['end_date_str']} ({int(last_ever['year'])})"
        
    # More transparent background (alpha=0.65)
    ax.text(0.02, 0.04, info_text, transform=ax.transAxes, fontsize=8, verticalalignment='bottom', 
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.65, edgecolor='#cccccc'), zorder=10)

def plot_station(ax, df, title_name):
    if df.empty: return
        
    df_first = df[df['cat'] == 'first']
    df_last = df[df['cat'] == 'last']
    df_inter = df[df['cat'] == 'inter']
    
    # Plot Grey Background Events (More transparent: 0.5)
    if not df_inter.empty:
        df_inter_2026 = df_inter[df_inter['is_2026']]
        df_inter_past = df_inter[~df_inter['is_2026']]
        ax.scatter(df_inter_past['year'], df_inter_past['plot_doy'], s=df_inter_past['t_sum']*8, color=color_inter, alpha=0.5, edgecolors='none', zorder=2)
        if not df_inter_2026.empty:
            ax.scatter(df_inter_2026['year'], df_inter_2026['plot_doy'], s=df_inter_2026['t_sum']*8, color=color_2026, alpha=0.85, edgecolors='none', zorder=5)
            
    # Plot First
    if not df_first.empty:
        df_first_2026 = df_first[df_first['is_2026']]
        df_first_past = df_first[~df_first['is_2026']]
        ax.scatter(df_first_past['year'], df_first_past['plot_doy'], s=df_first_past['t_sum']*8, color=color_first, alpha=0.75, edgecolors='w', linewidths=0.5, zorder=3)
        if not df_first_2026.empty:
            ax.scatter(df_first_2026['year'], df_first_2026['plot_doy'], s=df_first_2026['t_sum']*8, color=color_2026, alpha=0.85, edgecolors='none', zorder=6)

    # Plot Last
    if not df_last.empty:
        df_last_2026 = df_last[df_last['is_2026']]
        df_last_past = df_last[~df_last['is_2026']]
        ax.scatter(df_last_past['year'], df_last_past['plot_doy'], s=df_last_past['t_sum']*8, color=color_last, alpha=0.75, edgecolors='w', linewidths=0.5, zorder=4)
        if not df_last_2026.empty:
             ax.scatter(df_last_2026['year'], df_last_2026['plot_doy'], s=df_last_2026['t_sum']*8, color=color_2026, alpha=0.85, edgecolors='none', zorder=7)

    # Trendlines
    if len(df_first) > 5:
        l_f = lowess(df_first['plot_doy'], df_first['year'], frac=0.45)
        ax.plot(l_f[:, 0], l_f[:, 1], color=color_first, linewidth=2.5, zorder=8)
    if len(df_last) > 5:
        l_l = lowess(df_last['plot_doy'], df_last['year'], frac=0.45)
        ax.plot(l_l[:, 0], l_l[:, 1], color=color_last, linewidth=2.5, zorder=9)
        
    apply_styling(ax, title_name)
    add_header_info(ax, df)

# --- 3. GENERATE OUTPUTS ---

# Option A: PDF Carousel
print("Generating Carousel PDF...")
with PdfPages("1_Heatwave_Seasonality_Carousel_Final.pdf") as pdf:
    for station in stations_reversed:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        fig.subplots_adjust(bottom=0.20)
        df = compiled_data.get(station, pd.DataFrame())
        plot_station(ax, df, f"Evolution of heatwaves in {station.replace('-Main', '')}")
        
        ax.legend(handles=get_legend_handles(), loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=True, fontsize=9.5, facecolor='w', edgecolor='#dddddd')
        fig.text(0.5, 0.02, definition_text, ha='center', va='bottom', fontsize=8, color='#333333', bbox=dict(facecolor='#f9f9f9', edgecolor='#dddddd', boxstyle='round,pad=0.5'))
        
        pdf.savefig(fig)
        plt.close(fig)

# Option B: 3x1 Vertical Grid
print("Generating 3-Panel Grid...")
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 18), dpi=300, sharex=True)
fig2.subplots_adjust(bottom=0.10, hspace=0.15, top=0.95)

for idx, station in enumerate(transect_reversed):
    ax = axes2[idx]
    df = compiled_data.get(station, pd.DataFrame())
    plot_station(ax, df, f"{station.replace('-Main', '')}") 
    if idx < 2: ax.set_xlabel('')

axes2[-1].xaxis.set_major_locator(FixedLocator(axes2[-1].get_xticks()))
axes2[-1].set_xticklabels(axes2[-1].get_xticks().astype(int), fontsize=11)
fig2.legend(handles=get_legend_handles(), loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=5, frameon=True, fontsize=10.5, facecolor='w', edgecolor='#dddddd')
fig2.suptitle('Latitudinal Shift of Heatwave Seasonality (South to North)', fontsize=18, fontweight='bold', y=0.98)
fig2.text(0.5, 0.01, definition_text, ha='center', va='bottom', fontsize=9.5, color='#333333', bbox=dict(facecolor='#f9f9f9', edgecolor='#dddddd', boxstyle='round,pad=0.5'))

plt.savefig('2_Heatwave_Seasonality_3_Stations_Transect.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

# Option C: 9-Panel Grid (3x3)
print("Generating 9-Panel Grid...")
fig3, axes3 = plt.subplots(3, 3, figsize=(26, 18), dpi=300, sharex=True, sharey=True)
fig3.subplots_adjust(bottom=0.10, hspace=0.18, wspace=0.1, top=0.90)
axes3_flat = axes3.flatten()

for idx, station in enumerate(stations_reversed):
    ax = axes3_flat[idx]
    df = compiled_data.get(station, pd.DataFrame())
    plot_station(ax, df, f"{station.replace('-Main', '')}")
    if idx < 6: ax.set_xlabel('')
    if idx % 3 != 0: ax.set_ylabel('')

for ax in axes3[2, :]:
    ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
    ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=11)

fig3.legend(handles=get_legend_handles(), loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=5, frameon=True, fontsize=12, facecolor='w', edgecolor='#dddddd')
fig3.suptitle('Evolution of European Heatwaves: Shift in Seasonality and Intensity (1850–2026)', fontsize=26, fontweight='bold', y=0.96)
fig3.text(0.5, 0.01, definition_text, ha='center', va='bottom', fontsize=10.5, color='#333333', bbox=dict(facecolor='#f9f9f9', edgecolor='#dddddd', boxstyle='round,pad=0.5'))

plt.savefig('3_Heatwave_Seasonality_9_Stations_Panel.png', dpi=300, bbox_inches='tight')
plt.close(fig3)

print("All figures successfully generated.")