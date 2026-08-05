import xarray as xr
import regionmask
import pandas as pd
import numpy as np
import time
from pathlib import Path

def aggregate_by_country(persistence_file: str, output_csv: str):
    print("=== Starte Räumliche Aggregation (Länder-Zuschnitt) ===")
    start_time = time.time()
    
    # 1. Daten laden
    print("Lade Persistenzdaten...")
    ds = xr.open_dataset(persistence_file)
    
    # 2. Länder-Maske für das ERA5-Gitter erstellen
    print("Erstelle Länder-Masken (Natural Earth)...")
    # Zieht automatisch die Ländergrenzen und passt sie an unser Koordinatensystem an
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    mask = countries.mask(ds.longitude, ds.latitude)
    
    # Namen der Länder extrahieren, die in unserer Europa-Box liegen
    region_ids = np.unique(mask.values)
    region_ids = region_ids[~np.isnan(region_ids)] # Wasserflächen/NaN entfernen
    
    country_mapping = {int(idx): countries[int(idx)].name for idx in region_ids}
    print(f"{len(country_mapping)} europäische Staaten/Regionen in der Bounding Box gefunden.")
    
    # 3. Aggregation durchführen
    print("Berechne prozentualen Flächenanteil pro Land und Tag...")
    
    # Ein Event ist aktiv, wenn die Persistenz > 0 ist
    is_active = ds > 0
    
    # Gruppieren nach Land und Mittelwert berechnen. 
    # Da True=1 und False=0, entspricht der Mittelwert genau dem Flächenanteil.
    country_means = is_active.groupby(mask).mean() * 100
    
    # 4. In eine tabellarische Form (Pandas DataFrame) umwandeln für Streamlit
    print("Formatiere Daten für das Dashboard...")
    df_list = []
    
    for var_name in ds.data_vars:
        # Daten in Tabelle umwandeln
        df = country_means[var_name].to_dataframe().reset_index()
        
        # Jedes xarray-to-dataframe mit groupby hat die IDs in der allerersten Spalte.
        # Wir greifen uns die Spalte einfach über ihre Position (Index 0 = Zeit, Index 1 = Regions-ID)
        region_col_name = df.columns[1] 
        
        # Zuweisung der Ländernamen über die dynamisch ermittelte Spalte
        df['country'] = df[region_col_name].map(country_mapping)
        df['variable'] = var_name
        
        # Spalte umbenennen für die finale Tabelle
        df = df.rename(columns={var_name: 'area_percentage'})
        
        # Nur relevante Spalten behalten
        df = df[['valid_time', 'country', 'variable', 'area_percentage']]
        df_list.append(df)
        
    final_df = pd.concat(df_list, ignore_index=True)
    
    # Ozean / nicht zugeordnete Flächen entfernen
    final_df = final_df.dropna(subset=['country'])
    
    # 5. Speichern
    print(f"Speichere aggregierte Daten unter: {output_csv}")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    
    print(f"=== FERTIG in {(time.time() - start_time):.2f} Sekunden ===")
    return final_df

if __name__ == "__main__":
    # Input: Unsere sauberen Persistenzdaten von vorhin
    pers_file = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool/Persistence_Data/persistence_2026-01_test.nc"
    
    # Output: Eine klassische CSV-Tabelle
    out_csv = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool/Aggregated_Stats/country_stats_2026-01.csv"
    
    aggregate_by_country(pers_file, out_csv)