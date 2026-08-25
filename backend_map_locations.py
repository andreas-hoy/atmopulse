"""
AtmoPulse Geospatial Location & Top-10 Country Indexing (backend_map_locations.py)

This module handles the spatial hashing and intersection logic required for 
the AtmoPulse synoptic map hovers and the automated Top-10 impact tables. 
It ingests Natural Earth vector data, constructs highly efficient STRtree 
spatial indexes, and maps ERA5/IFS grid cells to specific European countries 
and marine regions.

Core functionalities:
- Pre-computes area-weighted fractional overlaps between meteorological grid 
  cells and national borders to ensure accurate Top-10 impact rankings.
- Excludes non-European nations and micro-territories from impact tables 
  based on a strict 3000 km² threshold and canonical naming aliases.
- Provides point-to-polygon lookups for interactive map hover labels 
  (e.g., identifying if a coordinate lies over a specific country or the Open Ocean).
"""
from __future__ import annotations

import requests
import numpy as np
from shapely.geometry import box, shape
from shapely.strtree import STRtree
from pyproj import Geod

EUROPE_BBOX = (-25.0, 30.0, 45.0, 72.0)
MIN_TOP10_LAND_AREA_KM2 = 3000
_GEOD = Geod(ellps="WGS84")

# Top-10 tables: exclude micro-territories and non-European overlap (by name).
_EXCLUDED_TOP10_NAMES = frozenset({
    "monaco", "guernsey", "jersey", "isle of man", "vatican", "vatican city",
    "holy see", "san marino", "andorra", "liechtenstein", "gibraltar",
    "faroe islands", "faeroe is.", "faeroe is", "åland", "sark", "malta",
    "akrotiri and dhekelia", "saint helena", "saint pierre and miquelon",
    # Non-European countries / territories appearing in the map bbox
    "morocco", "algeria", "tunisia", "libya", "egypt", "mauritania",
    "western sahara", "greenland", "syria", "lebanon", "israel",
    "palestine", "jordan", "iraq", "iran", "georgia", "armenia",
    "azerbaijan", "kazakhstan", "turkmenistan", "uzbekistan",
    "saudi arabia", "yemen", "sudan", "chad", "niger", "mali",
    "senegal", "gambia", "guinea-bissau", "guinea", "sierra leone",
    "liberia", "ivory coast", "côte d'ivoire", "ghana", "togo", "benin",
    "burkina faso", "nigeria", "cameroon", "equatorial guinea", "gabon",
})

_COUNTRY_TOP10_ALIASES = {
    "russia": "Russia (West)",
    "turkey": "Turkey",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "cyprus": "Cyprus",
    "northern cyprus": "Cyprus",
    "republic of cyprus": "Cyprus",
    "czechia": "Czech Republic",
    "north macedonia": "North Macedonia",
    "macedonia": "North Macedonia",
    "bosnia and herz.": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
}

COUNTRIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_50m_admin_0_countries.geojson"
)
MARINE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_50m_geography_marine_polys.geojson"
)

_LOCATION_INDEX: dict | None = None


def _geodesic_area_km2(geom) -> float:
    """Calculate the precise geodesic area of a geometry in square kilometers."""
    if geom.is_empty:
        return 0.0
    if geom.geom_type == "Polygon":
        area, _ = _GEOD.geometry_area_perimeter(geom)
        return abs(area) / 1e6
    if geom.geom_type == "MultiPolygon":
        return sum(_geodesic_area_km2(g) for g in geom.geoms)
    return 0.0


def _country_lookup_key(name: str) -> str:
    """Normalize country names for robust dictionary lookups."""
    return name.strip().lower().rstrip(".")


def _feature_name(props: dict, *keys: str) -> str | None:
    """Extract the primary feature name from GeoJSON properties using fallback keys."""
    for key in keys:
        val = props.get(key)
        if val:
            return str(val)
    return None


def _load_geojson(url: str) -> list[dict]:
    """Fetch and parse GeoJSON feature collections from a given URL."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["features"]


def _prepare_features(features: list[dict], name_keys: tuple[str, ...]) -> tuple[list[str], list, STRtree]:
    """
    Parse geometries, filter by the European bounding box, and build an 
    STRtree spatial index for highly efficient intersection queries.
    """
    names, geoms = [], []
    for feat in features:
        name = _feature_name(feat.get("properties", {}), *name_keys)
        if not name:
            continue
        try:
            geom = shape(feat["geometry"])
        except Exception:
            continue
        if geom.is_empty:
            continue
        
        # Fast bounding-box exclusion before adding to the spatial tree
        minx, miny, maxx, maxy = geom.bounds
        if maxx < EUROPE_BBOX[0] or minx > EUROPE_BBOX[2] or maxy < EUROPE_BBOX[1] or miny > EUROPE_BBOX[3]:
            continue
            
        names.append(name)
        geoms.append(geom)
        
    return names, geoms, STRtree(geoms)


def _get_location_index() -> dict:
    """
    Singleton spatial index initializer. Fetches global vector polygons, calculates 
    areas, and builds queryable trees for both terrestrial and marine regions.
    """
    global _LOCATION_INDEX
    if _LOCATION_INDEX is not None:
        return _LOCATION_INDEX

    raw_features = _load_geojson(COUNTRIES_URL)
    country_areas: dict[str, float] = {}
    
    for feat in raw_features:
        name = _feature_name(feat.get("properties", {}), "NAME", "ADMIN", "name")
        if not name:
            continue
        try:
            geom = shape(feat["geometry"])
        except Exception:
            continue
        if not geom.is_empty:
            country_areas[name] = _geodesic_area_km2(geom)

    country_names, country_geoms, country_tree = _prepare_features(
        raw_features, ("NAME", "ADMIN", "name")
    )
    sea_names, sea_geoms, sea_tree = _prepare_features(
        _load_geojson(MARINE_URL), ("NAME", "name")
    )
    
    _LOCATION_INDEX = {
        "country_names": country_names,
        "country_geoms": country_geoms,
        "country_tree": country_tree,
        "country_areas": country_areas,
        "sea_names": sea_names,
        "sea_geoms": sea_geoms,
        "sea_tree": sea_tree,
    }
    return _LOCATION_INDEX


def _hits_for_cell(cell, tree: STRtree, geoms: list, names: list[str]) -> list[tuple[float, str]]:
    """Query the STRtree for geometries intersecting the given cell and return their overlap areas."""
    hits: list[tuple[float, str]] = []
    for idx in tree.query(cell, predicate="intersects"):
        geom = geoms[int(idx)]
        inter = cell.intersection(geom)
        if inter.is_empty:
            continue
        area = inter.area
        if area > 0:
            hits.append((area, names[int(idx)]))
    hits.sort(reverse=True)
    return hits


def _canonical_top10_country(raw_name: str, area_km2: float | None = None) -> str | None:
    """Normalize Natural Earth country names for AtmoPulse Top-10 tables; None = exclude."""
    name = raw_name.strip()
    lower = _country_lookup_key(name)
    if lower in _EXCLUDED_TOP10_NAMES:
        return None
    if "cyprus" in lower:
        canonical = "Cyprus"
    elif lower in _COUNTRY_TOP10_ALIASES:
        canonical = _COUNTRY_TOP10_ALIASES[lower]
    else:
        canonical = name

    if area_km2 is not None and area_km2 < MIN_TOP10_LAND_AREA_KM2:
        return None
    return canonical


def label_grid_cell(lon: float, lat: float, cell_size: float = 0.25) -> str:
    """Return country/sea label for one ERA5 grid cell (center lon/lat)."""
    idx = _get_location_index()
    half = cell_size / 2.0
    cell = box(lon - half, lat - half, lon + half, lat + half)
    cell_area = cell.area

    land_hits = _hits_for_cell(cell, idx["country_tree"], idx["country_geoms"], idx["country_names"])
    if land_hits:
        significant = [name for area, name in land_hits if area / cell_area >= 0.05]
        if not significant:
            significant = [land_hits[0][1]]
        return "/".join(significant)

    sea_hits = _hits_for_cell(cell, idx["sea_tree"], idx["sea_geoms"], idx["sea_names"])
    if sea_hits:
        return sea_hits[0][1]

    return "Open Ocean"


def build_location_label_grid(lons, lats, cell_size: float = 0.25) -> np.ndarray:
    """Precompute location labels for the full synoptic map grid."""
    _get_location_index()
    labels = np.empty((len(lats), len(lons)), dtype=object)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            labels[i, j] = label_grid_cell(float(lon), float(lat), cell_size)
    return labels


def build_country_weight_grid(lons, lats, cell_size: float = 0.25) -> tuple[dict, dict]:
    """Precompute per-country land-area fractions for every grid cell.

    Returns (weights, sizes):
      - weights: {country_name: 2D array shape (len(lats), len(lons))}, where
        each entry is the fraction (0..1) of that grid cell's area lying
        within the country's land area. Sea fraction and any other
        country's share of the same cell are excluded, so cells straddling
        a border or coastline only contribute proportionally to each
        country they actually cover.
      - sizes: {country_name: summed weight across the whole grid}, used only
        to break ties (larger countries first) when impact percentages are
        equal.
    """
    idx = _get_location_index()
    half = cell_size / 2.0
    shape = (len(lats), len(lons))
    weights: dict[str, np.ndarray] = {}
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            cell = box(float(lon) - half, float(lat) - half, float(lon) + half, float(lat) + half)
            cell_area = cell.area
            
            for area, name in _hits_for_cell(cell, idx["country_tree"], idx["country_geoms"], idx["country_names"]):
                if area <= 0:
                    continue
                area_km2 = idx["country_areas"].get(name)
                canonical = _canonical_top10_country(name, area_km2)
                
                if canonical is None:
                    continue
                if canonical not in weights:
                    weights[canonical] = np.zeros(shape, dtype=np.float32)
                    
                weights[canonical][i, j] += area / cell_area

    sizes = {name: float(w.sum()) for name, w in weights.items()}
    return weights, sizes