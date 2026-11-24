"""Central configuration for RegionalFlash CERRA 5.5 km forecasting dataset & pipeline."""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Hugging Face Target & Source ------------------------------------------
HF_REPO_ID = "meteolibre-dev/regionalflash"
SOURCE_HF_CERRA_REPO = "JonasZimmerHHI/cerra_2010_2022"

# --- CERRA Full European Grid (5.5 km Lambert Conformal Conic) -------------
GRID_NY = 1069
GRID_NX = 1069
RESOLUTION_KM = 5.5
STEP_HOURS = 3

# --- Training Patch & Tiled Diffusion Inference ----------------------------
PATCH_SIZE = 128          # 128 x 128 px @ 5.5 km = 704 km x 704 km
BOUNDARY_HALO = 16        # Outer rim (16 px = 88 km) used for boundary/tile conditioning
TILE_STRIDE = 96          # Overlapping stride for tiled diffusion inference (32 px overlap)

# --- Temporal Splits (4 years train + 1 year val/test, 2018-2022 subset) ---
DATE_START = "2018-01-01"
DATE_END = "2022-12-31"
TRAIN_YEARS = (2018, 2019, 2020, 2021)
VAL_START = "2022-01-01"
VAL_END = "2022-06-30"
TEST_START = "2022-07-01"
TEST_END = "2022-12-31"

# --- Zarr v3 Storage Layout (Full Grid, Sharded for HF + Fast 128x128 IO) --
# Inner chunks align 1-to-1 with 128x128 patches; shards group 16 timesteps x 4x4 chunks
# so each half-year (~1,460 steps) fits in ~830 shard files (~150 GB).
ZARR_DTYPE = "float16"
ZARR_CHUNKS_TYX = (1, 128, 128)
ZARR_SHARDS_TYX = (16, 512, 512)

# --- CERRA Variables (Real-time compatible with ICON-EU / ECMWF Open Data) -
PRESSURE_LEVELS_HPA = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 100]

# Upper-air 3D variables from `reanalysis-cerra-pressure-levels` (analysis)
# Tuple: (cds_variable_name, canonical_short_name)
CERRA_PL_VARS = [
    ("temperature", "t"),
    ("u_component_of_wind", "u"),
    ("v_component_of_wind", "v"),
    ("relative_humidity", "r"),
    ("geopotential", "z"),
]

# Surface / single-level variables from `reanalysis-cerra-single-levels` (analysis)
# Note: CERRA provides 10m wind as speed + direction, which we convert to (u10, v10).
CERRA_SL_ANALYSIS_VARS = [
    "2m_temperature",
    "2m_relative_humidity",
    "10m_wind_speed",
    "10m_wind_direction",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_cloud_cover",
]

# Static invariant variables (fetched once from CERRA single-levels analysis)
CERRA_STATIC_VARS = [
    "orography",
    "land_sea_mask",
]

# Canonical ordered dynamic channel list (each pixel = 57-channel atmospheric column, analysis-only).
# Units are chosen so `float16` has high precision and zero overflow (max float16 = 65504):
#   - t_* , t2m            : degC (K - 273.15)
#   - u_* , v_* , u10, v10 : m/s
#   - r_* , r2 , tcc       : % (0..100)
#   - z_*                  : geopotential height in dam (m^2/s^2 / 98.0665)
#   - msl_anom_hpa         : hPa - 1013.25
#   - sp_anom_hpa          : hPa - 1000.0
# Precipitation will be supplied later via radar observations on the same (T, 1069, 1069) grid.
DYN_CHANNELS: list[str] = (
    [f"{short}_{lvl}" for _, short in CERRA_PL_VARS for lvl in PRESSURE_LEVELS_HPA]
    + [
        "t2m",
        "r2",
        "u10",
        "v10",
        "msl_anom_hpa",
        "sp_anom_hpa",
        "tcc",
    ]
)

STATIC_CHANNELS: list[str] = [
    "orography",
    "land_sea_mask",
    "sin_lat",
    "cos_lat",
    "sin_lon",
    "cos_lon",
]

N_DYN_CHANNELS = len(DYN_CHANNELS)       # 57
N_STATIC_CHANNELS = len(STATIC_CHANNELS) # 6
