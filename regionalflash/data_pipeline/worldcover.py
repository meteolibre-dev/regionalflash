"""Build 5.5 km ESA WorldCover 2021 fractional land-cover + coast-distance Zarr on the CERRA grid.

Streams ESA WorldCover 2021 v200 COG coarse overviews directly from public AWS S3
(same source as `regionalnext/static_250m.py`) and area-averages them onto the exact
`1069 x 1069` CERRA Lambert grid (`static_5km.zarr`).

Channels (7, 1069, 1069) float16 — strictly INPUT-ONLY conditioning:
  0: frac_tree   (WorldCover 10: Tree cover)
  1: frac_grass  (WorldCover 20/30/90/95/100: Shrubland, Grassland, Wetland, Mangrove, Moss)
  2: frac_crop   (WorldCover 40: Cropland)
  3: frac_urban  (WorldCover 50: Built-up / Urban)
  4: frac_bare   (WorldCover 60/70: Bare/sparse vegetation, Snow/Ice)
  5: frac_water  (WorldCover 80: Permanent water bodies + ocean outside tiles)
  6: log_dist_coast (log1p(dist_coast_km) / 5.0 from CERRA land_sea_mask)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import requests
import zarr
from scipy.ndimage import distance_transform_edt

from regionalflash import config

log = logging.getLogger("regionalflash.worldcover")

LANDCOVER_CHANNELS = [
    "frac_tree",
    "frac_grass",
    "frac_crop",
    "frac_urban",
    "frac_bare",
    "frac_water",
    "log_dist_coast",
]

# WorldCover v200 integer codes -> 6 fractional buckets
CLASS_BUCKETS = {
    0: (10,),                     # tree
    1: (20, 30, 90, 95, 100),     # grass / shrub / wetland / moss
    2: (40,),                     # crop
    3: (50,),                     # urban / built-up
    4: (60, 70),                  # bare / snow & ice
    5: (80,),                     # permanent water
}


def _worldcover_url(lat_sw: int, lon_sw: int) -> str:
    ns = f"N{lat_sw:02d}" if lat_sw >= 0 else f"S{abs(lat_sw):02d}"
    ew = f"E{lon_sw:03d}" if lon_sw >= 0 else f"W{abs(lon_sw):03d}"
    return (
        f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
        f"ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map.tif"
    )


def build_landcover_5km(
    data_dir: Path = config.PROCESSED_DIR,
    out_zarr: Path | None = None,
    overview_px: int = 360,
) -> Path:
    """Build `landcover_5km.zarr` aligned to `static_5km.zarr` (`1069 x 1069`).

    Reads each 3x3 deg WorldCover COG at `overview_px x overview_px` (~1 km effective
    resolution, ~5x finer than CERRA 5.5 km) and accumulates exact fractional class
    counts into every CERRA grid cell via nearest-neighbor KDTree / direct projection.
    """
    import rasterio
    from scipy.spatial import cKDTree

    out_zarr = out_zarr or (data_dir / "landcover_5km.zarr")
    st = zarr.open_group(str(data_dir / "static_5km.zarr"), mode="r")
    lat = np.asarray(st["latitude"][:], dtype=np.float32)    # (1069, 1069)
    lon = np.asarray(st["longitude"][:], dtype=np.float32)   # (1069, 1069)
    static_arr = np.asarray(st["static"][:], dtype=np.float32)
    lsm = static_arr[1]  # land_sea_mask in [0, 1]
    ny, nx = lat.shape

    # Normalize lon to [-180, 180]
    lon_w = ((lon + 180.0) % 360.0) - 180.0
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon_w.min()), float(lon_w.max())
    log.info("CERRA domain: lat=[%.2f, %.2f], lon=[%.2f, %.2f]", lat_min, lat_max, lon_min, lon_max)

    # Build 3D unit-sphere KDTree over CERRA pixels for exact fast binning of ~1km WorldCover pixels
    lat_r = np.deg2rad(lat.ravel())
    lon_r = np.deg2rad(lon_w.ravel())
    pts_cerra = np.column_stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ]).astype(np.float32)
    tree = cKDTree(pts_cerra)

    # Candidate 3x3 deg tiles covering Europe
    lat_starts = range(int(np.floor(lat_min / 3.0) * 3), int(np.ceil(lat_max / 3.0) * 3), 3)
    lon_starts = range(int(np.floor(lon_min / 3.0) * 3), int(np.ceil(lon_max / 3.0) * 3), 3)

    session = requests.Session()
    valid_tiles: list[tuple[int, int, str]] = []
    for la in lat_starts:
        for lo in lon_starts:
            url = _worldcover_url(la, lo)
            try:
                r = session.head(url, timeout=5)
                if r.status_code == 200:
                    valid_tiles.append((la, lo, url))
            except requests.RequestException:
                pass
    log.info("Found %d ESA WorldCover 2021 land tiles over CERRA domain", len(valid_tiles))

    counts = np.zeros((6, ny * nx), dtype=np.float32)
    totals = np.zeros(ny * nx, dtype=np.float32)

    # Lookup table from WorldCover code (0..100) -> bucket (0..5, or -1 for nodata)
    lut = np.full(256, -1, dtype=np.int8)
    for b_idx, codes in CLASS_BUCKETS.items():
        for code in codes:
            lut[code] = b_idx

    # Max angular chord distance (~10 km on unit sphere) to ignore samples outside CERRA domain
    max_chord = 10.0 / 6371.0

    for idx, (la, lo, url) in enumerate(valid_tiles, 1):
        try:
            with rasterio.open(url) as src:
                lc = src.read(1, out_shape=(overview_px, overview_px), resampling=rasterio.enums.Resampling.nearest)
                b = src.bounds
        except Exception as exc:
            log.warning("Skipping tile (%d,%d): %s", la, lo, exc)
            continue

        # Pixel centers in lat/lon
        ys = np.linspace(b.top, b.bottom, overview_px, endpoint=False) + (b.bottom - b.top) / (2.0 * overview_px)
        xs = np.linspace(b.left, b.right, overview_px, endpoint=False) + (b.right - b.left) / (2.0 * overview_px)
        lon_g, lat_g = np.meshgrid(xs, ys)

        buckets = lut[lc.ravel()]
        valid = buckets >= 0
        if not np.any(valid):
            continue

        la_v = np.deg2rad(lat_g.ravel()[valid])
        lo_v = np.deg2rad(lon_g.ravel()[valid])
        pts = np.column_stack([
            np.cos(la_v) * np.cos(lo_v),
            np.cos(la_v) * np.sin(lo_v),
            np.sin(la_v),
        ]).astype(np.float32)

        dists, nn_idx = tree.query(pts, workers=-1)
        inside = dists <= max_chord
        if not np.any(inside):
            continue

        nn_in = nn_idx[inside]
        bk_in = buckets[valid][inside]
        np.add.at(totals, nn_in, 1.0)
        for b_i in range(6):
            sel = bk_in == b_i
            if np.any(sel):
                np.add.at(counts[b_i], nn_in[sel], 1.0)

        if idx % 25 == 0 or idx == len(valid_tiles):
            log.info("WorldCover progress: %d/%d tiles", idx, len(valid_tiles))

    fracs = np.zeros((6, ny * nx), dtype=np.float32)
    has_data = totals > 0
    fracs[:, has_data] = counts[:, has_data] / totals[None, has_data]
    # Ocean pixels with no WorldCover tile -> 100% water (bucket 5)
    fracs[5, ~has_data] = 1.0
    fracs = fracs.reshape(6, ny, nx)

    # Distance to coast (km) from CERRA land_sea_mask (5.5 km pixels), log-scaled to ~[0, 1.5]
    dist_coast_km = distance_transform_edt(lsm > 0.5).astype(np.float32) * 5.5
    log_dist_coast = (np.log1p(dist_coast_km) / 5.0).astype(np.float32)[None, :, :]

    out = np.concatenate([fracs, log_dist_coast], axis=0).astype(np.float16)  # (7, 1069, 1069)

    root = zarr.open_group(str(out_zarr), mode="w")
    arr = root.create_array(
        "landcover",
        shape=out.shape,
        chunks=(7, 256, 256),
        dtype="float16",
    )
    arr[:] = out
    root.attrs["channels"] = LANDCOVER_CHANNELS
    log.info("Saved %s shape=%s (%.1f MB)", out_zarr, out.shape, out.nbytes / 1e6)
    return out_zarr


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build 5.5 km ESA WorldCover landcover_5km.zarr on CERRA grid")
    p.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--overview-px", type=int, default=360)
    args = p.parse_args()
    build_landcover_5km(args.data_dir, args.out, overview_px=args.overview_px)


if __name__ == "__main__":
    main()
