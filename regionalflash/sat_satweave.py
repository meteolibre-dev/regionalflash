"""Download Google DeepMind SatWeave (0.1 deg, 11 European Meteosat bands) at `t + 2h`
relative to CERRA, reproject to the exact 5.5 km CERRA (1069x1069) grid, write half-year
sharded Zarr v3 stores (`sat_5km_{YEAR}_H1/H2_plus2h.zarr`), and upload to Hugging Face.

Alignment:
  For each CERRA analysis timestamp `t` (00, 03, ..., 21 UTC), this fetches SatWeave at
  `t_sensor = t + 2h` (02, 05, ..., 23 UTC), matching OPERA radar & METAR (+2h) and
  real-time inference where NWP assimilation is ~2h delayed while satellite is live.

Channels (12 total, float16):
  Reflectance (%, clipped [0, 120], 0 at night / outside FOV):
    0: mosaic_640_nm    (VIS 0.6 um)
    1: mosaic_863_nm    (NIR 0.8 um)
    2: mosaic_1610_nm   (SWIR 1.6 um)
  Brightness Temperature (degC = K - 273.15, 0 where outside FOV):
    3: mosaic_3830_nm   (IR 3.9 um)
    4: mosaic_6200_nm   (WV 6.2 um)
    5: mosaic_7330_nm   (WV 7.3 um)
    6: mosaic_8590_nm   (IR 8.7 um)
    7: mosaic_9620_nm   (IR 9.7 um)
    8: mosaic_11200_nm  (IR 10.8 um)
    9: mosaic_12400_nm  (IR 12.0 um)
   10: mosaic_13300_nm  (IR 13.4 um)
  Validity mask:
   11: sat_valid_mask   (1.0 where IR 11.2 um is finite, 0.0 beyond geostationary horizon)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import shutil
import time

from huggingface_hub import HfApi
import numcodecs
import numpy as np
import pandas as pd
import requests
from scipy.ndimage import map_coordinates
import zarr
from zarr.codecs import BloscCodec

from . import config
from .cerra import HALF_MONTHS

log = logging.getLogger("regionalflash.sat_satweave")

SATWEAVE_BASE = "https://storage.googleapis.com/satweave_1/zarr/2016_to_2025/mosaic.zarr"
SATWEAVE_EPOCH = pd.Timestamp("2016-01-01 00:00:00", tz="UTC")
SENSOR_OFFSET_HOURS = 2

REFLECTANCE_BANDS = [
    "mosaic_640_nm",
    "mosaic_863_nm",
    "mosaic_1610_nm",
]
IR_BANDS = [
    "mosaic_3830_nm",
    "mosaic_6200_nm",
    "mosaic_7330_nm",
    "mosaic_8590_nm",
    "mosaic_9620_nm",
    "mosaic_11200_nm",
    "mosaic_12400_nm",
    "mosaic_13300_nm",
]
SAT_BANDS = REFLECTANCE_BANDS + IR_BANDS
SAT_CHANNELS = SAT_BANDS + ["sat_valid_mask"]
N_SAT_CHANNELS = len(SAT_CHANNELS)  # 12

_ZSTD = numcodecs.Zstd()


def _make_session(pool_size: int = 32) -> requests.Session:
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("https://", adapter)
    return s


def _build_satweave_to_cerra_coords(static_zarr: Path) -> np.ndarray:
    """Precompute exact (2, 1069, 1069) floating-point (row, col) lookup into SatWeave 0.1 deg grid.

    SatWeave grid: lat = 90.0 - row * 0.1 (1801 pts), lon = col * 0.1 (3600 pts, 0..359.9).
    We pad 1 extra longitude column at 360.0 (copy of 0.0) when sampling so Prime Meridian
    interpolation across lon=359.9..0.0 is seamless.
    """
    s = zarr.open_group(str(static_zarr), mode="r")
    lat = np.asarray(s["latitude"], dtype=np.float64)
    lon = np.asarray(s["longitude"], dtype=np.float64)
    lon_360 = lon % 360.0

    rows = ((90.0 - lat) * 10.0).astype(np.float32)
    cols = (lon_360 * 10.0).astype(np.float32)
    return np.stack([rows, cols], axis=0)


def _fetch_band_chunk(session: requests.Session, band: str, t_idx: int) -> np.ndarray | None:
    url = f"{SATWEAVE_BASE}/{band}/c/{t_idx}/0/0"
    for attempt in range(1, 6):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                buf = _ZSTD.decode(r.content)
                return np.frombuffer(buf, dtype="<f4").reshape(1801, 3600)
            if r.status_code == 404:
                return None
        except Exception as e:
            time.sleep(min(30, 3 * attempt))
            if attempt == 5:
                log.warning("SatWeave fetch failed for %s t=%d: %s", band, t_idx, e)
    return None


def _warp_band(grid_2d: np.ndarray, coords: np.ndarray, order: int = 1) -> np.ndarray:
    # Pad 1 column at lon=360.0 (= col 0) for seamless Prime Meridian bilinear interpolation
    padded = np.concatenate([grid_2d, grid_2d[:, :1]], axis=1)
    return map_coordinates(padded, coords, order=order, mode="nearest")


def _process_step(
    session: requests.Session,
    ts_sensor: pd.Timestamp,
    coords: np.ndarray,
) -> np.ndarray:
    """Fetch all 11 European SatWeave bands at `ts_sensor` and warp -> (1069, 1069, 12) float16."""
    t_idx = int((ts_sensor - SATWEAVE_EPOCH).total_seconds() // 1800)
    out = np.zeros((config.GRID_NY, config.GRID_NX, N_SAT_CHANNELS), dtype=np.float16)

    # Fetch IR 11.2 um first to determine geostationary FOV validity mask
    ir11 = _fetch_band_chunk(session, "mosaic_11200_nm", t_idx)
    if ir11 is None:
        return out

    valid_global = np.isfinite(ir11).astype(np.float32)
    valid_5km = _warp_band(valid_global, coords, order=0) > 0.5
    out[..., 11] = valid_5km.astype(np.float16)

    for bi, band in enumerate(SAT_BANDS):
        raw = ir11 if band == "mosaic_11200_nm" else _fetch_band_chunk(session, band, t_idx)
        if raw is None:
            continue

        is_refl = band in REFLECTANCE_BANDS
        if is_refl:
            clean = np.where(np.isfinite(raw), np.clip(raw, 0.0, 120.0), 0.0).astype(np.float32)
            warped = _warp_band(clean, coords, order=1)
            warped = np.where(valid_5km, np.clip(warped, 0.0, 120.0), 0.0)
        else:
            # Fill NaNs with local valid mean before bilinear warp to avoid horizon bleed, then mask
            fin = np.isfinite(raw)
            fill_v = float(raw[fin].mean()) if fin.any() else 260.0
            clean = np.where(fin, raw - 273.15, fill_v - 273.15).astype(np.float32)
            warped = _warp_band(clean, coords, order=1)
            warped = np.where(valid_5km, np.clip(warped, -120.0, 80.0), 0.0)

        out[..., bi] = warped.astype(np.float16)

    return out


def _init_or_open_sat_zarr(out_zarr: Path, period: str, n_times: int, t_shard: int) -> zarr.Group:
    if out_zarr.exists():
        root = zarr.open_group(str(out_zarr), mode="r+")
        if root["X"].shape[0] != n_times:
            raise ValueError(f"Existing {out_zarr} has T={root['X'].shape[0]}, expected {n_times}")
        return root

    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    root = zarr.open_group(str(out_zarr), mode="w")
    root.create_array(
        "X",
        shape=(n_times, config.GRID_NY, config.GRID_NX, N_SAT_CHANNELS),
        chunks=(1, 128, 128, N_SAT_CHANNELS),
        shards=(t_shard, 512, 512, N_SAT_CHANNELS),
        dtype="float16",
        compressors=compressor,
        fill_value=0.0,
    )
    root.attrs["period"] = period
    root.attrs["channels"] = SAT_CHANNELS
    root.attrs["sensor_offset_hours"] = SENSOR_OFFSET_HOURS
    root.attrs["source"] = SATWEAVE_BASE
    root.attrs["resolution_km"] = config.RESOLUTION_KM
    root.attrs["completed_shards"] = []
    root.attrs["units"] = {
        "reflectance_bands": "% (0..120, 0 at night)",
        "ir_bands": "degC (K - 273.15, 0 outside FOV)",
        "sat_valid_mask": "binary 0/1",
    }
    return root


def build_sat_period(
    year: int,
    half: str,
    session: requests.Session,
    coords: np.ndarray,
    timestamps_path: Path,
    processed_dir: Path,
    workers: int = 8,
    t_shard: int = 32,
) -> Path:
    period = f"{year}_{half}"
    target_months = HALF_MONTHS[half]

    ts_df = pd.read_parquet(timestamps_path)
    ts_df["timestamp_utc"] = pd.to_datetime(ts_df["timestamp_utc"], utc=True)
    ts_per = (
        ts_df[(ts_df["year"] == year) & (ts_df["month"].isin(target_months))]
        .sort_values("time_idx")
        .reset_index(drop=True)
    )
    ts_sensor = ts_per["timestamp_utc"] + pd.Timedelta(hours=SENSOR_OFFSET_HOURS)
    n_times = len(ts_sensor)

    out_zarr = processed_dir / f"sat_5km_{period}_plus2h.zarr"
    root = _init_or_open_sat_zarr(out_zarr, period=period, n_times=n_times, t_shard=t_shard)
    X = root["X"]
    done_shards = set(root.attrs.get("completed_shards", []))
    n_shards = (n_times + t_shard - 1) // t_shard
    pending = [s for s in range(n_shards) if s not in done_shards]
    log.info("SatWeave %s (+2h): %d/%d shards (%d steps) pending (workers=%d)",
             period, len(pending), n_shards, n_times, workers)

    t0_wall = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, shard_id in enumerate(pending, 1):
            s0 = shard_id * t_shard
            s1 = min(s0 + t_shard, n_times)
            shard_ts = [ts_sensor.iloc[i] for i in range(s0, s1)]

            frames = list(ex.map(lambda t_s: _process_step(session, t_s, coords), shard_ts))
            X[s0:s1] = np.stack(frames, axis=0)

            done_shards.add(shard_id)
            root.attrs["completed_shards"] = sorted(done_shards)

            if k % 5 == 0 or k == len(pending):
                elapsed = time.time() - t0_wall
                rate = (k * t_shard) / max(elapsed, 1e-3)
                eta_min = ((len(pending) - k) * t_shard) / max(rate, 1e-3) / 60.0
                log.info(
                    "SatWeave %s: shard %d/%d done (%.2f steps/s, ETA %.1f min)",
                    period, len(done_shards), n_shards, rate, eta_min,
                )

    return out_zarr


def run_all_sat(
    years: list[int],
    timestamps_path: Path = config.PROCESSED_DIR / "timestamps.parquet",
    static_zarr: Path = config.PROCESSED_DIR / "static_5km.zarr",
    processed_dir: Path = config.PROCESSED_DIR,
    workers: int = 8,
    upload_and_clean: bool = True,
    repo_id: str = config.HF_REPO_ID,
) -> None:
    session = _make_session(pool_size=max(workers * 3, 32))
    coords = _build_satweave_to_cerra_coords(static_zarr)
    log.info("Precomputed SatWeave 0.1 deg -> CERRA 5.5 km (1069x1069) lookup grid")

    state_path = processed_dir / "uploaded_sat_periods.json"
    uploaded = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    api = HfApi()

    for year in years:
        for half in ("H1", "H2"):
            period = f"{year}_{half}"
            if upload_and_clean and period in uploaded:
                log.info("SatWeave %s already uploaded & cleaned, skipping.", period)
                continue

            out_zarr = build_sat_period(
                year=year,
                half=half,
                session=session,
                coords=coords,
                timestamps_path=timestamps_path,
                processed_dir=processed_dir,
                workers=workers,
            )

            if upload_and_clean:
                log.info("Uploading %s -> hf://datasets/%s ...", out_zarr.name, repo_id)
                api.upload_folder(
                    folder_path=str(processed_dir),
                    repo_id=repo_id,
                    repo_type="dataset",
                    allow_patterns=[f"{out_zarr.name}/**"],
                    commit_message=f"Upload SatWeave 5.5 km satellite (+2h, {period}) sharded Zarr v3",
                )
                uploaded.add(period)
                state_path.write_text(json.dumps(sorted(uploaded), indent=2))
                log.info("Uploaded & cleaning local %s", out_zarr.name)
                shutil.rmtree(out_zarr, ignore_errors=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build & upload Google SatWeave 5.5 km (+2h) Zarr v3")
    p.add_argument("--years", type=int, nargs="+", default=[2018, 2019, 2020, 2021, 2022])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--repo-id", default=config.HF_REPO_ID)
    args = p.parse_args()
    run_all_sat(years=args.years, workers=args.workers, upload_and_clean=not args.no_upload, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
