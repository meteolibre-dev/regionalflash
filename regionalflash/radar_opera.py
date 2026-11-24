"""Download EUMETNET OPERA European radar archive (2018-2022) at `t + 2h` relative to CERRA,
reproject to the exact 5.5 km CERRA (1069x1069) grid, write sharded Zarr v3, and upload to HF.

Alignment:
  For each CERRA analysis timestamp `t` (00, 03, ..., 21 UTC), this fetches the live radar
  observation at `t_sensor = t + 2h` (02, 05, ..., 23 UTC), matching real-time inference where
  NWP assimilation is ~2h delayed while radar is live.

Output:
  `data/processed/radar_5km_plus2h.zarr`:
    - Array `X`: shape `(14608, 1069, 1069, 2)`, dtype `float16`
    - Channels: `["dbzh", "valid_mask"]`
      - `dbzh`: reflectivity in dBZ clipped to `[0, 75]` (`0.0` where clear-sky or outside coverage)
      - `valid_mask`: `1.0` where OPERA radar covers the pixel (`dbz != nodata`), `0.0` otherwise
    - Chunks `(1, 128, 128, 2)`, shards `(64, 512, 512, 2)`, Blosc-Zstd bitshuffle (~3-6 GB total).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import logging
from pathlib import Path
import time

import h5py
from huggingface_hub import HfApi
import numpy as np
import pandas as pd
from pyproj import Transformer
import requests
from scipy.ndimage import map_coordinates
import zarr
from zarr.codecs import BloscCodec

from . import config

log = logging.getLogger("regionalflash.radar_opera")

OPERA_ARCHIVE_BASE = "https://s3.waw3-1.cloudferro.com/openradar-archive"
SENSOR_OFFSET_HOURS = 2
RADAR_CHANNELS = ["dbzh", "valid_mask"]


def _make_session(pool_size: int = 24) -> requests.Session:
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("https://", adapter)
    return s


def _opera_url(ts: pd.Timestamp) -> str:
    y = ts.strftime("%Y")
    m = ts.strftime("%m")
    d = ts.strftime("%d")
    hhmm = ts.strftime("%H%M")
    return f"{OPERA_ARCHIVE_BASE}/{y}/{m}/{d}/OPERA/COMP/OPERA@{y}{m}{d}T{hhmm}@0@DBZH_QIND.h5"


def _fetch_opera_bytes(session: requests.Session, ts_sensor: pd.Timestamp) -> bytes | None:
    """Fetch OPERA DBZH_QIND.h5 at `ts_sensor` (with -15m / +15m fallback if exact slot missing)."""
    for delta_min in (0, -15, 15):
        t_try = ts_sensor + pd.Timedelta(minutes=delta_min)
        url = _opera_url(t_try)
        for attempt in range(1, 4):
            try:
                r = session.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 4096:
                    return r.content
                if r.status_code == 404:
                    break
            except Exception:
                time.sleep(2 * attempt)
    return None


def _build_opera_to_cerra_coords(
    sample_bytes: bytes,
    static_zarr: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute exact (2, 1069, 1069) floating-point (row, col) lookup into OPERA 2km grid."""
    s = zarr.open_group(str(static_zarr), mode="r")
    lat = np.asarray(s["latitude"], dtype=np.float64)
    lon = np.asarray(s["longitude"], dtype=np.float64)

    with h5py.File(io.BytesIO(sample_bytes), "r") as f:
        where = f["where"].attrs
        projdef = where["projdef"].decode() if isinstance(where["projdef"], bytes) else str(where["projdef"])
        xscale = float(where["xscale"])
        yscale = float(where["yscale"])
        ul_lat = float(where["UL_lat"])
        ul_lon = float(where["UL_lon"])

    tr = Transformer.from_crs("EPSG:4326", projdef, always_xy=True)
    ul_x, ul_y = tr.transform(ul_lon, ul_lat)
    gx, gy = tr.transform(lon, lat)
    cols = ((gx - ul_x) / xscale).astype(np.float32)
    rows = ((ul_y - gy) / yscale).astype(np.float32)
    return np.stack([rows, cols], axis=0), (projdef, xscale, yscale, ul_lat, ul_lon)


def _decode_and_reproject(raw_bytes: bytes | None, coords: np.ndarray) -> np.ndarray:
    """Decode OPERA HDF5 and warp onto (1069, 1069, 2) float16 `[dbzh, valid_mask]`."""
    out = np.zeros((config.GRID_NY, config.GRID_NX, 2), dtype=np.float16)
    if raw_bytes is None:
        return out
    try:
        with h5py.File(io.BytesIO(raw_bytes), "r") as f:
            dbz = f["dataset1/data1/data"][...]
            what1 = f["dataset1/what"].attrs
            nodata = float(what1.get("nodata", -9999000.0))
            undetect = float(what1.get("undetect", -8888000.0))
            gain = float(what1.get("gain", 1.0))
            offset = float(what1.get("offset", 0.0))
    except Exception:
        return out

    valid = (dbz != nodata) & np.isfinite(dbz)
    dbz_phys = dbz * gain + offset
    dbz_clean = np.where(~valid, 0.0, np.where(dbz == undetect, 0.0, np.clip(dbz_phys, 0.0, 75.0))).astype(np.float32)
    mask_clean = valid.astype(np.float32)

    dbz_5km = map_coordinates(dbz_clean, coords, order=1, mode="constant", cval=0.0)
    mask_5km = map_coordinates(mask_clean, coords, order=0, mode="constant", cval=0.0)
    # Zero out any bilinear bleed outside valid radar coverage
    dbz_5km = np.where(mask_5km > 0.5, np.clip(dbz_5km, 0.0, 75.0), 0.0)

    out[..., 0] = dbz_5km.astype(np.float16)
    out[..., 1] = (mask_5km > 0.5).astype(np.float16)
    return out


def build_radar_zarr(
    timestamps_path: Path = config.PROCESSED_DIR / "timestamps.parquet",
    static_zarr: Path = config.PROCESSED_DIR / "static_5km.zarr",
    out_zarr: Path = config.PROCESSED_DIR / "radar_5km_plus2h.zarr",
    workers: int = 16,
    t_shard: int = 64,
    upload: bool = True,
    repo_id: str = config.HF_REPO_ID,
) -> Path:
    ts_df = pd.read_parquet(timestamps_path)
    ts_cerra = pd.to_datetime(ts_df["timestamp_utc"], utc=True)
    ts_sensor = ts_cerra + pd.Timedelta(hours=SENSOR_OFFSET_HOURS)
    n_times = len(ts_sensor)

    session = _make_session(pool_size=max(workers + 4, 24))

    # Fetch one reference frame to precompute exact projection mapping
    sample_bytes = None
    for probe_ts in (ts_sensor.iloc[len(ts_sensor) // 2], ts_sensor.iloc[0], ts_sensor.iloc[-1]):
        sample_bytes = _fetch_opera_bytes(session, probe_ts)
        if sample_bytes is not None:
            break
    if sample_bytes is None:
        raise RuntimeError("Could not fetch any reference OPERA frame from CloudFerro archive.")

    coords, proj_info = _build_opera_to_cerra_coords(sample_bytes, static_zarr)
    log.info("Precomputed OPERA 2km -> CERRA 5.5km (1069x1069) lookup grid (proj=%s)", proj_info[0])

    if out_zarr.exists():
        root = zarr.open_group(str(out_zarr), mode="r+")
        if root["X"].shape[0] != n_times:
            raise ValueError(f"Existing {out_zarr} has T={root['X'].shape[0]}, expected {n_times}")
    else:
        out_zarr.parent.mkdir(parents=True, exist_ok=True)
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
        root = zarr.open_group(str(out_zarr), mode="w")
        root.create_array(
            "X",
            shape=(n_times, config.GRID_NY, config.GRID_NX, 2),
            chunks=(1, 128, 128, 2),
            shards=(t_shard, 512, 512, 2),
            dtype="float16",
            compressors=compressor,
            fill_value=0.0,
        )
        root.attrs["channels"] = RADAR_CHANNELS
        root.attrs["sensor_offset_hours"] = SENSOR_OFFSET_HOURS
        root.attrs["source"] = OPERA_ARCHIVE_BASE
        root.attrs["resolution_km"] = config.RESOLUTION_KM
        root.attrs["completed_shards"] = []
        root.attrs["missing_steps"] = 0

    X = root["X"]
    done_shards = set(root.attrs.get("completed_shards", []))
    missing_total = int(root.attrs.get("missing_steps", 0))
    n_shards = (n_times + t_shard - 1) // t_shard
    pending = [s for s in range(n_shards) if s not in done_shards]
    log.info("OPERA radar (+2h): %d/%d shards (%d steps) pending (workers=%d)",
             len(pending), n_shards, n_times, workers)

    t0_wall = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, shard_id in enumerate(pending, 1):
            s0 = shard_id * t_shard
            s1 = min(s0 + t_shard, n_times)
            shard_ts = [ts_sensor.iloc[i] for i in range(s0, s1)]

            def _process_one(t_s: pd.Timestamp) -> tuple[np.ndarray, bool]:
                raw = _fetch_opera_bytes(session, t_s)
                return _decode_and_reproject(raw, coords), (raw is None)

            results = list(ex.map(_process_one, shard_ts))
            frames = np.stack([r[0] for r in results], axis=0)
            n_miss = sum(1 for r in results if r[1])
            missing_total += n_miss

            X[s0:s1] = frames
            done_shards.add(shard_id)
            root.attrs["completed_shards"] = sorted(done_shards)
            root.attrs["missing_steps"] = missing_total

            if k % 10 == 0 or k == len(pending):
                elapsed = time.time() - t0_wall
                rate = (k * t_shard) / max(elapsed, 1e-3)
                eta_min = ((len(pending) - k) * t_shard) / max(rate, 1e-3) / 60.0
                log.info(
                    "OPERA shard %d/%d done (%.1f frames/s, missing=%d, ETA %.1f min)",
                    len(done_shards), n_shards, rate, missing_total, eta_min,
                )

    if upload:
        api = HfApi()
        log.info("Uploading %s -> hf://datasets/%s ...", out_zarr.name, repo_id)
        api.upload_folder(
            folder_path=str(out_zarr.parent),
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[f"{out_zarr.name}/**"],
            commit_message="Upload EUMETNET OPERA 5.5 km radar (+2h offset, 2018-2022) sharded Zarr v3",
        )
        log.info("OPERA radar uploaded: https://huggingface.co/datasets/%s", repo_id)

    return out_zarr


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build & upload EUMETNET OPERA 5.5 km radar (+2h) Zarr v3")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--repo-id", default=config.HF_REPO_ID)
    args = p.parse_args()
    build_radar_zarr(workers=args.workers, upload=not args.no_upload, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
