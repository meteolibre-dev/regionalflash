"""Step 2: Fetch full-grid CERRA (5.5 km, 1069x1069) half-year by half-year into sharded Zarr v3.

Half-year workflow (H1 = Jan..Jun, H2 = Jul..Dec, ~155 GB each — keeps disk safely < 180 GB):
  For each period `{YEAR}_H1`, `{YEAR}_H2` in 2021..2025:
    1. Build `data/processed/cerra_5km_{YEAR}_{H}.zarr` (analysis-only, 57 channels) month-by-month,
       keeping all 3 CDS slots saturated and deleting raw monthly GRIB files right after streaming.
    2. Save sufficient statistics (`stats_accum_{YEAR}_{H}.json`) for train periods and update
       merged `norm_stats.json`.
    3. Optionally (`--upload-and-clean`): push `cerra_5km_{YEAR}_{H}.zarr` + metadata to Hugging Face
       (`meteolibre-dev/regionalflash`) and delete local half-year Zarr before starting the next one.
"""
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import cdsapi
import cfgrib
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from zarr.codecs import BloscCodec

from . import config

log = logging.getLogger("regionalflash.cerra")

ALL_HOURS_3H = [f"{h:02d}:00" for h in range(0, 24, config.STEP_HOURS)]
HALF_MONTHS = {"H1": [1, 2, 3, 4, 5, 6], "H2": [7, 8, 9, 10, 11, 12]}


def _days_in_month(year: int, month: int) -> list[str]:
    n_days = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, n_days + 1)]


def _existing_or_target(raw_dir: Path, stem: str, preferred_ext: str) -> Path:
    for ext in ("nc", "grib", "grib2"):
        cand = raw_dir / f"{stem}.{ext}"
        if cand.exists() and cand.stat().st_size > 1024:
            return cand
    return raw_dir / f"{stem}.{preferred_ext}"


def _cds_retrieve(dataset: str, req: dict, out_path: Path, label: str) -> Path:
    if out_path.exists() and out_path.stat().st_size > 1024:
        log.info("Reusing cached %s (%s)", out_path.name, label)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    for attempt in range(1, 61):
        try:
            client = cdsapi.Client(quiet=True)
            client.retrieve(dataset, req, str(tmp_path))
            tmp_path.replace(out_path)
            log.info("Downloaded %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
            return out_path
        except Exception as e:
            msg = str(e)
            if any(tok in msg for tok in ("Maximum number queued", "429", "502", "503", "504", "Timeout", "Connection", "too many")):
                wait_s = min(120, 30 * attempt)
                log.warning("CDS busy for %s (attempt %d, %s), sleeping %ds...", label, attempt, msg.splitlines()[0][:120], wait_s)
                time.sleep(wait_s)
                continue
            raise
    raise RuntimeError(f"CDS download failed after retries for {label}")


def download_static(raw_dir: Path, data_format: str = "grib") -> Path:
    ext = "nc" if data_format == "netcdf" else "grib"
    out_path = _existing_or_target(raw_dir, "cerra_static", ext)
    req = {
        "data_type": ["reanalysis"],
        "product_type": ["analysis"],
        "level_type": ["surface_or_atmosphere"],
        "variable": config.CERRA_STATIC_VARS,
        "year": ["2021"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "data_format": data_format,
    }
    return _cds_retrieve("reanalysis-cerra-single-levels", req, out_path, "static")


def _month_requests(year: int, month: int, raw_dir: Path, data_format: str) -> dict[str, tuple[str, dict, Path, str]]:
    """Build the 2 analysis-only CDS requests (pl + sl_an) for (year, month)."""
    ext = "nc" if data_format == "netcdf" else "grib"
    days = _days_in_month(year, month)

    pl_path = _existing_or_target(raw_dir, f"cerra_pl_{year}_{month:02d}", ext)
    sl_an_path = _existing_or_target(raw_dir, f"cerra_sl_an_{year}_{month:02d}", ext)

    req_pl = {
        "data_type": ["reanalysis"],
        "product_type": ["analysis"],
        "variable": [v for v, _ in config.CERRA_PL_VARS],
        "pressure_level": [str(lvl) for lvl in config.PRESSURE_LEVELS_HPA],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "time": ALL_HOURS_3H,
        "data_format": data_format,
    }
    req_sl_an = {
        "data_type": ["reanalysis"],
        "product_type": ["analysis"],
        "level_type": ["surface_or_atmosphere"],
        "variable": config.CERRA_SL_ANALYSIS_VARS,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "time": ALL_HOURS_3H,
        "data_format": data_format,
    }

    return {
        "pl": ("reanalysis-cerra-pressure-levels", req_pl, pl_path, f"pl {year}-{month:02d}"),
        "sl_an": ("reanalysis-cerra-single-levels", req_sl_an, sl_an_path, f"sl_an {year}-{month:02d}"),
    }


def _normalize_time_coord(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.coords and "time" in ds.dims:
        vt = ds["valid_time"]
        if vt.dims == ("time",):
            ds = ds.assign_coords(time=vt.to_numpy()).drop_vars("valid_time", errors="ignore")
    elif "time" not in ds.dims and "valid_time" in ds.coords:
        ds = ds.expand_dims(time=np.atleast_1d(ds["valid_time"].to_numpy()))
    elif "time" in ds.coords and "time" not in ds.dims:
        ds = ds.expand_dims(time=np.atleast_1d(ds["time"].to_numpy()))
    return ds


def _open_var_map(path: Path) -> dict[str, xr.DataArray]:
    def _open_one(p: Path) -> list[xr.Dataset]:
        if p.suffix in (".grib", ".grib2", ".grb"):
            return cfgrib.open_datasets(str(p), backend_kwargs={"indexpath": ""})
        return [xr.open_dataset(p, engine="netcdf4")]

    if zipfile.is_zipfile(path):
        td = tempfile.mkdtemp(prefix="cerra_zip_")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(td)
            dsets = []
            for name in sorted(zf.namelist()):
                if name.endswith((".nc", ".grib", ".grib2")):
                    dsets.extend(_open_one(Path(td) / name))
    else:
        dsets = _open_one(path)

    vmap: dict[str, xr.DataArray] = {}
    for ds in dsets:
        ds = _normalize_time_coord(ds)
        for v in ds.data_vars:
            vmap[v] = ds[v]
        if "latitude" in ds.coords and "_latitude" not in vmap:
            vmap["_latitude"] = ds["latitude"]
        if "longitude" in ds.coords and "_longitude" not in vmap:
            vmap["_longitude"] = ds["longitude"]
    return vmap


def _pick_da(vmap: dict[str, xr.DataArray], candidates: tuple[str, ...]) -> xr.DataArray:
    for name in candidates:
        if name in vmap:
            return vmap[name]
    raise KeyError(f"None of {candidates} found in variables: {[k for k in vmap if not k.startswith('_')]}")


def build_static_zarr(static_raw: Path, out_zarr: Path) -> Path:
    if out_zarr.exists():
        log.info("Static Zarr already exists: %s", out_zarr)
        return out_zarr

    vmap = _open_var_map(static_raw)
    orog = np.squeeze(_pick_da(vmap, ("orog", "orography", "z")).to_numpy()).astype(np.float32)
    lsm = np.squeeze(_pick_da(vmap, ("lsm", "land_sea_mask")).to_numpy()).astype(np.float32)
    lat = np.squeeze(vmap["_latitude"].to_numpy()).astype(np.float32)
    lon = np.squeeze(vmap["_longitude"].to_numpy()).astype(np.float32)
    lon = ((lon + 180.0) % 360.0) - 180.0

    assert orog.shape == (config.GRID_NY, config.GRID_NX), f"Unexpected CERRA grid shape {orog.shape}"

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    static_stack = np.stack(
        [orog, lsm, np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
        axis=0,
    ).astype(np.float32)

    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    root = zarr.open_group(str(out_zarr), mode="w")
    root.create_array("static", data=static_stack, chunks=(config.N_STATIC_CHANNELS, 256, 256), compressors=compressor)
    root.create_array("latitude", data=lat, chunks=(256, 256), compressors=compressor)
    root.create_array("longitude", data=lon, chunks=(256, 256), compressors=compressor)
    root.attrs["channels"] = config.STATIC_CHANNELS
    root.attrs["resolution_km"] = config.RESOLUTION_KM
    root.attrs["grid"] = "CERRA Lambert Conformal Conic (1069x1069)"
    log.info("Wrote static Zarr -> %s (shape=%s)", out_zarr, static_stack.shape)
    return out_zarr


def _init_or_open_period_zarr(out_zarr: Path, period: str, n_times: int, dtype: str) -> zarr.Group:
    if out_zarr.exists():
        root = zarr.open_group(str(out_zarr), mode="r+")
        if root["X"].shape[0] != n_times:
            raise ValueError(f"Existing {out_zarr} has T={root['X'].shape[0]}, expected T={n_times}.")
        return root

    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    chunks = (*config.ZARR_CHUNKS_TYX, config.N_DYN_CHANNELS)
    shards = (*config.ZARR_SHARDS_TYX, config.N_DYN_CHANNELS)

    root = zarr.open_group(str(out_zarr), mode="w")
    root.create_array(
        "X",
        shape=(n_times, config.GRID_NY, config.GRID_NX, config.N_DYN_CHANNELS),
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        compressors=compressor,
        fill_value=np.nan,
    )
    root.attrs["period"] = period
    root.attrs["channels"] = config.DYN_CHANNELS
    root.attrs["pressure_levels_hpa"] = config.PRESSURE_LEVELS_HPA
    root.attrs["step_hours"] = config.STEP_HOURS
    root.attrs["resolution_km"] = config.RESOLUTION_KM
    root.attrs["completed_months"] = []
    root.attrs["units"] = {
        "t_*": "degC",
        "u_*": "m/s",
        "v_*": "m/s",
        "r_*": "%",
        "z_*": "dam (z / 98.0665)",
        "t2m": "degC",
        "r2": "%",
        "u10": "m/s",
        "v10": "m/s",
        "msl_anom_hpa": "hPa - 1013.25",
        "sp_anom_hpa": "hPa - 1000.0",
        "tcc": "%",
    }
    log.info(
        "Initialized %s (shape=(%d,%d,%d,%d), chunks=%s, shards=%s, dtype=%s)",
        out_zarr.name, n_times, config.GRID_NY, config.GRID_NX, config.N_DYN_CHANNELS, chunks, shards, dtype,
    )
    return root


def _stream_month_to_zarr(
    files: dict[str, Path],
    target_times_naive: np.ndarray,
    local_idx: np.ndarray,
    X: zarr.Array,
    dtype: str,
    t_chunk: int = 16,
) -> None:
    """Stream one month into Zarr in 16-step slices (peak RAM < 8 GB)."""
    v_pl = _open_var_map(files["pl"])
    v_an = _open_var_map(files["sl_an"])

    pl_candidates = {
        "t": ("t", "temperature"),
        "u": ("u", "u_component_of_wind"),
        "v": ("v", "v_component_of_wind"),
        "r": ("r", "relative_humidity"),
        "z": ("z", "geopotential"),
    }
    pl_das = {short: _pick_da(v_pl, pl_candidates[short]) for _, short in config.CERRA_PL_VARS}
    sample_pl = next(iter(pl_das.values()))
    lvl_dim = "pressure_level" if "pressure_level" in sample_pl.dims else "isobaricInhPa"
    pl_das = {k: da.sel({lvl_dim: config.PRESSURE_LEVELS_HPA}) for k, da in pl_das.items()}

    da_t2m = _pick_da(v_an, ("t2m", "2m_temperature"))
    da_r2 = _pick_da(v_an, ("r2", "2m_relative_humidity"))
    da_ws10 = _pick_da(v_an, ("si10", "ws10", "10m_wind_speed"))
    da_wdir10 = _pick_da(v_an, ("wdir10", "10m_wind_direction"))
    da_msl = _pick_da(v_an, ("msl", "mean_sea_level_pressure"))
    da_sp = _pick_da(v_an, ("sp", "surface_pressure"))
    da_tcc = _pick_da(v_an, ("tcc", "total_cloud_cover"))

    n_total = len(target_times_naive)
    for start in range(0, n_total, t_chunk):
        end = min(start + t_chunk, n_total)
        t_slice = target_times_naive[start:end]
        dt_len = end - start
        cube = np.empty((dt_len, config.GRID_NY, config.GRID_NX, config.N_DYN_CHANNELS), dtype=np.float32)

        ch = 0
        for _, short in config.CERRA_PL_VARS:
            arr = pl_das[short].sel(time=t_slice).to_numpy().astype(np.float32)
            if short == "t":
                arr -= 273.15
            elif short == "z":
                arr /= 98.0665
            elif short == "r":
                np.clip(arr, 0.0, 100.0, out=arr)
            for li in range(len(config.PRESSURE_LEVELS_HPA)):
                cube[..., ch] = arr[:, li, :, :]
                ch += 1

        t2m = da_t2m.sel(time=t_slice).to_numpy().astype(np.float32) - 273.15
        r2 = np.clip(da_r2.sel(time=t_slice).to_numpy().astype(np.float32), 0.0, 100.0)
        ws10 = da_ws10.sel(time=t_slice).to_numpy().astype(np.float32)
        wdir_rad = np.deg2rad(da_wdir10.sel(time=t_slice).to_numpy().astype(np.float32))
        u10 = -ws10 * np.sin(wdir_rad)
        v10 = -ws10 * np.cos(wdir_rad)
        msl_anom = da_msl.sel(time=t_slice).to_numpy().astype(np.float32) / 100.0 - 1013.25
        sp_anom = da_sp.sel(time=t_slice).to_numpy().astype(np.float32) / 100.0 - 1000.0
        tcc = np.clip(da_tcc.sel(time=t_slice).to_numpy().astype(np.float32), 0.0, 100.0)

        for field in (t2m, r2, u10, v10, msl_anom, sp_anom, tcc):
            cube[..., ch] = field
            ch += 1

        assert ch == config.N_DYN_CHANNELS
        X[int(local_idx[start]) : int(local_idx[end - 1]) + 1] = cube.astype(dtype)


def accumulate_period_stats(
    period_zarr: Path,
    period: str,
    processed_dir: Path,
    stride_t: int = 4,
    stride_xy: int = 4,
) -> Path:
    """Save sufficient statistics for one half-year so norm_stats can be merged after cleanup."""
    root = zarr.open_group(str(period_zarr), mode="r")
    X = root["X"]
    n_t = X.shape[0]
    C = config.N_DYN_CHANNELS

    sum_x = np.zeros(C, dtype=np.float64)
    sum_x2 = np.zeros(C, dtype=np.float64)
    count = 0
    sum_dx2 = np.zeros(C, dtype=np.float64)
    count_dx = 0

    for idx in range(0, n_t, stride_t):
        arr = np.asarray(X[idx, ::stride_xy, ::stride_xy, :], dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            continue
        flat = arr.reshape(-1, C)
        sum_x += flat.sum(axis=0)
        sum_x2 += (flat * flat).sum(axis=0)
        count += flat.shape[0]

        if idx + 1 < n_t:
            nxt = np.asarray(X[idx + 1, ::stride_xy, ::stride_xy, :], dtype=np.float64)
            if np.all(np.isfinite(nxt)):
                dx = (nxt - arr).reshape(-1, C)
                sum_dx2 += (dx * dx).sum(axis=0)
                count_dx += dx.shape[0]

    accum = {
        "period": period,
        "sum_x": sum_x.tolist(),
        "sum_x2": sum_x2.tolist(),
        "count": int(count),
        "sum_dx2": sum_dx2.tolist(),
        "count_dx": int(count_dx),
    }
    out_path = processed_dir / f"stats_accum_{period}.json"
    out_path.write_text(json.dumps(accum))
    log.info("Saved period %s sufficient stats (%d sampled pixels) -> %s", period, count, out_path.name)
    return out_path


def merge_norm_stats(
    processed_dir: Path = config.PROCESSED_DIR,
    static_zarr: Path = config.PROCESSED_DIR / "static_5km.zarr",
    out_path: Path = config.PROCESSED_DIR / "norm_stats.json",
) -> dict:
    """Merge all train-period sufficient stats into exact `norm_stats.json`."""
    C = config.N_DYN_CHANNELS
    sum_x = np.zeros(C, dtype=np.float64)
    sum_x2 = np.zeros(C, dtype=np.float64)
    count = 0
    sum_dx2 = np.zeros(C, dtype=np.float64)
    count_dx = 0
    used_periods = []

    train_periods = [f"{y}_{h}" for y in config.TRAIN_YEARS for h in ("H1", "H2")]
    for per in train_periods:
        p = processed_dir / f"stats_accum_{per}.json"
        if p.exists():
            d = json.loads(p.read_text())
            sum_x += np.asarray(d["sum_x"], dtype=np.float64)
            sum_x2 += np.asarray(d["sum_x2"], dtype=np.float64)
            count += int(d["count"])
            sum_dx2 += np.asarray(d["sum_dx2"], dtype=np.float64)
            count_dx += int(d["count_dx"])
            used_periods.append(per)

    mean = sum_x / max(count, 1)
    var = np.maximum(sum_x2 / max(count, 1) - mean * mean, 1e-8)
    std = np.sqrt(var)
    delta_std = np.sqrt(np.maximum(sum_dx2 / max(count_dx, 1), 1e-8))

    s_root = zarr.open_group(str(static_zarr), mode="r")
    s_arr = np.asarray(s_root["static"], dtype=np.float64)
    s_mean = s_arr.mean(axis=(1, 2))
    s_std = np.maximum(s_arr.std(axis=(1, 2)), 1e-6)

    stats = {
        "channels": config.DYN_CHANNELS,
        "train_periods_included": used_periods,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "delta_std_3h": delta_std.tolist(),
        "n_train_pixels": int(count),
        "static_channels": config.STATIC_CHANNELS,
        "static_mean": s_mean.tolist(),
        "static_std": s_std.tolist(),
    }
    out_path.write_text(json.dumps(stats, indent=2))
    log.info("Merged norm_stats.json from train periods %s (%d pixels)", used_periods, count)
    return stats


def build_period(
    year: int,
    half: str,
    timestamps_path: Path = config.PROCESSED_DIR / "timestamps.parquet",
    raw_dir: Path = config.RAW_DIR / "cerra",
    processed_dir: Path = config.PROCESSED_DIR,
    workers: int = 3,
    data_format: str = "grib",
    dtype: str = config.ZARR_DTYPE,
    delete_raw: bool = True,
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
    if ts_per.empty:
        raise ValueError(f"No timestamps found for period {period} in {timestamps_path}")

    static_zarr = processed_dir / "static_5km.zarr"
    if not static_zarr.exists():
        static_raw = download_static(raw_dir, data_format=data_format)
        build_static_zarr(static_raw, static_zarr)
        if delete_raw:
            static_raw.unlink(missing_ok=True)

    period_zarr = processed_dir / f"cerra_5km_{period}.zarr"
    root = _init_or_open_period_zarr(period_zarr, period=period, n_times=len(ts_per), dtype=dtype)
    completed = set(root.attrs.get("completed_months", []))

    pending = [m for m in target_months if f"{year}-{m:02d}" not in completed]
    log.info("Period %s: %d/%d months pending (workers=%d, format=%s, dtype=%s)",
             period, len(pending), len(target_months), workers, data_format, dtype)

    X = root["X"]
    # Queue all pending months of this half-year into the 3-worker pool so all 3 CDS slots
    # stay 100% saturated with zero idle gaps between months.
    ready_files: dict[int, dict[str, Path]] = {m: {} for m in pending}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for m in pending:
            reqs = _month_requests(year, m, raw_dir, data_format)
            for key, (ds_name, req, out_path, label) in reqs.items():
                fut = ex.submit(_cds_retrieve, ds_name, req, out_path, label)
                futs[fut] = (m, key)

        for fut in as_completed(futs):
            m, key = futs[fut]
            ready_files[m][key] = fut.result()

            if len(ready_files[m]) == 2:
                sub = ts_per[ts_per["month"] == m]
                local_idx = sub.index.to_numpy(dtype=np.int64)
                t_naive = np.array(
                    [t.tz_convert("UTC").tz_localize(None).to_datetime64() for t in sub["timestamp_utc"]]
                )
                log.info("Streaming %04d-%02d (%d steps) -> %s ...", year, m, len(local_idx), period_zarr.name)
                _stream_month_to_zarr(ready_files[m], t_naive, local_idx, X, dtype=dtype)

                completed.add(f"{year}-{m:02d}")
                root.attrs["completed_months"] = sorted(completed)
                log.info(
                    "Wrote %04d-%02d (%d steps -> X[%d:%d]) [%d/%d months done in %s]",
                    year, m, len(local_idx), int(local_idx[0]), int(local_idx[-1]) + 1,
                    len(completed), len(target_months), period,
                )
                if delete_raw:
                    for p in ready_files[m].values():
                        p.unlink(missing_ok=True)

    if year in config.TRAIN_YEARS:
        accumulate_period_stats(period_zarr, period, processed_dir)
        merge_norm_stats(processed_dir, static_zarr)

    return period_zarr


def run_all_periods(
    years: list[int],
    timestamps_path: Path = config.PROCESSED_DIR / "timestamps.parquet",
    raw_dir: Path = config.RAW_DIR / "cerra",
    processed_dir: Path = config.PROCESSED_DIR,
    workers: int = 3,
    data_format: str = "grib",
    dtype: str = config.ZARR_DTYPE,
    delete_raw: bool = True,
    upload_and_clean: bool = False,
    repo_id: str = config.HF_REPO_ID,
    hf_workers: int = 8,
) -> None:
    from .push_hf import push_period_to_hub

    state_path = processed_dir / "uploaded_periods.json"
    uploaded = set(json.loads(state_path.read_text())) if state_path.exists() else set()

    for year in years:
        for half in ("H1", "H2"):
            period = f"{year}_{half}"
            if upload_and_clean and period in uploaded:
                log.info("Period %s already uploaded & cleaned, skipping.", period)
                continue

            period_zarr = build_period(
                year=year,
                half=half,
                timestamps_path=timestamps_path,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                workers=workers,
                data_format=data_format,
                dtype=dtype,
                delete_raw=delete_raw,
            )

            if upload_and_clean:
                push_period_to_hub(
                    period=period,
                    processed_dir=processed_dir,
                    repo_id=repo_id,
                    num_workers=hf_workers,
                )
                uploaded.add(period)
                state_path.write_text(json.dumps(sorted(uploaded), indent=2))

                log.info("Cleaning local %s to keep disk free...", period_zarr)
                shutil.rmtree(period_zarr, ignore_errors=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build full-grid CERRA Zarr v3 half-year by half-year (H1/H2)")
    p.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    p.add_argument("--timestamps", type=Path, default=config.PROCESSED_DIR / "timestamps.parquet")
    p.add_argument("--raw-dir", type=Path, default=config.RAW_DIR / "cerra")
    p.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--workers", type=int, default=3, help="Concurrent CDS download workers (CDS cap is 3 queued jobs)")
    p.add_argument("--hf-workers", type=int, default=8, help="Concurrent HF upload workers")
    p.add_argument("--format", choices=["grib", "netcdf"], default="grib")
    p.add_argument("--dtype", choices=["float16", "float32"], default=config.ZARR_DTYPE)
    p.add_argument("--keep-raw", action="store_true", help="Keep raw monthly files instead of deleting after Zarr write")
    p.add_argument("--upload-and-clean", action="store_true", help="Upload each half-year to HF and delete local Zarr")
    p.add_argument("--repo-id", default=config.HF_REPO_ID)
    args = p.parse_args()

    run_all_periods(
        years=args.years,
        timestamps_path=args.timestamps,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        workers=args.workers,
        data_format=args.format,
        dtype=args.dtype,
        delete_raw=not args.keep_raw,
        upload_and_clean=args.upload_and_clean,
        repo_id=args.repo_id,
        hf_workers=args.hf_workers,
    )


if __name__ == "__main__":
    main()
