"""Download European METAR airport observations (2018-2022) from IEM ASOS archive,
snap to `t + 2h` relative to CERRA timestamps, project to exact CERRA (y, x) pixels,
and upload `metar_stations.parquet` + `metar_obs_plus2h.parquet` to Hugging Face.

Fast & rate-limit-safe strategy (zero 429s, ~10-20 min for all 5 years):
  1. Discover all IEM stations inside the CERRA 1069x1069 grid using 2 probe days (2018 & 2022).
  2. Query IEM sequentially (`workers=1`) in batches of 40 European stations x 1 year at a time
     (instead of downloading 30 MB/day of global stations x 1,826 days).
  3. Snap each station's reports to `t_cerra + 2h` within `[-35min, +10min]` (latest report wins).
  4. Write `metar_stations.parquet` + `metar_obs_plus2h.parquet` and upload to Hugging Face.
"""
import argparse
import datetime as dt
import logging
from pathlib import Path
import time

from huggingface_hub import HfApi
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
import zarr

from . import config

log = logging.getLogger("regionalflash.metar")

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_FIELDS = ["tmpc", "dwpc", "mslp", "p01m", "drct", "sknt"]
SENSOR_OFFSET_HOURS = 2
EARTH_RADIUS_KM = 6371.0088


def _latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    cos_lat = np.cos(lat)
    return np.stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=-1)


def _build_cerra_kdtree(static_zarr: Path) -> tuple[cKDTree, np.ndarray, float, float, float, float]:
    s = zarr.open_group(str(static_zarr), mode="r")
    lat = np.asarray(s["latitude"], dtype=np.float64)
    lon = np.asarray(s["longitude"], dtype=np.float64)
    orog = np.asarray(s["static"][0], dtype=np.float32)
    xyz = _latlon_to_xyz(lat.ravel(), lon.ravel())
    tree = cKDTree(xyz)
    return tree, orog, float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())


def _iem_get(session: requests.Session, params: list[tuple[str, str]], label: str) -> str:
    for attempt in range(1, 10):
        try:
            r = session.get(IEM_ASOS_URL, params=params, timeout=240)
            if r.status_code == 429 or r.status_code >= 500:
                wait_s = min(90, 10 * attempt)
                log.warning("IEM %d for %s (attempt %d), sleeping %ds...", r.status_code, label, attempt, wait_s)
                time.sleep(wait_s)
                continue
            r.raise_for_status()
            return r.text
        except Exception as e:
            wait_s = min(90, 10 * attempt)
            log.warning("IEM retry %d for %s (%s), sleeping %ds...", attempt, label, e, wait_s)
            time.sleep(wait_s)
    raise RuntimeError(f"IEM request failed after retries for {label}")


def _parse_iem_text(
    text: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> pd.DataFrame:
    rows = []
    header = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if header is None:
            header = line.split(",")
            idx_st = header.index("station")
            idx_val = header.index("valid")
            idx_lon = header.index("lon")
            idx_lat = header.index("lat")
            idx_tmp = header.index("tmpc")
            idx_dwp = header.index("dwpc")
            idx_msl = header.index("mslp")
            idx_p01 = header.index("p01m")
            idx_dir = header.index("drct")
            idx_spd = header.index("sknt")
            continue

        parts = line.split(",")
        if len(parts) < len(header):
            continue
        try:
            lat = float(parts[idx_lat])
            lon = float(parts[idx_lon])
        except ValueError:
            continue
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            continue

        def _f(s: str) -> float:
            if not s or s == "M" or s == "T":
                return np.nan
            try:
                v = float(s)
                return np.nan if v == -9999.0 else v
            except ValueError:
                return np.nan

        tmpc = _f(parts[idx_tmp])
        dwpc = _f(parts[idx_dwp])
        mslp = _f(parts[idx_msl])
        if np.isnan(tmpc) and np.isnan(mslp):
            continue

        wspd_kt = _f(parts[idx_spd])
        wdir = _f(parts[idx_dir])
        if not np.isnan(wspd_kt) and not np.isnan(wdir):
            wspd_ms = wspd_kt * 0.514444
            wrad = np.deg2rad(wdir)
            u10 = -wspd_ms * np.sin(wrad)
            v10 = -wspd_ms * np.cos(wrad)
        else:
            u10, v10 = np.nan, np.nan

        rows.append((
            parts[idx_st],
            parts[idx_val],
            lat,
            lon,
            tmpc,
            dwpc,
            mslp,
            u10,
            v10,
            _f(parts[idx_p01]),
        ))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["station", "valid", "lat", "lon", "tmpc", "dwpc", "mslp", "u10", "v10", "p01m"],
    )
    df["valid_utc"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    df = df.dropna(subset=["valid_utc"]).drop(columns=["valid"])
    return df


def _discover_cerra_stations(
    session: requests.Session,
    tree: cKDTree,
    orog: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    probe_dfs = []
    for d in (dt.date(2018, 6, 15), dt.date(2020, 6, 15), dt.date(2022, 6, 15)):
        nxt = d + dt.timedelta(days=1)
        params = [
            ("tz", "UTC"),
            ("year1", str(d.year)), ("month1", str(d.month)), ("day1", str(d.day)),
            ("year2", str(nxt.year)), ("month2", str(nxt.month)), ("day2", str(nxt.day)),
            ("format", "comma"), ("latlon", "yes"), ("direct", "no"),
            ("report_type", "1"), ("report_type", "2"),
        ] + [("data", f) for f in IEM_FIELDS]
        txt = _iem_get(session, params, f"probe {d}")
        df = _parse_iem_text(txt, lat_min, lat_max, lon_min, lon_max)
        if not df.empty:
            probe_dfs.append(df[["station", "lat", "lon"]])
        time.sleep(1.0)

    all_st = pd.concat(probe_dfs, ignore_index=True)
    st_meta = all_st.groupby("station", as_index=False).agg(lat=("lat", "median"), lon=("lon", "median"))
    st_xyz = _latlon_to_xyz(st_meta["lat"].to_numpy(), st_meta["lon"].to_numpy())
    dists, flat_idx = tree.query(st_xyz, workers=-1)
    dist_km = dists * EARTH_RADIUS_KM
    ys = (flat_idx // config.GRID_NX).astype(np.int16)
    xs = (flat_idx % config.GRID_NX).astype(np.int16)

    st_meta["y"] = ys
    st_meta["x"] = xs
    st_meta["dist_km"] = dist_km.astype(np.float32)
    st_meta["cerra_orog_m"] = orog[ys, xs].astype(np.float32)
    st_meta = st_meta[st_meta["dist_km"] <= 5.5].sort_values("station").reset_index(drop=True)
    st_meta["station_idx"] = np.arange(len(st_meta), dtype=np.int32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    st_meta.to_parquet(cache_path, index=False)
    log.info("Discovered %d European METAR stations inside CERRA 5.5 km grid", len(st_meta))
    return st_meta


def _snap_to_sensor_times(
    df: pd.DataFrame,
    sensor_unix_s: np.ndarray,
    time_indices: np.ndarray,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    obs_s = (df["valid_utc"].astype("int64") // 10**9).to_numpy()
    pos = np.searchsorted(sensor_unix_s, obs_s)
    pos_right = np.clip(pos, 0, len(sensor_unix_s) - 1)
    pos_left = np.clip(pos - 1, 0, len(sensor_unix_s) - 1)

    d_right = obs_s - sensor_unix_s[pos_right]
    d_left = obs_s - sensor_unix_s[pos_left]

    use_right = np.abs(d_right) <= np.abs(d_left)
    best_pos = np.where(use_right, pos_right, pos_left)
    delta_s = np.where(use_right, d_right, d_left)

    keep = (delta_s >= -35 * 60) & (delta_s <= 10 * 60)
    if not np.any(keep):
        return pd.DataFrame()

    out = df.loc[keep].copy()
    out["time_idx"] = time_indices[best_pos[keep]].astype(np.int32)
    out = out.sort_values(["time_idx", "station", "valid_utc"])
    out = out.drop_duplicates(subset=["time_idx", "station"], keep="last")
    return out


def build_metar_dataset(
    timestamps_path: Path = config.PROCESSED_DIR / "timestamps.parquet",
    static_zarr: Path = config.PROCESSED_DIR / "static_5km.zarr",
    raw_dir: Path = config.RAW_DIR / "metar_yearly",
    processed_dir: Path = config.PROCESSED_DIR,
    batch_size: int = 40,
    upload: bool = True,
    repo_id: str = config.HF_REPO_ID,
) -> tuple[Path, Path]:
    ts_df = pd.read_parquet(timestamps_path)
    ts_cerra = pd.to_datetime(ts_df["timestamp_utc"], utc=True)
    ts_sensor = ts_cerra + pd.Timedelta(hours=SENSOR_OFFSET_HOURS)
    sensor_unix_s = (ts_sensor.astype("int64") // 10**9).to_numpy()
    time_indices = ts_df["time_idx"].to_numpy(dtype=np.int32)

    log.info("Building CERRA 5.5 km cKDTree for exact METAR station projection...")
    tree, orog, lat_min, lat_max, lon_min, lon_max = _build_cerra_kdtree(static_zarr)

    session = requests.Session()
    session.headers.update({"User-Agent": "RegionalFlash/1.0 (meteolibre-dev)"})

    raw_dir.mkdir(parents=True, exist_ok=True)
    st_meta = _discover_cerra_stations(
        session, tree, orog, lat_min, lat_max, lon_min, lon_max, raw_dir / "discovered_stations.parquet"
    )
    station_ids = st_meta["station"].tolist()
    years = sorted(int(y) for y in ts_df["year"].unique())

    yearly_parquets: list[Path] = []
    for yr in years:
        out_y = raw_dir / f"metar_{yr}.parquet"
        if out_y.exists() and out_y.stat().st_size > 1024:
            log.info("Reusing cached %s", out_y.name)
            yearly_parquets.append(out_y)
            continue

        year_dfs = []
        n_batches = (len(station_ids) + batch_size - 1) // batch_size
        t0_yr = time.time()
        for bi in range(n_batches):
            st_batch = station_ids[bi * batch_size : (bi + 1) * batch_size]
            params = [
                ("tz", "UTC"),
                ("year1", str(yr)), ("month1", "1"), ("day1", "1"),
                ("year2", str(yr + 1)), ("month2", "1"), ("day2", "2"),
                ("format", "comma"), ("latlon", "yes"), ("direct", "no"),
                ("report_type", "1"), ("report_type", "2"),
            ] + [("data", f) for f in IEM_FIELDS] + [("station", s) for s in st_batch]

            txt = _iem_get(session, params, f"{yr} batch {bi+1}/{n_batches}")
            df = _parse_iem_text(txt, lat_min, lat_max, lon_min, lon_max)
            if not df.empty:
                snapped = _snap_to_sensor_times(df, sensor_unix_s, time_indices)
                if not snapped.empty:
                    year_dfs.append(snapped)

            if (bi + 1) % 5 == 0 or (bi + 1) == n_batches:
                log.info("METAR %d: batch %d/%d done (%.1fs elapsed)", yr, bi + 1, n_batches, time.time() - t0_yr)
            time.sleep(0.8)

        y_df = pd.concat(year_dfs, ignore_index=True)
        y_df = y_df.sort_values(["time_idx", "station", "valid_utc"]).drop_duplicates(
            subset=["time_idx", "station"], keep="last"
        )
        y_df.to_parquet(out_y, index=False, compression="zstd")
        yearly_parquets.append(out_y)
        log.info("Saved METAR %d: %d snapped (+2h) obs -> %s", yr, len(y_df), out_y.name)

    full_df = pd.concat([pd.read_parquet(p) for p in sorted(yearly_parquets)], ignore_index=True)
    full_df = full_df.sort_values(["time_idx", "station", "valid_utc"]).drop_duplicates(
        subset=["time_idx", "station"], keep="last"
    )

    st_counts = full_df.groupby("station").size().rename("n_obs")
    st_meta = st_meta.merge(st_counts, on="station", how="left")
    st_meta["n_obs"] = st_meta["n_obs"].fillna(0).astype(np.int32)

    obs_df = full_df.merge(st_meta[["station", "station_idx", "y", "x"]], on="station", how="inner")
    obs_df["mslp_anom_hpa"] = (obs_df["mslp"] - 1013.25).astype(np.float32)
    for col in ("tmpc", "dwpc", "u10", "v10", "p01m"):
        obs_df[col] = obs_df[col].astype(np.float32)

    obs_df = obs_df[
        ["time_idx", "station_idx", "y", "x", "tmpc", "dwpc", "mslp_anom_hpa", "u10", "v10", "p01m", "valid_utc"]
    ].sort_values(["time_idx", "station_idx"]).reset_index(drop=True)

    stations_out = processed_dir / "metar_stations.parquet"
    obs_out = processed_dir / "metar_obs_plus2h.parquet"
    st_meta.to_parquet(stations_out, index=False, compression="zstd")
    obs_df.to_parquet(obs_out, index=False, compression="zstd")

    log.info(
        "Wrote %d CERRA-domain METAR stations -> %s and %d (+2h) obs -> %s (%.1f MB)",
        len(st_meta), stations_out.name, len(obs_df), obs_out.name, obs_out.stat().st_size / 1e6,
    )

    if upload:
        api = HfApi()
        log.info("Uploading METAR parquet tables -> hf://datasets/%s ...", repo_id)
        api.upload_folder(
            folder_path=str(processed_dir),
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["metar_stations.parquet", "metar_obs_plus2h.parquet"],
            commit_message="Upload European METAR airport observations (+2h offset, 2018-2022)",
        )
        log.info("METAR uploaded: https://huggingface.co/datasets/%s", repo_id)

    return stations_out, obs_out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Download & upload European METAR (+2h) Parquet tables")
    p.add_argument("--batch-size", type=int, default=40, help="Stations per IEM yearly request")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--repo-id", default=config.HF_REPO_ID)
    args = p.parse_args()
    build_metar_dataset(batch_size=args.batch_size, upload=not args.no_upload, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
