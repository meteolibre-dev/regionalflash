"""Fast Hugging Face -> Sharded Zarr v3 CERRA repacker (zero CDS queue!).

Streams a 5-year subset (`2018-2022`: 4 yrs train `2018-2021` + 1 yr val/test `2022`)
directly from `JonasZimmerHHI/cerra_2010_2022` (Anemoi Zarr v2, 1 timestep/chunk) in RAM
with zero local download-cache bloat, repacks into 128x128-chunked sharded Zarr v3
(`cerra_5km_{YEAR}_H1/H2.zarr`, float16), computes exact train stats, pushes each half-year
to `meteolibre-dev/regionalflash`, and cleans local Zarr before starting the next half-year.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import shutil
import time

from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers
import numcodecs
import numpy as np
import pandas as pd
import requests
import zarr

from . import config
from .cerra import (
    HALF_MONTHS,
    _init_or_open_period_zarr,
    accumulate_period_stats,
    merge_norm_stats,
)
from .push_hf import push_period_to_hub
from .timestamps import build_timestamps

log = logging.getLogger("regionalflash.from_hf_cerra")

_BLOSC = numcodecs.Blosc()


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(build_hf_headers())
    adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
    s.mount("https://", adapter)
    return s


def _fetch_bytes(session: requests.Session, repo_id: str, filename: str) -> bytes:
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
    for attempt in range(1, 10):
        try:
            r = session.get(url, timeout=180)
            r.raise_for_status()
            return r.content
        except Exception as e:
            wait_s = min(60, 5 * attempt)
            log.warning("HF fetch retry %d for %s (%s), sleeping %ds...", attempt, filename, e, wait_s)
            time.sleep(wait_s)
    raise RuntimeError(f"Failed to fetch {repo_id}/{filename}")


def _load_source_index(session: requests.Session, repo_id: str) -> tuple[list[str], dict[pd.Timestamp, int]]:
    meta = json.loads(_fetch_bytes(session, repo_id, ".zmetadata"))["metadata"]
    variables = list(meta[".zattrs"]["variables"])
    n_dates = int(meta["dates/.zarray"]["shape"][0])

    d0 = np.frombuffer(_BLOSC.decode(_fetch_bytes(session, repo_id, "dates/0")), dtype="<M8[s]")
    d1 = np.frombuffer(_BLOSC.decode(_fetch_bytes(session, repo_id, "dates/1")), dtype="<M8[s]")
    dates = np.concatenate([d0, d1])[:n_dates]
    ts_idx = pd.to_datetime(dates, utc=True)
    date_to_src_idx = {t: i for i, t in enumerate(ts_idx)}
    log.info("Loaded source index from %s: %d steps (%s .. %s)", repo_id, len(ts_idx), ts_idx[0], ts_idx[-1])
    return variables, date_to_src_idx


def _decode_step_to_57ch(
    raw_bytes: bytes,
    src_idx_map: dict[str, int],
) -> np.ndarray:
    """Decode one compressed (81, 1069*1069) float32 step -> (1069, 1069, 57) float16."""
    buf = _BLOSC.decode(raw_bytes)
    arr = np.frombuffer(buf, dtype="<f4").reshape(81, config.GRID_NY, config.GRID_NX)

    out = np.empty((config.GRID_NY, config.GRID_NX, config.N_DYN_CHANNELS), dtype=np.float16)
    ch = 0

    for _, short in config.CERRA_PL_VARS:
        for lvl in config.PRESSURE_LEVELS_HPA:
            v = arr[src_idx_map[f"{short}_{lvl}"]]
            if short == "t":
                out[..., ch] = (v - 273.15).astype(np.float16)
            elif short == "z":
                out[..., ch] = (v / 98.0665).astype(np.float16)
            elif short == "r":
                out[..., ch] = np.clip(v, 0.0, 100.0).astype(np.float16)
            else:
                out[..., ch] = v.astype(np.float16)
            ch += 1

    # Surface variables (7 channels)
    out[..., ch] = (arr[src_idx_map["2m_temperature"]] - 273.15).astype(np.float16)
    ch += 1
    out[..., ch] = np.clip(arr[src_idx_map["2m_relative_humidity"]], 0.0, 100.0).astype(np.float16)
    ch += 1

    ws10 = arr[src_idx_map["10m_wind_speed"]]
    wdir_rad = np.deg2rad(arr[src_idx_map["10m_wind_direction"]])
    out[..., ch] = (-ws10 * np.sin(wdir_rad)).astype(np.float16)
    ch += 1
    out[..., ch] = (-ws10 * np.cos(wdir_rad)).astype(np.float16)
    ch += 1

    out[..., ch] = (arr[src_idx_map["mean_sea_level_pressure"]] / 100.0 - 1013.25).astype(np.float16)
    ch += 1
    out[..., ch] = (arr[src_idx_map["surface_pressure"]] / 100.0 - 1000.0).astype(np.float16)
    ch += 1
    out[..., ch] = np.clip(arr[src_idx_map["total_cloud_cover"]], 0.0, 100.0).astype(np.float16)
    ch += 1

    assert ch == config.N_DYN_CHANNELS
    return out


def build_period_from_hf(
    year: int,
    half: str,
    session: requests.Session,
    src_repo: str,
    src_idx_map: dict[str, int],
    date_to_src_idx: dict[pd.Timestamp, int],
    timestamps_path: Path,
    processed_dir: Path,
    dl_workers: int = 8,
    t_shard: int = 16,
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
    n_times = len(ts_per)
    src_indices = [date_to_src_idx[t] for t in ts_per["timestamp_utc"]]

    period_zarr = processed_dir / f"cerra_5km_{period}.zarr"
    root = _init_or_open_period_zarr(period_zarr, period=period, n_times=n_times, dtype=config.ZARR_DTYPE)
    done_shards = set(root.attrs.get("completed_shards", []))
    X = root["X"]

    n_shards = (n_times + t_shard - 1) // t_shard
    pending_shards = [s for s in range(n_shards) if s not in done_shards]
    log.info("Period %s: %d/%d shards (%d steps) pending from %s (dl_workers=%d)",
             period, len(pending_shards), n_shards, n_times, src_repo, dl_workers)

    t0_wall = time.time()
    with ThreadPoolExecutor(max_workers=dl_workers) as ex:
        for k, shard_id in enumerate(pending_shards, 1):
            s0 = shard_id * t_shard
            s1 = min(s0 + t_shard, n_times)
            shard_src_ids = src_indices[s0:s1]

            def _dl_and_decode(si: int) -> np.ndarray:
                raw = _fetch_bytes(session, src_repo, f"data/{si}/0/0/0")
                return _decode_step_to_57ch(raw, src_idx_map)

            steps = list(ex.map(_dl_and_decode, shard_src_ids))
            X[s0:s1] = np.stack(steps, axis=0)

            done_shards.add(shard_id)
            root.attrs["completed_shards"] = sorted(done_shards)

            if k % 5 == 0 or k == len(pending_shards):
                elapsed = time.time() - t0_wall
                rate = (k * t_shard) / max(elapsed, 1e-3)
                eta_min = ((len(pending_shards) - k) * t_shard) / max(rate, 1e-3) / 60.0
                log.info(
                    "Period %s: shard %d/%d done (%.2f steps/s, ETA %.1f min)",
                    period, len(done_shards), n_shards, rate, eta_min,
                )

    root.attrs["completed_months"] = [f"{year}-{m:02d}" for m in target_months]

    static_zarr = processed_dir / "static_5km.zarr"
    if year in config.TRAIN_YEARS:
        if not (processed_dir / f"stats_accum_{period}.json").exists():
            accumulate_period_stats(period_zarr, period, processed_dir)
        merge_norm_stats(processed_dir, static_zarr)

    return period_zarr


def run_all(
    years: list[int],
    src_repo: str = config.SOURCE_HF_CERRA_REPO,
    dst_repo: str = config.HF_REPO_ID,
    processed_dir: Path = config.PROCESSED_DIR,
    dl_workers: int = 8,
    hf_workers: int = 8,
    upload_and_clean: bool = True,
) -> None:
    timestamps_path = processed_dir / "timestamps.parquet"
    build_timestamps(f"{min(years)}-01-01", f"{max(years)}-12-31", config.STEP_HOURS, timestamps_path)

    session = _make_session()
    variables, date_to_src_idx = _load_source_index(session, src_repo)
    src_idx_map = {v: i for i, v in enumerate(variables)}

    state_path = processed_dir / "uploaded_periods.json"
    uploaded = set(json.loads(state_path.read_text())) if state_path.exists() else set()

    for year in years:
        for half in ("H1", "H2"):
            period = f"{year}_{half}"
            if upload_and_clean and period in uploaded:
                log.info("Period %s already uploaded & cleaned, skipping.", period)
                continue

            period_zarr = build_period_from_hf(
                year=year,
                half=half,
                session=session,
                src_repo=src_repo,
                src_idx_map=src_idx_map,
                date_to_src_idx=date_to_src_idx,
                timestamps_path=timestamps_path,
                processed_dir=processed_dir,
                dl_workers=dl_workers,
            )

            if upload_and_clean:
                push_period_to_hub(
                    period=period,
                    processed_dir=processed_dir,
                    repo_id=dst_repo,
                    num_workers=hf_workers,
                )
                uploaded.add(period)
                state_path.write_text(json.dumps(sorted(uploaded), indent=2))

                log.info("Cleaning local %s to keep disk free...", period_zarr)
                shutil.rmtree(period_zarr, ignore_errors=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Repack CERRA 2018-2022 from Hugging Face into sharded Zarr v3")
    p.add_argument("--years", type=int, nargs="+", default=[2018, 2019, 2020, 2021, 2022])
    p.add_argument("--src-repo", default=config.SOURCE_HF_CERRA_REPO)
    p.add_argument("--dst-repo", default=config.HF_REPO_ID)
    p.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--dl-workers", type=int, default=8, help="Concurrent HF download threads")
    p.add_argument("--hf-workers", type=int, default=8, help="Concurrent HF upload threads")
    p.add_argument("--no-upload", action="store_true", help="Skip HF upload & local cleanup")
    args = p.parse_args()

    run_all(
        years=args.years,
        src_repo=args.src_repo,
        dst_repo=args.dst_repo,
        processed_dir=args.processed_dir,
        dl_workers=args.dl_workers,
        hf_workers=args.hf_workers,
        upload_and_clean=not args.no_upload,
    )


if __name__ == "__main__":
    main()
