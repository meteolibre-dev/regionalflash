"""Step 3: Push half-year sharded CERRA Zarr v3 stores (`cerra_5km_{YYYY}_H1/H2.zarr`) to Hugging Face Hub.

Uses `HfApi.upload_large_folder` for resumable, multi-commit uploads of large Zarr stores.
Target repo: https://huggingface.co/datasets/meteolibre-dev/regionalflash
"""
import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi
import pandas as pd
import zarr

from . import config

log = logging.getLogger("regionalflash.push_hf")


def _build_dataset_card(processed_dir: Path, repo_id: str, periods_present: list[str]) -> str:
    ts = pd.read_parquet(processed_dir / "timestamps.parquet")
    splits = ts["split_time"].value_counts().to_dict()
    stats_path = processed_dir / "norm_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    n_pix = stats.get("n_train_pixels", 0)

    sample_zarr = None
    for per in periods_present:
        p = processed_dir / f"cerra_5km_{per}.zarr"
        if p.exists():
            sample_zarr = zarr.open_group(str(p), mode="r")
            break

    chunks = sample_zarr["X"].chunks if sample_zarr is not None else (*config.ZARR_CHUNKS_TYX, config.N_DYN_CHANNELS)
    shards = sample_zarr["X"].shards if sample_zarr is not None else (*config.ZARR_SHARDS_TYX, config.N_DYN_CHANNELS)
    dtype = str(sample_zarr["X"].dtype) if sample_zarr is not None else config.ZARR_DTYPE

    period_lines = "\n".join(f"- `cerra_5km_{per}.zarr/`" for per in sorted(periods_present))

    return f"""---
license: cc-by-4.0
task_categories:
  - video-prediction
  - image-to-image
tags:
  - weather
  - forecasting
  - meteorology
  - cerra
  - zarr
  - europe
  - diffusion
pretty_name: RegionalFlash 5.5 km European Regional Weather Forecasting Dataset (CERRA)
---

# {repo_id}

ML-ready **5.5 km** pan-European regional weather forecasting dataset built from **Copernicus CERRA** (`1069 × 1069` Lambert Conformal Conic grid, 3-hourly analyses), stored as **half-year sharded Zarr v3** (`H1` Jan–Jun, `H2` Jul–Dec, ~155 GB each) for fast on-the-fly `128 × 128` (`704 km × 704 km`) spatio-temporal patching and tiled diffusion inference.

## Dataset Layout

### Half-Year Dynamic Cubes (`(T_half, 1069, 1069, {config.N_DYN_CHANNELS})`)
{period_lines}

- **Dtype:** `{dtype}` (Blosc-Zstd bitshuffle)
- **Inner chunks:** `{chunks}` (1-to-1 aligned with `128 × 128` training patches)
- **Shards:** `{shards}`
- **Dynamic channels (`C = {config.N_DYN_CHANNELS}`):** `{config.DYN_CHANNELS}`

### Shared Static & Metadata
- `static_5km.zarr/` — Static terrain & coordinate arrays (`(6, 1069, 1069)`: `{config.STATIC_CHANNELS}` + `latitude`, `longitude`)
- `timestamps.parquet` — **{len(ts):,}** contiguous 3-hourly timestamps (`splits`: `{splits}`)
- `norm_stats.json` — Train-split channel `mean`, `std`, and 3h tendency `delta_std_3h` computed over `{n_pix:,}` sampled training pixels

## Physical Units (`float16`-safe)
- `t_*`, `t2m`: `°C` (`K - 273.15`)
- `u_*`, `v_*`, `u10`, `v10`: `m/s`
- `r_*`, `r2`, `tcc`: `%` (`0..100`)
- `z_*`: geopotential height in `dam` (`geopotential / 98.0665`)
- `msl_anom_hpa`: `hPa - 1013.25`
- `sp_anom_hpa`: `hPa - 1000.0`

## Temporal Splits (Zero Leakage)
- **`train`:** `2018_H1` .. `2021_H2` (4 years)
- **`val`:** `2022_H1` (6 months)
- **`test`:** `2022_H2` (6 months)

## Future Enrichment
The `(T_half, 1069, 1069)` Zarr layout is designed so `radar_5km_{{period}}.zarr` and `sat_5km_{{period}}.zarr` can be added later as aligned companion Zarr stores without regenerating CERRA.

## Source & License
- Copernicus Climate Change Service (C3S) — CERRA Regional Reanalysis (repacked from `JonasZimmerHHI/cerra_2010_2022`).
"""


def push_period_to_hub(
    period: str,
    processed_dir: Path = config.PROCESSED_DIR,
    repo_id: str = config.HF_REPO_ID,
    private: bool = False,
    num_workers: int = 8,
) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    state_path = processed_dir / "uploaded_periods.json"
    uploaded = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    periods_present = sorted(uploaded | {period})

    card = _build_dataset_card(processed_dir, repo_id, periods_present)
    (processed_dir / "README.md").write_text(card)

    patterns = [
        f"cerra_5km_{period}.zarr/**",
        "static_5km.zarr/**",
        "timestamps.parquet",
        "norm_stats.json",
        "README.md",
    ]
    log.info("Uploading period %s -> hf://datasets/%s ...", period, repo_id)
    api.upload_folder(
        folder_path=str(processed_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        commit_message=f"Upload CERRA 5.5 km {period} sharded Zarr v3 + metadata",
    )
    log.info("Period %s uploaded: https://huggingface.co/datasets/%s", period, repo_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Push RegionalFlash half-year Zarr v3 dataset to Hugging Face Hub")
    p.add_argument("--period", required=True, help="Period to upload (e.g. 2021_H1)")
    p.add_argument("--repo-id", default=config.HF_REPO_ID)
    p.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--private", action="store_true")
    args = p.parse_args()
    push_period_to_hub(args.period, args.processed_dir, args.repo_id, args.private, args.workers)


if __name__ == "__main__":
    main()
