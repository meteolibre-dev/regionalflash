"""Step 1: Generate contiguous 3-hourly timestamp table for CERRA forecasting splits.

Splits (zero future leakage):
  - train : 2021-01-01 .. 2024-12-31  (4 full years, 11,688 steps @ 3h)
  - val   : 2025-01-01 .. 2025-06-30  (6 months, 1,448 steps @ 3h)
  - test  : 2025-07-01 .. 2025-12-31  (6 months, 1,472 steps @ 3h)
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger("regionalflash.timestamps")


def build_timestamps(
    date_start: str = config.DATE_START,
    date_end: str = config.DATE_END,
    step_hours: int = config.STEP_HOURS,
    out_path: Path = config.PROCESSED_DIR / "timestamps.parquet",
) -> pd.DataFrame:
    start_ts = pd.Timestamp(f"{date_start} 00:00:00", tz="UTC")
    end_ts = pd.Timestamp(f"{date_end} 21:00:00", tz="UTC")
    ts = pd.date_range(start=start_ts, end=end_ts, freq=f"{step_hours}h")

    val_start = pd.Timestamp(f"{config.VAL_START} 00:00:00", tz="UTC")
    test_start = pd.Timestamp(f"{config.TEST_START} 00:00:00", tz="UTC")

    splits = []
    for t in ts:
        if t < val_start:
            splits.append("train")
        elif t < test_start:
            splits.append("val")
        else:
            splits.append("test")

    df = pd.DataFrame(
        {
            "time_idx": range(len(ts)),
            "timestamp_utc": ts,
            "prev_timestamp_utc": ts - pd.Timedelta(hours=step_hours),
            "year": ts.year.astype("int16"),
            "month": ts.month.astype("int8"),
            "day": ts.day.astype("int8"),
            "hour": ts.hour.astype("int8"),
            "split_time": splits,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    counts = df["split_time"].value_counts().to_dict()
    log.info("Wrote %d contiguous %dh timestamps -> %s (%s)", len(df), step_hours, out_path, counts)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build contiguous 3-hourly CERRA timestamp index")
    p.add_argument("--start", default=config.DATE_START, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=config.DATE_END, help="End date YYYY-MM-DD")
    p.add_argument("--out", type=Path, default=config.PROCESSED_DIR / "timestamps.parquet")
    args = p.parse_args()
    build_timestamps(args.start, args.end, config.STEP_HOURS, args.out)


if __name__ == "__main__":
    main()
