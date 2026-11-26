"""Multimodal 2-frame PyTorch Dataset for RegionalFlash (CERRA + SatWeave + OPERA + METAR).

For each anchor `t0` and `128x128` crop `(y0, x0)` on the `1069x1069` (5.5 km) CERRA grid:
  - Frame 0 (known conditional context):
      CERRA(t0) [57 ch] + SatWeave(t0 + 2h) [11 ch + mask] + OPERA(t0 + 2h) [1 ch + mask]
      + sparse METAR(t0 + 2h) [6 ch + mask] + Static [6 ch]
  - Frame 1 (target to denoise, +3h):
      CERRA(t0 + 3h) [57 ch] + SatWeave(t0 + 5h) [11 ch + mask] + OPERA(t0 + 5h) [1 ch + mask]
      + sparse METAR(t0 + 5h) [6 ch + mask]

Missing/outside-FOV/non-station pixels are strictly set to neutral `0.0` AFTER normalization,
paired with explicit binary validity masks (`sat_mask`, `radar_mask`, `metar_mask`).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from . import config
from .solar import solar_step_features

METAR_COLS = ["tmpc", "dwpc", "mslp_anom_hpa", "u10", "v10", "p01m"]
# Simple physical normalization constants for METAR (tmpc, dwpc, mslp_anom, u10, v10, log1p(p01m))
METAR_MEAN = np.array([10.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
METAR_STD = np.array([10.0, 10.0, 12.0, 6.0, 6.0, 1.0], dtype=np.float32)

# SatWeave normalization: 3 reflectance bands (%) + 8 IR brightness temp bands (degC)
SAT_MEAN = np.array([20.0] * 3 + [-20.0] * 8, dtype=np.float32)
SAT_STD = np.array([25.0] * 3 + [25.0] * 8, dtype=np.float32)

RADAR_SCALE = 20.0  # dBZ / 20.0 -> [0, 3.75]


class _MetarIndex:
    """Compact CSR-like in-memory index for O(1) sparse METAR lookup by `time_idx`."""

    def __init__(self, obs_parquet: Path, n_times: int) -> None:
        df = pd.read_parquet(obs_parquet)
        t_idx = df["time_idx"].to_numpy(dtype=np.int32)
        order = np.argsort(t_idx, kind="mergesort")
        t_idx = t_idx[order]

        self.y = df["y"].to_numpy(dtype=np.int16)[order]
        self.x = df["x"].to_numpy(dtype=np.int16)[order]

        vals = np.stack([df[c].to_numpy(dtype=np.float32)[order] for c in METAR_COLS], axis=-1)
        # Compress heavy precipitation tail with log1p(clip(p01m, 0, 100))
        p01 = vals[:, 5]
        vals[:, 5] = np.where(np.isfinite(p01), np.log1p(np.clip(p01, 0.0, 100.0)), np.nan)
        self.vals = vals  # (N_obs, 6) float32, NaN where unreported

        self.offsets = np.searchsorted(t_idx, np.arange(n_times + 1, dtype=np.int32))

    def rasterize_patch(
        self,
        t_idx: int,
        y0: int,
        x0: int,
        patch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(metar_norm (6, P, P), metar_mask (6, P, P) bool)` for one patch."""
        out = np.zeros((6, patch_size, patch_size), dtype=np.float32)
        mask = np.zeros((6, patch_size, patch_size), dtype=bool)
        s = int(self.offsets[t_idx])
        e = int(self.offsets[t_idx + 1])
        if s >= e:
            return out, mask

        ys = self.y[s:e]
        xs = self.x[s:e]
        in_box = (ys >= y0) & (ys < y0 + patch_size) & (xs >= x0) & (xs < x0 + patch_size)
        if not np.any(in_box):
            return out, mask

        ly = ys[in_box] - y0
        lx = xs[in_box] - x0
        v = self.vals[s:e][in_box]  # (K, 6)
        fin = np.isfinite(v)
        v_norm = np.where(fin, np.clip((v - METAR_MEAN) / METAR_STD, -6.0, 10.0), 0.0)

        for c in range(6):
            mc = fin[:, c]
            if np.any(mc):
                out[c, ly[mc], lx[mc]] = v_norm[mc, c]
                mask[c, ly[mc], lx[mc]] = True
        return out, mask


class RegionalFlashMultimodalDataset:
    """On-the-fly 2-frame `128x128` multimodal dataset (`CERRA + SatWeave + OPERA + METAR`)."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        patch_size: int = config.PATCH_SIZE,
        tile_stride: int = config.TILE_STRIDE,
        random_crop: bool | None = None,
        samples_per_epoch: int | None = None,
        seed: int = 42,
        shard_reuse: int = 1,
    ) -> None:
        root = Path(data_dir)
        self.patch_size = patch_size
        self.random_crop = (split == "train") if random_crop is None else random_crop
        self.samples_per_epoch = samples_per_epoch
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.shard_reuse = max(1, int(shard_reuse))
        self._shard_rem = 0
        self._shard_anchor_lo = 0
        self._shard_anchor_hi = 1

        # 1. Open all available half-year CERRA stores
        self.cerra_arrays: dict[str, zarr.Array] = {}
        for zp in sorted(root.glob("cerra_5km_*_H*.zarr")):
            per = zp.stem.replace("cerra_5km_", "")
            self.cerra_arrays[per] = zarr.open_group(str(zp), mode="r")["X"]
        if not self.cerra_arrays:
            raise FileNotFoundError(f"No cerra_5km_*_H*.zarr stores found in {root}")

        sample_c = next(iter(self.cerra_arrays.values()))
        _, self.ny, self.nx, self.n_cerra = sample_c.shape

        # 2. Open all available half-year SatWeave stores (optional fallback to zeros if period absent)
        self.sat_arrays: dict[str, zarr.Array] = {}
        for zp in sorted(root.glob("sat_5km_*_plus2h.zarr")):
            per = zp.stem.replace("sat_5km_", "").replace("_plus2h", "")
            self.sat_arrays[per] = zarr.open_group(str(zp), mode="r")["X"]

        # 3. Open OPERA radar Zarr (single 5-year store)
        radar_path = root / "radar_5km_plus2h.zarr"
        self.radar_arr = zarr.open_group(str(radar_path), mode="r")["X"] if radar_path.exists() else None

        # 4. Preload static grid & lat/lon (~36 MB total)
        s_root = zarr.open_group(str(root / "static_5km.zarr"), mode="r")
        static_raw = np.asarray(s_root["static"], dtype=np.float32)
        self.lat = np.asarray(s_root["latitude"], dtype=np.float32)
        self.lon = np.asarray(s_root["longitude"], dtype=np.float32)

        # 4b. Optional ESA WorldCover fractional land-cover + coast-distance (~32 MB, input-only)
        lc_path = root / "landcover_5km.zarr"
        self.landcover: np.ndarray | None = (
            np.asarray(zarr.open_group(str(lc_path), mode="r")["landcover"], dtype=np.float32)
            if lc_path.exists() else None
        )

        stats = json.loads((root / "norm_stats.json").read_text())
        self.cerra_mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
        self.cerra_std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
        self.cerra_delta_std = np.asarray(stats["delta_std_3h"], dtype=np.float32)

        s_mean = np.asarray(stats["static_mean"], dtype=np.float32)[:, None, None]
        s_std = np.asarray(stats["static_std"], dtype=np.float32)[:, None, None]
        self.static = (static_raw - s_mean) / s_std

        # 5. Timestamps & valid 2-frame anchors `(t0, t0 + 1)`
        ts_df = pd.read_parquet(root / "timestamps.parquet")
        ts_df["timestamp_utc"] = pd.to_datetime(ts_df["timestamp_utc"], utc=True)
        half_col = np.where(ts_df["month"] <= 6, "H1", "H2")
        ts_df["period"] = ts_df["year"].astype(str) + "_" + half_col
        ts_df["local_idx"] = ts_df.groupby("period").cumcount()

        self.periods_by_t = ts_df["period"].to_numpy()
        self.local_idx_by_t = ts_df["local_idx"].to_numpy(dtype=np.int64)
        self.n_times = len(ts_df)

        hour_rad = 2.0 * np.pi * ts_df["hour"].to_numpy(dtype=np.float32) / 24.0
        doy_rad = 2.0 * np.pi * (ts_df["timestamp_utc"].dt.dayofyear.to_numpy(dtype=np.float32) - 1.0) / 365.25
        self.time_feats = np.stack(
            [np.sin(hour_rad), np.cos(hour_rad), np.sin(doy_rad), np.cos(doy_rad)],
            axis=-1,
        ).astype(np.float32)

        # Precompute Unix seconds for sensor context (t0+2h), sensor target (t0+5h), and UTC noon
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        ts_ctx_sensor = ts_df["timestamp_utc"] + pd.Timedelta(hours=2)
        ts_tgt_sensor = ts_df["timestamp_utc"] + pd.Timedelta(hours=5)
        ts_noon = ts_ctx_sensor.dt.normalize() + pd.Timedelta(hours=12)
        self.ctx_unix_s = (ts_ctx_sensor - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
        self.tgt_unix_s = (ts_tgt_sensor - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
        self.noon_unix_s = (ts_noon - epoch).dt.total_seconds().to_numpy(dtype=np.float64)

        # 6. Sparse METAR CSR index
        metar_path = root / "metar_obs_plus2h.parquet"
        self.metar_idx = _MetarIndex(metar_path, self.n_times) if metar_path.exists() else None

        split_mask = (
            (ts_df["split_time"] == split).to_numpy()
            if split in ("train", "val", "test")
            else np.ones(self.n_times, dtype=bool)
        )
        avail_mask = ts_df["period"].isin(self.cerra_arrays.keys()).to_numpy()
        valid = split_mask & avail_mask
        self.anchors = np.asarray(
            [t0 for t0 in range(self.n_times - 1) if valid[t0] and valid[t0 + 1]],
            dtype=np.int64,
        )

        # Deterministic tile coordinates for validation
        ys = list(range(0, self.ny - patch_size + 1, tile_stride))
        if ys[-1] != self.ny - patch_size:
            ys.append(self.ny - patch_size)
        xs = list(range(0, self.nx - patch_size + 1, tile_stride))
        if xs[-1] != self.nx - patch_size:
            xs.append(self.nx - patch_size)
        self.tile_coords = [(y0, x0) for y0 in ys for x0 in xs]

    def _read_cerra_pair(self, t0: int, y0: int, x0: int) -> tuple[np.ndarray, np.ndarray]:
        y1, x1 = y0 + self.patch_size, x0 + self.patch_size
        p0, p1 = str(self.periods_by_t[t0]), str(self.periods_by_t[t0 + 1])
        l0, l1 = int(self.local_idx_by_t[t0]), int(self.local_idx_by_t[t0 + 1])
        if p0 == p1:
            seq = np.asarray(self.cerra_arrays[p0][l0 : l0 + 2, y0:y1, x0:x1, :], dtype=np.float32)
            c0, c1 = seq[0], seq[1]
        else:
            c0 = np.asarray(self.cerra_arrays[p0][l0, y0:y1, x0:x1, :], dtype=np.float32)
            c1 = np.asarray(self.cerra_arrays[p1][l1, y0:y1, x0:x1, :], dtype=np.float32)
        c0 = (np.transpose(c0, (2, 0, 1)) - self.cerra_mean) / self.cerra_std
        c1 = (np.transpose(c1, (2, 0, 1)) - self.cerra_mean) / self.cerra_std
        return np.clip(c0, -8.0, 8.0), np.clip(c1, -8.0, 8.0)

    def _read_sat_frame(self, t_idx: int, y0: int, x0: int) -> tuple[np.ndarray, np.ndarray]:
        P = self.patch_size
        per = str(self.periods_by_t[t_idx])
        if per not in self.sat_arrays:
            return np.zeros((11, P, P), dtype=np.float32), np.zeros((1, P, P), dtype=np.float32)
        li = int(self.local_idx_by_t[t_idx])
        raw = np.asarray(self.sat_arrays[per][li, y0 : y0 + P, x0 : x0 + P, :], dtype=np.float32)
        vals = np.transpose(raw[..., :11], (2, 0, 1))
        mask = (raw[..., 11:12] > 0.5).transpose(2, 0, 1).astype(np.float32)
        norm = (vals - SAT_MEAN[:, None, None]) / SAT_STD[:, None, None]
        norm = np.where(mask > 0.5, np.clip(norm, -5.0, 5.0), 0.0)
        return norm, mask

    def _read_radar_pair(self, t0: int, y0: int, x0: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        P = self.patch_size
        if self.radar_arr is None:
            z = np.zeros((1, P, P), dtype=np.float32)
            return z, z, z, z
        seq = np.asarray(self.radar_arr[t0 : t0 + 2, y0 : y0 + P, x0 : x0 + P, :], dtype=np.float32)
        m0 = (seq[0, ..., 1:2] > 0.5).transpose(2, 0, 1).astype(np.float32)
        m1 = (seq[1, ..., 1:2] > 0.5).transpose(2, 0, 1).astype(np.float32)
        r0 = np.where(m0 > 0.5, np.clip(seq[0, ..., 0:1].transpose(2, 0, 1) / RADAR_SCALE, 0.0, 4.0), 0.0)
        r1 = np.where(m1 > 0.5, np.clip(seq[1, ..., 0:1].transpose(2, 0, 1) / RADAR_SCALE, 0.0, 4.0), 0.0)
        return r0, m0, r1, m1

    def __len__(self) -> int:
        if self.random_crop:
            return self.samples_per_epoch if self.samples_per_epoch is not None else len(self.anchors)
        return len(self.anchors) * len(self.tile_coords)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        P = self.patch_size
        if self.random_crop:
            if self.shard_reuse > 1:
                if self._shard_rem <= 0:
                    # Pick a random 16-step Zarr temporal shard block and reuse it `shard_reuse` times
                    # (massive OS page-cache hit rate for 256x256 crops while remaining uniform over the epoch)
                    n_anc = len(self.anchors)
                    ai = int(self.rng.integers(0, n_anc))
                    lo = (ai // 16) * 16
                    hi = min(n_anc, lo + 16)
                    self._shard_anchor_lo = lo
                    self._shard_anchor_hi = hi
                    self._shard_rem = self.shard_reuse
                self._shard_rem -= 1
                t0 = int(self.anchors[int(self.rng.integers(self._shard_anchor_lo, self._shard_anchor_hi))])
            else:
                t0 = int(self.anchors[int(self.rng.integers(0, len(self.anchors)))])
            y0 = int(self.rng.integers(0, self.ny - P + 1))
            x0 = int(self.rng.integers(0, self.nx - P + 1))
        else:
            n_tiles = len(self.tile_coords)
            t0 = int(self.anchors[idx // n_tiles])
            y0, x0 = self.tile_coords[idx % n_tiles]

        c0, c1 = self._read_cerra_pair(t0, y0, x0)
        s0, sm0 = self._read_sat_frame(t0, y0, x0)
        s1, sm1 = self._read_sat_frame(t0 + 1, y0, x0)
        r0, rm0, r1, rm1 = self._read_radar_pair(t0, y0, x0)

        if self.metar_idx is not None:
            m0, mm0 = self.metar_idx.rasterize_patch(t0, y0, x0, P)
            m1, mm1 = self.metar_idx.rasterize_patch(t0 + 1, y0, x0, P)
        else:
            m0 = np.zeros((6, P, P), dtype=np.float32)
            mm0 = np.zeros((6, P, P), dtype=bool)
            m1, mm1 = m0, mm0

        grid_ctx = np.concatenate([c0, s0, r0], axis=0)  # (69, P, P)
        grid_tgt = np.concatenate([c1, s1, r1], axis=0)  # (69, P, P)
        static_patch = self.static[:, y0 : y0 + P, x0 : x0 + P]

        cy, cx = y0 + P // 2, x0 + P // 2
        c_lat = float(self.lat[cy, cx])
        c_lon = float(self.lon[cy, cx])
        lat_norm = np.float32((c_lat - 48.0) / 15.0)
        lon_norm = np.float32((c_lon - 10.0) / 25.0)
        solar_8d = solar_step_features(
            self.ctx_unix_s[t0],
            self.tgt_unix_s[t0],
            self.noon_unix_s[t0],
            c_lon,
            c_lat,
        )
        global_cond = np.concatenate(
            [self.time_feats[t0], np.array([lat_norm, lon_norm], dtype=np.float32), solar_8d]
        )  # (14,) float32

        sample = {
            "grid_ctx": grid_ctx,
            "grid_tgt": grid_tgt,
            "metar_ctx": m0,
            "metar_tgt": m1,
            "sat_mask_ctx": sm0,
            "sat_mask_tgt": sm1,
            "radar_mask_ctx": rm0,
            "radar_mask_tgt": rm1,
            "metar_mask_ctx": mm0,
            "metar_mask_tgt": mm1,
            "static": static_patch,
            "global_cond": global_cond,
            "time_idx": np.int64(t0),
            "yx": np.asarray([y0, x0], dtype=np.int64),
        }
        if self.landcover is not None:
            sample["landcover"] = self.landcover[:, y0 : y0 + P, x0 : x0 + P]
        return sample


def seed_worker(worker_id: int) -> None:
    """Give each PyTorch DataLoader worker an independent NumPy RNG stream."""
    import torch

    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        base_seed = int(getattr(info.dataset, "seed", 42))
        info.dataset.rng = np.random.default_rng(np.random.SeedSequence([base_seed, int(worker_id)]))


class RegionalFlashSequenceDataset(RegionalFlashMultimodalDataset):
    """Multi-step `(max_steps + 1)`-frame sequence dataset for STRATA-style AR fine-tuning.

    Reads `L = max_steps + 1` consecutive 3-hourly frames (`t0 .. t0 + max_steps`) for a
    `128x128` patch in a single contiguous Zarr slice whenever all frames lie in the same
    half-year store (almost identical I/O cost to 2-frame reads!).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        max_steps: int = 4,
        patch_size: int = config.PATCH_SIZE,
        tile_stride: int = config.TILE_STRIDE,
        random_crop: bool | None = None,
        samples_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            split=split,
            patch_size=patch_size,
            tile_stride=tile_stride,
            random_crop=random_crop,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
        )
        self.max_steps = int(max_steps)
        # Filter anchors so all frames t0 .. t0 + max_steps are valid & in-split
        valid_set = set(int(a) for a in self.anchors)
        self.anchors = np.asarray(
            [t0 for t0 in self.anchors if all((int(t0) + k) in valid_set for k in range(self.max_steps))],
            dtype=np.int64,
        )

    def _read_cerra_seq(self, t0: int, L: int, y0: int, x0: int) -> np.ndarray:
        y1, x1 = y0 + self.patch_size, x0 + self.patch_size
        p_first = str(self.periods_by_t[t0])
        p_last = str(self.periods_by_t[t0 + L - 1])
        if p_first == p_last:
            l0 = int(self.local_idx_by_t[t0])
            raw = np.asarray(self.cerra_arrays[p_first][l0 : l0 + L, y0:y1, x0:x1, :], dtype=np.float32)
        else:
            frames = [
                np.asarray(
                    self.cerra_arrays[str(self.periods_by_t[t0 + k])][int(self.local_idx_by_t[t0 + k]), y0:y1, x0:x1, :],
                    dtype=np.float32,
                )
                for k in range(L)
            ]
            raw = np.stack(frames, axis=0)
        # (L, H, W, 57) -> (L, 57, H, W)
        c = (np.transpose(raw, (0, 3, 1, 2)) - self.cerra_mean[None]) / self.cerra_std[None]
        return np.clip(c, -8.0, 8.0)

    def _read_sat_seq(self, t0: int, L: int, y0: int, x0: int) -> tuple[np.ndarray, np.ndarray]:
        P = self.patch_size
        p_first = str(self.periods_by_t[t0])
        p_last = str(self.periods_by_t[t0 + L - 1])
        if p_first == p_last and p_first in self.sat_arrays:
            l0 = int(self.local_idx_by_t[t0])
            raw = np.asarray(self.sat_arrays[p_first][l0 : l0 + L, y0 : y0 + P, x0 : x0 + P, :], dtype=np.float32)
            vals = np.transpose(raw[..., :11], (0, 3, 1, 2))
            mask = (raw[..., 11:12] > 0.5).transpose(0, 3, 1, 2).astype(np.float32)
            norm = (vals - SAT_MEAN[None, :, None, None]) / SAT_STD[None, :, None, None]
            norm = np.where(mask > 0.5, np.clip(norm, -5.0, 5.0), 0.0)
            return norm, mask
        s_list, m_list = [], []
        for k in range(L):
            sk, mk = self._read_sat_frame(t0 + k, y0, x0)
            s_list.append(sk)
            m_list.append(mk)
        return np.stack(s_list, axis=0), np.stack(m_list, axis=0)

    def _read_radar_seq(self, t0: int, L: int, y0: int, x0: int) -> tuple[np.ndarray, np.ndarray]:
        P = self.patch_size
        if self.radar_arr is None:
            z = np.zeros((L, 1, P, P), dtype=np.float32)
            return z, z
        raw = np.asarray(self.radar_arr[t0 : t0 + L, y0 : y0 + P, x0 : x0 + P, :], dtype=np.float32)
        mask = (raw[..., 1:2] > 0.5).transpose(0, 3, 1, 2).astype(np.float32)
        rad = np.where(mask > 0.5, np.clip(raw[..., 0:1].transpose(0, 3, 1, 2) / RADAR_SCALE, 0.0, 4.0), 0.0)
        return rad, mask

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        P = self.patch_size
        L = self.max_steps + 1
        if self.random_crop:
            t0 = int(self.anchors[int(self.rng.integers(0, len(self.anchors)))])
            y0 = int(self.rng.integers(0, self.ny - P + 1))
            x0 = int(self.rng.integers(0, self.nx - P + 1))
        else:
            n_tiles = len(self.tile_coords)
            t0 = int(self.anchors[idx // n_tiles])
            y0, x0 = self.tile_coords[idx % n_tiles]

        c_seq = self._read_cerra_seq(t0, L, y0, x0)            # (L, 57, P, P)
        s_seq, sm_seq = self._read_sat_seq(t0, L, y0, x0)      # (L, 11, P, P), (L, 1, P, P)
        r_seq, rm_seq = self._read_radar_seq(t0, L, y0, x0)    # (L, 1, P, P), (L, 1, P, P)
        grid_seq = np.concatenate([c_seq, s_seq, r_seq], axis=1)  # (L, 69, P, P)

        if self.metar_idx is not None:
            m_list, mm_list = [], []
            for k in range(L):
                mk, mmk = self.metar_idx.rasterize_patch(t0 + k, y0, x0, P)
                m_list.append(mk)
                mm_list.append(mmk)
            metar_seq = np.stack(m_list, axis=0)       # (L, 6, P, P)
            metar_mask_seq = np.stack(mm_list, axis=0)  # (L, 6, P, P) bool
        else:
            metar_seq = np.zeros((L, 6, P, P), dtype=np.float32)
            metar_mask_seq = np.zeros((L, 6, P, P), dtype=bool)

        static_patch = self.static[:, y0 : y0 + P, x0 : x0 + P]
        cy, cx = y0 + P // 2, x0 + P // 2
        c_lat = float(self.lat[cy, cx])
        c_lon = float(self.lon[cy, cx])
        lat_norm = np.float32((c_lat - 48.0) / 15.0)
        lon_norm = np.float32((c_lon - 10.0) / 25.0)

        # Global conditioning for each transition step k = 0 .. max_steps - 1 (t0+k -> t0+k+1)
        g_cond_seq = np.stack(
            [
                np.concatenate([
                    self.time_feats[t0 + k],
                    np.array([lat_norm, lon_norm], dtype=np.float32),
                    solar_step_features(
                        self.ctx_unix_s[t0 + k],
                        self.tgt_unix_s[t0 + k],
                        self.noon_unix_s[t0 + k],
                        c_lon,
                        c_lat,
                    ),
                ])
                for k in range(self.max_steps)
            ],
            axis=0,
        )  # (max_steps, 14)

        sample = {
            "grid_seq": grid_seq,
            "metar_seq": metar_seq,
            "sat_mask_seq": sm_seq,
            "radar_mask_seq": rm_seq,
            "metar_mask_seq": metar_mask_seq,
            "static": static_patch,
            "global_cond_seq": g_cond_seq,
            "time_idx": np.int64(t0),
            "yx": np.asarray([y0, x0], dtype=np.int64),
        }
        if self.landcover is not None:
            sample["landcover"] = self.landcover[:, y0 : y0 + P, x0 : x0 + P]
        return sample
