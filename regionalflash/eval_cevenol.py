"""Autoregressive Multi-Diffusion rollout & visualization for an Épisode Cévenol.

Supports both regional crops (e.g. 384x384) and the FULL 1069x1069 pan-European grid!
Compares predicted vs observed across +3h .. +24h:
  - OPERA Radar reflectivity (dBZ) + Cévennes CSI at 20 / 35 dBZ
  - SatWeave IR 11.2 um cloud-top signal
  - CERRA 2m temperature (°C)
  - METAR airport 2m temperature (°C)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from regionalflash import config
from regionalflash.dataset import (
    METAR_STD,
    RADAR_SCALE,
    SAT_MEAN,
    SAT_STD,
    RegionalFlashMultimodalDataset,
)
from regionalflash.flow import project_ar_feedback, sample_tiled_europe
from regionalflash.model import RegionalFlashJiT3D

log = logging.getLogger(__name__)


def _read_region_frame(
    ds: RegionalFlashMultimodalDataset,
    t_idx: int,
    y0: int,
    x0: int,
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read normalized (grid, metar, sat_mask, radar_mask, metar_mask) for a (H, W) region at t_idx."""
    per = str(ds.periods_by_t[t_idx])
    li = int(ds.local_idx_by_t[t_idx])

    raw_c = np.asarray(ds.cerra_arrays[per][li, y0 : y0 + H, x0 : x0 + W, :], dtype=np.float32)
    c = np.clip((np.transpose(raw_c, (2, 0, 1)) - ds.cerra_mean) / ds.cerra_std, -8.0, 8.0)

    if per in ds.sat_arrays:
        raw_s = np.asarray(ds.sat_arrays[per][li, y0 : y0 + H, x0 : x0 + W, :], dtype=np.float32)
        vals_s = np.transpose(raw_s[..., :11], (2, 0, 1))
        sm = (raw_s[..., 11:12] > 0.5).transpose(2, 0, 1).astype(np.float32)
        s = np.where(sm > 0.5, np.clip((vals_s - SAT_MEAN[:, None, None]) / SAT_STD[:, None, None], -5.0, 5.0), 0.0)
    else:
        s = np.zeros((11, H, W), dtype=np.float32)
        sm = np.zeros((1, H, W), dtype=np.float32)

    if ds.radar_arr is not None:
        raw_r = np.asarray(ds.radar_arr[t_idx, y0 : y0 + H, x0 : x0 + W, :], dtype=np.float32)
        rm = (raw_r[..., 1:2] > 0.5).transpose(2, 0, 1).astype(np.float32)
        r = np.where(rm > 0.5, np.clip(raw_r[..., 0:1].transpose(2, 0, 1) / RADAR_SCALE, 0.0, 4.0), 0.0)
    else:
        r = np.zeros((1, H, W), dtype=np.float32)
        rm = np.zeros((1, H, W), dtype=np.float32)

    if ds.metar_idx is not None:
        m, mm = ds.metar_idx.rasterize_patch(t_idx, y0, x0, H)
    else:
        m = np.zeros((6, H, W), dtype=np.float32)
        mm = np.zeros((6, H, W), dtype=bool)

    grid = np.concatenate([c, s, r], axis=0).astype(np.float32)
    return grid, m.astype(np.float32), sm.astype(np.float32), rm.astype(np.float32), mm.astype(np.float32)


def _csi(pred_dbz: np.ndarray, true_dbz: np.ndarray, thresh: float) -> float:
    p = pred_dbz >= thresh
    t = true_dbz >= thresh
    hits = int((p & t).sum())
    denom = int((p | t).sum())
    return hits / denom if denom > 0 else float("nan")


def _plot_grid(
    out_png: Path,
    title: str,
    gt_grids: list[np.ndarray],
    pred_grids: np.ndarray,
    gt_s_masks: list[np.ndarray],
    lsm: np.ndarray,
    orog: np.ndarray,
    ts_df: pd.DataFrame,
    start_idx: int,
    show_steps: list[int],
    yslice: slice,
    xslice: slice,
    rad_i: int,
    ir11_i: int,
) -> None:
    ir_mean, ir_std = float(SAT_MEAN[8]), float(SAT_STD[8])
    h_span = (yslice.stop or lsm.shape[0]) - (yslice.start or 0)
    pstep = 2 if h_span > 512 else 1
    ys = slice(yslice.start, yslice.stop, pstep)
    xs = slice(xslice.start, xslice.stop, pstep)
    lsm_z = lsm[ys, xs]
    orog_z = orog[ys, xs]

    # Auto-calibrate IR11 colormap from observed valid pixels
    ref_ir = (gt_grids[1][ir11_i] * ir_std + ir_mean)[ys, xs]
    ref_sm = gt_s_masks[1][0][ys, xs] > 0.5
    if ref_sm.any():
        ir_vmin, ir_vmax = np.percentile(ref_ir[ref_sm], [1, 99])
    else:
        ir_vmin, ir_vmax = float(ref_ir.min()), float(ref_ir.max())

    fig, axes = plt.subplots(4, len(show_steps), figsize=(3.8 * len(show_steps), 14.0), constrained_layout=True)
    for col, step in enumerate(show_steps):
        lead_h = 3 * (step + 1)
        t_str = str(ts_df.loc[start_idx + step + 1, "timestamp_utc"])[5:16]
        gt_dbz = (gt_grids[step + 1][rad_i] * RADAR_SCALE)[ys, xs]
        pr_dbz = (pred_grids[step, rad_i] * RADAR_SCALE)[ys, xs]
        gt_ir = (gt_grids[step + 1][ir11_i] * ir_std + ir_mean)[ys, xs]
        pr_ir = (pred_grids[step, ir11_i] * ir_std + ir_mean)[ys, xs]

        for r_i, (arr, stitle) in enumerate([
            (gt_dbz, f"+{lead_h}h ({t_str})\nOBS Radar (dBZ)"),
            (pr_dbz, "Forecast Radar (dBZ)"),
        ]):
            axes[r_i, col].set_facecolor("#11161f")
            im_rad = axes[r_i, col].imshow(np.ma.masked_less(arr, 5.0), origin="lower", cmap="turbo", vmin=5, vmax=55)
            axes[r_i, col].contour(lsm_z, levels=[0.5], colors=["white"], linewidths=0.55, alpha=0.8)
            axes[r_i, col].contour(orog_z, levels=[0.8, 1.8], colors=["#cccccc"], linewidths=0.35, alpha=0.4)
            axes[r_i, col].set_title(stitle, fontsize=10)

        for r_i, (arr, stitle) in enumerate([
            (gt_ir, "OBS SatWeave IR 11.2µm"),
            (pr_ir, "Forecast SatWeave IR 11.2µm"),
        ], start=2):
            axes[r_i, col].imshow(arr, origin="lower", cmap="gray_r", vmin=ir_vmin, vmax=ir_vmax)
            axes[r_i, col].contour(lsm_z, levels=[0.5], colors=["cyan"], linewidths=0.5, alpha=0.7)
            axes[r_i, col].set_title(stitle, fontsize=10)

        for r_i in range(4):
            axes[r_i, col].set_xticks([])
            axes[r_i, col].set_yticks([])

    fig.colorbar(im_rad, ax=axes[:2, :], shrink=0.7, label="OPERA Radar Reflectivity (dBZ)")
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info("Saved figure -> %s", out_png)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Evaluate RegionalFlash on an Épisode Cévenol")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--out-dir", type=Path, default=Path("eval_cevenol"))
    p.add_argument("--start-idx", type=int, default=12254, help="Starting time_idx (default: 2022-03-12 18:00 UTC)")
    p.add_argument("--lead-steps", type=int, default=8, help="Number of 3h autoregressive steps (8 = 24h)")
    p.add_argument("--n-members", type=int, default=1, help="Ensemble members")
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--domain-size", type=int, default=1069, help="Domain size (1069 = full Europe!)")
    p.add_argument("--stride", type=int, default=0, help="Tile stride (0 = auto 3/4 of model img_size)")
    p.add_argument("--tile-batch-size", type=int, default=16)
    p.add_argument("--deterministic", action="store_true", help="1-step STRATA deterministic mode (t=1.0, xt=0)")
    p.add_argument("--split", default="val", choices=["val", "test"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    c_args = ckpt.get("args", {})
    ema_sd = ckpt["ema"]
    lc_w = ema_sd.get("landcover_embed.soft_emb.0.weight")
    c_landcover = int(lc_w.shape[1]) if lc_w is not None else 0
    tile_size = int(c_args.get("img_size", config.PATCH_SIZE))
    stride = args.stride if args.stride > 0 else (3 * tile_size) // 4
    model = RegionalFlashJiT3D(
        img_size=(tile_size, tile_size),
        patch_size=(1, c_args.get("patch_xy", 8), c_args.get("patch_xy", 8)),
        embed_dim=c_args.get("embed_dim", 1024),
        depth=c_args.get("depth", 12),
        num_heads=c_args.get("num_heads", 16),
        kv_ctx_noise=0.0,
        c_landcover=c_landcover,
    ).to(device)
    model.load_state_dict(ema_sd)
    model.eval()
    log.info(
        "Loaded EMA checkpoint %s (epoch=%s, best_val=%.4f, tile=%d, stride=%d, landcover=%d)",
        args.ckpt, ckpt.get("epoch"), ckpt.get("best_val", float("nan")), tile_size, stride, c_landcover,
    )

    ds = RegionalFlashMultimodalDataset(args.data_dir, split=args.split, random_crop=False)
    if c_landcover > 0 and ds.landcover is None:
        raise FileNotFoundError(f"Checkpoint expects c_landcover={c_landcover}, but landcover_5km.zarr was not found in {args.data_dir}")
    ts_df = pd.read_parquet(Path(args.data_dir) / "timestamps.parquet")

    cy, cx = 418, 474
    S = min(args.domain_size, ds.ny, ds.nx)
    y0 = 0 if S == ds.ny else max(0, min(ds.ny - S, cy - S // 2))
    x0 = 0 if S == ds.nx else max(0, min(ds.nx - S, cx - S // 2))
    log.info(
        "Rollout domain: y=[%d:%d], x=[%d:%d] (%dx%d = ~%dkm square)",
        y0, y0 + S, x0, x0 + S, S, S, int(S * 5.5),
    )

    static_sub = torch.from_numpy(ds.static[:, y0 : y0 + S, x0 : x0 + S]).to(device)
    landcover_sub = (
        torch.from_numpy(ds.landcover[:, y0 : y0 + S, x0 : x0 + S]).to(device)
        if c_landcover > 0 and ds.landcover is not None
        else None
    )
    lat_sub = torch.from_numpy(ds.lat[y0 : y0 + S, x0 : x0 + S]).to(device)
    lon_sub = torch.from_numpy(ds.lon[y0 : y0 + S, x0 : x0 + S]).to(device)
    lsm_sub = ds.static[1, y0 : y0 + S, x0 : x0 + S]
    orog_sub = ds.static[0, y0 : y0 + S, x0 : x0 + S]

    lat_np = ds.lat[y0 : y0 + S, x0 : x0 + S]
    lon_np = ds.lon[y0 : y0 + S, x0 : x0 + S]
    cev_mask = (lat_np >= 43.2) & (lat_np <= 45.0) & (lon_np >= 2.8) & (lon_np <= 5.0)

    gt_grids, gt_metars, gt_s_masks, gt_r_masks, gt_m_masks = [], [], [], [], []
    for k in range(args.lead_steps + 1):
        g, m, sm, rm, mm = _read_region_frame(ds, args.start_idx + k, y0, x0, S, S)
        gt_grids.append(g)
        gt_metars.append(m)
        gt_s_masks.append(sm)
        gt_r_masks.append(rm)
        gt_m_masks.append(mm)

    t0_str = str(ts_df.loc[args.start_idx, "timestamp_utc"])
    log.info(
        "Initial condition t0=%d (%s), rolling out %d steps (%dh) with %d member(s)...",
        args.start_idx, t0_str, args.lead_steps, 3 * args.lead_steps, args.n_members,
    )

    ens_grids = np.zeros((args.n_members, args.lead_steps, 69, S, S), dtype=np.float32)
    ens_metars = np.zeros((args.n_members, args.lead_steps, 6, S, S), dtype=np.float32)

    for mem in range(args.n_members):
        cur_g = torch.from_numpy(gt_grids[0]).to(device)
        cur_m = torch.from_numpy(gt_metars[0]).to(device)
        cur_mm = torch.from_numpy(gt_m_masks[0]).to(device)
        for step in range(args.lead_steps):
            t_cur = args.start_idx + step
            sm_t = torch.from_numpy(gt_s_masks[step]).to(device)
            rm_t = torch.from_numpy(gt_r_masks[step]).to(device)
            sm_next = torch.from_numpy(gt_s_masks[step + 1]).to(device)
            rm_next = torch.from_numpy(gt_r_masks[step + 1]).to(device)
            tf = torch.from_numpy(ds.time_feats[t_cur]).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred_g, pred_m = sample_tiled_europe(
                    model,
                    cur_g,
                    cur_m,
                    sm_t,
                    rm_t,
                    cur_mm,
                    static_sub,
                    tf,
                    lat_sub,
                    lon_sub,
                    ctx_unix_s=float(ds.ctx_unix_s[t_cur]),
                    tgt_unix_s=float(ds.tgt_unix_s[t_cur]),
                    noon_unix_s=float(ds.noon_unix_s[t_cur]),
                    patch_size=tile_size,
                    stride=stride,
                    n_steps=args.ode_steps,
                    tile_batch_size=args.tile_batch_size,
                    use_residual=True,
                    seed=1000 + mem * 100 + step,
                    landcover_full=landcover_sub,
                    sat_mask_tgt_full=sm_next,
                    radar_mask_tgt_full=rm_next,
                    deterministic=args.deterministic,
                )
            pred_g_f = pred_g.float()
            pred_m_f = pred_m.float()
            ens_grids[mem, step] = pred_g_f.cpu().numpy()
            ens_metars[mem, step] = pred_m_f.cpu().numpy()
            # On-manifold AR feedback (ported from FlashEdges):
            # clamp CERRA/Sat/Radar/METAR, zero off-coverage sat & radar, and
            # re-sparsify METAR back to station pixels so steps 2+ stay on-manifold.
            cur_g, cur_m = project_ar_feedback(pred_g_f, pred_m_f, sm_next, rm_next, cur_mm)
            log.info("  member %d/%d | step %d/%d (+%02dh) done", mem + 1, args.n_members, step + 1, args.lead_steps, 3 * (step + 1))

    t2m_i = config.DYN_CHANNELS.index("t2m")
    t2m_mean, t2m_std = float(ds.cerra_mean[t2m_i, 0, 0]), float(ds.cerra_std[t2m_i, 0, 0])
    ir11_i = 57 + 8
    rad_i = 68
    met_t2m_std = float(METAR_STD[0])

    mean_grids = ens_grids.mean(axis=0)
    mean_metars = ens_metars.mean(axis=0)

    log.info("=== 24h Full-Coverage Verification (Domain %dx%d & Cévennes Box) ===", S, S)
    for step in range(args.lead_steps):
        lead_h = 3 * (step + 1)
        gt_g = gt_grids[step + 1]
        rm = gt_r_masks[step + 1][0] > 0.5

        gt_t2m = gt_g[t2m_i] * t2m_std + t2m_mean
        pr_t2m = mean_grids[step, t2m_i] * t2m_std + t2m_mean

        gt_dbz = gt_g[rad_i] * RADAR_SCALE
        pr_dbz = ens_grids[0, step, rad_i] * RADAR_SCALE

        t2m_mae = float(np.abs(pr_t2m - gt_t2m).mean())
        cev_valid = cev_mask & rm
        csi20 = _csi(pr_dbz[cev_valid], gt_dbz[cev_valid], 20.0)
        csi35 = _csi(pr_dbz[cev_valid], gt_dbz[cev_valid], 35.0)
        gt_max_cev = float(gt_dbz[cev_valid].max())
        pr_max_cev = float(pr_dbz[cev_valid].max())

        mm = gt_m_masks[step + 1][0] > 0.5
        met_mae = (
            float(np.abs(mean_metars[step, 0][mm] - gt_metars[step + 1][0][mm]).mean()) * met_t2m_std
            if mm.any() else float("nan")
        )
        t_str = str(ts_df.loc[args.start_idx + step + 1, "timestamp_utc"])[:16]
        log.info(
            "+%02dh (%s) | Pan-EU T2m MAE=%.2f°C | Pan-EU METAR MAE=%.2f°C | "
            "Cévennes Radar: max(gt=%.1f, pred=%.1f) dBZ | CSI20=%.2f CSI35=%.2f",
            lead_h, t_str, t2m_mae, met_mae, gt_max_cev, pr_max_cev, csi20, csi35,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    show_steps = [s for s in [0, 1, 3, 5, 7] if s < args.lead_steps]

    # 1. Full-domain pan-European view
    _plot_grid(
        args.out_dir / f"cevenol_full_{S}_t{args.start_idx}.png",
        f"RegionalFlash 171M — Full Pan-European {S}x{S} Multi-Diffusion Rollout (init {t0_str[:16]} UTC)",
        gt_grids, ens_grids[0], gt_s_masks, lsm_sub, orog_sub, ts_df, args.start_idx, show_steps,
        slice(0, S), slice(0, S), rad_i, ir11_i,
    )

    # 2. Zoomed Cévennes / France view from the exact same full-Europe rollout
    if S >= 512:
        zy0, zy1 = max(0, (cy - y0) - 128), min(S, (cy - y0) + 128)
        zx0, zx1 = max(0, (cx - x0) - 128), min(S, (cx - x0) + 128)
        _plot_grid(
            args.out_dir / f"cevenol_full_{S}_zoom_t{args.start_idx}.png",
            f"RegionalFlash 171M — Cévennes Zoom from Full {S}x{S} Europe Rollout (init {t0_str[:16]} UTC)",
            gt_grids, ens_grids[0], gt_s_masks, lsm_sub, orog_sub, ts_df, args.start_idx, show_steps,
            slice(zy0, zy1), slice(zx0, zx1), rad_i, ir11_i,
        )


if __name__ == "__main__":
    main()
