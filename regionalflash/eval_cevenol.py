"""Autoregressive Multi-Diffusion rollout & visualization for an Épisode Cévenol.

Runs a multi-step (e.g. 24h = 8 x 3h) tiled Multi-Diffusion forecast over a
384x384 (~2100 km x 2100 km) domain centered on the Cévennes / Western Mediterranean,
comparing predicted vs observed:
  - OPERA Radar reflectivity (dBZ) + Cévennes CSI at 20 / 35 dBZ
  - SatWeave IR 11.2 um cloud-top brightness temperature (K)
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
from regionalflash.flow import sample_tiled_europe
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

    # CERRA (57, H, W)
    raw_c = np.asarray(ds.cerra_arrays[per][li, y0 : y0 + H, x0 : x0 + W, :], dtype=np.float32)
    c = np.clip((np.transpose(raw_c, (2, 0, 1)) - ds.cerra_mean) / ds.cerra_std, -8.0, 8.0)

    # SatWeave (11, H, W) + mask (1, H, W)
    if per in ds.sat_arrays:
        raw_s = np.asarray(ds.sat_arrays[per][li, y0 : y0 + H, x0 : x0 + W, :], dtype=np.float32)
        vals_s = np.transpose(raw_s[..., :11], (2, 0, 1))
        sm = (raw_s[..., 11:12] > 0.5).transpose(2, 0, 1).astype(np.float32)
        s = np.where(sm > 0.5, np.clip((vals_s - SAT_MEAN[:, None, None]) / SAT_STD[:, None, None], -5.0, 5.0), 0.0)
    else:
        s = np.zeros((11, H, W), dtype=np.float32)
        sm = np.zeros((1, H, W), dtype=np.float32)

    # OPERA Radar (1, H, W) + mask (1, H, W)
    if ds.radar_arr is not None:
        raw_r = np.asarray(ds.radar_arr[t_idx, y0 : y0 + H, x0 : x0 + W, :], dtype=np.float32)
        rm = (raw_r[..., 1:2] > 0.5).transpose(2, 0, 1).astype(np.float32)
        r = np.where(rm > 0.5, np.clip(raw_r[..., 0:1].transpose(2, 0, 1) / RADAR_SCALE, 0.0, 4.0), 0.0)
    else:
        r = np.zeros((1, H, W), dtype=np.float32)
        rm = np.zeros((1, H, W), dtype=np.float32)

    # METAR (6, H, W) + mask (6, H, W)
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Evaluate RegionalFlash on an Épisode Cévenol")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--out-dir", type=Path, default=Path("eval_cevenol"))
    p.add_argument("--start-idx", type=int, default=12254, help="Starting time_idx (default: 2022-03-12 18:00 UTC)")
    p.add_argument("--lead-steps", type=int, default=8, help="Number of 3h autoregressive steps (8 = 24h)")
    p.add_argument("--n-members", type=int, default=4, help="Ensemble members (member 0 shown + ensemble mean)")
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--domain-size", type=int, default=384, help="Square rollout domain size (multiple of 32)")
    p.add_argument("--stride", type=int, default=96)
    p.add_argument("--split", default="val", choices=["val", "test"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    c_args = ckpt.get("args", {})
    model = RegionalFlashJiT3D(
        img_size=(config.PATCH_SIZE, config.PATCH_SIZE),
        patch_size=(1, c_args.get("patch_xy", 8), c_args.get("patch_xy", 8)),
        embed_dim=c_args.get("embed_dim", 1024),
        depth=c_args.get("depth", 12),
        num_heads=c_args.get("num_heads", 16),
        kv_ctx_noise=0.0,
    ).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    log.info(
        "Loaded EMA checkpoint %s (epoch=%s, best_val=%.4f)",
        args.ckpt, ckpt.get("epoch"), ckpt.get("best_val", float("nan")),
    )

    ds = RegionalFlashMultimodalDataset(args.data_dir, split=args.split, random_crop=False)
    ts_df = pd.read_parquet(Path(args.data_dir) / "timestamps.parquet")

    # Center domain on Cévennes (lat ~44.15N, lon ~3.9E) -> grid y~418, x~474
    cy, cx = 418, 474
    S = args.domain_size
    y0 = max(0, min(ds.ny - S, cy - S // 2))
    x0 = max(0, min(ds.nx - S, cx - S // 2))
    log.info(
        "Rollout domain: y=[%d:%d], x=[%d:%d] (%dx%d = ~%dkm square)",
        y0, y0 + S, x0, x0 + S, S, S, int(S * 5.5),
    )

    static_sub = torch.from_numpy(ds.static[:, y0 : y0 + S, x0 : x0 + S]).to(device)
    lat_sub = torch.from_numpy(ds.lat[y0 : y0 + S, x0 : x0 + S]).to(device)
    lon_sub = torch.from_numpy(ds.lon[y0 : y0 + S, x0 : x0 + S]).to(device)
    lsm_sub = ds.static[1, y0 : y0 + S, x0 : x0 + S]
    orog_sub = ds.static[0, y0 : y0 + S, x0 : x0 + S]

    lat_np = ds.lat[y0 : y0 + S, x0 : x0 + S]
    lon_np = ds.lon[y0 : y0 + S, x0 : x0 + S]
    cev_mask = (lat_np >= 43.2) & (lat_np <= 45.0) & (lon_np >= 2.8) & (lon_np <= 5.0)

    # Read ground truth trajectory t0 .. t0 + lead_steps
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
        "Initial condition t0=%d (%s), rolling out %d steps (%dh) with %d ensemble members...",
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
                    stride=args.stride,
                    n_steps=args.ode_steps,
                    tile_batch_size=32,
                    use_residual=True,
                    seed=1000 + mem * 100 + step,
                )
            cur_g = pred_g.float()
            cur_m = pred_m.float()
            ens_grids[mem, step] = cur_g.cpu().numpy()
            ens_metars[mem, step] = cur_m.cpu().numpy()
        log.info("Finished ensemble member %d/%d", mem + 1, args.n_members)

    t2m_i = config.DYN_CHANNELS.index("t2m")
    t2m_mean, t2m_std = float(ds.cerra_mean[t2m_i, 0, 0]), float(ds.cerra_std[t2m_i, 0, 0])
    ir11_i = 57 + 8  # SatWeave 11200 nm (index 8 of 11 bands)
    ir_mean, ir_std = float(SAT_MEAN[8]), float(SAT_STD[8])
    rad_i = 68
    met_t2m_std = float(METAR_STD[0])

    mean_grids = ens_grids.mean(axis=0)
    mean_metars = ens_metars.mean(axis=0)

    log.info("=== 24h Épisode Cévenol Verification (Domain %dx%d & Cévennes Box) ===", S, S)
    for step in range(args.lead_steps):
        lead_h = 3 * (step + 1)
        gt_g = gt_grids[step + 1]
        rm = gt_r_masks[step + 1][0] > 0.5

        gt_t2m = gt_g[t2m_i] * t2m_std + t2m_mean - 273.15
        pr_t2m_m0 = ens_grids[0, step, t2m_i] * t2m_std + t2m_mean - 273.15
        pr_t2m_ens = mean_grids[step, t2m_i] * t2m_std + t2m_mean - 273.15

        gt_dbz = gt_g[rad_i] * RADAR_SCALE
        pr_dbz_m0 = ens_grids[0, step, rad_i] * RADAR_SCALE
        pr_dbz_ens = mean_grids[step, rad_i] * RADAR_SCALE

        gt_ir = gt_g[ir11_i] * ir_std + ir_mean
        pr_ir_ens = mean_grids[step, ir11_i] * ir_std + ir_mean

        t2m_mae_m0 = float(np.abs(pr_t2m_m0 - gt_t2m).mean())
        t2m_mae_ens = float(np.abs(pr_t2m_ens - gt_t2m).mean())
        ir_mae = float(np.abs(pr_ir_ens - gt_ir).mean())

        cev_valid = cev_mask & rm
        csi20_m0 = _csi(pr_dbz_m0[cev_valid], gt_dbz[cev_valid], 20.0)
        csi20_ens = _csi(pr_dbz_ens[cev_valid], gt_dbz[cev_valid], 20.0)
        csi35_m0 = _csi(pr_dbz_m0[cev_valid], gt_dbz[cev_valid], 35.0)
        gt_max_cev = float(gt_dbz[cev_valid].max())
        pr_max_cev = float(pr_dbz_m0[cev_valid].max())

        mm = gt_m_masks[step + 1][0] > 0.5
        if mm.any():
            met_mae_m0 = float(np.abs(ens_metars[0, step, 0][mm] - gt_metars[step + 1][0][mm]).mean()) * met_t2m_std
            met_mae_ens = float(np.abs(mean_metars[step, 0][mm] - gt_metars[step + 1][0][mm]).mean()) * met_t2m_std
        else:
            met_mae_m0 = met_mae_ens = float("nan")

        t_str = str(ts_df.loc[args.start_idx + step + 1, "timestamp_utc"])[:16]
        log.info(
            "+%02dh (%s) | T2m MAE: m0=%.2f°C ens=%.2f°C | METAR: m0=%.2f°C ens=%.2f°C | IR11=%.1fK | "
            "Cévennes Radar: max(gt=%.1f, m0=%.1f) dBZ | CSI20(m0=%.2f, ens=%.2f) CSI35(m0=%.2f)",
            lead_h, t_str, t2m_mae_m0, t2m_mae_ens, met_mae_m0, met_mae_ens, ir_mae,
            gt_max_cev, pr_max_cev, csi20_m0, csi20_ens, csi35_m0,
        )

    # Save multi-panel figure zoomed onto SE France / Cévennes / Gulf of Lion (center 192x192)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    z0, z1 = S // 4, 3 * S // 4
    lsm_z = lsm_sub[z0:z1, z0:z1]
    orog_z = orog_sub[z0:z1, z0:z1]
    show_steps = [s for s in [0, 1, 3, 5, 7] if s < args.lead_steps]

    fig, axes = plt.subplots(4, len(show_steps), figsize=(3.6 * len(show_steps), 13.5), constrained_layout=True)
    for col, step in enumerate(show_steps):
        lead_h = 3 * (step + 1)
        t_str = str(ts_df.loc[args.start_idx + step + 1, "timestamp_utc"])[5:16]
        gt_dbz = (gt_grids[step + 1][rad_i] * RADAR_SCALE)[z0:z1, z0:z1]
        m0_dbz = (ens_grids[0, step, rad_i] * RADAR_SCALE)[z0:z1, z0:z1]
        en_dbz = (mean_grids[step, rad_i] * RADAR_SCALE)[z0:z1, z0:z1]
        gt_ir = (gt_grids[step + 1][ir11_i] * ir_std + ir_mean)[z0:z1, z0:z1]
        m0_ir = (ens_grids[0, step, ir11_i] * ir_std + ir_mean)[z0:z1, z0:z1]

        for r_i, (arr, title) in enumerate([
            (gt_dbz, f"+{lead_h}h ({t_str})\nOBS Radar (dBZ)"),
            (m0_dbz, "Forecast Member 0 (dBZ)"),
            (en_dbz, f"Forecast {args.n_members}-Mem Mean (dBZ)"),
        ]):
            axes[r_i, col].set_facecolor("#11161f")
            im0 = axes[r_i, col].imshow(np.ma.masked_less(arr, 5.0), origin="lower", cmap="turbo", vmin=5, vmax=55)
            axes[r_i, col].contour(lsm_z, levels=[0.5], colors=["white"], linewidths=0.7, alpha=0.8)
            axes[r_i, col].contour(orog_z, levels=[0.8, 1.8], colors=["#cccccc"], linewidths=0.4, alpha=0.45)
            axes[r_i, col].set_title(title, fontsize=10)

        axes[3, col].imshow(m0_ir, origin="lower", cmap="gray_r", vmin=210, vmax=290)
        axes[3, col].contour(lsm_z, levels=[0.5], colors=["yellow"], linewidths=0.6, alpha=0.7)
        axes[3, col].contour(gt_ir, levels=[230], colors=["cyan"], linewidths=0.8)
        axes[3, col].set_title("Forecast IR 11.2µm (cyan=OBS 230K)", fontsize=9)

        for r_i in range(4):
            axes[r_i, col].set_xticks([])
            axes[r_i, col].set_yticks([])

    fig.colorbar(im0, ax=axes[:3, :], shrink=0.6, label="OPERA Radar Reflectivity (dBZ)")
    fig.suptitle(
        f"RegionalFlash 171M (24h Multi-Diffusion) — Épisode Cévenol (init {t0_str[:16]} UTC)",
        fontsize=13,
    )
    out_png = args.out_dir / f"cevenol_t{args.start_idx}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info("Saved Cévenol comparison figure -> %s", out_png)


if __name__ == "__main__":
    main()
