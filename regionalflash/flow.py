"""JiT x-prediction Rectified Flow training step & tiled-diffusion sampler for RegionalFlash.

Ported from FlashEdges `rectified_flow_satellite_metar_v1.py`, adapted for our 2-frame
multimodal regional setup (`CERRA(t) + Sat(t+2h) + Radar(t+2h) + METAR(t+2h)` -> next 3h):
  - Linear rectified flow interpolation: `x_t = (1 - t) * x0 + t * eps`
  - Stratified 32-bin t sampling + clamped `1 / (t + 0.01)^2` x-prediction weighting
  - Optional structured Gaussian noise (`rho` shared + independent)
  - Residual target (`x0 = target - context`) on slowly-varying CERRA + METAR `(tmpc, dwpc, mslp)`
    channels; absolute target on intermittent Sat / Radar / METAR `(u10, v10, p01m)`
  - Masked per-channel MSE with FastNet-style inverse tendency variance weights (`delta_std_3h`)
  - Station-count-weighted METAR loss + METAR context dropout (`metar_drop_frac`) for spatial fill
  - Euler & Heun samplers + overlapping tiled-diffusion full-Europe inference (`sample_tiled_europe`).
"""
import math

import torch

from . import config
from .solar import solar_step_features

C_CERRA = config.N_DYN_CHANNELS  # 57
C_SAT = 11                       # 11 European SatWeave bands
C_RADAR = 1                      # OPERA dbzh
C_GRID = C_CERRA + C_SAT + C_RADAR  # 69
C_METAR = 6                      # tmpc, dwpc, mslp_anom_hpa, u10, v10, p01m

# Residual targets: all 57 CERRA channels + METAR [tmpc(0), dwpc(1), mslp_anom(2)]
METAR_RESIDUAL_IDX = (0, 1, 2)


def structured_gaussian_noise(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    rho: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Partially shared 2D Gaussian noise across channels (Theiss et al. 2024)."""
    B, C, H, W = shape
    if rho <= 0.0:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
    if rho >= 1.0:
        return torch.randn((B, 1, H, W), device=device, dtype=dtype, generator=generator).expand(B, C, H, W)
    shared = torch.randn((B, 1, H, W), device=device, dtype=dtype, generator=generator).expand(B, C, H, W)
    indep = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * indep


def _build_grid_residual_mask(device: torch.device) -> torch.Tensor:
    """True for CERRA (0..56), False for Sat (57..67) and Radar (68)."""
    m = torch.zeros(C_GRID, dtype=torch.bool, device=device)
    m[:C_CERRA] = True
    return m.view(1, C_GRID, 1, 1)


def _build_metar_residual_mask(device: torch.device) -> torch.Tensor:
    """True for METAR tmpc(0), dwpc(1), mslp_anom(2); False for u10, v10, p01m."""
    m = torch.zeros(C_METAR, dtype=torch.bool, device=device)
    for idx in METAR_RESIDUAL_IDX:
        m[idx] = True
    return m.view(1, C_METAR, 1, 1)


def trainer_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    cerra_ch_weights: torch.Tensor | None = None,
    use_residual: bool = True,
    metar_drop_frac: float = 0.15,
    metar_loss_weight: float = 1.0,
    sat_loss_weight: float = 0.35,
    radar_loss_weight: float = 0.25,
    metar_count_denom: float = 15.0,
    noise_rho: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One JiT x-prediction Rectified Flow training step on a 2-frame batch.

    Expected batch keys (all tensors on `device`, already normalized, neutral 0 where masked):
      grid_ctx:        (B, 69, H, W)  clean frame 0 `[CERRA(57), Sat(11), Radar(1)]`
      grid_tgt:        (B, 69, H, W)  clean frame 1 `[CERRA(57), Sat(11), Radar(1)]`
      metar_ctx:       (B, 6, H, W)   clean frame 0 sparse METAR (0 where missing)
      metar_tgt:       (B, 6, H, W)   clean frame 1 sparse METAR (0 where missing)
      sat_mask_ctx:    (B, 1, H, W)   float 0/1
      sat_mask_tgt:    (B, 1, H, W)   float 0/1
      radar_mask_ctx:  (B, 1, H, W)   float 0/1
      radar_mask_tgt:  (B, 1, H, W)   float 0/1
      metar_mask_ctx:  (B, 6, H, W)   bool/float valid per METAR channel at frame 0
      metar_mask_tgt:  (B, 6, H, W)   bool/float valid per METAR channel at frame 1
      static:          (B, 6, H, W)   normalized static terrain & coords
      global_cond:     (B, 6)         `[sin_h, cos_h, sin_doy, cos_doy, lat_norm, lon_norm]`
    """
    grid_ctx = batch["grid_ctx"].to(device, non_blocking=True)
    grid_tgt = batch["grid_tgt"].to(device, non_blocking=True)
    metar_ctx = batch["metar_ctx"].to(device, non_blocking=True)
    metar_tgt = batch["metar_tgt"].to(device, non_blocking=True)
    sat_m_ctx = batch["sat_mask_ctx"].to(device, non_blocking=True)
    sat_m_tgt = batch["sat_mask_tgt"].to(device, non_blocking=True)
    rad_m_ctx = batch["radar_mask_ctx"].to(device, non_blocking=True)
    rad_m_tgt = batch["radar_mask_tgt"].to(device, non_blocking=True)
    met_m_ctx = batch["metar_mask_ctx"].to(device, non_blocking=True)
    met_m_tgt = batch["metar_mask_tgt"].to(device, non_blocking=True)
    static = batch["static"].to(device, non_blocking=True)
    landcover = batch["landcover"].to(device, non_blocking=True) if "landcover" in batch else None
    g_cond = batch["global_cond"].to(device, non_blocking=True)

    B, _, H, W = grid_ctx.shape

    # Build clean x0 targets (residual for CERRA + slow METAR vars, absolute for Sat/Radar/fast METAR)
    if use_residual:
        g_rmask = _build_grid_residual_mask(device)
        m_rmask = _build_metar_residual_mask(device)
        x0_grid = torch.where(g_rmask, grid_tgt - grid_ctx, grid_tgt)
        # Only take METAR residual where context station is also valid (else fallback to absolute)
        both_valid = met_m_ctx.bool() & met_m_tgt.bool()
        x0_metar = torch.where(m_rmask & both_valid, metar_tgt - metar_ctx, metar_tgt)
    else:
        x0_grid = grid_tgt
        x0_metar = metar_tgt

    # METAR context station dropout (teaches spatial interpolation to unseen airports)
    metar_ctx_in = metar_ctx
    met_any_ctx = met_m_ctx.any(dim=1, keepdim=True)  # (B, 1, H, W)
    if metar_drop_frac > 0.0 and model.training:
        drop = (torch.rand(B, 1, H, W, device=device) < metar_drop_frac) & met_any_ctx
        if drop.any():
            metar_ctx_in = torch.where(drop, torch.zeros_like(metar_ctx), metar_ctx)
            met_any_ctx = met_any_ctx & ~drop

    # Stratified 32-bin flow time t in [0, 1]
    n_bins = 32
    bin_size = 1.0 / n_bins
    bin_idx = torch.randperm(n_bins, device=device).repeat_interleave((B + n_bins - 1) // n_bins)[:B]
    t = (bin_idx.float() + torch.rand(B, device=device)) * bin_size
    t = t[torch.randperm(B, device=device)]
    t_4d = t.view(B, 1, 1, 1)

    # Rectified flow linear interpolation: x_t = (1 - t) * x0 + t * eps
    eps_grid = structured_gaussian_noise(x0_grid.shape, device, x0_grid.dtype, rho=noise_rho)
    eps_metar = torch.randn_like(x0_metar)
    xt_grid = (1.0 - t_4d) * x0_grid + t_4d * eps_grid
    xt_metar = (1.0 - t_4d) * x0_metar + t_4d * eps_metar

    # Assemble 2-frame 5D input (B, C_in, 2, H, W)
    frame0 = torch.cat([grid_ctx, metar_ctx_in, sat_m_ctx, rad_m_ctx, met_any_ctx.float(), static], dim=1)
    frame1 = torch.cat([xt_grid, xt_metar, sat_m_tgt, rad_m_tgt, met_any_ctx.float(), static], dim=1)
    x_in = torch.stack([frame0, frame1], dim=2)
    global_cond = torch.cat([g_cond, t.unsqueeze(1)], dim=1)  # t LAST for sinusoidal time emb

    pred_grid, pred_metar = model(
        x_in=x_in,
        global_cond=global_cond,
        ctx_metar=metar_ctx_in,
        noisy_metar=xt_metar,
        ctx_metar_mask=met_any_ctx.float(),
        static=static,
        landcover=landcover,
    )

    # JiT x-prediction loss weighting: upweight data-end (t -> 0)
    w_t = (1.0 / (t_4d + 1e-2) ** 2).clamp(0.9, 10.0)

    # 1. CERRA loss (57 ch, unmasked, FastNet inverse tendency variance weights)
    cerra_err = w_t * (pred_grid[:, :C_CERRA] - x0_grid[:, :C_CERRA]) ** 2
    cerra_per_ch = cerra_err.mean(dim=(0, 2, 3))
    if cerra_ch_weights is not None:
        loss_cerra = (cerra_per_ch * cerra_ch_weights).mean()
    else:
        loss_cerra = cerra_per_ch.mean()

    # 2. SatWeave loss (11 ch, masked by geostationary FOV)
    sat_err = w_t * (pred_grid[:, C_CERRA : C_CERRA + C_SAT] - x0_grid[:, C_CERRA : C_CERRA + C_SAT]) ** 2
    sat_m = sat_m_tgt.float()
    loss_sat = (sat_err * sat_m).sum() / (sat_m.sum() * C_SAT).clamp(min=1.0)

    # 3. OPERA Radar loss (1 ch, masked by valid radar coverage)
    rad_err = w_t * (pred_grid[:, C_CERRA + C_SAT :] - x0_grid[:, C_CERRA + C_SAT :]) ** 2
    rad_m = rad_m_tgt.float()
    loss_radar = (rad_err * rad_m).sum() / rad_m.sum().clamp(min=1.0)

    # 4. METAR sparse airport loss (6 ch, station-count weighted)
    met_err = w_t * (pred_metar - x0_metar) ** 2
    met_m = met_m_tgt.float()
    if metar_count_denom > 0:
        n_i = met_m[:, :1].sum(dim=(1, 2, 3))  # (B,) stations per patch
        count_factor = n_i / float(metar_count_denom)
        cnt_ic = met_m.sum(dim=(2, 3)).clamp(min=1.0)  # (B, 6)
        patch_mean_ic = (met_err * met_m).sum(dim=(2, 3)) / cnt_ic
        loss_metar = (patch_mean_ic * count_factor[:, None]).mean()
    else:
        loss_metar = (met_err * met_m).sum() / met_m.sum().clamp(min=1.0)

    total_loss = (
        loss_cerra
        + sat_loss_weight * loss_sat
        + radar_loss_weight * loss_radar
        + metar_loss_weight * loss_metar
    )

    metrics = {
        "loss": float(total_loss.detach().item()),
        "loss_cerra": float(loss_cerra.detach().item()),
        "loss_sat": float(loss_sat.detach().item()),
        "loss_radar": float(loss_radar.detach().item()),
        "loss_metar": float(loss_metar.detach().item()),
    }
    return total_loss, metrics


@torch.no_grad()
def sample_next_step(
    model: torch.nn.Module,
    grid_ctx: torch.Tensor,
    metar_ctx: torch.Tensor,
    sat_mask_ctx: torch.Tensor,
    sat_mask_tgt: torch.Tensor,
    radar_mask_ctx: torch.Tensor,
    radar_mask_tgt: torch.Tensor,
    metar_mask_ctx: torch.Tensor,
    static: torch.Tensor,
    global_cond_base: torch.Tensor,
    n_steps: int = 20,
    use_residual: bool = True,
    noise_rho: float = 0.0,
    seed: int | None = None,
    landcover: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler rectified-flow x-prediction ODE sampler from t=1 -> 0 for one 128x128 patch.

    Velocity from x-prediction: `v(x_t, t) = (x_t - x0_pred) / max(t, eps)`.
    Step update: `x_{t - dt} = x_t - dt * v = (1 - dt/t) * x_t + (dt/t) * x0_pred`.
    """
    device = grid_ctx.device
    B, _, H, W = grid_ctx.shape
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    xt_grid = structured_gaussian_noise((B, C_GRID, H, W), device, grid_ctx.dtype, rho=noise_rho, generator=gen)
    xt_metar = torch.randn((B, C_METAR, H, W), device=device, dtype=metar_ctx.dtype, generator=gen)
    met_any_ctx = metar_mask_ctx.any(dim=1, keepdim=True).float()
    frame0 = torch.cat([grid_ctx, metar_ctx, sat_mask_ctx, radar_mask_ctx, met_any_ctx, static], dim=1)

    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t_cur = ts[i]
        t_nxt = ts[i + 1]
        t_vec = torch.full((B, 1), float(t_cur.item()), device=device, dtype=global_cond_base.dtype)
        global_cond = torch.cat([global_cond_base, t_vec], dim=1)

        frame1 = torch.cat([xt_grid, xt_metar, sat_mask_tgt, radar_mask_tgt, met_any_ctx, static], dim=1)
        x_in = torch.stack([frame0, frame1], dim=2)

        x0_g, x0_m = model(x_in, global_cond, metar_ctx, xt_metar, met_any_ctx, static, landcover=landcover)

        # On-manifold forcing (FlashEdges): zero x0 at off-coverage SatWeave & OPERA Radar
        # pixels so x_t follows exact t*noise -> 0 decay there.
        x0_g = x0_g.clone()
        x0_g[:, C_CERRA : C_CERRA + C_SAT] = torch.where(
            sat_mask_tgt > 0.5, x0_g[:, C_CERRA : C_CERRA + C_SAT], torch.zeros_like(x0_g[:, C_CERRA : C_CERRA + C_SAT])
        )
        x0_g[:, C_CERRA + C_SAT :] = torch.where(
            radar_mask_tgt > 0.5, x0_g[:, C_CERRA + C_SAT :], torch.zeros_like(x0_g[:, C_CERRA + C_SAT :])
        )

        if i == n_steps - 1:
            # Hybrid METAR forcing released on final step to keep dense spatial-fill forecast
            xt_grid, xt_metar = x0_g, x0_m
        else:
            # Hybrid METAR forcing: zero non-station x0_m on intermediate steps so xt_metar
            # stays on the exact t*noise -> 0 training marginal and never contaminates the trunk.
            x0_m_step = torch.where(met_any_ctx > 0.5, x0_m, torch.zeros_like(x0_m))
            ratio = float(((t_cur - t_nxt) / t_cur.clamp(min=1e-5)).item())
            xt_grid = (1.0 - ratio) * xt_grid + ratio * x0_g
            xt_metar = (1.0 - ratio) * xt_metar + ratio * x0_m_step

    if use_residual:
        g_rmask = _build_grid_residual_mask(device)
        m_rmask = _build_metar_residual_mask(device)
        pred_grid = torch.where(g_rmask, xt_grid + grid_ctx, xt_grid)
        pred_metar = torch.where(m_rmask & metar_mask_ctx.bool(), xt_metar + metar_ctx, xt_metar)
    else:
        pred_grid, pred_metar = xt_grid, xt_metar

    return pred_grid, pred_metar


@torch.no_grad()
def sample_tiled_europe(
    model: torch.nn.Module,
    grid_ctx_full: torch.Tensor,
    metar_ctx_full: torch.Tensor,
    sat_mask_full: torch.Tensor,
    radar_mask_full: torch.Tensor,
    metar_mask_full: torch.Tensor,
    static_full: torch.Tensor,
    time_feats: torch.Tensor,
    lat_full: torch.Tensor,
    lon_full: torch.Tensor,
    ctx_unix_s: float,
    tgt_unix_s: float,
    noon_unix_s: float,
    patch_size: int = config.PATCH_SIZE,
    stride: int = config.TILE_STRIDE,
    n_steps: int = 20,
    tile_batch_size: int = 32,
    use_residual: bool = True,
    seed: int = 42,
    landcover_full: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-Diffusion / Tiled Diffusion over the full 1069x1069 European grid.

    At EVERY denoising step `t_k`, overlapping `128x128` tiles (`stride=96`, 32px overlap)
    are denoised in parallel and blended back onto the global canvas with a 2D raised-cosine
    weight window so adjacent tiles communicate seamlessly across the entire continent.
    """
    device = grid_ctx_full.device
    _, H_full, W_full = grid_ctx_full.shape

    ys = list(range(0, H_full - patch_size + 1, stride))
    if ys[-1] != H_full - patch_size:
        ys.append(H_full - patch_size)
    xs = list(range(0, W_full - patch_size + 1, stride))
    if xs[-1] != W_full - patch_size:
        xs.append(W_full - patch_size)
    tiles = [(y0, x0) for y0 in ys for x0 in xs]

    # 2D raised-cosine blending window (strictly positive everywhere)
    w1d = 0.5 * (1.0 - torch.cos(2.0 * math.pi * (torch.arange(patch_size, device=device) + 0.5) / patch_size))
    win = (w1d[:, None] * w1d[None, :]).clamp(min=1e-3)  # (P, P)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    xt_grid_full = torch.randn((C_GRID, H_full, W_full), device=device, dtype=grid_ctx_full.dtype, generator=gen)
    xt_metar_full = torch.randn((C_METAR, H_full, W_full), device=device, dtype=metar_ctx_full.dtype, generator=gen)
    met_any_full = metar_mask_full.any(dim=0, keepdim=True).float()

    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    for step_i in range(n_steps):
        t_cur = ts[step_i]
        t_nxt = ts[step_i + 1]
        ratio = 1.0 if step_i == n_steps - 1 else float(((t_cur - t_nxt) / t_cur.clamp(min=1e-5)).item())

        acc_grid = torch.zeros_like(xt_grid_full)
        acc_metar = torch.zeros_like(xt_metar_full)
        acc_w = torch.zeros((1, H_full, W_full), device=device, dtype=xt_grid_full.dtype)

        for bi in range(0, len(tiles), tile_batch_size):
            batch_tiles = tiles[bi : bi + tile_batch_size]
            f0_list, f1_list, cm_list, nm_list, cmm_list, st_list, gc_list, lc_list = [], [], [], [], [], [], [], []
            for y0, x0 in batch_tiles:
                y1, x1 = y0 + patch_size, x0 + patch_size
                g0 = grid_ctx_full[:, y0:y1, x0:x1]
                m0 = metar_ctx_full[:, y0:y1, x0:x1]
                sm = sat_mask_full[:, y0:y1, x0:x1]
                rm = radar_mask_full[:, y0:y1, x0:x1]
                ma = met_any_full[:, y0:y1, x0:x1]
                st = static_full[:, y0:y1, x0:x1]
                gn = xt_grid_full[:, y0:y1, x0:x1]
                mn = xt_metar_full[:, y0:y1, x0:x1]

                f0_list.append(torch.cat([g0, m0, sm, rm, ma, st], dim=0))
                f1_list.append(torch.cat([gn, mn, sm, rm, ma, st], dim=0))
                cm_list.append(m0)
                nm_list.append(mn)
                cmm_list.append(ma)
                st_list.append(st)
                if landcover_full is not None:
                    lc_list.append(landcover_full[:, y0:y1, x0:x1])

                cy, cx = y0 + patch_size // 2, x0 + patch_size // 2
                c_lat = float(lat_full[cy, cx].item())
                c_lon = float(lon_full[cy, cx].item())
                lat_n = torch.tensor([(c_lat - 48.0) / 15.0], device=device, dtype=time_feats.dtype)
                lon_n = torch.tensor([(c_lon - 10.0) / 25.0], device=device, dtype=time_feats.dtype)
                sol_t = torch.from_numpy(
                    solar_step_features(ctx_unix_s, tgt_unix_s, noon_unix_s, c_lon, c_lat)
                ).to(device=device, dtype=time_feats.dtype)
                gc_list.append(torch.cat([time_feats, lat_n, lon_n, sol_t, t_cur.view(1)], dim=0))

            x_in = torch.stack([torch.stack(f0_list, dim=0), torch.stack(f1_list, dim=0)], dim=2)
            x0_g, x0_m = model(
                x_in,
                torch.stack(gc_list, dim=0),
                torch.stack(cm_list, dim=0),
                torch.stack(nm_list, dim=0),
                torch.stack(cmm_list, dim=0),
                torch.stack(st_list, dim=0),
                landcover=(torch.stack(lc_list, dim=0) if lc_list else None),
            )

            for ti, (y0, x0) in enumerate(batch_tiles):
                y1, x1 = y0 + patch_size, x0 + patch_size
                acc_grid[:, y0:y1, x0:x1] += x0_g[ti] * win
                acc_metar[:, y0:y1, x0:x1] += x0_m[ti] * win
                acc_w[:, y0:y1, x0:x1] += win

        x0_grid_full = torch.nan_to_num(acc_grid / acc_w, nan=0.0, posinf=8.0, neginf=-8.0)
        x0_metar_full = torch.nan_to_num(acc_metar / acc_w, nan=0.0, posinf=10.0, neginf=-6.0)

        # On-manifold forcing (FlashEdges): zero off-coverage SatWeave & OPERA Radar x0 at every step
        x0_grid_full[C_CERRA : C_CERRA + C_SAT] = torch.where(
            sat_mask_full > 0.5,
            x0_grid_full[C_CERRA : C_CERRA + C_SAT],
            torch.zeros_like(x0_grid_full[C_CERRA : C_CERRA + C_SAT]),
        )
        x0_grid_full[C_CERRA + C_SAT :] = torch.where(
            radar_mask_full > 0.5,
            x0_grid_full[C_CERRA + C_SAT :],
            torch.zeros_like(x0_grid_full[C_CERRA + C_SAT :]),
        )

        # Hybrid METAR forcing: zero non-station x0_metar on every step EXCEPT the last
        if step_i < n_steps - 1:
            x0_metar_step = torch.where(met_any_full > 0.5, x0_metar_full, torch.zeros_like(x0_metar_full))
        else:
            x0_metar_step = x0_metar_full

        xt_grid_full = (1.0 - ratio) * xt_grid_full + ratio * x0_grid_full
        xt_metar_full = (1.0 - ratio) * xt_metar_full + ratio * x0_metar_step

    if use_residual:
        g_rmask = _build_grid_residual_mask(device).squeeze(0)
        m_rmask = _build_metar_residual_mask(device).squeeze(0)
        pred_grid_full = torch.where(g_rmask, xt_grid_full + grid_ctx_full, xt_grid_full)
        pred_metar_full = torch.where(m_rmask & metar_mask_full.bool(), xt_metar_full + metar_ctx_full, xt_metar_full)
    else:
        pred_grid_full, pred_metar_full = xt_grid_full, xt_metar_full

    return pred_grid_full, pred_metar_full


@torch.no_grad()
def project_ar_feedback(
    pred_grid: torch.Tensor,
    pred_metar: torch.Tensor,
    sat_mask: torch.Tensor,
    radar_mask: torch.Tensor,
    metar_station_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project predicted (grid, metar) back onto the exact training context manifold for AR rollouts.

    Matches `dataset.py` normalization & FlashEdges `inference_engine.py` on-manifold AR feedback:
      - CERRA (0:57): clamp to [-8, 8]
      - SatWeave (57:68): clamp to [-5, 5] and zero outside `sat_mask`
      - OPERA Radar (68:69): clamp to [0, 4] and zero outside `radar_mask`
      - METAR (0:6): clamp to [-6, 10] and re-sparsify to `metar_station_mask`
        (zeroing non-station pixels so AR steps 2+ stay on the sparse training manifold)
    Works on both unbatched (C, H, W) and batched (B, C, H, W) tensors.
    """
    g = torch.nan_to_num(pred_grid, nan=0.0).clone()
    m = torch.nan_to_num(pred_metar, nan=0.0).clone()

    sm = (sat_mask > 0.5)
    rm = (radar_mask > 0.5)
    mm = (metar_station_mask > 0.5)

    g[..., :C_CERRA, :, :] = g[..., :C_CERRA, :, :].clamp(-8.0, 8.0)
    sat_slice = g[..., C_CERRA : C_CERRA + C_SAT, :, :].clamp(-5.0, 5.0)
    g[..., C_CERRA : C_CERRA + C_SAT, :, :] = torch.where(sm, sat_slice, torch.zeros_like(sat_slice))
    rad_slice = g[..., C_CERRA + C_SAT :, :, :].clamp(0.0, 4.0)
    g[..., C_CERRA + C_SAT :, :, :] = torch.where(rm, rad_slice, torch.zeros_like(rad_slice))

    m = torch.where(mm, m.clamp(-6.0, 10.0), torch.zeros_like(m))
    return g, m
