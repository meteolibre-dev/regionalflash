"""STRATA-inspired Multi-Step Autoregressive (AR) Fine-Tuning for RegionalFlash JiT-3D.

Inspired by STRATA (arXiv:2606.31248): after 1-step pretraining, fine-tune the model
autoregressively for up to K=4 steps (+12h) so it learns to suppress rollout drift,
phase lag, and compounding artifacts over 24h rollouts.

Supports two AR modes (`--ar-mode`):
  1. `flow_rollout` (DEFAULT — recommended for JiT-3D Rectified Flow):
     - Preserves full generative sharpness & ensemble spread (no blurry deterministic collapse!).
     - Samples horizon `K` from curriculum `1 .. cur_max_steps` (with `p_1step=0.25` clean anchor).
     - Rolls the model `K - 1` steps under `no_grad` with fast Euler ODE (`--rollout-ode-steps 6`),
       hybrid METAR forcing, and FlashEdges on-manifold `project_ar_feedback`.
     - Computes the full Rectified Flow x-prediction loss on step `K` predicting ground-truth
       `t0 + K` conditioned on the model's own rolled-out state `pred_{t0 + K - 1}`.
  2. `strata_det` (exact STRATA deterministic multi-step recipe):
     - Unrolls `K` differentiable deterministic steps (`t_flow = 1.0`, `xt = 0`) using gradient
       checkpointing and computes channel-normalized Huber loss (`delta=1.0`) on the final frame `t0 + K`.

Logs true 4-step (+3h, +6h, +9h, +12h) autoregressive validation T2m MAE (CERRA & METAR)
every epoch and saves `best.pt` on the lowest +12h AR error.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from . import config
from .dataset import METAR_STD, RegionalFlashSequenceDataset, seed_worker
from .flow import (
    C_CERRA,
    C_GRID,
    C_METAR,
    C_SAT,
    _build_grid_residual_mask,
    _build_metar_residual_mask,
    project_ar_feedback,
    sample_next_step,
    trainer_step,
)
from .model import RegionalFlashJiT3D
from .train import build_cerra_channel_weights, update_ema

log = logging.getLogger(__name__)


def _strata_det_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    k_steps: int,
    cerra_ch_weights: torch.Tensor | None = None,
    huber_delta: float = 1.0,
    sat_loss_weight: float = 0.35,
    radar_loss_weight: float = 0.25,
    metar_loss_weight: float = 1.0,
    metar_count_denom: float = 8.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact STRATA-style differentiable K-step rollout with Huber loss on the final frame."""
    grid_seq = batch["grid_seq"].to(device, non_blocking=True)             # (B, L, 69, H, W)
    metar_seq = batch["metar_seq"].to(device, non_blocking=True)           # (B, L, 6, H, W)
    sat_m_seq = batch["sat_mask_seq"].to(device, non_blocking=True)        # (B, L, 1, H, W)
    rad_m_seq = batch["radar_mask_seq"].to(device, non_blocking=True)      # (B, L, 1, H, W)
    met_m_seq = batch["metar_mask_seq"].to(device, non_blocking=True)      # (B, L, 6, H, W)
    static = batch["static"].to(device, non_blocking=True)                 # (B, 6, H, W)
    landcover = batch["landcover"].to(device, non_blocking=True) if "landcover" in batch else None
    g_cond_seq = batch["global_cond_seq"].to(device, non_blocking=True)    # (B, K_max, 14)

    B, _, _, H, W = grid_seq.shape
    g_rmask = _build_grid_residual_mask(device)
    m_rmask = _build_metar_residual_mask(device)

    cur_g = grid_seq[:, 0]
    cur_m = metar_seq[:, 0]
    met_m0 = met_m_seq[:, 0].bool()
    met_any0 = met_m0.any(dim=1, keepdim=True).float()

    zeros_g = torch.zeros_like(cur_g)
    zeros_m = torch.zeros_like(cur_m)
    t_one = torch.ones((B, 1), device=device, dtype=g_cond_seq.dtype)

    def _one_step(g_in, m_in, sm_c, sm_t, rm_c, rm_t, gc_base):
        f0 = torch.cat([g_in, m_in, sm_c, rm_c, met_any0, static], dim=1)
        f1 = torch.cat([zeros_g, zeros_m, sm_t, rm_t, met_any0, static], dim=1)
        x_in = torch.stack([f0, f1], dim=2)
        gc = torch.cat([gc_base, t_one], dim=1)
        x0_g, x0_m = model(x_in, gc, m_in, zeros_m, met_any0, static, landcover=landcover)
        pg = torch.where(g_rmask, x0_g + g_in, x0_g)
        pm = torch.where(m_rmask & met_m0, x0_m + m_in, x0_m)
        return pg, pm

    for s in range(k_steps):
        sm_c, sm_t = sat_m_seq[:, s], sat_m_seq[:, s + 1]
        rm_c, rm_t = rad_m_seq[:, s], rad_m_seq[:, s + 1]
        gc_base = g_cond_seq[:, s]
        if s < k_steps - 1:
            pg, pm = checkpoint(_one_step, cur_g, cur_m, sm_c, sm_t, rm_c, rm_t, gc_base, use_reentrant=False)
            # Differentiable on-manifold clamping & masking for intermediate AR steps
            cur_g = torch.cat([
                pg[:, :C_CERRA].clamp(-8.0, 8.0),
                torch.where(sm_t > 0.5, pg[:, C_CERRA : C_CERRA + C_SAT].clamp(-5.0, 5.0), torch.zeros_like(pg[:, C_CERRA : C_CERRA + C_SAT])),
                torch.where(rm_t > 0.5, pg[:, C_CERRA + C_SAT :].clamp(0.0, 4.0), torch.zeros_like(pg[:, C_CERRA + C_SAT :])),
            ], dim=1)
            cur_m = torch.where(met_m0, pm.clamp(-6.0, 10.0), torch.zeros_like(pm))
        else:
            pred_g, pred_m = _one_step(cur_g, cur_m, sm_c, sm_t, rm_c, rm_t, gc_base)

    # STRATA channel-normalized Huber loss (delta=1.0) on FINAL step k_steps
    gt_g = grid_seq[:, k_steps]
    gt_m = metar_seq[:, k_steps]
    sat_m = sat_m_seq[:, k_steps].float()
    rad_m = rad_m_seq[:, k_steps].float()
    met_m = met_m_seq[:, k_steps].float()

    cerra_err = F.huber_loss(pred_g[:, :C_CERRA], gt_g[:, :C_CERRA], reduction="none", delta=huber_delta)
    cerra_per_ch = cerra_err.mean(dim=(0, 2, 3))
    loss_cerra = (cerra_per_ch * cerra_ch_weights).mean() if cerra_ch_weights is not None else cerra_per_ch.mean()

    sat_err = F.huber_loss(pred_g[:, C_CERRA : C_CERRA + C_SAT], gt_g[:, C_CERRA : C_CERRA + C_SAT], reduction="none", delta=huber_delta)
    loss_sat = (sat_err * sat_m).sum() / (sat_m.sum() * C_SAT).clamp(min=1.0)

    rad_err = F.huber_loss(pred_g[:, C_CERRA + C_SAT :], gt_g[:, C_CERRA + C_SAT :], reduction="none", delta=huber_delta)
    loss_radar = (rad_err * rad_m).sum() / rad_m.sum().clamp(min=1.0)

    met_err = F.huber_loss(pred_m, gt_m, reduction="none", delta=huber_delta)
    n_i = met_m[:, :1].sum(dim=(1, 2, 3))
    cnt_ic = met_m.sum(dim=(2, 3)).clamp(min=1.0)
    loss_metar = (((met_err * met_m).sum(dim=(2, 3)) / cnt_ic) * (n_i / float(metar_count_denom))[:, None]).mean()

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


def _flow_rollout_step(
    train_model: torch.nn.Module,
    rollout_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    k_steps: int,
    rollout_ode_steps: int,
    cerra_ch_weights: torch.Tensor | None,
    metar_drop_frac: float,
    noise_rho: float,
    step_seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rollout-Conditioned Rectified Flow: roll K-1 steps (no_grad), then flow-match step K."""
    grid_seq = batch["grid_seq"].to(device, non_blocking=True)
    metar_seq = batch["metar_seq"].to(device, non_blocking=True)
    sat_m_seq = batch["sat_mask_seq"].to(device, non_blocking=True)
    rad_m_seq = batch["radar_mask_seq"].to(device, non_blocking=True)
    met_m_seq = batch["metar_mask_seq"].to(device, non_blocking=True)
    static = batch["static"].to(device, non_blocking=True)
    landcover = batch["landcover"].to(device, non_blocking=True) if "landcover" in batch else None
    g_cond_seq = batch["global_cond_seq"].to(device, non_blocking=True)

    cur_g = grid_seq[:, 0]
    cur_m = metar_seq[:, 0]
    cur_mm = met_m_seq[:, 0]

    # Roll K - 1 steps autoregressively under no_grad with exact on-manifold AR feedback
    if k_steps > 1:
        was_training = rollout_model.training
        rollout_model.eval()
        with torch.no_grad():
            for s in range(k_steps - 1):
                pg, pm = sample_next_step(
                    rollout_model,
                    grid_ctx=cur_g,
                    metar_ctx=cur_m,
                    sat_mask_ctx=sat_m_seq[:, s],
                    sat_mask_tgt=sat_m_seq[:, s + 1],
                    radar_mask_ctx=rad_m_seq[:, s],
                    radar_mask_tgt=rad_m_seq[:, s + 1],
                    metar_mask_ctx=cur_mm,
                    static=static,
                    global_cond_base=g_cond_seq[:, s],
                    n_steps=rollout_ode_steps,
                    use_residual=True,
                    noise_rho=noise_rho,
                    seed=step_seed + s,
                    landcover=landcover,
                )
                cur_g, cur_m = project_ar_feedback(pg, pm, sat_m_seq[:, s + 1], rad_m_seq[:, s + 1], cur_mm)
        if was_training:
            rollout_model.train()

    # Step K (with gradients!): Rectified Flow x-prediction conditioned on rolled-out state
    step_batch = {
        "grid_ctx": cur_g,
        "grid_tgt": grid_seq[:, k_steps],
        "metar_ctx": cur_m,
        "metar_tgt": metar_seq[:, k_steps],
        "sat_mask_ctx": sat_m_seq[:, k_steps - 1],
        "sat_mask_tgt": sat_m_seq[:, k_steps],
        "radar_mask_ctx": rad_m_seq[:, k_steps - 1],
        "radar_mask_tgt": rad_m_seq[:, k_steps],
        "metar_mask_ctx": cur_mm,
        "metar_mask_tgt": met_m_seq[:, k_steps],
        "static": static,
        "global_cond": g_cond_seq[:, k_steps - 1],
    }
    if landcover is not None:
        step_batch["landcover"] = landcover

    return trainer_step(
        train_model,
        step_batch,
        device,
        cerra_ch_weights=cerra_ch_weights,
        use_residual=True,
        metar_drop_frac=(metar_drop_frac if k_steps == 1 else 0.0),
        noise_rho=noise_rho,
    )


@torch.no_grad()
def evaluate_ar(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_steps: int,
    t2m_std: float,
    sample_steps: int = 12,
    max_batches: int = 6,
) -> dict[str, float]:
    """Run true multi-step (+3h .. +3*max_steps h) AR rollouts on val and measure physical T2m MAE."""
    model.eval()
    cerra_t2m_idx = config.DYN_CHANNELS.index("t2m")
    metar_t2m_std = float(METAR_STD[0])

    c_err = [0.0] * max_steps
    c_cnt = [0] * max_steps
    m_err = [0.0] * max_steps
    m_cnt = [0] * max_steps

    for bi, batch in enumerate(val_loader):
        if bi >= max_batches:
            break
        grid_seq = batch["grid_seq"].to(device)
        metar_seq = batch["metar_seq"].to(device)
        sat_m_seq = batch["sat_mask_seq"].to(device)
        rad_m_seq = batch["radar_mask_seq"].to(device)
        met_m_seq = batch["metar_mask_seq"].to(device)
        static = batch["static"].to(device)
        landcover = batch["landcover"].to(device) if "landcover" in batch else None
        g_cond_seq = batch["global_cond_seq"].to(device)

        cur_g = grid_seq[:, 0]
        cur_m = metar_seq[:, 0]
        cur_mm = met_m_seq[:, 0]

        for s in range(max_steps):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pg, pm = sample_next_step(
                    model,
                    grid_ctx=cur_g,
                    metar_ctx=cur_m,
                    sat_mask_ctx=sat_m_seq[:, s],
                    sat_mask_tgt=sat_m_seq[:, s + 1],
                    radar_mask_ctx=rad_m_seq[:, s],
                    radar_mask_tgt=rad_m_seq[:, s + 1],
                    metar_mask_ctx=cur_mm,
                    static=static,
                    global_cond_base=g_cond_seq[:, s],
                    n_steps=sample_steps,
                    use_residual=True,
                    seed=200 + bi * 10 + s,
                    landcover=landcover,
                )
            gt_t2m = grid_seq[:, s + 1, cerra_t2m_idx]
            c_err[s] += float((pg[:, cerra_t2m_idx] - gt_t2m).abs().sum().item()) * t2m_std
            c_cnt[s] += gt_t2m.numel()

            mm = met_m_seq[:, s + 1, 0].bool()
            if mm.any():
                m_err[s] += float((pm[:, 0][mm] - metar_seq[:, s + 1, 0][mm]).abs().sum().item()) * metar_t2m_std
                m_cnt[s] += int(mm.sum().item())

            cur_g, cur_m = project_ar_feedback(pg.float(), pm.float(), sat_m_seq[:, s + 1], rad_m_seq[:, s + 1], cur_mm)

    out: dict[str, float] = {}
    for s in range(max_steps):
        lead_h = 3 * (s + 1)
        out[f"cerra_t2m_{lead_h}h"] = c_err[s] / max(c_cnt[s], 1)
        out[f"metar_t2m_{lead_h}h"] = m_err[s] / max(m_cnt[s], 1)
    model.train()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="STRATA-inspired Multi-Step AR Fine-Tuning for RegionalFlash")
    p.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--init-from", type=Path, required=True, help="Pretrained 1-step checkpoint (best.pt)")
    p.add_argument("--out-dir", type=Path, default=config.ROOT_DIR / "runs" / "jit3d_ar_ft")
    p.add_argument("--ar-mode", choices=["flow_rollout", "strata_det"], default="flow_rollout")
    p.add_argument("--max-steps", type=int, default=4, help="Max AR steps (4 = 12h rollout)")
    p.add_argument("--rollout-ode-steps", type=int, default=6, help="ODE steps during K-1 rollout in flow_rollout mode")
    p.add_argument("--p-1step", type=float, default=0.25, help="Fraction of batches kept at K=1 anchor in flow_rollout")
    p.add_argument("--huber-delta", type=float, default=1.0, help="Huber delta in strata_det mode")

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-samples-per-epoch", type=int, default=16384)
    p.add_argument("--val-samples", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--metar-drop-frac", type=float, default=0.15)
    p.add_argument("--noise-rho", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=1)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
    c_args = ckpt.get("args", {})
    ema_sd = ckpt["ema"]
    lc_w = ema_sd.get("landcover_embed.soft_emb.0.weight")
    c_landcover = int(lc_w.shape[1]) if lc_w is not None else 0

    train_ds = RegionalFlashSequenceDataset(
        args.data_dir, split="train", max_steps=args.max_steps, random_crop=True, samples_per_epoch=args.train_samples_per_epoch
    )
    val_ds = RegionalFlashSequenceDataset(
        args.data_dir, split="val", max_steps=args.max_steps, random_crop=True, samples_per_epoch=args.val_samples, seed=123
    )
    log.info("Loaded AR Sequence dataset (max_steps=%d): %d train anchors, %d val anchors",
             args.max_steps, len(train_ds.anchors), len(val_ds.anchors))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        worker_init_fn=(seed_worker if args.num_workers > 0 else None),
    )
    val_workers = min(4, max(2, args.num_workers // 2)) if args.num_workers > 0 else 0
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        prefetch_factor=(1 if val_workers > 0 else None),
        worker_init_fn=(seed_worker if val_workers > 0 else None),
    )

    model = RegionalFlashJiT3D(
        img_size=(config.PATCH_SIZE, config.PATCH_SIZE),
        patch_size=(1, c_args.get("patch_xy", 8), c_args.get("patch_xy", 8)),
        embed_dim=c_args.get("embed_dim", 1024),
        depth=c_args.get("depth", 12),
        num_heads=c_args.get("num_heads", 16),
        kv_ctx_noise=0.0,
        c_landcover=c_landcover,
    ).to(device)
    model.load_state_dict(ema_sd, strict=True)
    ema_model = copy.deepcopy(model).eval()
    for p_ema in ema_model.parameters():
        p_ema.requires_grad_(False)
    log.info("Loaded pretrained EMA weights from %s (landcover=%d, ar_mode=%s)", args.init_from, c_landcover, args.ar_mode)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    train_model = torch.compile(model) if args.compile else model
    cerra_ch_weights = build_cerra_channel_weights(args.data_dir, device)
    t2m_std = float(train_ds.cerra_std[config.DYN_CHANNELS.index("t2m"), 0, 0])

    total_steps = args.epochs * len(train_loader)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    best_ar_score = float("inf")

    # Baseline 4-step AR evaluation before fine-tuning!
    base_val = evaluate_ar(ema_model, val_loader, device, args.max_steps, t2m_std)
    log.info(
        "BASELINE (Ep 00) AR VAL | CERRA T2m: 3h=%.2f°C 6h=%.2f°C 9h=%.2f°C 12h=%.2f°C | "
        "METAR T2m: 3h=%.2f°C 6h=%.2f°C 9h=%.2f°C 12h=%.2f°C",
        base_val.get("cerra_t2m_3h", float("nan")), base_val.get("cerra_t2m_6h", float("nan")),
        base_val.get("cerra_t2m_9h", float("nan")), base_val.get("cerra_t2m_12h", float("nan")),
        base_val.get("metar_t2m_3h", float("nan")), base_val.get("metar_t2m_6h", float("nan")),
        base_val.get("metar_t2m_9h", float("nan")), base_val.get("metar_t2m_12h", float("nan")),
    )

    for epoch in range(args.epochs):
        train_model.train()
        t0_ep = time.time()
        running: dict[str, float] = {}

        # STRATA-style curriculum: ramp max rollout horizon 2 -> 3 -> 4 across first 40% of epochs
        cur_max_k = min(args.max_steps, 2 + int(epoch * (args.max_steps - 1) / max(int(0.4 * args.epochs), 1)))
        log.info("Starting AR Ep %02d/%02d (curriculum max_k=%d, mode=%s)", epoch + 1, args.epochs, cur_max_k, args.ar_mode)

        for step_i, batch in enumerate(train_loader, 1):
            if global_step < args.warmup_steps:
                lr = args.lr * (global_step + 1) / max(args.warmup_steps, 1)
            else:
                prog = (global_step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
                lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * min(prog, 1.0)))
            for pg in opt.param_groups:
                pg["lr"] = lr

            # Sample horizon K for this batch
            if args.ar_mode == "flow_rollout" and torch.rand(1).item() < args.p_1step:
                k_steps = 1
            else:
                k_steps = int(torch.randint(2, cur_max_k + 1, (1,)).item())

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                if args.ar_mode == "flow_rollout":
                    loss, metrics = _flow_rollout_step(
                        train_model=train_model,
                        rollout_model=ema_model,
                        batch=batch,
                        device=device,
                        k_steps=k_steps,
                        rollout_ode_steps=args.rollout_ode_steps,
                        cerra_ch_weights=cerra_ch_weights,
                        metar_drop_frac=args.metar_drop_frac,
                        noise_rho=args.noise_rho,
                        step_seed=global_step * 17,
                    )
                else:
                    loss, metrics = _strata_det_step(
                        model=train_model,
                        batch=batch,
                        device=device,
                        k_steps=k_steps,
                        cerra_ch_weights=cerra_ch_weights,
                        huber_delta=args.huber_delta,
                    )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema_model, model, args.ema_decay)
            global_step += 1

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v

            if step_i % 50 == 0 or step_i == len(train_loader):
                avg = {k: v / step_i for k, v in running.items()}
                log.info(
                    "AR Ep %02d [%d/%d] k=%d lr=%.2e | loss=%.4f (cerra=%.4f sat=%.4f rad=%.4f metar=%.4f)",
                    epoch + 1, step_i, len(train_loader), k_steps, lr,
                    avg["loss"], avg["loss_cerra"], avg["loss_sat"], avg["loss_radar"], avg["loss_metar"],
                )

        val_ar = evaluate_ar(ema_model, val_loader, device, args.max_steps, t2m_std)
        lead_max = 3 * args.max_steps
        ar_score = val_ar[f"cerra_t2m_{lead_max}h"] + 0.5 * val_ar[f"metar_t2m_{lead_max}h"]
        log.info(
            "AR Ep %02d DONE (%.1fs) | CERRA T2m: 3h=%.2f°C 6h=%.2f°C 9h=%.2f°C 12h=%.2f°C | "
            "METAR T2m: 3h=%.2f°C 6h=%.2f°C 9h=%.2f°C 12h=%.2f°C",
            epoch + 1, time.time() - t0_ep,
            val_ar.get("cerra_t2m_3h", float("nan")), val_ar.get("cerra_t2m_6h", float("nan")),
            val_ar.get("cerra_t2m_9h", float("nan")), val_ar.get("cerra_t2m_12h", float("nan")),
            val_ar.get("metar_t2m_3h", float("nan")), val_ar.get("metar_t2m_6h", float("nan")),
            val_ar.get("metar_t2m_9h", float("nan")), val_ar.get("metar_t2m_12h", float("nan")),
        )

        merged_args = {**c_args, **vars(args)}
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val": min(best_ar_score, ar_score),
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "opt": opt.state_dict(),
            "args": merged_args,
        }
        torch.save(state, args.out_dir / "last.pt")
        if ar_score < best_ar_score:
            best_ar_score = ar_score
            state["best_val"] = best_ar_score
            torch.save(state, args.out_dir / "best.pt")
            log.info("New best AR checkpoint -> %s (ar_score=%.4f)", args.out_dir / "best.pt", best_ar_score)


if __name__ == "__main__":
    main()
