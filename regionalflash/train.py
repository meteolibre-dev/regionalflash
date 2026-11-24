"""Training script for RegionalFlash JiT-3D DiT (x-prediction Rectified Flow).

Features:
  - Optional `--hf-download` to pull selected periods (e.g. `2018_H1` + `2022_H1`) plus
    radar, METAR, static, and metadata from `meteolibre-dev/regionalflash`
  - FastNet-style inverse 3h-tendency variance channel weights (`norm_stats.json`)
  - bfloat16 AMP, AdamW, cosine LR schedule with linear warmup, gradient clipping, EMA weights
  - Validation loop tracking flow loss AND actual 10-step ODE sampled physical metrics
    (CERRA T2m MAE in degC & METAR airport T2m MAE in degC!)
"""
import argparse
import copy
import json
import logging
import math
from pathlib import Path
import time

from huggingface_hub import snapshot_download
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import METAR_STD, RegionalFlashMultimodalDataset
from .flow import sample_next_step, trainer_step
from .model import RegionalFlashJiT3D

log = logging.getLogger("regionalflash.train")


def download_from_hf(
    data_dir: Path,
    periods: list[str],
    repo_id: str = config.HF_REPO_ID,
    include_sat: bool = True,
) -> None:
    patterns = [
        "static_5km.zarr/**",
        "radar_5km_plus2h.zarr/**",
        "metar_stations.parquet",
        "metar_obs_plus2h.parquet",
        "timestamps.parquet",
        "norm_stats.json",
    ]
    for per in periods:
        patterns.append(f"cerra_5km_{per}.zarr/**")
        if include_sat:
            patterns.append(f"sat_5km_{per}_plus2h.zarr/**")
    log.info("Downloading periods %s from hf://datasets/%s -> %s ...", periods, repo_id, data_dir)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=patterns,
    )


def build_cerra_channel_weights(data_dir: Path, device: torch.device) -> torch.Tensor:
    """FastNet-style inverse normalized 3h-tendency variance weights (mean = 1)."""
    stats = json.loads((data_dir / "norm_stats.json").read_text())
    std = np.asarray(stats["std"], dtype=np.float64)
    dstd = np.asarray(stats["delta_std_3h"], dtype=np.float64)
    # Tendency std in normalized space
    norm_dvar = np.maximum((dstd / np.maximum(std, 1e-6)) ** 2, 1e-3)
    w = 1.0 / norm_dvar
    w = np.clip(w / np.mean(w), 0.2, 5.0)
    w = w / np.mean(w)
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.lerp_(p.detach(), 1.0 - decay)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cerra_ch_weights: torch.Tensor,
    t2m_std: float,
    max_batches: int = 32,
    sample_batches: int = 4,
    sample_steps: int = 12,
    use_residual: bool = True,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n_b = 0

    cerra_t2m_idx = config.DYN_CHANNELS.index("t2m")
    metar_t2m_std = float(METAR_STD[0])

    cerra_t2m_err_sum, cerra_t2m_cnt = 0.0, 0
    metar_t2m_err_sum, metar_t2m_cnt = 0.0, 0

    for bi, batch in enumerate(val_loader):
        if max_batches > 0 and bi >= max_batches:
            break
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, metrics = trainer_step(
                model,
                batch,
                device,
                cerra_ch_weights=cerra_ch_weights,
                use_residual=use_residual,
                blur_sigma=0.0,
                metar_drop_frac=0.0,
            )
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v
        n_b += 1

        # Run actual ODE sampling on first `sample_batches` to measure physical T2m MAE (degC)
        if bi < sample_batches:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pg, pm = sample_next_step(
                    model,
                    grid_ctx=batch["grid_ctx"].to(device),
                    metar_ctx=batch["metar_ctx"].to(device),
                    sat_mask_ctx=batch["sat_mask_ctx"].to(device),
                    sat_mask_tgt=batch["sat_mask_tgt"].to(device),
                    radar_mask_ctx=batch["radar_mask_ctx"].to(device),
                    radar_mask_tgt=batch["radar_mask_tgt"].to(device),
                    metar_mask_ctx=batch["metar_mask_ctx"].to(device),
                    static=batch["static"].to(device),
                    global_cond_base=batch["global_cond"].to(device),
                    n_steps=sample_steps,
                    use_residual=use_residual,
                    seed=100 + bi,
                )
            gt_t2m = batch["grid_tgt"][:, cerra_t2m_idx].to(device)
            pr_t2m = pg[:, cerra_t2m_idx]
            cerra_t2m_err_sum += float((pr_t2m - gt_t2m).abs().sum().item()) * t2m_std
            cerra_t2m_cnt += gt_t2m.numel()

            mm = batch["metar_mask_tgt"][:, 0].to(device).bool()
            if mm.any():
                gt_m = batch["metar_tgt"][:, 0].to(device)[mm]
                pr_m = pm[:, 0][mm]
                metar_t2m_err_sum += float((pr_m - gt_m).abs().sum().item()) * metar_t2m_std
                metar_t2m_cnt += int(mm.sum().item())

    out = {k: v / max(n_b, 1) for k, v in sums.items()}
    if cerra_t2m_cnt > 0:
        out["val_cerra_t2m_mae_c"] = cerra_t2m_err_sum / cerra_t2m_cnt
    if metar_t2m_cnt > 0:
        out["val_metar_t2m_mae_c"] = metar_t2m_err_sum / metar_t2m_cnt
    model.train()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Train RegionalFlash JiT-3D Rectified Flow model")
    p.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--out-dir", type=Path, default=config.ROOT_DIR / "runs" / "jit3d_v1")
    p.add_argument("--hf-download", action="store_true", help="Download periods from Hugging Face before training")
    p.add_argument("--hf-periods", nargs="+", default=["2018_H1", "2022_H1"], help="Periods to fetch if --hf-download")
    p.add_argument("--repo-id", default=config.HF_REPO_ID)

    # Model size
    p.add_argument("--embed-dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--patch-xy", type=int, default=8)
    p.add_argument("--kv-ctx-noise", type=float, default=0.05)

    # Training recipe
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-samples-per-epoch", type=int, default=16384)
    p.add_argument("--val-samples", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--blur-sigma", type=float, default=0.5)
    p.add_argument("--metar-drop-frac", type=float, default=0.15)
    p.add_argument("--noise-rho", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--compile", action="store_true", help="Use torch.compile on model")
    p.add_argument("--resume", type=Path, default=None)
    args = p.parse_args()

    if args.hf_download:
        download_from_hf(args.data_dir, args.hf_periods, repo_id=args.repo_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_ds = RegionalFlashMultimodalDataset(
        args.data_dir, split="train", random_crop=True, samples_per_epoch=args.train_samples_per_epoch
    )
    val_ds = RegionalFlashMultimodalDataset(
        args.data_dir, split="val", random_crop=True, samples_per_epoch=args.val_samples, seed=123
    )
    log.info("Loaded RegionalFlash dataset: %d train anchors, %d val anchors", len(train_ds.anchors), len(val_ds.anchors))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(2, args.num_workers // 2),
        pin_memory=True,
    )

    model = RegionalFlashJiT3D(
        img_size=(config.PATCH_SIZE, config.PATCH_SIZE),
        patch_size=(1, args.patch_xy, args.patch_xy),
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        kv_ctx_noise=args.kv_ctx_noise,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("RegionalFlashJiT3D: %.2fM params (dim=%d, depth=%d, heads=%d, patch=(1,%d,%d))",
             n_params, args.embed_dim, args.depth, args.num_heads, args.patch_xy, args.patch_xy)

    ema_model = copy.deepcopy(model).eval()
    for p_ema in ema_model.parameters():
        p_ema.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["global_step"])
        best_val = float(ckpt.get("best_val", float("inf")))
        log.info("Resumed from %s (epoch=%d, step=%d)", args.resume, start_epoch, global_step)

    train_model = torch.compile(model) if args.compile else model
    cerra_ch_weights = build_cerra_channel_weights(args.data_dir, device)
    t2m_std = float(train_ds.cerra_std[config.DYN_CHANNELS.index("t2m"), 0, 0])

    total_steps = args.epochs * len(train_loader)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        train_model.train()
        t0_ep = time.time()
        running: dict[str, float] = {}

        for step_i, batch in enumerate(train_loader, 1):
            # Cosine LR schedule with linear warmup
            if global_step < args.warmup_steps:
                lr = args.lr * (global_step + 1) / max(args.warmup_steps, 1)
            else:
                prog = (global_step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
                lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * min(prog, 1.0)))
            for pg in opt.param_groups:
                pg["lr"] = lr

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss, metrics = trainer_step(
                    train_model,
                    batch,
                    device,
                    cerra_ch_weights=cerra_ch_weights,
                    use_residual=True,
                    blur_sigma=args.blur_sigma,
                    metar_drop_frac=args.metar_drop_frac,
                    noise_rho=args.noise_rho,
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
                    "Ep %02d [%d/%d] lr=%.2e | loss=%.4f (cerra=%.4f sat=%.4f rad=%.4f metar=%.4f)",
                    epoch + 1, step_i, len(train_loader), lr,
                    avg["loss"], avg["loss_cerra"], avg["loss_sat"], avg["loss_radar"], avg["loss_metar"],
                )

        val_metrics = evaluate(ema_model, val_loader, device, cerra_ch_weights, t2m_std)
        log.info(
            "Ep %02d DONE (%.1fs) | VAL loss=%.4f (cerra=%.4f sat=%.4f rad=%.4f metar=%.4f) | "
            "T2m MAE: CERRA=%.2f°C METAR=%.2f°C",
            epoch + 1,
            time.time() - t0_ep,
            val_metrics["loss"],
            val_metrics["loss_cerra"],
            val_metrics["loss_sat"],
            val_metrics["loss_radar"],
            val_metrics["loss_metar"],
            val_metrics.get("val_cerra_t2m_mae_c", float("nan")),
            val_metrics.get("val_metar_t2m_mae_c", float("nan")),
        )

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "opt": opt.state_dict(),
            "best_val": min(best_val, val_metrics["loss"]),
            "args": vars(args),
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(ckpt, args.out_dir / "best.pt")
            log.info("New best checkpoint -> %s (val_loss=%.4f)", args.out_dir / "best.pt", best_val)


if __name__ == "__main__":
    main()
