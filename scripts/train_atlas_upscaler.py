from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from fontnn.models.atlas_upscaler import AtlasSuperResolutionNet, UpscalerConfig


VALID_SUFFIXES = {".webp", ".png", ".jpg", ".jpeg"}


@dataclass
class AtlasUpscalerTrainConfig:
    dataset_dir: Path
    output_dir: Path
    perceptual_checkpoint: Path
    perceptual_model_name: str = "timm/convnextv2_tiny.fcmae_ft_in22k_in1k"
    upscale_factor: int = 2
    hr_patch_size: int = 256
    val_ratio: float = 0.05
    samples_per_image_train: int = 24
    samples_per_image_val: int = 3
    batch_size: int = 20
    epochs: int = 40
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    precision: str = "bf16"
    num_workers: int = 6
    seed: int = 42
    log_every_steps: int = 50
    save_every_epochs: int = 1

    in_channels: int = 1
    out_channels: int = 1
    feature_channels: int = 96
    num_residual_blocks: int = 16
    residual_expansion: int = 4
    dropout: float = 0.0

    loss_l1_weight: float = 1.0
    loss_edge_weight: float = 0.25
    loss_perceptual_weight: float = 0.08
    perceptual_layers: tuple[int, ...] = (1, 2, 3)
    perceptual_layer_weights: tuple[float, ...] = (0.25, 0.5, 1.0)
    resume_checkpoint: Path | None = None

    def ensure_paths(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


def _filter_dataclass_kwargs(dc: type, raw: dict[str, Any]) -> dict[str, Any]:
    keys = {f.name for f in fields(dc)}
    return {k: v for k, v in raw.items() if k in keys}


def load_config(path: Path) -> AtlasUpscalerTrainConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    cleaned = _filter_dataclass_kwargs(AtlasUpscalerTrainConfig, data)
    for key in ("dataset_dir", "output_dir", "perceptual_checkpoint", "resume_checkpoint"):
        if key in cleaned:
            if cleaned[key] is not None:
                cleaned[key] = Path(cleaned[key])
    if "perceptual_layers" in cleaned:
        cleaned["perceptual_layers"] = tuple(int(v) for v in cleaned["perceptual_layers"])
    if "perceptual_layer_weights" in cleaned:
        cleaned["perceptual_layer_weights"] = tuple(float(v) for v in cleaned["perceptual_layer_weights"])
    cfg = AtlasUpscalerTrainConfig(**cleaned)
    cfg.ensure_paths()
    return cfg


class AtlasPatchDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        hr_patch_size: int,
        scale_factor: int,
        samples_per_image: int,
        train: bool,
    ) -> None:
        self.image_paths = list(image_paths)
        self.hr_patch_size = int(hr_patch_size)
        self.scale_factor = int(scale_factor)
        self.samples_per_image = max(1, int(samples_per_image))
        self.train = bool(train)

        if self.hr_patch_size % self.scale_factor != 0:
            raise ValueError("hr_patch_size must be divisible by scale_factor")

    def __len__(self) -> int:
        return len(self.image_paths) * self.samples_per_image

    def _open_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    def _pick_patch(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape
        ps = self.hr_patch_size
        if h < ps or w < ps:
            new_h = max(ps, h)
            new_w = max(ps, w)
            pil = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
            pil = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            h, w = arr.shape

        if self.train:
            y0 = random.randint(0, h - ps)
            x0 = random.randint(0, w - ps)
        else:
            y0 = max(0, (h - ps) // 2)
            x0 = max(0, (w - ps) // 2)
        patch = arr[y0 : y0 + ps, x0 : x0 + ps]
        return patch

    @staticmethod
    def _downsample(hr: torch.Tensor, scale_factor: int) -> torch.Tensor:
        size = (hr.shape[-2] // scale_factor, hr.shape[-1] // scale_factor)
        try:
            return F.interpolate(hr, size=size, mode="bicubic", align_corners=False, antialias=True)
        except TypeError:
            return F.interpolate(hr, size=size, mode="bicubic", align_corners=False)

    def _degrade_lr(self, lr: torch.Tensor) -> torch.Tensor:
        if not self.train:
            return lr
        if random.random() < 0.30:
            noise_std = random.uniform(0.002, 0.012)
            lr = (lr + torch.randn_like(lr) * noise_std).clamp(0.0, 1.0)
        if random.random() < 0.20:
            lr = F.avg_pool2d(lr, kernel_size=3, stride=1, padding=1)
        return lr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.image_paths[idx % len(self.image_paths)]
        arr = self._open_image(path)
        patch = self._pick_patch(arr)
        hr = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
        lr = self._downsample(hr, self.scale_factor)
        lr = self._degrade_lr(lr)
        return {
            "lr": lr.squeeze(0),
            "hr": hr.squeeze(0),
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def _module_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    base = getattr(module, "_orig_mod", None)
    return base.state_dict() if base is not None else module.state_dict()


def _best_remapped_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_keys: set[str],
) -> tuple[dict[str, torch.Tensor], int]:
    def to_features_only_convnext_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        remapped: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            nk = re.sub(r"^stem\.(\d+)\.", r"stem_\1.", k)
            nk = re.sub(r"^stages\.(\d+)\.", r"stages_\1.", nk)
            remapped[nk] = v
        return remapped

    candidates: list[dict[str, torch.Tensor]] = [state_dict]
    candidates.append(to_features_only_convnext_keys(state_dict))

    for prefix in ("model.", "backbone.", "module.", "_orig_mod."):
        if any(k.startswith(prefix) for k in state_dict):
            stripped = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if stripped:
                candidates.append(stripped)
                candidates.append(to_features_only_convnext_keys(stripped))
                candidates.append({f"model.{k}": v for k, v in stripped.items()})
                candidates.append({f"backbone.{k}": v for k, v in stripped.items()})

    candidates.append({f"model.{k}": v for k, v in state_dict.items()})
    candidates.append({f"backbone.{k}": v for k, v in state_dict.items()})

    best = state_dict
    best_overlap = -1
    for cand in candidates:
        overlap = len(set(cand.keys()) & target_keys)
        if overlap > best_overlap:
            best = cand
            best_overlap = overlap

    return best, best_overlap


def _load_backbone_state_with_remap(
    model: nn.Module,
    raw_state: dict[str, torch.Tensor],
    *,
    min_loaded_ratio: float = 0.85,
) -> tuple[list[str], list[str], float]:
    model_keys = set(model.state_dict().keys())
    candidate, overlap = _best_remapped_state_dict(raw_state, model_keys)
    missing, unexpected = model.load_state_dict(candidate, strict=False)
    loaded_ratio = 0.0 if len(model_keys) == 0 else float(overlap) / float(len(model_keys))

    if loaded_ratio < min_loaded_ratio:
        raise RuntimeError(
            "Perceptual checkpoint is not compatible with the current encoder. "
            f"Loaded ratio={loaded_ratio:.3f}, overlap={overlap}/{len(model_keys)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return missing, unexpected, loaded_ratio


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        try:
            from torch.serialization import safe_globals

            with safe_globals([pathlib.PosixPath, pathlib.WindowsPath]):
                try:
                    return torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    return torch.load(path, map_location="cpu")
        except Exception:
            # Fallback for legacy checkpoints serialized with pathlib.PosixPath on Windows.
            pathlib.PosixPath = pathlib.WindowsPath
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(path, map_location="cpu")


def _split_paths(paths: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if len(paths) < 2:
        return paths, []
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * max(0.0, min(0.5, val_ratio)))
    if n_val < 1:
        n_val = 1
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    if not train:
        train = val
        val = []
    return train, val


def _find_latest_resume_checkpoint(checkpoints_dir: Path) -> Path | None:
    last_path = checkpoints_dir / "upscaler_x2_last.pt"
    if last_path.exists():
        return last_path

    candidates: list[tuple[int, Path]] = []
    for p in checkpoints_dir.glob("upscaler_x2_epoch_*.pt"):
        stem = p.stem
        try:
            epoch_id = int(stem.rsplit("_", 1)[-1])
        except Exception:
            continue
        candidates.append((epoch_id, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _resolve_resume_checkpoint(
    *,
    cfg: AtlasUpscalerTrainConfig,
    cli_resume: Path | None,
    auto_resume: bool,
) -> Path | None:
    if cli_resume is not None and auto_resume:
        raise SystemExit("Use either --resume or --auto-resume, not both.")
    if cli_resume is not None:
        return cli_resume
    if cfg.resume_checkpoint is not None:
        return cfg.resume_checkpoint
    if auto_resume:
        return _find_latest_resume_checkpoint(cfg.output_dir / "checkpoints")
    return None


def _resume_training_state(
    *,
    resume_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
) -> tuple[int, int, float, int | None]:
    checkpoint = _safe_torch_load(resume_path)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported resume checkpoint format: {type(checkpoint).__name__}")

    state = checkpoint.get("model")
    if not isinstance(state, dict):
        state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"No model state found in checkpoint: {resume_path}")

    model.load_state_dict(state, strict=True)

    if isinstance(checkpoint.get("optimizer"), dict):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if isinstance(checkpoint.get("scheduler"), dict):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler.is_enabled() and isinstance(checkpoint.get("scaler"), dict):
        scaler.load_state_dict(checkpoint["scaler"])

    last_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf"))))
    resume_total_steps = checkpoint.get("total_steps")
    if resume_total_steps is not None:
        try:
            resume_total_steps = int(resume_total_steps)
        except Exception:
            resume_total_steps = None
    start_epoch = last_epoch + 1

    print(
        f"Resumed from {resume_path}: "
        f"last_epoch={last_epoch} start_epoch={start_epoch} global_step={global_step} "
        f"best_val_loss={best_val_loss:.5f}"
    )
    return start_epoch, global_step, best_val_loss, resume_total_steps


class ConvNeXtPerceptualLoss(nn.Module):
    def __init__(
        self,
        model_name: str,
        checkpoint_path: Path,
        out_indices: tuple[int, ...],
        layer_weights: tuple[float, ...],
    ) -> None:
        super().__init__()
        try:
            import timm
        except Exception as exc:
            raise RuntimeError("timm is required for ConvNeXt perceptual loss.") from exc

        if len(out_indices) != len(layer_weights):
            raise ValueError("perceptual_layers and perceptual_layer_weights must have the same length")
        self.out_indices = tuple(int(i) for i in out_indices)
        self.layer_weights = tuple(float(w) for w in layer_weights)

        self.encoder = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=self.out_indices,
        )

        checkpoint = _safe_torch_load(checkpoint_path)
        backbone_state = checkpoint.get("backbone_state")
        if backbone_state is None:
            backbone_state = checkpoint.get("model") or checkpoint.get("state_dict")
        if backbone_state is None:
            raise ValueError(f"Invalid perceptual checkpoint: {checkpoint_path}")
        missing, unexpected, loaded_ratio = _load_backbone_state_with_remap(self.encoder, backbone_state)
        print(
            "Perceptual encoder load: "
            f"loaded_ratio={loaded_ratio:.3f} missing={len(missing)} unexpected={len(unexpected)}"
        )

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x3 = x.repeat(1, 3, 1, 1)
        return (x3 - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feats = self.encoder(self._prep(pred))
        with torch.no_grad():
            target_feats = self.encoder(self._prep(target))
        loss = pred.new_tensor(0.0)
        for w, pf, tf in zip(self.layer_weights, pred_feats, target_feats):
            loss = loss + float(w) * F.l1_loss(pf, tf)
        return loss


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    mse = max(mse, 1e-12)
    return 10.0 * math.log10(1.0 / mse)


def _save_checkpoint(
    cfg: AtlasUpscalerTrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    val_loss: float,
    best_val_loss: float,
    total_steps: int,
    filename: str,
) -> Path:
    model_cfg = UpscalerConfig(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        feature_channels=cfg.feature_channels,
        num_residual_blocks=cfg.num_residual_blocks,
        residual_expansion=cfg.residual_expansion,
        dropout=cfg.dropout,
        upscale_factor=cfg.upscale_factor,
    )
    payload = {
        "type": "font_atlas_upscaler",
        "model": _module_state_dict(model),
        "config": asdict(model_cfg),
        "training_config": asdict(cfg),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),
        "total_steps": int(total_steps),
    }
    out_path = cfg.output_dir / "checkpoints" / filename
    torch.save(payload, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train x2 atlas upscaler with ConvNeXt perceptual loss.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to training checkpoint to resume from.")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from output_dir/checkpoints/upscaler_x2_last.pt or the latest epoch checkpoint.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    _seed_everything(cfg.seed)

    if cfg.upscale_factor != 2:
        raise SystemExit("This trainer is specialized for x2 and expects upscale_factor=2.")
    if not cfg.dataset_dir.exists():
        raise SystemExit(f"Dataset path not found: {cfg.dataset_dir}")
    if not cfg.perceptual_checkpoint.exists():
        raise SystemExit(f"Perceptual checkpoint not found: {cfg.perceptual_checkpoint}")

    image_paths = sorted([p for p in cfg.dataset_dir.rglob("*") if p.suffix.lower() in VALID_SUFFIXES])
    if not image_paths:
        raise SystemExit(f"No atlas images found in: {cfg.dataset_dir}")

    train_paths, val_paths = _split_paths(image_paths, cfg.val_ratio, cfg.seed)
    train_ds = AtlasPatchDataset(
        train_paths,
        hr_patch_size=cfg.hr_patch_size,
        scale_factor=cfg.upscale_factor,
        samples_per_image=cfg.samples_per_image_train,
        train=True,
    )
    val_ds = AtlasPatchDataset(
        val_paths if val_paths else train_paths,
        hr_patch_size=cfg.hr_patch_size,
        scale_factor=cfg.upscale_factor,
        samples_per_image=cfg.samples_per_image_val,
        train=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and cfg.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
    )
    if len(train_loader) == 0:
        raise SystemExit("Train loader has zero batches. Reduce batch_size or increase samples_per_image_train.")

    model_cfg = UpscalerConfig(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        feature_channels=cfg.feature_channels,
        num_residual_blocks=cfg.num_residual_blocks,
        residual_expansion=cfg.residual_expansion,
        dropout=cfg.dropout,
        upscale_factor=cfg.upscale_factor,
    )
    model = AtlasSuperResolutionNet(model_cfg).to(device)

    perceptual = ConvNeXtPerceptualLoss(
        model_name=cfg.perceptual_model_name,
        checkpoint_path=cfg.perceptual_checkpoint,
        out_indices=cfg.perceptual_layers,
        layer_weights=cfg.perceptual_layer_weights,
    ).to(device)
    perceptual.eval()

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, cfg.epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and cfg.precision == "fp16"))

    history_path = cfg.output_dir / "history.jsonl"
    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    resume_path = _resolve_resume_checkpoint(
        cfg=cfg,
        cli_resume=args.resume,
        auto_resume=bool(args.auto_resume),
    )
    if resume_path is not None:
        if not resume_path.exists():
            raise SystemExit(f"Resume checkpoint not found: {resume_path}")
        start_epoch, global_step, best_val_loss, resume_total_steps = _resume_training_state(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if resume_total_steps is not None and resume_total_steps != total_steps:
            print(
                "WARNING: total_steps differs from resume checkpoint "
                f"({resume_total_steps} -> {total_steps}). "
                "If you changed epochs/batch/samples_per_image, LR schedule shape will differ."
            )

    if start_epoch > cfg.epochs:
        print(
            f"Nothing to train: start_epoch={start_epoch} is greater than cfg.epochs={cfg.epochs}. "
            "Increase epochs in config to continue."
        )
        return

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        train_total = 0.0
        train_l1_total = 0.0
        train_edge_total = 0.0
        train_perc_total = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{cfg.epochs}")
        for batch in pbar:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.amp.autocast(amp_device_type, enabled=use_amp, dtype=amp_dtype)
            with autocast_ctx:
                pred = model(lr)
                l1_loss = F.l1_loss(pred, hr)
                edge_loss = F.l1_loss(_sobel_edges(pred), _sobel_edges(hr))
                perc_loss = perceptual(pred, hr)
                loss = (
                    cfg.loss_l1_weight * l1_loss
                    + cfg.loss_edge_weight * edge_loss
                    + cfg.loss_perceptual_weight * perc_loss
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
            scheduler.step()

            global_step += 1
            train_steps += 1
            train_total += float(loss.detach())
            train_l1_total += float(l1_loss.detach())
            train_edge_total += float(edge_loss.detach())
            train_perc_total += float(perc_loss.detach())

            avg_loss = train_total / train_steps
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                l1=f"{train_l1_total / train_steps:.4f}",
                edge=f"{train_edge_total / train_steps:.4f}",
                perc=f"{train_perc_total / train_steps:.4f}",
            )

            if cfg.log_every_steps > 0 and train_steps % cfg.log_every_steps == 0:
                pbar.write(
                    f"Train epoch={epoch} step={train_steps} global_step={global_step} "
                    f"loss={avg_loss:.5f}"
                )

        model.eval()
        val_total = 0.0
        val_psnr_total = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val {epoch}/{cfg.epochs}"):
                lr = batch["lr"].to(device, non_blocking=True)
                hr = batch["hr"].to(device, non_blocking=True)
                pred = model(lr)
                l1_loss = F.l1_loss(pred, hr)
                edge_loss = F.l1_loss(_sobel_edges(pred), _sobel_edges(hr))
                perc_loss = perceptual(pred, hr)
                loss = (
                    cfg.loss_l1_weight * l1_loss
                    + cfg.loss_edge_weight * edge_loss
                    + cfg.loss_perceptual_weight * perc_loss
                )
                val_total += float(loss)
                val_psnr_total += _psnr(pred, hr)
                val_steps += 1

        avg_train = train_total / max(1, train_steps)
        avg_val = val_total / max(1, val_steps)
        avg_val_psnr = val_psnr_total / max(1, val_steps)

        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": avg_train,
            "train_l1": train_l1_total / max(1, train_steps),
            "train_edge": train_edge_total / max(1, train_steps),
            "train_perceptual": train_perc_total / max(1, train_steps),
            "val_loss": avg_val,
            "val_psnr": avg_val_psnr,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        if epoch % max(1, cfg.save_every_epochs) == 0:
            _save_checkpoint(
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                val_loss=avg_val,
                best_val_loss=min(best_val_loss, avg_val),
                total_steps=total_steps,
                filename=f"upscaler_x2_epoch_{epoch:03d}.pt",
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            _save_checkpoint(
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                val_loss=avg_val,
                best_val_loss=best_val_loss,
                total_steps=total_steps,
                filename="upscaler_x2_best.pt",
            )

        print(
            f"Epoch {epoch}: train_loss={avg_train:.5f} "
            f"val_loss={avg_val:.5f} val_psnr={avg_val_psnr:.2f}dB"
        )

    final_path = _save_checkpoint(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=cfg.epochs,
        global_step=global_step,
        val_loss=best_val_loss if math.isfinite(best_val_loss) else 0.0,
        best_val_loss=best_val_loss if math.isfinite(best_val_loss) else 0.0,
        total_steps=total_steps,
        filename="upscaler_x2_last.pt",
    )
    print(f"Saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
