from __future__ import annotations

import argparse
import json
import math
import random
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


VALID_SUFFIXES = {".webp", ".png", ".jpg", ".jpeg"}


@dataclass
class ConvNextPerceptualConfig:
    dataset_dir: Path
    output_dir: Path
    model_name: str = "timm/convnextv2_tiny.fcmae_ft_in22k_in1k"
    patch_size: int = 224
    val_ratio: float = 0.05
    samples_per_image_train: int = 16
    samples_per_image_val: int = 2
    batch_size: int = 48
    epochs: int = 12
    lr: float = 2e-4
    weight_decay: float = 1e-2
    temperature: float = 0.1
    proj_dim: int = 256
    grad_clip_norm: float = 1.0
    precision: str = "bf16"
    num_workers: int = 6
    seed: int = 42
    log_every_steps: int = 50
    save_every_epochs: int = 1

    def ensure_paths(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


def _filter_dataclass_kwargs(dc: type, raw: dict[str, Any]) -> dict[str, Any]:
    keys = {f.name for f in fields(dc)}
    return {k: v for k, v in raw.items() if k in keys}


def load_config(path: Path) -> ConvNextPerceptualConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    cleaned = _filter_dataclass_kwargs(ConvNextPerceptualConfig, data)
    for key in ("dataset_dir", "output_dir"):
        if key in cleaned:
            cleaned[key] = Path(cleaned[key])
    cfg = ConvNextPerceptualConfig(**cleaned)
    cfg.ensure_paths()
    return cfg


class AtlasContrastiveDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        patch_size: int,
        samples_per_image: int,
        train: bool,
    ) -> None:
        try:
            from torchvision import transforms as T
        except Exception as exc:
            raise RuntimeError("torchvision is required for contrastive data augmentation.") from exc

        self.image_paths = list(image_paths)
        self.patch_size = int(patch_size)
        self.samples_per_image = max(1, int(samples_per_image))
        self.train = bool(train)

        self.train_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=self.patch_size,
                    scale=(0.45, 1.0),
                    ratio=(0.85, 1.15),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.35),
                T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.eval_transform = T.Compose(
            [
                T.Resize(self.patch_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(self.patch_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        if self.train:
            return len(self.image_paths) * self.samples_per_image
        return max(1, len(self.image_paths) * self.samples_per_image)

    def _load_rgb(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("L")
        return image.convert("RGB")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_idx = idx % len(self.image_paths)
        image = self._load_rgb(self.image_paths[image_idx])
        if self.train:
            v1 = self.train_transform(image)
            v2 = self.train_transform(image)
        else:
            # Keep mild randomness in validation to avoid collapse on identical pairs.
            v1 = self.train_transform(image)
            v2 = self.train_transform(image)
        return {"view1": v1, "view2": v2}


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def simclr_nt_xent_loss(z: torch.Tensor, temperature: float) -> torch.Tensor:
    if z.ndim != 2 or z.size(0) % 2 != 0:
        raise ValueError(f"Expected z shape (2N, D), got {tuple(z.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    n2 = z.size(0)
    n = n2 // 2
    sim = (z @ z.t()) / float(temperature)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    diag = torch.eye(n2, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, float("-inf"))

    pos = torch.arange(n2, device=z.device)
    pos = torch.where(pos < n, pos + n, pos - n)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return (-log_prob[torch.arange(n2, device=z.device), pos]).mean()


def simclr_retrieval_topk(z: torch.Tensor, k: int = 1) -> float:
    if z.ndim != 2 or z.size(0) % 2 != 0:
        return 0.0
    if k <= 0:
        return 0.0
    n2 = z.size(0)
    n = n2 // 2
    sim = z @ z.t()
    diag = torch.eye(n2, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, float("-inf"))
    pos = torch.arange(n2, device=z.device)
    pos = torch.where(pos < n, pos + n, pos - n)
    topk = torch.topk(sim, k=min(k, n2 - 1), dim=1).indices
    return float(topk.eq(pos.unsqueeze(-1)).any(dim=-1).float().mean().item())


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


def _run_epoch(
    *,
    loader: DataLoader,
    backbone: nn.Module,
    projector: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler,
    temperature: float,
    grad_clip_norm: float,
    device: torch.device,
    log_every_steps: int,
    start_global_step: int,
    is_train: bool,
    epoch_title: str,
    amp_device_type: str,
) -> tuple[float, float, float, int]:
    if is_train:
        backbone.train()
        projector.train()
    else:
        backbone.eval()
        projector.eval()

    total_loss = 0.0
    retr1_sum = 0.0
    retr5_sum = 0.0
    steps = 0
    global_step = start_global_step

    pbar = tqdm(loader, desc=epoch_title)
    for batch in pbar:
        x1 = batch["view1"].to(device, non_blocking=True)
        x2 = batch["view2"].to(device, non_blocking=True)
        x = torch.cat([x1, x2], dim=0)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        autocast_ctx = torch.amp.autocast(amp_device_type, enabled=use_amp, dtype=amp_dtype)
        with autocast_ctx:
            emb = backbone(x)
            z = projector(emb)
            z = F.normalize(z.float(), dim=-1)
            loss = simclr_nt_xent_loss(z, temperature=temperature)

        if is_train and optimizer is not None:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(backbone.parameters()) + list(projector.parameters()),
                        grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(backbone.parameters()) + list(projector.parameters()),
                        grad_clip_norm,
                    )
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            retr1 = simclr_retrieval_topk(z, k=1)
            retr5 = simclr_retrieval_topk(z, k=5)

        steps += 1
        global_step += 1
        total_loss += float(loss.detach())
        retr1_sum += retr1
        retr5_sum += retr5

        avg_loss = total_loss / max(1, steps)
        avg_r1 = retr1_sum / max(1, steps)
        avg_r5 = retr5_sum / max(1, steps)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", r1=f"{avg_r1:.3f}", r5=f"{avg_r5:.3f}")

        if log_every_steps > 0 and (steps % log_every_steps == 0):
            pbar.write(
                f"{epoch_title}: step={steps} global_step={global_step} "
                f"loss={avg_loss:.4f} r1={avg_r1:.3f} r5={avg_r5:.3f}"
            )

    return (
        total_loss / max(1, steps),
        retr1_sum / max(1, steps),
        retr5_sum / max(1, steps),
        global_step,
    )


def _save_checkpoint(
    output_dir: Path,
    filename: str,
    *,
    cfg: ConvNextPerceptualConfig,
    backbone: nn.Module,
    projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    global_step: int,
    val_loss: float | None,
) -> Path:
    payload = {
        "type": "convnext_font_perceptual",
        "model_name": cfg.model_name,
        "backbone_state": _module_state_dict(backbone),
        "projection_state": _module_state_dict(projector),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": asdict(cfg),
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
    }
    out_path = output_dir / "checkpoints" / filename
    torch.save(payload, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain ConvNeXtV2 tiny for font-atlas perceptual features.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _seed_everything(cfg.seed)

    try:
        import timm
    except Exception as exc:
        raise SystemExit("timm is required. Install dependencies from requirements.txt first.") from exc

    if not cfg.dataset_dir.exists():
        raise SystemExit(f"Dataset path not found: {cfg.dataset_dir}")

    image_paths = sorted([p for p in cfg.dataset_dir.rglob("*") if p.suffix.lower() in VALID_SUFFIXES])
    if not image_paths:
        raise SystemExit(f"No atlas images found in: {cfg.dataset_dir}")

    train_paths, val_paths = _split_paths(image_paths, cfg.val_ratio, cfg.seed)
    train_ds = AtlasContrastiveDataset(
        train_paths,
        patch_size=cfg.patch_size,
        samples_per_image=cfg.samples_per_image_train,
        train=True,
    )
    val_ds = AtlasContrastiveDataset(
        val_paths if val_paths else train_paths,
        patch_size=cfg.patch_size,
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

    backbone = timm.create_model(cfg.model_name, pretrained=True, num_classes=0).to(device)
    feature_dim = int(getattr(backbone, "num_features", 0))
    if feature_dim <= 0:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.patch_size, cfg.patch_size, device=device)
            feature_dim = int(backbone(dummy).shape[-1])
    projector = ProjectionHead(feature_dim, cfg.proj_dim).to(device)

    optimizer = AdamW(
        list(backbone.parameters()) + list(projector.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, cfg.epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and cfg.precision == "fp16"))

    history_path = cfg.output_dir / "history.jsonl"
    best_val = float("inf")
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_r1, train_r5, global_step = _run_epoch(
            loader=train_loader,
            backbone=backbone,
            projector=projector,
            optimizer=optimizer,
            scheduler=scheduler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            temperature=cfg.temperature,
            grad_clip_norm=cfg.grad_clip_norm,
            device=device,
            log_every_steps=cfg.log_every_steps,
            start_global_step=global_step,
            is_train=True,
            epoch_title=f"Train {epoch}/{cfg.epochs}",
            amp_device_type=amp_device_type,
        )

        with torch.no_grad():
            val_loss, val_r1, val_r5, _ = _run_epoch(
                loader=val_loader,
                backbone=backbone,
                projector=projector,
                optimizer=None,
                scheduler=None,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                temperature=cfg.temperature,
                grad_clip_norm=0.0,
                device=device,
                log_every_steps=0,
                start_global_step=global_step,
                is_train=False,
                epoch_title=f"Val {epoch}/{cfg.epochs}",
                amp_device_type=amp_device_type,
            )

        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(train_loss),
            "train_r1": float(train_r1),
            "train_r5": float(train_r5),
            "val_loss": float(val_loss),
            "val_r1": float(val_r1),
            "val_r5": float(val_r5),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        if epoch % max(1, cfg.save_every_epochs) == 0:
            _save_checkpoint(
                cfg.output_dir,
                filename=f"convnext_epoch_{epoch:03d}.pt",
                cfg=cfg,
                backbone=backbone,
                projector=projector,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                val_loss=val_loss,
            )

        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(
                cfg.output_dir,
                filename="convnext_perceptual_best.pt",
                cfg=cfg,
                backbone=backbone,
                projector=projector,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                val_loss=val_loss,
            )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"train_r1={train_r1:.3f} val_r1={val_r1:.3f}"
        )

    final_path = _save_checkpoint(
        cfg.output_dir,
        filename="convnext_perceptual_last.pt",
        cfg=cfg,
        backbone=backbone,
        projector=projector,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=cfg.epochs,
        global_step=global_step,
        val_loss=best_val if math.isfinite(best_val) else None,
    )
    print(f"Saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
