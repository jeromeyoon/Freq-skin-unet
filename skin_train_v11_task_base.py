from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import skin_train_v11 as v11
from skin_dataset import _load_gray, _load_rgb, skin_collate_fn
from skin_net_v2 import build_analyzer_v2


class TaskSpecificSkinDataset(torch.utils.data.Dataset):
    """Single-task dataset with task-specific input folder routing.

    - brown/red: rgb_cross + rgb_parallel + mask
    - wrinkle:  rgb_cross + rgb_parallel_wrinkle + mask_wrinkle
    """

    def __init__(self, patch_dir: str, task: str, img_size: int = 256, augment: bool = False):
        if task not in {"brown", "red", "wrinkle"}:
            raise ValueError(f"unsupported task: {task}")
        self.patch_dir = Path(patch_dir)
        self.task = task
        self.img_size = int(img_size)
        self.augment = bool(augment)

        manifest_path = self.patch_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.stems = [
            stem for stem, info in self.manifest.items()
            if info.get(f"has_{self.task}", False)
        ]
        if not self.stems:
            raise RuntimeError(f"No samples with has_{self.task}=True in {patch_dir}")

        self.parallel_dir = "rgb_parallel"
        self.mask_dir = "mask"
        if self.task == "wrinkle":
            self.parallel_dir = "rgb_parallel_wrinkle"
            if not (self.patch_dir / self.parallel_dir).exists():
                raise FileNotFoundError(
                    f"wrinkle task requires rgb_parallel_wrinkle folder: {self.patch_dir / self.parallel_dir}"
                )
            if (self.patch_dir / "mask_wrinkle").exists():
                self.mask_dir = "mask_wrinkle"

        print(
            f"[TaskSpecificSkinDataset] task={self.task} samples={len(self.stems)} "
            f"parallel_dir={self.parallel_dir} mask_dir={self.mask_dir}"
        )

    def __len__(self):
        return len(self.stems)

    def _apply_augment(self, *tensors):
        import torchvision.transforms.functional as TF

        do_hflip = random.random() > 0.5
        do_vflip = random.random() > 0.5
        out = []
        for t in tensors:
            if t is None:
                out.append(None)
                continue
            if do_hflip:
                t = TF.hflip(t)
            if do_vflip:
                t = TF.vflip(t)
            out.append(t)
        return tuple(out)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        rgb_cross = _load_rgb(self.patch_dir / "rgb_cross" / f"{stem}.png", self.img_size)
        rgb_parallel = _load_rgb(self.patch_dir / self.parallel_dir / f"{stem}.png", self.img_size)

        mask_path = self.patch_dir / self.mask_dir / f"{stem}.png"
        mask = _load_gray(mask_path, self.img_size) if mask_path.exists() else torch.ones(1, self.img_size, self.img_size)

        gt_path = self.patch_dir / self.task / f"{stem}.png"
        gt = _load_gray(gt_path, self.img_size) if gt_path.exists() else None

        brown = gt if self.task == "brown" else None
        red = gt if self.task == "red" else None
        wrinkle = gt if self.task == "wrinkle" else None

        if self.augment:
            rgb_cross, rgb_parallel, mask, brown, red, wrinkle = self._apply_augment(
                rgb_cross, rgb_parallel, mask, brown, red, wrinkle
            )

        return {
            "rgb_cross": rgb_cross,
            "rgb_parallel": rgb_parallel,
            "mask": mask,
            "brown": brown,
            "red": red,
            "wrinkle": wrinkle,
            "has_brown": brown is not None,
            "has_red": red is not None,
            "has_wrinkle": wrinkle is not None,
            "stem": stem,
        }


def _task_tensors(task: str, result, batch: dict, device: torch.device):
    if task == "brown":
        return result.brown_mask.float(), batch["brown"].to(device, non_blocking=True).float(), batch["has_brown"]
    if task == "red":
        return result.red_mask.float(), batch["red"].to(device, non_blocking=True).float(), batch["has_red"]
    return result.wrinkle_mask.float(), batch["wrinkle"].to(device, non_blocking=True).float(), batch["has_wrinkle"]


def _simple_seg_loss(
    logits: torch.Tensor,
    gt: torch.Tensor,
    face_mask: torch.Tensor,
    has_gt: list[bool],
    mode: str,
    bce_weight: float,
) -> torch.Tensor:
    valid = torch.tensor(has_gt, dtype=torch.bool, device=logits.device)
    if not valid.any():
        return logits.new_tensor(0.0)

    valid4d = valid.view(-1, 1, 1, 1).float()
    gt_m = gt * face_mask

    bce_map = F.binary_cross_entropy_with_logits(logits, gt_m, reduction="none")
    bce = (bce_map * face_mask * valid4d).sum() / (face_mask * valid4d).sum().clamp(min=1.0)
    if mode == "bce":
        return bce

    prob = torch.sigmoid(logits) * face_mask
    prob = prob[valid].flatten(1)
    gt_f = gt_m[valid].flatten(1)
    inter = (prob * gt_f).sum(dim=1)
    denom = prob.sum(dim=1) + gt_f.sum(dim=1)
    dice_loss = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()

    return float(bce_weight) * bce + (1.0 - float(bce_weight)) * dice_loss


@torch.no_grad()
def _dice_metric(logits: torch.Tensor, gt: torch.Tensor, face_mask: torch.Tensor, has_gt: list[bool], threshold=0.5):
    valid = torch.tensor(has_gt, dtype=torch.bool, device=logits.device)
    if not valid.any():
        return 0.0, 0

    pred = (torch.sigmoid(logits) >= threshold).float() * face_mask
    gt = (gt >= 0.5).float() * face_mask
    pred = pred[valid].flatten(1)
    gt = gt[valid].flatten(1)
    inter = (pred * gt).sum(dim=1)
    denom = pred.sum(dim=1) + gt.sum(dim=1)
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return float(dice.sum().item()), int(valid.sum().item())


def train_one_epoch_single_task(model, loader, optimizer, scaler, device, cfg, epoch, total_epochs, task, loss_mode, bce_weight):
    model.train()
    total_loss = 0.0
    processed_steps = 0
    amp_enabled = device.type == "cuda"

    aug_kwargs = dict(
        intensity_range=cfg["intensity_range"],
        color_temp_range=cfg["color_temp_range"],
        ambient_fog_alpha_range=cfg.get("ambient_fog_alpha_range", (0.0, 0.05)),
        ceiling_gradient_prob=cfg.get("ceiling_gradient_prob", 0.5),
    )

    pbar = tqdm(loader, desc=f"Train {task} {epoch:03d}/{total_epochs}", leave=False)
    for batch in pbar:
        rgb_cross = batch["rgb_cross"].to(device, non_blocking=True)
        rgb_parallel = batch["rgb_parallel"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True).float()

        rgb_cross_aug, rgb_parallel_aug = v11.apply_paired_batch_illumination_aug_fast(
            rgb_cross, rgb_parallel, **aug_kwargs
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            result = model(rgb_cross_aug, rgb_parallel_aug, mask)

        logits, gt, has_gt = _task_tensors(task, result, batch, device)
        loss = _simple_seg_loss(logits, gt, mask, has_gt, mode=loss_mode, bce_weight=bce_weight)
        if not loss.requires_grad:
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        processed_steps += 1
        total_loss += float(loss.item())
        pbar.set_postfix(loss=f"{total_loss / max(processed_steps, 1):.4f}")

    return {"total": total_loss / max(processed_steps, 1), "processed_steps": processed_steps}


@torch.inference_mode()
def validate_single_task(model, loader, device, task, loss_mode, bce_weight, threshold=0.5):
    model.eval()
    total_loss = 0.0
    steps = 0
    dice_sum = 0.0
    dice_cnt = 0
    amp_enabled = device.type == "cuda"

    for batch in loader:
        rgb_cross = batch["rgb_cross"].to(device, non_blocking=True)
        rgb_parallel = batch["rgb_parallel"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            result = model(rgb_cross, rgb_parallel, mask)

        logits, gt, has_gt = _task_tensors(task, result, batch, device)
        loss = _simple_seg_loss(logits, gt, mask, has_gt, mode=loss_mode, bce_weight=bce_weight)
        total_loss += float(loss.item())
        steps += 1

        dsum, dcnt = _dice_metric(logits, gt, mask, has_gt, threshold=threshold)
        dice_sum += dsum
        dice_cnt += dcnt

    return {
        "total": total_loss / max(steps, 1),
        f"{task}_dice": dice_sum / max(dice_cnt, 1),
    }


def run_single_task(task: str) -> None:
    parser = argparse.ArgumentParser(description=f"V11 single-task training ({task})")
    parser.add_argument("--patch_dir", type=str, default=v11.CFG["patch_dir"])
    parser.add_argument("--checkpoint_dir", type=str, default=f"./checkpoints_v11_{task}")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=v11.CFG["epochs"])
    parser.add_argument("--batch_size", type=int, default=v11.CFG["batch_size"])
    parser.add_argument("--loss_mode", type=str, choices=["bce", "dice_bce"], default="dice_bce")
    parser.add_argument("--bce_weight", type=float, default=0.5)
    args = parser.parse_args()

    cfg = copy.deepcopy(v11.CFG)
    cfg.update(
        patch_dir=args.patch_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device} | task={task} | patch_dir={cfg['patch_dir']} | "
        f"loss_mode={args.loss_mode} | bce_weight={args.bce_weight:.2f}"
    )

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_analyzer_v2(base_ch=cfg["base_ch"], low_r=cfg["low_r"], high_r=cfg["high_r"]).to(device)

    full_ds = TaskSpecificSkinDataset(cfg["patch_dir"], task=task, img_size=cfg["img_size"], augment=True)
    if len(full_ds) <= 1:
        raise RuntimeError("Need at least 2 samples for train/val split")

    val_size = max(1, int(len(full_ds) * cfg["val_ratio"]))
    if val_size >= len(full_ds):
        val_size = len(full_ds) - 1

    train_ds, val_ds = random_split(
        full_ds,
        [len(full_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    val_ds.dataset = copy.copy(full_ds)
    val_ds.dataset.augment = False

    prefetch_factor = cfg.get("prefetch_factor", 2) if cfg["num_workers"] > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["num_workers"] > 0,
        prefetch_factor=prefetch_factor,
        collate_fn=skin_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["num_workers"] > 0,
        prefetch_factor=prefetch_factor,
        collate_fn=skin_collate_fn,
    )

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 1
    best_dice = float("-inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", float("-inf"))

    print(f"Start {task}-only training...")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_log = train_one_epoch_single_task(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
            epoch,
            cfg["epochs"],
            task,
            args.loss_mode,
            args.bce_weight,
        )
        val_log = validate_single_task(
            model,
            val_loader,
            device,
            task,
            args.loss_mode,
            args.bce_weight,
            threshold=cfg.get("dice_threshold", 0.5),
        )
        scheduler.step()

        task_dice = val_log[f"{task}_dice"]
        is_best = task_dice > best_dice
        if is_best:
            best_dice = task_dice

        payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_dice": best_dice,
            "task": task,
            "loss_mode": args.loss_mode,
            "bce_weight": args.bce_weight,
            "train_loss": train_log,
            "val_loss": val_log,
            "cfg": cfg,
        }
        torch.save(payload, ckpt_dir / "last_checkpoint.pth")
        if is_best:
            torch.save(payload, ckpt_dir / f"best_{task}.pth")

        print(
            f"[{epoch:03d}/{cfg['epochs']}] trn={train_log['total']:.4f} "
            f"val={val_log['total']:.4f} {task}_dice={task_dice:.4f} "
            f"steps={train_log['processed_steps']}" + (" *" if is_best else "")
        )

    print("Training complete")
