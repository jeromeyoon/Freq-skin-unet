from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from skin_dataset import _load_gray, _load_rgb
from skin_net_v2 import build_analyzer_v2


class TaskInferenceDataset(Dataset):
    """Task-specific inference dataset.

    - brown/red: rgb_cross + rgb_parallel + (optional) mask
    - wrinkle: rgb_cross + rgb_parallel_wrinkle + (optional) mask_wrinkle
    """

    def __init__(self, input_dir: str, task: str, img_size: int = 256):
        if task not in {"brown", "red", "wrinkle"}:
            raise ValueError(f"unsupported task: {task}")

        self.input_dir = Path(input_dir)
        self.task = task
        self.img_size = int(img_size)

        self.cross_dir = self.input_dir / "rgb_cross"
        self.parallel_dir = self.input_dir / ("rgb_parallel_wrinkle" if task == "wrinkle" else "rgb_parallel")
        self.mask_dir = self.input_dir / ("mask_wrinkle" if task == "wrinkle" else "mask")

        if not self.cross_dir.exists():
            raise FileNotFoundError(f"rgb_cross folder not found: {self.cross_dir}")
        if not self.parallel_dir.exists():
            raise FileNotFoundError(f"parallel folder not found: {self.parallel_dir}")

        cross_stems = {p.stem for p in self.cross_dir.glob("*.png")}
        parallel_stems = {p.stem for p in self.parallel_dir.glob("*.png")}
        self.stems = sorted(cross_stems & parallel_stems)
        if not self.stems:
            raise RuntimeError("No valid samples found for inference")

        print(
            f"[TaskInferenceDataset] task={self.task} samples={len(self.stems)} "
            f"parallel_dir={self.parallel_dir.name} mask_dir={self.mask_dir.name if self.mask_dir.exists() else '(none)'}"
        )

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        rgb_cross = _load_rgb(self.cross_dir / f"{stem}.png", self.img_size)
        rgb_parallel = _load_rgb(self.parallel_dir / f"{stem}.png", self.img_size)

        mask_path = self.mask_dir / f"{stem}.png"
        if self.mask_dir.exists() and mask_path.exists():
            mask = _load_gray(mask_path, self.img_size)
        else:
            mask = torch.ones(1, self.img_size, self.img_size)

        return {
            "stem": stem,
            "rgb_cross": rgb_cross,
            "rgb_parallel": rgb_parallel,
            "mask": mask,
        }


def _collate(batch):
    return {
        "stem": [b["stem"] for b in batch],
        "rgb_cross": torch.stack([b["rgb_cross"] for b in batch]),
        "rgb_parallel": torch.stack([b["rgb_parallel"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }


def _save_prob_png(prob: torch.Tensor, path: Path):
    arr = (prob.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def run_task_inference(task: str):
    parser = argparse.ArgumentParser(description=f"V11 single-task inference ({task})")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_analyzer_v2().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = TaskInferenceDataset(args.input_dir, task=task, img_size=args.img_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=_collate,
    )

    out_root = Path(args.output_dir) if args.output_dir else (Path(args.input_dir) / f"infer_{task}")
    out_prob = out_root / "prob"
    out_bin = out_root / "bin"
    out_prob.mkdir(parents=True, exist_ok=True)
    out_bin.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Infer {task}"):
            rgb_cross = batch["rgb_cross"].to(device, non_blocking=True)
            rgb_parallel = batch["rgb_parallel"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            result = model(rgb_cross, rgb_parallel, mask)
            if task == "brown":
                logits = result.brown_mask
            elif task == "red":
                logits = result.red_mask
            else:
                logits = result.wrinkle_mask

            prob = torch.sigmoid(logits) * mask
            pred = (prob >= args.threshold).float()

            for i, stem in enumerate(batch["stem"]):
                _save_prob_png(prob[i, 0], out_prob / f"{stem}.png")
                _save_prob_png(pred[i, 0], out_bin / f"{stem}.png")

    print(f"Done. saved to: {out_root}")

