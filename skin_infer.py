"""
skin_infer.py
=============
SkinAnalyzer 추론 스크립트

기능
----
  1. 단일 이미지 쌍 추론
  2. 디렉토리 일괄 추론 (배치 처리)
  3. 결과 시각화 저장 (입력 | brown | red | wrinkle 가로 나열)
  4. 스코어 JSON 출력

실행 예시
---------
  # 단일 이미지 쌍
  python skin_infer.py \
      --cross  ./samples/cross.png \
      --parallel ./samples/parallel.png \
      --checkpoint ./checkpoints/best_skin_analyzer.pth \
      --out_dir ./results

  # 디렉토리 일괄 추론 (rgb_cross / rgb_parallel 폴더 구조 가정)
  python skin_infer.py \
      --input_dir ./patches \
      --checkpoint ./checkpoints/best_skin_analyzer.pth \
      --out_dir ./results \
      --batch_size 8

  # 마스크 적용
  python skin_infer.py \
      --cross  ./samples/cross.png \
      --parallel ./samples/parallel.png \
      --mask   ./samples/mask.png \
      --checkpoint ./checkpoints/best_skin_analyzer.pth \
      --out_dir ./results
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from skin_net import build_analyzer, SkinAnalyzer


# ══════════════════════════════════════════════════════════════════════════════
# 이미지 로드 유틸
# ══════════════════════════════════════════════════════════════════════════════
def load_rgb(path: Path, img_size: int) -> torch.Tensor:
    """RGB 이미지 로드 → [3, H, W] 텐서 (0~1)"""
    img = Image.open(path).convert('RGB').resize(
        (img_size, img_size), Image.BILINEAR)
    return TF.to_tensor(img)


def load_gray(path: Path, img_size: int) -> torch.Tensor:
    """Grayscale 이미지 로드 → [1, H, W] 텐서 (0~1)"""
    img = Image.open(path).convert('L').resize(
        (img_size, img_size), Image.BILINEAR)
    return TF.to_tensor(img)


# ══════════════════════════════════════════════════════════════════════════════
# 모델 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_model(checkpoint_path: str, device: torch.device) -> tuple[SkinAnalyzer, dict]:
    """
    체크포인트에서 모델 로드.

    Returns
    -------
    model : SkinAnalyzer  (eval 모드)
    cfg   : dict          학습 시 사용된 설정값
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt.get('cfg', {})

    model = build_analyzer(
        base_ch = cfg.get('base_ch', 64),
        low_r   = cfg.get('low_r',   0.1),
        high_r  = cfg.get('high_r',  0.4),
    ).to(device)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    epoch = ckpt.get('epoch', '?')
    val   = ckpt.get('val_loss', {})
    print(f"체크포인트 로드 완료  epoch={epoch}  "
          f"val_loss={val.get('total', '?'):.4f}")
    return model, cfg


# ══════════════════════════════════════════════════════════════════════════════
# 시각화 저장
# ══════════════════════════════════════════════════════════════════════════════
def save_visualization(
    rgb_cross:    torch.Tensor,   # [3, H, W]  0~1
    brown_prob:   torch.Tensor,   # [1, H, W]  0~1
    red_prob:     torch.Tensor,   # [1, H, W]  0~1
    wrinkle_prob: torch.Tensor,   # [1, H, W]  0~1
    save_path:    Path,
    brown_score:  float,
    red_score:    float,
    wrinkle_score: float,
) -> None:
    """
    [입력 | brown | red | wrinkle] 가로 격자 이미지 저장.
    각 맵 위에 스코어를 표시하기 위해 텍스트는 파일명에 포함.
    """
    inp_vis    = rgb_cross.clamp(0, 1)
    brown_vis  = brown_prob.repeat(3, 1, 1)
    red_vis    = red_prob.repeat(3, 1, 1)
    wrinkle_vis = wrinkle_prob.repeat(3, 1, 1)

    grid = vutils.make_grid(
        torch.stack([inp_vis, brown_vis, red_vis, wrinkle_vis]),
        nrow=4, padding=2, normalize=False,
    )
    img = TF.to_pil_image(grid)
    img.save(save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 단일 추론
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer_single(
    model:        SkinAnalyzer,
    cross_path:   Path,
    parallel_path: Path,
    mask_path:    Path | None,
    img_size:     int,
    device:       torch.device,
) -> dict:
    """
    단일 이미지 쌍 추론.

    Returns
    -------
    dict with keys:
        brown_mask, red_mask, wrinkle_mask  : torch.Tensor [1, H, W] (0~1 확률)
        brown_score, red_score, wrinkle_score : float (0~100)
    """
    rgb_cross    = load_rgb(cross_path,    img_size).unsqueeze(0).to(device)
    rgb_parallel = load_rgb(parallel_path, img_size).unsqueeze(0).to(device)

    if mask_path is not None and mask_path.exists():
        mask = load_gray(mask_path, img_size).unsqueeze(0).to(device)
    else:
        mask = None

    result = model(rgb_cross, rgb_parallel, mask)

    return {
        'brown_mask'    : torch.sigmoid(result.brown_mask[0]).cpu(),
        'red_mask'      : torch.sigmoid(result.red_mask[0]).cpu(),
        'wrinkle_mask'  : torch.sigmoid(result.wrinkle_mask[0]).cpu(),
        'brown_score'   : result.brown_score[0].item(),
        'red_score'     : result.red_score[0].item(),
        'wrinkle_score' : result.wrinkle_score[0].item(),
        'rgb_cross'     : rgb_cross[0].cpu(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 배치 추론 (디렉토리)
# ══════════════════════════════════════════════════════════════════════════════
class InferDataset(torch.utils.data.Dataset):
    """
    디렉토리 기반 추론 데이터셋.

    input_dir/
      rgb_cross/    {stem}.png
      rgb_parallel/ {stem}.png
      mask/         {stem}.png  (optional)
    """

    def __init__(self, input_dir: Path, img_size: int):
        self.input_dir = input_dir
        self.img_size  = img_size

        cross_dir = input_dir / 'rgb_cross'
        if not cross_dir.exists():
            raise FileNotFoundError(
                f"rgb_cross 폴더를 찾을 수 없습니다: {cross_dir}\n"
                "디렉토리 구조: input_dir/rgb_cross/, input_dir/rgb_parallel/"
            )

        self.stems = sorted(p.stem for p in cross_dir.glob('*.png'))
        if not self.stems:
            raise RuntimeError(f"rgb_cross 폴더에 .png 파일이 없습니다: {cross_dir}")

        print(f"추론 대상: {len(self.stems)}장")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        rgb_cross    = load_rgb(
            self.input_dir / 'rgb_cross'    / f'{stem}.png', self.img_size)
        rgb_parallel = load_rgb(
            self.input_dir / 'rgb_parallel' / f'{stem}.png', self.img_size)

        mask_path = self.input_dir / 'mask' / f'{stem}.png'
        if mask_path.exists():
            mask = load_gray(mask_path, self.img_size)
        else:
            mask = torch.ones(1, self.img_size, self.img_size)

        return {
            'rgb_cross'   : rgb_cross,
            'rgb_parallel': rgb_parallel,
            'mask'        : mask,
            'stem'        : stem,
        }


def _infer_collate(batch: list) -> dict:
    return {
        'rgb_cross'   : torch.stack([b['rgb_cross']    for b in batch]),
        'rgb_parallel': torch.stack([b['rgb_parallel'] for b in batch]),
        'mask'        : torch.stack([b['mask']         for b in batch]),
        'stem'        : [b['stem'] for b in batch],
    }


@torch.no_grad()
def infer_directory(
    model:      SkinAnalyzer,
    input_dir:  Path,
    out_dir:    Path,
    img_size:   int,
    batch_size: int,
    device:     torch.device,
    num_workers: int = 4,
) -> list[dict]:
    """
    디렉토리 일괄 추론.

    Returns
    -------
    list of dicts: [{'stem', 'brown_score', 'red_score', 'wrinkle_score'}, ...]
    """
    ds = InferDataset(input_dir, img_size)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = device.type == 'cuda',
        collate_fn  = _infer_collate,
    )

    vis_dir   = out_dir / 'visualizations'
    mask_dir  = out_dir / 'masks'
    vis_dir.mkdir(parents=True, exist_ok=True)
    (mask_dir / 'brown').mkdir(parents=True, exist_ok=True)
    (mask_dir / 'red').mkdir(parents=True, exist_ok=True)
    (mask_dir / 'wrinkle').mkdir(parents=True, exist_ok=True)

    all_scores = []

    pbar = tqdm(loader, desc='추론 중', unit='batch', dynamic_ncols=True)
    for batch in pbar:
        rgb_cross    = batch['rgb_cross'].to(device)
        rgb_parallel = batch['rgb_parallel'].to(device)
        mask         = batch['mask'].to(device)
        stems        = batch['stem']

        result = model(rgb_cross, rgb_parallel, mask)

        for i, stem in enumerate(stems):
            brown_prob   = torch.sigmoid(result.brown_mask[i]).cpu()    # [1,H,W]
            red_prob     = torch.sigmoid(result.red_mask[i]).cpu()      # [1,H,W]
            wrinkle_prob = torch.sigmoid(result.wrinkle_mask[i]).cpu()  # [1,H,W]

            b_score = result.brown_score[i].item()
            r_score = result.red_score[i].item()
            w_score = result.wrinkle_score[i].item()

            # 시각화 저장
            save_visualization(
                rgb_cross    = rgb_cross[i].cpu(),
                brown_prob   = brown_prob,
                red_prob     = red_prob,
                wrinkle_prob = wrinkle_prob,
                save_path    = vis_dir / f'{stem}_vis.png',
                brown_score  = b_score,
                red_score    = r_score,
                wrinkle_score = w_score,
            )

            # 개별 마스크 PNG 저장
            TF.to_pil_image(brown_prob).save(mask_dir / 'brown'   / f'{stem}.png')
            TF.to_pil_image(red_prob).save(  mask_dir / 'red'     / f'{stem}.png')
            TF.to_pil_image(wrinkle_prob).save(mask_dir / 'wrinkle' / f'{stem}.png')

            all_scores.append({
                'stem'         : stem,
                'brown_score'  : round(b_score, 2),
                'red_score'    : round(r_score, 2),
                'wrinkle_score': round(w_score, 2),
            })

        pbar.set_postfix(
            brown=f"{result.brown_score.mean().item():.1f}",
            red=f"{result.red_score.mean().item():.1f}",
            wrinkle=f"{result.wrinkle_score.mean().item():.1f}",
        )

    return all_scores


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='SkinAnalyzer 추론 스크립트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 공통 인수
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='학습된 체크포인트 경로 (.pth)')
    parser.add_argument('--out_dir',    type=str, default='./results',
                        help='결과 저장 디렉토리 (기본: ./results)')
    parser.add_argument('--img_size',   type=int, default=256,
                        help='추론 해상도 (기본: 256)')
    parser.add_argument('--device',     type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='추론 장치 (기본: auto)')

    # 단일 이미지 모드
    single = parser.add_argument_group('단일 이미지 모드')
    single.add_argument('--cross',    type=str, default=None,
                        help='교차 편광 이미지 경로')
    single.add_argument('--parallel', type=str, default=None,
                        help='평행 편광 이미지 경로')
    single.add_argument('--mask',     type=str, default=None,
                        help='피부 영역 마스크 경로 (optional)')

    # 디렉토리 일괄 추론 모드
    batch = parser.add_argument_group('디렉토리 일괄 추론 모드')
    batch.add_argument('--input_dir',   type=str, default=None,
                       help='입력 디렉토리 (rgb_cross/, rgb_parallel/ 하위 폴더 포함)')
    batch.add_argument('--batch_size',  type=int, default=8,
                       help='배치 크기 (기본: 8)')
    batch.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader 워커 수 (기본: 4)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 장치 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # 출력 디렉토리 생성
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    model, cfg = load_model(args.checkpoint, device)
    img_size   = args.img_size or cfg.get('img_size', 256)

    # ── 단일 이미지 모드 ──────────────────────────────────────────────────────
    if args.cross is not None and args.parallel is not None:
        cross_path    = Path(args.cross)
        parallel_path = Path(args.parallel)
        mask_path     = Path(args.mask) if args.mask else None

        if not cross_path.exists():
            raise FileNotFoundError(f"교차 편광 이미지를 찾을 수 없습니다: {cross_path}")
        if not parallel_path.exists():
            raise FileNotFoundError(f"평행 편광 이미지를 찾을 수 없습니다: {parallel_path}")

        print(f"\n단일 추론: {cross_path.name}")
        pred = infer_single(
            model        = model,
            cross_path   = cross_path,
            parallel_path = parallel_path,
            mask_path    = mask_path,
            img_size     = img_size,
            device       = device,
        )

        # 시각화 저장
        stem = cross_path.stem
        vis_path = out_dir / f'{stem}_vis.png'
        save_visualization(
            rgb_cross     = pred['rgb_cross'],
            brown_prob    = pred['brown_mask'],
            red_prob      = pred['red_mask'],
            wrinkle_prob  = pred['wrinkle_mask'],
            save_path     = vis_path,
            brown_score   = pred['brown_score'],
            red_score     = pred['red_score'],
            wrinkle_score = pred['wrinkle_score'],
        )

        # 개별 마스크 저장
        TF.to_pil_image(pred['brown_mask']).save(  out_dir / f'{stem}_brown.png')
        TF.to_pil_image(pred['red_mask']).save(    out_dir / f'{stem}_red.png')
        TF.to_pil_image(pred['wrinkle_mask']).save(out_dir / f'{stem}_wrinkle.png')

        # 스코어 출력
        scores = {
            'stem'         : stem,
            'brown_score'  : round(pred['brown_score'],   2),
            'red_score'    : round(pred['red_score'],     2),
            'wrinkle_score': round(pred['wrinkle_score'], 2),
        }
        print(f"\n결과 스코어:")
        print(f"  갈색 반점(brown)  : {scores['brown_score']:6.2f} / 100")
        print(f"  적색 반점(red)    : {scores['red_score']:6.2f} / 100")
        print(f"  주름 (wrinkle)    : {scores['wrinkle_score']:6.2f} / 100")

        score_path = out_dir / f'{stem}_scores.json'
        with open(score_path, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)

        print(f"\n저장 완료:")
        print(f"  시각화  : {vis_path}")
        print(f"  스코어  : {score_path}")

    # ── 디렉토리 일괄 추론 모드 ───────────────────────────────────────────────
    elif args.input_dir is not None:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")

        print(f"\n일괄 추론: {input_dir}")
        all_scores = infer_directory(
            model       = model,
            input_dir   = input_dir,
            out_dir     = out_dir,
            img_size    = img_size,
            batch_size  = args.batch_size,
            device      = device,
            num_workers = args.num_workers,
        )

        # 전체 스코어 JSON 저장
        scores_path = out_dir / 'scores.json'
        with open(scores_path, 'w', encoding='utf-8') as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)

        # 요약 통계
        if all_scores:
            n = len(all_scores)
            avg_b = sum(s['brown_score']   for s in all_scores) / n
            avg_r = sum(s['red_score']     for s in all_scores) / n
            avg_w = sum(s['wrinkle_score'] for s in all_scores) / n
            print(f"\n추론 완료  총 {n}장")
            print(f"  갈색 반점 평균: {avg_b:.2f}")
            print(f"  적색 반점 평균: {avg_r:.2f}")
            print(f"  주름 평균     : {avg_w:.2f}")
            print(f"  스코어 저장  : {scores_path}")
            print(f"  시각화 저장  : {out_dir / 'visualizations'}")
            print(f"  마스크 저장  : {out_dir / 'masks'}")

    else:
        raise ValueError(
            "추론 모드를 지정하세요.\n"
            "  단일: --cross <path> --parallel <path>\n"
            "  일괄: --input_dir <dir>"
        )


if __name__ == '__main__':
    main()
