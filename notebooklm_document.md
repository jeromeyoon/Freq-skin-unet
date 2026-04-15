# FreqAwareUNet: 주변광 강인 피부 분석 모델
## NotebookLM 학습 문서 — 배경 지식 + 코드 설명

---

# 1부: 배경 지식

---

## 1.1 피부의 발색단 (Skin Chromophores)

피부의 색은 크게 세 가지 발색단(chromophore)에 의해 결정된다.

### 멜라닌 (Melanin)
- 피부 기저층(basal layer)과 극세포층(stratum spinosum)에 존재하는 색소
- 갈색~흑색 계열 색상 담당
- 자외선(UV)을 흡수해 피부를 보호하는 역할
- 과다 생성 시 기미(melasma), 잡티(age spot), 주근깨(freckle) 등 과색소 병변 발생
- 피부 분석에서는 "brown channel" 또는 "melanin map"으로 표현

### 헤모글로빈 (Hemoglobin)
- 진피(dermis)의 혈관 내 적혈구에 존재하는 붉은 색소
- 산화 헤모글로빈(oxy-Hb): 밝은 붉은색
- 환원 헤모글로빈(deoxy-Hb): 어두운 보라색
- 홍조(redness), 혈관 확장증, 여드름 흉터 등과 관련
- 피부 분석에서는 "red channel" 또는 "hemoglobin map"으로 표현

### 주름 (Wrinkle)
- 콜라겐(collagen)과 엘라스틴(elastin) 섬유의 분해로 형성
- 진피와 표피의 경계에서 발생하는 기계적 변형
- 주름의 깊이와 방향은 피부 노화의 주요 지표
- 주름은 피부 표면의 미세 요철(texture)이므로 고주파 성분에 해당
- 피부 분석에서는 "wrinkle map"으로 표현

---

## 1.2 Beer-Lambert 법칙 (Beer-Lambert Law)

### 기본 원리
Beer-Lambert 법칙은 빛이 물질을 통과할 때 흡수되는 정도를 설명하는 물리 법칙이다.

**수식:**
```
I_out = I_in × exp(-α × c × d)
```
- I_in: 입사 광 강도
- I_out: 투과 광 강도
- α: 흡수 계수 (absorptivity)
- c: 발색단 농도 (concentration)
- d: 광 경로 길이 (path length)

### 피부에 적용
피부에서 RGB 이미지는 다음과 같이 표현된다:

```
RGB = Illuminant × exp(-OD_tissue)
```
- Illuminant: 조명 강도 (조명 조건에 따라 변함)
- OD_tissue: 조직의 Optical Density (발색단 정보)

### Optical Density (OD) 변환
```
OD = -log(RGB) = -log(Illuminant) + OD_tissue
```

**핵심 통찰:**
- log 공간에서 조명(Illuminant)은 단순한 **덧셈 오프셋**
- OD_tissue = -log(RGB) + log(Illuminant)
- 만약 조명을 알거나 제거할 수 있다면 발색단 정보만 추출 가능

### 발색단 분리
```
OD_tissue = α_mel × c_mel + α_hem × c_hem + (기타)
```
- α_mel: 멜라닌 흡수 계수 (R:0.28, G:0.18, B:0.09)
- α_hem: 헤모글로빈 흡수 계수 (R:0.10, G:0.35, B:0.05)
- c_mel, c_hem: 멜라닌, 헤모글로빈 농도

### 조명 불변 특징: 채널 로그 비율
```
log(R/G) = OD_R - OD_G = (α_mel_R - α_mel_G) × c_mel + ...
```
- 중립 조명(neutral illuminant) 가정 시, 채널 비율은 조명에 무관
- log(R/G), log(G/B)는 조명 불변(illumination-invariant) 특징

---

## 1.3 편광 이미지 (Polarized Light Imaging)

### 편광의 원리
빛은 파동으로, 특정 방향으로 진동할 수 있다. 편광 필터를 사용하면 특정 방향의 빛만 통과시킬 수 있다.

### 교차 편광 (Cross Polarization)
- 조명과 카메라 앞에 각각 편광 필터를 설치
- 두 필터를 **90도** 회전시켜 배치 (서로 교차)
- 피부 표면에서 반사되는 빛(정반사, specular reflection)은 편광 방향이 유지되어 카메라로 들어오지 못함
- **표면 반사 제거 → 피부 하부(진피)의 발색단 정보만 포착**
- 멜라닌, 헤모글로빈 등 발색단 분석에 적합
- Beer-Lambert 법칙이 물리적으로 잘 적용됨

### 평행 편광 (Parallel Polarization)
- 두 편광 필터를 **같은 방향**으로 배치
- 표면 반사 빛도 카메라로 입력됨
- **표면 텍스처 정보 포함 → 주름, 모공 등 표면 구조 포착에 적합**
- 주름은 피부 표면의 기하학적 변형이므로 평행 편광 이미지에서 더 잘 보임

### 우리 기기의 특징
VISIA와 달리 **외부 조명에 노출된 환경**에서 촬영:
- 주변광 세기, 색온도, 방향이 촬영마다 다름
- → 조명 변화에 강인한 모델이 필요

---

## 1.4 주파수 도메인 분석 (Frequency Domain Analysis)

### 주파수와 이미지 정보
이미지를 2D 푸리에 변환(FFT)하면 공간 주파수별로 분해할 수 있다:

| 주파수 대역 | 포함 정보 | 피부 분석 의미 |
|------------|---------|-------------|
| 저주파 (low) | 전체적인 밝기, 색조, 그라데이션 | **조명 성분** |
| 중주파 (mid) | 넓은 영역의 색 변화, 패치 | **발색단 (melanin, hemoglobin)** |
| 고주파 (high) | 세밀한 텍스처, 경계, 요철 | **주름, 모공** |

### Retinex 이론과의 연관
Retinex 이론에서:
```
log(Image) = log(Illumination) + log(Reflectance)
```
- Illumination = 저주파 성분 (공간적으로 천천히 변함)
- Reflectance = 중/고주파 성분 (발색단, 텍스처)

→ 저주파를 억제하면 조명 영향을 줄일 수 있음

### FrequencyGate 개념
```
FFT2D → fftshift → 반경별 마스크 → 학습 가능한 gate → ifft2D
```
- low_r = 0.1: 정규화 반경 0~0.1 → 조명 성분 (억제)
- high_r = 0.4: 정규화 반경 0.4~1.0 → 주름 성분 (보존)
- 중간 (0.1~0.4): 발색단 성분 (보존)

**게이트 초기화:**
- low_logit = -2.0 → sigmoid ≈ 0.12 (조명 강하게 억제)
- mid_logit = +2.0 → sigmoid ≈ 0.88 (발색단 보존)
- high_logit = +2.0 → sigmoid ≈ 0.88 (주름 보존)
- 학습 중 데이터 기반으로 최적값 자동 수렴

---

## 1.5 Multi-Task Learning

### 기본 개념
하나의 네트워크에서 여러 task를 동시에 학습:
- **공유 특징 추출기(Encoder)**: 공통 정보 학습
- **Task-specific Head**: 각 task별 특화 예측

### 장점
1. 데이터 효율성: 공유 표현 학습으로 적은 데이터로도 효과적
2. 정규화 효과: 여러 task가 서로 규제 역할
3. 관련 task 간 지식 공유

### 부분 GT 학습 (Partial Label Learning)
현실에서는 모든 task의 GT가 항상 존재하지 않음:
- 일부 환자는 갈색 반점 GT만 있음
- 일부는 주름 GT만 있음
→ GT가 있는 task만 loss 계산, 없는 task는 0으로 처리
→ 모든 데이터 활용 가능

---

# 2부: 코드 설명

---

## 2.1 전체 아키텍처 개요

```
입력
 ├── rgb_cross    [B,3,H,W]  교차 편광 이미지
 └── rgb_parallel [B,3,H,W]  평행 편광 이미지

Stage 1-A: CrossPolEncoder (ir_encoder.py)
  ODPreprocess → Encoder(4 level) → FrequencyGate × 3
  출력: chroma_feats (mid-freq), bottleneck

Stage 1-B: ParallelPolEncoder (parallel_encoder.py)
  ParallelPreprocess → Encoder(4 level) → HighFreqGate × 3
  출력: texture_feats (high-freq), bottleneck

Stage 2: Task Heads (task_heads.py)
  brown_head   ← chroma_feats → brown_mask [B,1,H,W] + brown_score [B]
  red_head     ← chroma_feats → red_mask   [B,1,H,W] + red_score   [B]
  wrinkle_head ← texture_feats → wrinkle_mask [B,1,H,W] + wrinkle_score [B]
```

---

## 2.2 ir_encoder.py — 교차 편광 인코더

### ODPreprocess
교차 편광 이미지를 Beer-Lambert 기반으로 전처리한다.

```
입력: RGB [B,3,H,W]
처리:
  1. OD = -log(RGB)                   → [B,3,H,W]
  2. log(R/G) = OD_R - OD_G          → [B,1,H,W]
  3. log(G/B) = OD_G - OD_B          → [B,1,H,W]
  4. 5채널 concat [OD, R/G, G/B]     → [B,5,H,W]
  5. 1×1 Conv(5→3)                   → [B,3,H,W]
출력: [B,3,H,W]
```

초기화 전략:
- 1×1 Conv의 OD 채널 부분: identity 초기화 (OD 물리 보장)
- log-ratio 채널: 0 초기화 (학습으로 발견)

### FrequencyGate
FFT 기반 주파수 대역 분리:

```
입력: [B,C,H,W]
처리:
  1. fft2d + fftshift → 주파수 공간
  2. 반경별 마스크 생성 (low/mid/high)
  3. 학습 가능한 gate × mask
  4. mid-freq 성분: ifft2d → x_chroma
  5. high-freq 성분: ifft2d → x_texture
출력: (x_chroma, x_texture)
```

### IlluminationRobustEncoder (= CrossPolEncoder)
4-level U-Net Encoder:

```
입력: RGB [B,3,H,W]
  ↓ ODPreprocess
enc1: ConvBlock(3→64)     [B,64,H,W]   + FrequencyGate → chroma1, texture1
  ↓ MaxPool2d
enc2: ConvBlock(64→128)   [B,128,H/2]  + FrequencyGate → chroma2, texture2
  ↓ MaxPool2d
enc3: ConvBlock(128→256)  [B,256,H/4]  + FrequencyGate → chroma3, texture3
  ↓ MaxPool2d
enc4: ConvBlock(256→512)  [B,512,H/8]  bottleneck

출력: EncoderOutput(chroma=[c1,c2,c3], texture=[t1,t2,t3], bottleneck)
```

ConvBlock 구조: Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU

---

## 2.3 parallel_encoder.py — 평행 편광 인코더

### ParallelPreprocess
평행 편광에는 OD 변환 대신 Instance Normalization 사용:

```
이유:
- OD(-log) 변환은 Beer-Lambert (투과 발색단) 기반
- 평행 편광은 표면 반사 포함 → OD 가정 위반
- Instance Normalization: 각 이미지의 평균/분산 제거
  → 조명 밝기 차이 직접 정규화
```

```
입력: RGB [B,3,H,W]
처리:
  1. InstanceNorm2d(3, affine=True)  → 조명 통계 제거
  2. 1×1 Conv(3→3)                  → 채널 혼합 (학습 가능)
출력: [B,3,H,W]
```

### HighFreqGate
평행 편광용 고주파 특화 게이트:

```
low_logit  = -2.0 → sigmoid ≈ 0.12 (조명 억제)
high_logit = +2.0 → sigmoid ≈ 0.88 (주름 보존)

forward(x) → x_texture  (고주파 성분만 반환)
```

주름은 표면 텍스처(고주파)이므로 high-freq만 추출.

---

## 2.4 task_heads.py — 예측 헤드

### SpotHead (갈색/적색 반점)
교차 편광의 chroma features로 반점 예측:

```
입력: chroma_feats=[c1,c2,c3], bottleneck
디코더:
  up3: UpBlock(512, skip=256, out=256)
  up2: UpBlock(256, skip=128, out=128)
  up1: UpBlock(128, skip=64,  out=64)
  head: Conv(64→32) → ReLU → Conv(32→1) → sigmoid

출력:
  mask  [B,1,H,W]  반점 확률 맵 (0~1)
  score [B]        스코어 0~100

UpBlock 구조:
  ConvTranspose2d(×2) → concat(skip) → ConvBlock
```

스코어 계산:
```
area_ratio = (mask > 0.5).sum() / skin_mask.sum()
density    = (mask × skin_mask).max()
score      = (area_ratio × 0.6 + density × 0.4) × 100
```

### WrinkleHead
평행 편광의 texture features로 주름 예측:

```
입력: texture_feats=[t1,t2,t3]
처리:
  t2, t3를 t1 해상도로 upsample
  concat: [t1, t2_up, t3_up] → [B, 64+128+256, H, W]
  Conv(448→64) → BN → ReLU → Conv(64→32) → ReLU → Conv(32→1) → sigmoid

출력:
  mask  [B,1,H,W]
  score [B]
```

---

## 2.5 skin_net.py — SkinAnalyzer (통합 모델)

전체 모델을 통합:

```python
class SkinAnalyzer(nn.Module):
    def __init__(self, base_ch=64, low_r=0.1, high_r=0.4):
        self.cross_encoder    = CrossPolEncoder(...)   # 교차 편광
        self.parallel_encoder = ParallelPolEncoder(...)  # 평행 편광
        self.brown_head   = SpotHead(base_ch)
        self.red_head     = SpotHead(base_ch)
        self.wrinkle_head = WrinkleHead(base_ch)

    def forward(self, rgb_cross, rgb_parallel, skin_mask=None):
        cross_out    = self.cross_encoder(rgb_cross)
        parallel_out = self.parallel_encoder(rgb_parallel)

        brown_mask,   brown_score   = self.brown_head(cross_out.chroma, ...)
        red_mask,     red_score     = self.red_head(cross_out.chroma, ...)
        wrinkle_mask, wrinkle_score = self.wrinkle_head(parallel_out.texture, ...)

        return SkinResult(
            brown_mask, brown_score,
            red_mask, red_score,
            wrinkle_mask, wrinkle_score
        )
```

두 인코더가 독립적인 이유:
- 교차 편광: 발색단 정보 (OD 전처리 적합)
- 평행 편광: 표면 텍스처 정보 (InstanceNorm 적합)
- 물리적으로 다른 정보 → 가중치 공유 불필요

---

## 2.6 skin_loss.py — 손실 함수

### 학습 3단계

**Phase 1 (0~40%): 기본 수렴**
```
L = w_brown × L_brown + w_red × L_red + w_wrinkle × L_wrinkle
  + w_recon × L_recon
```
- 빠른 초기 수렴 목적
- Beer-Lambert recon으로 물리 일관성 강제

**Phase 2 (40~80%): 조명 불변성**
```
L = Phase1 + w_consist × L_consistency
```
- 조명 augmentation 적용한 이미지의 예측이 원본과 동일해야 함
- 모델이 조명 변화에 강인해지도록 학습

**Phase 3 (80~100%): 주파수 정규화**
```
L = Phase2 + w_freq_reg × L_freq_reg
```
- FrequencyGate의 low_logit을 0으로 억제
- 저주파(조명) 게이트가 완전히 닫히도록 강제

### Beer-Lambert 재구성 Loss
```
OD_recon = brown_mask × mel_abs + red_mask × hem_abs
RGB_recon = exp(-OD_recon)
L_recon = L1(RGB_recon, rgb_cross)  [교차 편광 기준]
```
이 loss가 있으면 모델이 Beer-Lambert 물리 법칙을 따르도록 유도됨.

### 부분 GT Loss (Partial Label Loss)
```
GT 가용 마스크: gt_avail [B,1,1,1] (has_brown: True/False per sample)
L_partial = Σ(|pred - gt| × face_mask × gt_avail) / Σ(face_mask × gt_avail)
```
- GT가 없는 샘플은 loss 기여 0
- GT가 있는 샘플만으로 평균 계산

---

## 2.7 skin_dataset.py — 데이터셋

### manifest.json 구조
```json
{
  "M100010006_0001": {
    "subject_name": "M100010006-M7910",
    "patch_idx": 1,
    "patch_pos": [256, 0],
    "has_brown": true,
    "has_red": false,
    "has_wrinkle": true
  }
}
```

### 부분 GT 처리
```python
def load_gt(task):
    if not info[f'has_{task}']:
        return None                   # GT 없음 → None 반환
    return load_gray(path)            # GT 있음 → 로드

brown   = load_gt('brown')   # None 가능
red     = load_gt('red')     # None 가능
wrinkle = load_gt('wrinkle') # None 가능
```

### skin_collate_fn
DataLoader에서 None을 처리하기 위한 커스텀 collate 함수:
- None GT → zeros 텐서로 대체
- has_* 플래그로 실제 GT 유무 전달
- Loss에서 has_* 플래그 기반으로 마스킹

---

## 2.8 data_prep.py — 데이터 준비

### 처리 파이프라인
```
1. input_path 스캔: 모든 ID_name 폴더 탐색
2. F_10.jpg (교차), F_11.jpg (평행) 확인
3. GT ID 매칭 (5단계 우선순위)
4. 얼굴 마스크 생성 (MediaPipe)
5. 패치 추출 (stride, patch_size)
6. 얼굴 마스크 커버리지 필터 (min_mask_coverage)
7. 패치 저장 + manifest.json 생성
```

### ID 매칭 전략 (5단계)
```
입력 폴더: M100010006-M7910
GT 폴더:   M100010006

1단계: 정확히 일치    → X
2단계: - 앞 부분     "M100010006" == GT?  → ✓
3단계: _ 앞 부분     (해당 없음)
4단계: GT ⊂ input   (해당 없음)
5단계: 숫자 추출     (fallback)
```

### MediaPipe 얼굴 마스크
```
Face Mesh 36개 윤곽선 랜드마크 → 폴리곤 생성
→ fillPoly → 침식(erode) × 2회
→ 귀, 목 자동 제외
```
MediaPipe 미설치 시: 이미지 중앙 타원형 fallback

---

## 2.9 전체 학습 흐름 요약

```
[데이터 준비]
python data_prep.py --input_path ./raw --gt_path ./gt --patch_dir ./patches

  F_10.jpg (교차) ─┐
  F_11.jpg (평행) ─┤→ 패치 추출 + 얼굴 마스크 + manifest.json
  GT(brown/red/wrinkle) ─┘

[학습]
python skin_train.py

  Epoch 1~40  (Phase 1):
    rgb_cross → CrossPolEncoder → brown/red_head
    rgb_parallel → ParallelPolEncoder → wrinkle_head
    Loss = L_supervised + L_recon

  Epoch 41~80 (Phase 2):
    + Illumination Consistency Loss
    (rgb_aug_cross, rgb_aug_parallel → same prediction)

  Epoch 81~100 (Phase 3):
    + Frequency Gate Regularization
    (low_gate → 0: 조명 성분 완전 억제)

[출력]
  checkpoints/best_skin_analyzer.pth
  → brown_mask + brown_score
  → red_mask   + red_score
  → wrinkle_mask + wrinkle_score
```

---

## 2.10 파라미터 및 설정값

| 파라미터 | 기본값 | 의미 |
|---------|--------|------|
| base_ch | 64 | U-Net 기본 채널 수 |
| low_r | 0.1 | 저주파(조명) 반경 |
| high_r | 0.4 | 고주파(주름) 시작 반경 |
| patch_size | 256 | 패치 크기 (픽셀) |
| stride | 256 | 패치 추출 보폭 |
| min_mask_coverage | 0.3 | 최소 얼굴 마스크 비율 |
| epochs | 100 | 학습 에폭 수 |
| batch_size | 16 | 배치 크기 |
| lr | 1e-4 | AdamW 학습률 |
| w_brown/red/wrinkle | 1.0 | 발색단/주름 loss 가중치 |
| w_recon | 0.3 | Beer-Lambert recon 가중치 |
| w_consist | 0.3 | 조명 일관성 가중치 |
| w_freq_reg | 0.1 | 주파수 게이트 정규화 가중치 |

---

## 2.11 기존 방법과의 차별점

| 항목 | 일반 U-Net | DINOv2 기반 | FreqAwareUNet (본 모델) |
|------|-----------|------------|----------------------|
| 조명 강인성 | 없음 | 사전학습 의존 | 물리적 주파수 분리 |
| 입력 | RGB 1장 | RGB 1장 | 편광 2장 (교차+평행) |
| 전처리 | 없음 | patch embed | OD + InstanceNorm |
| Skip 연결 | 표준 concat | DPT 구조 | 주파수 대역별 분리 |
| 주름 처리 | 단일 디코더 | 단일 head | 평행 편광 전용 head |
| 물리 기반 loss | 없음 | 없음 | Beer-Lambert recon |
| 외부 의존성 | 없음 | DINOv2 필요 | 없음 (순수 CNN) |
| 파라미터 수 | ~30M | ~300M+ | ~10M |

---

## 2.12 향후 개선 방향

1. **Face Segmentation 강화**: MediaPipe 대신 face_parsing (BiSeNet)으로 더 정밀한 마스크
2. **Attention Mechanism**: cross-attention으로 교차/평행 편광 정보 융합
3. **Self-supervised Pre-training**: VISIA GT 없이 OD reconstruction으로 사전학습
4. **Score Calibration**: 임상 데이터로 스코어 0~100 보정
5. **Uncertainty Estimation**: 예측 불확실성 출력 (의료 신뢰성)

---

## 용어 정리

| 용어 | 설명 |
|------|------|
| Chromophore (발색단) | 특정 파장의 빛을 흡수하여 색을 내는 물질 |
| OD (Optical Density) | 광학 밀도, OD = -log(투과율) |
| Beer-Lambert Law | 빛의 흡수량이 농도와 경로에 비례하는 법칙 |
| Cross Polarization | 교차 편광, 표면 반사 제거, 발색단 포착 |
| Parallel Polarization | 평행 편광, 표면 텍스처 포착 |
| FrequencyGate | FFT 기반 학습 가능한 주파수 대역 필터 |
| Partial Label | 일부 샘플에만 GT가 있는 학습 방식 |
| Multi-task Learning | 하나의 모델로 여러 task 동시 학습 |
| Instance Normalization | 각 이미지 독립적으로 정규화, 조명 제거 효과 |
| Collate Function | DataLoader에서 배치 구성 방식을 커스텀하는 함수 |
| Manifest | 패치별 메타 정보를 담은 JSON 파일 |
