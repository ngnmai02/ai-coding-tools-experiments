# Data Flow Through Pyramid Feature Attention Network

This document traces how data transforms from input to output through the entire pipeline.

**Assumptions:**
- Input image size: 256×256×3 (as specified in train.py)
- Batch size: N (variable, shown as first dimension)
- All operations use 'same' padding to preserve spatial dimensions where applicable

---

## Stage 1: Input Preprocessing (data.py)

```
Input Image (Original)
├─ Shape: (H, W, 3) - Variable size
├─ Operations:
│  ├─ BGR to RGB conversion
│  ├─ Mean subtraction: [103.939, 116.779, 123.68] per channel
│  ├─ Padding to square (if needed)
│  └─ Resize to (256, 256)
└─ Output: (256, 256, 3) - Float32, normalized
```

**Tensor Shape:** `(N, 256, 256, 3)`

---

## Stage 2: VGG16 Backbone - Feature Extraction

### Block 1: Early Features (Fine Details)
```
Input: (N, 256, 256, 3)
├─ Conv2D(64, 3×3) + ReLU
│  └─ Output: (N, 256, 256, 64)
├─ Conv2D(64, 3×3) + ReLU
│  └─ Output: (N, 256, 256, 64)
├─ C1 = (N, 256, 256, 64) ← SAVED
└─ MaxPooling2D(2×2, stride=2)
   └─ Output: (N, 128, 128, 64)
```

### Block 2: Mid-Level Features
```
Input: (N, 128, 128, 64)
├─ [Optional] Dropout(0.3)
├─ Conv2D(128, 3×3) + ReLU
│  └─ Output: (N, 128, 128, 128)
├─ Conv2D(128, 3×3) + ReLU
│  └─ Output: (N, 128, 128, 128)
├─ C2 = (N, 128, 128, 128) ← SAVED
└─ MaxPooling2D(2×2, stride=2)
   └─ Output: (N, 64, 64, 128)
```

### Block 3: Deep Features
```
Input: (N, 64, 64, 128)
├─ [Optional] Dropout(0.3)
├─ Conv2D(256, 3×3) + ReLU
│  └─ Output: (N, 64, 64, 256)
├─ Conv2D(256, 3×3) + ReLU
│  └─ Output: (N, 64, 64, 256)
├─ Conv2D(256, 3×3) + ReLU
│  └─ Output: (N, 64, 64, 256)
├─ C3 = (N, 64, 64, 256) ← SAVED
└─ MaxPooling2D(2×2, stride=2)
   └─ Output: (N, 32, 32, 256)
```

### Block 4: Deeper Features
```
Input: (N, 32, 32, 256)
├─ [Optional] Dropout(0.3)
├─ Conv2D(512, 3×3) + ReLU
│  └─ Output: (N, 32, 32, 512)
├─ Conv2D(512, 3×3) + ReLU
│  └─ Output: (N, 32, 32, 512)
├─ Conv2D(512, 3×3) + ReLU
│  └─ Output: (N, 32, 32, 512)
├─ C4 = (N, 32, 32, 512) ← SAVED
└─ MaxPooling2D(2×2, stride=2)
   └─ Output: (N, 16, 16, 512)
```

### Block 5: Deepest Features
```
Input: (N, 16, 16, 512)
├─ [Optional] Dropout(0.3)
├─ Conv2D(512, 3×3) + ReLU
│  └─ Output: (N, 16, 16, 512)
├─ Conv2D(512, 3×3) + ReLU
│  └─ Output: (N, 16, 16, 512)
├─ Conv2D(512, 3×3) + ReLU
│  └─ Output: (N, 16, 16, 512)
└─ C5 = (N, 16, 16, 512) ← SAVED
```

**Summary of Saved Features:**
- C1: `(N, 256, 256, 64)` - Full resolution, fine details
- C2: `(N, 128, 128, 128)` - Half resolution, mid-level features
- C3: `(N, 64, 64, 256)` - Quarter resolution, deep features
- C4: `(N, 32, 32, 512)` - Eighth resolution, deeper features
- C5: `(N, 16, 16, 512)` - Sixteenth resolution, deepest features

---

## Stage 3: Feature Refinement

### C1 and C2 Processing
```
C1: (N, 256, 256, 64)
├─ Conv2D(64, 3×3)
│  └─ Output: (N, 256, 256, 64)
└─ BatchNorm + ReLU
   └─ C1_refined: (N, 256, 256, 64)

C2: (N, 128, 128, 128)
├─ Conv2D(64, 3×3)
│  └─ Output: (N, 128, 128, 64)
└─ BatchNorm + ReLU
   └─ C2_refined: (N, 128, 128, 64)
```

---

## Stage 4: Context Pyramid Feature Extraction (CPFE)

### CFE Function Breakdown
For each input feature map, CFE applies:
1. **1×1 Convolution**: `Conv2D(32, 1×1)` → 32 channels
2. **Atrous Conv (rate=3)**: `Conv2D(32, 3×3, dilation=3)` → 32 channels
3. **Atrous Conv (rate=5)**: `Conv2D(32, 3×3, dilation=5)` → 32 channels
4. **Atrous Conv (rate=7)**: `Conv2D(32, 3×3, dilation=7)` → 32 channels
5. **Concatenate**: All 4 branches → 128 channels total
6. **BatchNorm + ReLU**: Final output

### Applying CFE to C3, C4, C5

```
C3: (N, 64, 64, 256)
└─ CFE(32 filters)
   └─ C3_cfe: (N, 64, 64, 128)  [4 branches × 32 channels]

C4: (N, 32, 32, 512)
└─ CFE(32 filters)
   └─ C4_cfe: (N, 32, 32, 128)

C5: (N, 16, 16, 512)
└─ CFE(32 filters)
   └─ C5_cfe: (N, 16, 16, 128)
```

### Upsampling to Common Scale
```
C5_cfe: (N, 16, 16, 128)
└─ BilinearUpsampling(4×)  [16×4 = 64]
   └─ C5_cfe_up: (N, 64, 64, 128)

C4_cfe: (N, 32, 32, 128)
└─ BilinearUpsampling(2×)  [32×2 = 64]
   └─ C4_cfe_up: (N, 64, 64, 128)

C3_cfe: (N, 64, 64, 128)
└─ (No upsampling needed)
```

### Concatenation
```
C3_cfe: (N, 64, 64, 128)
C4_cfe_up: (N, 64, 64, 128)
C5_cfe_up: (N, 64, 64, 128)
└─ Concatenate(axis=-1)
   └─ C345: (N, 64, 64, 384)  [128+128+128 = 384 channels]
```

---

## Stage 5: Channel-Wise Attention (CA)

```
C345: (N, 64, 64, 384)
├─ GlobalAveragePooling2D
│  └─ Output: (N, 384)  [Global average per channel]
├─ Dense(96) + ReLU  [384/4 = 96]
│  └─ Output: (N, 96)
├─ Dense(384) + Sigmoid  [Channel attention weights]
│  └─ Output: (N, 384)
├─ Reshape to (N, 1, 1, 384)
│  └─ Output: (N, 1, 1, 384)
├─ Repeat to (N, 64, 64, 384)
│  └─ Attention weights: (N, 64, 64, 384)
└─ Multiply with original C345
   └─ C345_attended: (N, 64, 64, 384)
```

**Effect:** Each channel is weighted by its global importance, emphasizing informative features.

---

## Stage 6: C345 Final Processing

```
C345_attended: (N, 64, 64, 384)
├─ Conv2D(64, 1×1)  [Channel reduction]
│  └─ Output: (N, 64, 64, 64)
├─ BatchNorm + ReLU
│  └─ Output: (N, 64, 64, 64)
└─ BilinearUpsampling(4×)  [64×4 = 256]
   └─ C345_final: (N, 256, 256, 64)
```

---

## Stage 7: Spatial Attention (SA)

```
C345_final: (N, 256, 256, 64)
├─ Branch 1 (Horizontal):
│  ├─ Conv2D(32, (1, 9))  [Horizontal kernel]
│  │  └─ Output: (N, 256, 256, 32)
│  ├─ BatchNorm + ReLU
│  ├─ Conv2D(1, (9, 1))  [Vertical kernel]
│  │  └─ Output: (N, 256, 256, 1)
│  └─ BatchNorm
│
├─ Branch 2 (Vertical):
│  ├─ Conv2D(32, (9, 1))  [Vertical kernel]
│  │  └─ Output: (N, 256, 256, 32)
│  ├─ BatchNorm + ReLU
│  ├─ Conv2D(1, (1, 9))  [Horizontal kernel]
│  │  └─ Output: (N, 256, 256, 1)
│  └─ BatchNorm
│
├─ Add both branches
│  └─ Output: (N, 256, 256, 1)
├─ Sigmoid activation
│  └─ Output: (N, 256, 256, 1)  [Attention map: 0-1]
└─ Repeat to (N, 256, 256, 64)
   └─ SA: (N, 256, 256, 64)  [Spatial attention weights]
```

**Effect:** Highlights important spatial locations across all channels.

---

## Stage 8: C12 Processing with Spatial Attention

```
C2_refined: (N, 128, 128, 64)
└─ BilinearUpsampling(2×)  [128×2 = 256]
   └─ C2_up: (N, 256, 256, 64)

C1_refined: (N, 256, 256, 64)
C2_up: (N, 256, 256, 64)
└─ Concatenate(axis=-1)
   └─ C12: (N, 256, 256, 128)  [64+64 = 128]

C12: (N, 256, 256, 128)
├─ Conv2D(64, 3×3)
│  └─ Output: (N, 256, 256, 64)
├─ BatchNorm + ReLU
│  └─ Output: (N, 256, 256, 64)
└─ Multiply with SA
   └─ C12_attended: (N, 256, 256, 64)
```

**Effect:** Fine detail features (C1, C2) are spatially weighted by attention from deep features.

---

## Stage 9: Final Fusion and Prediction

```
C12_attended: (N, 256, 256, 64)  [Fine details with spatial attention]
C345_final: (N, 256, 256, 64)    [Deep semantics]
└─ Concatenate(axis=-1)
   └─ fea: (N, 256, 256, 128)  [64+64 = 128]

fea: (N, 256, 256, 128)
└─ Conv2D(1, 3×3)  [Final prediction layer]
   └─ sa: (N, 256, 256, 1)  [Saliency map - logits]
```

---

## Stage 10: Output Post-Processing

During inference:
```
sa (logits): (N, 256, 256, 1)
└─ Sigmoid activation
   └─ Saliency map: (N, 256, 256, 1)  [Values: 0.0 - 1.0]
```

**Final Output:** Single-channel saliency map where each pixel value (0-1) represents the probability/importance of that location being salient.

---

## Complete Data Flow Summary

```
Input Image (256×256×3)
    ↓
VGG16 Backbone
    ├─→ C1 (256×256×64) ────────────────┐
    ├─→ C2 (128×128×128) ───────────┐   │
    ├─→ C3 (64×64×256) ────┐        │   │
    ├─→ C4 (32×32×512) ──┐ │        │   │
    └─→ C5 (16×16×512) ┐ │ │        │   │
                        │ │ │        │   │
CPFE Processing         │ │ │        │   │
    ├─→ C3_cfe (64×64×128) │        │   │
    ├─→ C4_cfe (32×32×128) ─→ up2×  │   │
    └─→ C5_cfe (16×16×128) ─→ up4×  │   │
                        │ │ │        │   │
    Concatenate → C345 (64×64×384)   │   │
                        │            │   │
    Channel Attention → (64×64×384)  │   │
                        │            │   │
    Conv(1×1) → (64×64×64)           │   │
                        │            │   │
    Upsample 4× → C345_final (256×256×64) │
                        │            │   │
    Spatial Attention → SA (256×256×64)   │
                        │            │   │
C1 Processing           │            │   │
    Conv(3×3) → (256×256×64) ────────┘   │
                        │                 │
C2 Processing           │                 │
    Conv(3×3) → (128×128×64)             │
    Upsample 2× → (256×256×64) ──────────┘
                        │
    Concatenate → C12 (256×256×128)
                        │
    Conv(3×3) → (256×256×64)
                        │
    Apply SA → C12_attended (256×256×64)
                        │
    Concatenate with C345_final
                        ↓
    Final Features (256×256×128)
                        ↓
    Conv(3×3, 1 channel) → Saliency Map (256×256×1)
```

---

## Key Transformations

1. **Spatial Resolution Changes:**
   - Input: 256×256
   - Through VGG: 256 → 128 → 64 → 32 → 16 (downsampling)
   - Through Upsampling: 16 → 64 → 256 (upsampling back)

2. **Channel Count Changes:**
   - Input: 3 channels (RGB)
   - VGG features: 64 → 128 → 256 → 512 → 512
   - After refinement: All reduced to 64 channels
   - CPFE: 4 parallel branches × 32 = 128 channels per level
   - Final fusion: 64 + 64 = 128 channels
   - Output: 1 channel (saliency)

3. **Information Flow:**
   - **Bottom-up**: VGG extracts hierarchical features
   - **Multi-scale**: CPFE captures context at different scales
   - **Attention**: CA and SA refine features
   - **Top-down**: Upsampling and fusion combine all levels
   - **Final**: Single prediction layer produces saliency map

