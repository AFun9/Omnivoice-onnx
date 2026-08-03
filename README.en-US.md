

# OmniVoice ONNX Export / Quantization / Inference

[English README](README_EN.md)

Converts [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) (PyTorch) to ONNX, covering the full pipeline: **export → quantization (INT8 / INT8-HQ / INT4 / FP16) → numerical verification → end-to-end inference → performance benchmarking**.

> **Current production recommendation**: `int8hq` (LM 587→611 MB, audio output head kept in FP32). Numerical cosine similarity > 0.9999, argmax consistency > 80%, and audio quality is nearly indistinguishable from FP32. See [§7.2](#72--int8-hq推荐生产) for details.

## Prerequisites

This repository only contains **export / quantization / inference scripts**, and does not include model weights or test audio. Before running, you need:

1. **OmniVoice PyTorch Model** —— Clone the main repository from [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) and download the pre-trained weights to obtain an `OmniVoice/` directory (containing `model.safetensors` + `audio_tokenizer/`).
2. **`tts` conda environment** —— Install dependencies following the OmniVoice main repository's README (`torch`, `transformers>=5.5`, `onnx`, `onnxruntime`, `onnxsim`, `onnxconverter_common`, `torchaudio`).
3. **Reference Audio** (optional, only required for `voice_clone` demo) —— Any 24 kHz mono WAV file. This document defaults to `zero_short_prompt`.

Recommended layout:

```
your-workspace/
├── OmniVoice/                # k2-fsa/OmniVoice main repo (contains model weights)
│   ├── OmniVoice/             # Model weights directory (PT_MODEL_DIR)
│   ├── omnivoice/             # Python package
│   └── onnx_export/           # ← Clone this repository here
│       ├── _common.py
│       ├── export_lm.py
│       ├── zero_short_prompt     # voice_clone reference audio
│       └── ...
```

The top of `_common.py` defines paths: `PROJECT_ROOT = Path(__file__).parent.parent`, `PT_MODEL_DIR = PROJECT_ROOT / "OmniVoice"`. If your layout differs, simply update the two constants in `_common.py`.

## Table of Contents

1. [Overview](#1-概览)
2. [Directory Layout](#2-目录布局)
3. [One-Click Workflow](#3-一键流程)
4. [ONNX IO Conventions](#4-onnx-io-约定)
5. [Numerical Verification](#5-数值校验)
6. [Graph Optimization](#6-图优化)
7. [Quantization Strategies](#7-量化方案)
8. [End-to-End Inference](#8-端到端推理)
9. [Performance / RTF / Multi-threading](#9-性能--rtf--多线程)
10. [Troubleshooting & Rollback](#10-故障排查与回滚)

---

## 1. Overview

OmniVoice is a text-to-audio TTS model composed of two parts:

- **LM**: A Qwen3-like 600M parameter language model, outputting audio token logits with shape `[B, 8 codebooks, S, 1025]`
- **Audio Tokenizer**: HiggsAudioV2, split into an encoder (audio → 8 codebook codes) and a decoder (codes → audio)

This directory exports each of these three components as an independent ONNX file and provides 5 precision variants:

| variant | LM Size | Recommended Use Case |
|---------|---------|----------|
| `fp32` | 2.28 GB | Baseline / reference |
| `int8` | 587 MB | Size-priority, tolerates slight quality loss |
| **`int8hq`** | **611 MB** | **★ Production Recommended** (nearly lossless quality, only +24 MB size) |
| `int4` | 586 MB | Experimental: tighter size/speed, but noticeable quality drop |
| `fp16` | 1.14 GB | GPU deployment only (actually slowest on CPU) |

---

## 2. Directory Layout

Repository source (runtime artifacts are generated under `output/`, which is `.gitignore`d):

```
.
├── README.md                  # The document you are reading
├── .gitignore
├── _common.py                 # Shared paths / devices / loading / error printing / variant_paths
├── export_lm.py               # Export LM
├── export_audio_tokenizer.py  # Export audio tokenizer encoder + decoder
├── quantize.py                # ★ One-stop INT8 / INT8-HQ / INT4 / FP16 quantization
├── verify_lm.py               # ONNX vs PyTorch numerical verification (LM, per variant)
├── verify_audio_tokenizer.py  # ONNX vs PyTorch numerical verification (encoder / decoder)
├── test_pipeline.py           # End-to-end encode→decode test using assert/andelie.wav
├── infer_onnx.py              # ★ End-to-end TTS inference: LM + tokenizer fully ONNX
├── zero_short_prompt          # voice_clone reference audio
└── benchmark_lm.py            # ★ LM single-step latency micro-benchmark (thread sweep)
```

Automatically created after running (**not committed**):

```
output/
├── omnivoice_lm/                  ← FP32 LM (baseline)
├── omnivoice_lm_int8/             ← INT8 weight-only (W8A32)
├── omnivoice_lm_int8_hq/          ← ★ INT8-HQ: W8A32 + audio_heads kept FP32
├── omnivoice_lm_int4/             ← INT4 group-wise (MatMulNBits, RTN/HQQ)
├── omnivoice_lm_fp16/             ← FP16 LM (GPU path)
├── audio_tokenizer_encoder/        ← FP32 encoder
├── audio_tokenizer_encoder_int8/   ← INT8 encoder (shared by int8 / int8hq / int4)
├── audio_tokenizer_decoder/        ← FP32 decoder
├── audio_tokenizer_decoder_int8/   ← INT8 decoder (shared by int8 / int8hq / int4)
├── test_outputs/                   ← test_pipeline artifacts
└── inference_demo/
    ├── fp32/   demo_auto.wav, demo_voice_clone.wav, demo_voice_design.wav
    ├── int8/   ...
    ├── int8hq/ ...     ← ★ Listening comparison baseline
    ├── int4/   ...
    └── fp16/   ...
```

> All quantization results are in separate subdirectories and will not overwrite each other. Deleting a subdirectory fully rolls back that variant.

---

## 3. One-Click Workflow

```bash
conda activate tts
cd path/to/OmniVoice/onnx_export   # See §Prerequisites recommended layout

# 1. Export FP32 (one-time, ~5-10 mins)
python export_audio_tokenizer.py
python export_lm.py

# 2. Numerical verification (FP32)
python verify_audio_tokenizer.py
python verify_lm.py --variant fp32

# 3. End-to-end pipeline test (using real audio assert/andelie.wav)
python test_pipeline.py

# 4. Quantize to recommended int8hq (LM only; AT model uses standard int8)
python quantize.py --targets at_encoder,at_decoder --methods int8 --no-smoke
python quantize.py --targets lm --methods int8 --int8-mode weight_only_hq --no-smoke

# 5. int8hq numerical verification
python verify_lm.py --variant int8hq

# 6. ★ End-to-end TTS inference (generates 3 demo clips for listening test)
python infer_onnx.py --variant int8hq
```

---

## 4. ONNX IO Conventions

### 4.1 `omnivoice_lm/model.onnx` (interface shared across all variants)

| name | dtype | shape |
|------|-------|-------|
| **in** `input_ids` | int64 | `[batch, 8, sequence]` |
| **in** `audio_mask` | bool | `[batch, sequence]` |
| **in** `attention_mask` | bool | `[batch, 1, sequence, sequence]` |
| **in** `position_ids` | int64 | `[batch, sequence]` |
| **out** `logits` | float32 | `[batch, 8, sequence, 1025]` |

The interface maps 1:1 to how `OmniVoice._generate_iterative` internally calls `self.forward()`. The generate loop constructs a `[2B, 1, S, S]` 4-D boolean block-diagonal mask (first B are cond, next B are uncond, for CFG), making this ONNX a true drop-in replacement drivable directly by the generator.

### 4.2 `audio_tokenizer_encoder/model.onnx`

| name | dtype | shape |
|------|-------|-------|
| **in** `audio` | float32 | `[batch, 1, num_samples]`, 24 kHz mono, length must be a multiple of `hop_length=960` |
| **out** `audio_codes` | int64 | `[batch, 8, num_frames]`, `num_frames = num_samples / 960` |

### 4.3 `audio_tokenizer_decoder/model.onnx`

| name | dtype | shape |
|------|-------|-------|
| **in** `audio_codes` | int64 | `[batch, 8, num_frames]` |
| **out** `audio` | float32 | `[batch, 1, num_samples]`, `num_samples = num_frames * 960` |

> The 8 codebooks output by the encoder map 1:1 to the LM's `num_audio_codebook=8`, allowing direct interchange.

---

## 5. Numerical Verification

| Target | FP32 Threshold | int8hq Threshold | Notes |
|---------|----------|-------------|------|
| LM logits | `max\|Δ\| ≤ 5e-3`, `cos ≥ 0.999`, argmax ≥ 99.9 % | `max\|Δ\| ≤ 5`, `cos ≥ 0.9999`, argmax ≥ 80 % | Thresholds for all variants are in `verify_lm.py::TOLERANCES` |
| Audio codes (encoder) | exact-match ≥ 99 % | — | Int quantization is sensitive |
| Decoded audio (decoder) | `max\|Δ\| ≤ 5e-3`, `cos ≥ 0.999` | — | — |

Exit code non-zero from verification scripts indicates failure.

---

## 6. Graph Optimization

Each export script applies the following optimizations by default:

1. `torch.onnx.export(..., do_constant_folding=True)`: Torch's built-in constant folding.
2. **onnxsim**: Constant folding, Identity elimination, shape inference, dead-node pruning.
3. **onnxruntime.transformers.optimizer** (LM only): `gpt2` preset for RMSNorm / SkipLayerNorm / RotaryEmbedding fusion.

LM node counts before/after optimization:

| Stage | Node Count | Key Changes |
|------|--------|----------|
| Raw export | 7092 | `Constant` 2380, `Pow/ReduceMean/Sqrt` 113 each (RMSNorm fully expanded) |
| onnxsim + ORT optimizer | **3176** (-55 %) | `Constant` 0, RMSNorm fused into 57 `SimplifiedLayerNormalization` ops |

Add `--no-optimize` to skip steps 2 & 3.

---

## 7. Quantization Strategies

### 7.0 Overview

| variant | LM Size | LM cos | argmax | Audio Quality | Command |
|---------|---------|--------|--------|------|------|
| FP32 | 2.28 GB | 1.0 | 100 % | — | (baseline) |
| INT8 dynamic | 587 MB | 0.99987 | 36–42 % | Audible distortion | `--int8-mode dynamic` |
| INT8 weight-only | 587 MB | 0.99999 | 71–75 % | Close to FP32, occasional slight metallic artifact | `--int8-mode weight_only` (default) |
| **★ INT8-HQ** | **611 MB** | **0.99999+** | **80–91 %** | **Indistinguishable from FP32** | `--int8-mode weight_only_hq` |
| INT4 (RTN) | 586 MB | 0.9990 | ~5 % | Obvious distortion | `--methods int4 --int4-algo rtn` |
| INT4 (HQQ) | 586 MB | 0.9991 | ~6 % | Still distorted | `--methods int4 --int4-algo hqq` (default) |
| FP16 | 1.14 GB | 0.99995 | 85 % | Indistinguishable from FP32 | `--methods fp16` |

> The INT8 build for AT-encoder/decoder (`audio_tokenizer_*_int8/`) is shared across `int8`, `int8hq`, and `int4` LM variants. Audio tokenizer quantization has far less impact on audio quality than the LM, so a separate `hq` variant is unnecessary.

### 7.1 Weight-Only INT8 (W8A32)

Performs **per-row symmetric INT8** quantization on all 2D FP32 weights ≥ 1 MB, inserting a `DequantizeLinear` node before each weight. Activations remain FP32 (W8A32 = 8-bit weights + 32-bit activations).

```bash
python quantize.py --targets lm,at_encoder,at_decoder --methods int8
```

Numerical performance:
- 198 LM weights quantized (includes `embed_tokens`, `audio_embeddings`, all attention/MLP MatMul)
- max\|Δ\| 0.87–3.17, cos 0.99999+, argmax consistency 71–75 %
- AT encoder/decoder processed similarly (some convs remain FP32 due to ≥ 3D shapes, but size still drops to ~60 %)

### 7.2 ★ INT8-HQ (Recommended for Production)

#### Motivation

Standard INT8 weight-only preserves 95 % of the logit direction (cos > 0.99999), but **argmax consistency is only 71–75 %**. Why?

Diagnostics show the issue concentrates in the **LM's audio output head**:

- It's a `[1024, 8200] = 8 codebooks × 1025 vocab` MatMul that directly produces logits for the sampler
- The top-k logits for 1025 audio tokens often have spacing < 0.05; per-row INT8 quantization noise is sufficient to flip argmax
- Quantization noise in the other 197 layers gets "averaged out" by subsequent non-linearities and doesn't propagate 1:1 to final logits

**INT8-HQ = Standard INT8 + selective MatMul nodes kept in FP32**. By default, only `audio_heads` are protected. Size increases by only 24 MB (611 vs 587), boosting argmax consistency from 75 % to 80–91 % and making audio quality indistinguishable from FP32.

#### Export Command

```bash
# Default: protect audio_heads
python quantize.py --targets lm --methods int8 --int8-mode weight_only_hq

# Protect more layers (comma-separated node name substrings)
python quantize.py --targets lm --methods int8 --int8-mode weight_only_hq \
                   --int8hq-exclude "audio_heads,o_proj,down_proj"

# Disable exclusions completely (equivalent to --int8-mode weight_only, but outputs to _hq dir)
python quantize.py --targets lm --methods int8 --int8-mode weight_only_hq \
                   --int8hq-exclude ""
```

Implementation details (`quantize.py`):

| Function | Purpose |
|------|------|
| `_init_names_for_node_patterns(model, patterns, op_types=("MatMul","Gemm"))` | Scans the computation graph, finds MatMul/Gemm nodes whose names contain any substring in `patterns`, returns the set of initializer names they use |
| `quantize_initializers_int8(model, exclude_init_names=...)` | Adds `exclude_init_names` parameter to the base W8A32 quantization function; excluded weights remain FP32 |
| `quantize_int8_weight_only(..., exclude_node_patterns=...)` | High-level entry: passes node name patterns, resolves to initializer names internally, then calls the above functions |
| `--int8-mode weight_only_hq` | Uses dedicated job list `JOBS_INT8HQ` in main (LM only, outputs to `omnivoice_lm_int8_hq/`), preventing pollution of the standard `int8` directory |

> AT models are not re-quantized in hq mode. They directly reuse the `audio_tokenizer_*_int8/` directory. This is controlled by `_common.py::variant_paths("int8hq")`.

#### Testing Protocol

Complete testing follows three layers, increasing in importance:

**A. Offline Numerical Verification** (~5 seconds)

```bash
python verify_lm.py --variant int8hq
```

Runs 4 `(batch, seq)` configurations, comparing against PyTorch FP32:

```
  [logits] shape=(1, 8,  32, 1025)  max|Δ|=6.6e-01  cos=0.999999   argmax 80.9 %
  [logits] shape=(1, 8,  64, 1025)  max|Δ|=1.1e+00  cos=0.999999   argmax 91.0 %
  [logits] shape=(1, 8, 128, 1025)  max|Δ|=7.7e-01  cos=1.000000   argmax 87.3 %
  [logits] shape=(2, 8,  96, 1025)  max|Δ|=3.1e+00  cos=1.000003   argmax 87.2 %
ALL CASES PASSED ✓
```

Thresholds in `verify_lm.py::TOLERANCES["int8hq"]`: `max|Δ| ≤ 5`, `cos ≥ 0.9999`, `argmax ≥ 0.80`.

**B. Computation Graph Self-Check** (verify exclusion rules actually applied)

```bash
python -c "
import onnx
m = onnx.load('output/omnivoice_lm_int8_hq/model.onnx', load_external_data=False)
n_dq = sum(1 for n in m.graph.node if n.op_type == 'DequantizeLinear')
n_int8 = sum(1 for i in m.graph.initializer if i.data_type == 3)
fp32_big = [(i.name, list(i.dims)) for i in m.graph.initializer
            if i.data_type == 1 and len(i.dims) == 2 and i.dims[0]*i.dims[1]*4 >= 1024*1024]
print(f'DequantizeLinear nodes: {n_dq}')
print(f'INT8 initializers: {n_int8}')
print(f'FP32 2-D weights >= 1MB still in graph ({len(fp32_big)}):')
for nm, dims in fp32_big:
    print(f'  {nm}: {dims} ({dims[0]*dims[1]*4/1024/1024:.1f} MB)')
"
```

Expected output (default exclude `audio_heads`):

```
DequantizeLinear nodes: 198
INT8 initializers: 198
FP32 2-D weights >= 1MB still in graph (1):
  onnx::MatMul_9011: [1024, 8200] (32.0 MB)   ← audio output head
```

**C. End-to-End Listening Test** (final criterion, ~100 seconds each)

```bash
python infer_onnx.py --variant int8hq
```

Generates three WAV files to `output/inference_demo/int8hq/`:

| File | Mode | Text | Key |
|------|------|------|------|
| `demo_auto.wav` | auto | Long English sentence | No reference, model auto-selects voice |
| `demo_voice_clone.wav` | voice clone | Short Chinese sentence | Uses `zero_short_prompt.wav` (Chinese female) as reference |
| `demo_voice_design.wav` | voice design | Short English sentence | `instruct = "male, british accent, low pitch, middle-aged"` |

**Comparison Method**: Listen to the `int8/`, `int8hq/`, and `fp32/` versions of the same demo sequentially. Focus on:
- Metallic/hissing artifacts on vowels (primary degradation point for INT8)
- High-frequency crispness (INT8 often dulls this)
- Naturalness of phrase endings
- Tone/prosody under Chinese cloning

Expectation: `int8hq` is indistinguishable from `fp32`; standard `int8` may exhibit occasional metallic artifacts and slight clicks.

#### Rollback

```bash
rm -rf output/omnivoice_lm_int8_hq output/inference_demo/int8hq
```

This will not affect other variants.

### 7.3 INT8 Dynamic (Not Recommended for LM)

```bash
python quantize.py --int8-mode dynamic
```

- Uses ORT's `quantize_dynamic`; both weights and activations are INT8
- Pros: ~2× faster on CPU than weight_only (due to INT8 GEMM fused ops)
- Cons: Qwen3-like models have many activation outliers; per-tensor absmax scaling amplifies quantization error, causing obvious quality degradation (argmax consistency only 36–42 %)
- Only recommended for AT-encoder/decoder (conv activation distributions are more uniform). **Do not use for LM**.

### 7.4 INT4 Group-wise (Experimental)

```bash
# HQQ (default, better quality)
python quantize.py --methods int4 --int4-algo hqq --int4-block-size 16

# RTN (faster, slightly worse quality)
python quantize.py --methods int4 --int4-algo rtn --int4-block-size 16
```

- Uses `onnxruntime.quantization.matmul_4bits_quantizer.MatMul4BitsQuantizer`, outputs `MatMulNBits` custom op
- Default `block_size=16`, HQQ asymmetric, `audio_heads` excluded (same output head protection as int8hq)
- Size 586 MB, but argmax consistency only ~5–6 %, with obvious quality degradation. **Not recommended for production**.
- Kept for research: exploring `MatMul4BitsQuantizer` API with different `block_size` / algorithms.

### 7.5 FP16

```bash
python quantize.py --methods fp16
```

- `onnxconverter_common.float16.convert_float_to_float16`, `keep_io_types=True`, `op_block_list=[]`
- LM size 1.14 GB (2× reduction)
- High numerical precision (nearly identical to PyTorch FP32)
- **Actually slowest on CPU**: x86 lacks native FP16 SIMD; every fp16 op casts to fp32 at runtime before computation, and casting itself dominates overhead.
- Only worthwhile on GPU (`--ort-provider CUDAExecutionProvider`).

---

## 8. End-to-End Inference

`infer_onnx.py` design: Preserves PyTorch's `OmniVoice.generate(...)` framework (handles tokenization / text preprocessing / diffusion sampling loop / CFG / audio post-processing), and monkey-patches three neural network entry points:

| PyTorch Call | Replaced With |
|---|---|
| `model.forward(input_ids, audio_mask, attention_mask, ...)` → `.logits` | `omnivoice_lm[_int8 / _int8_hq / _int4 / _fp16]/model.onnx` |
| `model.audio_tokenizer.encode(audio)` → `.audio_codes` | `audio_tokenizer_encoder[_int8]/model.onnx` |
| `model.audio_tokenizer.decode(audio_codes)` → `.audio_values` | `audio_tokenizer_decoder[_int8]/model.onnx` |

Variant is resolved to concrete triplet paths via `_common.py::variant_paths(variant)`.

```bash
# Full 3 demos, outputs to output/inference_demo/<variant>/
python infer_onnx.py --variant int8hq

# Run only one specific demo
python infer_onnx.py --variant int8hq --only demo_voice_clone

# Adjust sampling steps (default 32; fewer = faster, slightly lower quality)
python infer_onnx.py --variant int8hq --num-step 16
```

**`instruct` field convention**: In `voice_design` mode, the `instruct` field must use the fixed vocabulary supported by the model (see `omnivoice/models/omnivoice.py::_resolve_instruct`). Free-text is not accepted. Common enums:
- English: `american accent / british accent / indian accent / male / female / low pitch / high pitch / whisper / young adult / middle-aged / elderly`
- Chinese: `男 / 女 / 老年 / 中年 / 高音调 / 低音调 / 河南话` etc.

---

## 9. Performance / RTF / Multi-threading

### 9.1 RTF Definition

`RTF = wall-clock time / audio duration`. Lower is better; `< 1.0` means faster than real-time.

### 9.2 LM Single-Step Benchmark (`benchmark_lm.py`)

The LM accounts for ~95 % of the total generation time (one LM forward pass per sampling step), so benchmarking the LM alone directly predicts overall RTF.

```bash
python benchmark_lm.py --variant int8hq --threads 1,4,8,12,16,24 --shapes 1x256,1x512,1x1024
```

**i9-14900KF (8 P-cores with hyper-threading + 16 E-cores) measured LM 1024-token forward**:

| intra_op_num_threads | median latency | relative to 1 thread | marginal gain |
|---:|---:|---:|---:|
| 1 | 7926 ms | 1.0× | — |
| 4 | 2457 ms | 3.2× | primary gain |
| **8** | **1633 ms** | **4.9×** | **★ optimal efficiency** |
| 12 | 1563 ms | 5.1× | +4 % |
| 16 | 1512 ms | 5.2× | +3 % |
| 24 | 1506 ms | 5.3× | ~0 % |

**Conclusion**:

- **8 threads is the optimal efficiency point**: exactly fills the 8 physical P-cores
- The ~7 % gain from 8→16 comes from P-core hyper-threading; further yields diminish
- 16→24 is nearly saturated, as additional threads schedule to E-cores (4.4 GHz vs P-cores 6.0 GHz, IPC also 30 % lower), which actually drags down BLAS execution rhythm

### 9.3 Recommended Deployment Config

In `infer_onnx.py::_make_session`, the default is `intra_op_num_threads = 0` (lets ORT decide, usually equals physical cores = 24). To reliably hit the optimum:

```python
so.intra_op_num_threads = 8     # For 14900K / 13900K class machines
# Or pin to P-cores: use taskset -c 0-15 python infer_onnx.py ... at startup
```

Or via CLI:

```bash
taskset -c 0-15 python infer_onnx.py --variant int8hq
```

> Optimal thread counts for different CPUs:
> - Intel 12/13/14th gen consumer K: physical P-core count (8, 6, 8)
> - AMD Ryzen / Threadripper: physical core count (no P/E hybrid)
> - Xeon / EPYC servers: physical core count, but avoid cross-node NUMA scheduling

### 9.4 End-to-End RTF (i9-14900KF default config, `--num-step 32`)

| variant | demo_auto (9.2 s) | demo_voice_clone (7.8 s) | demo_voice_design (6.4 s) |
|---------|------------------:|-------------------------:|---------------------------:|
| FP32 | 1.34 | 3.84 | 1.45 |
| INT8 weight-only | 2.92 | 6.74 | 3.63 |
| **INT8-HQ** | **2.79** | **6.69** | **3.42** |
| INT8 dynamic | 1.19 | 3.36 | 1.32 |
| FP16 (CPU) | 5.43 | 14.82 | 6.68 |

> Notes:
> - `voice_clone` RTF is significantly higher due to the extra ref-audio encode step (~5–10 s one-time overhead) and Chinese token count being ~2× English.
> - INT8 weight-only / int8hq are ~2× slower than INT8 dynamic because ORT lacks a `DequantizeLinear+MatMul` fused op; weights must be dequantized back to fp32 before each MatMul. However, audio quality is significantly better—this is a quality vs. speed tradeoff.
> - The first run will take 5–8 seconds longer than steady state (ORT graph optimization + PyTorch loading).

### 9.5 GPU Path (Estimated)

Current INT8/INT8-HQ ONNX models are optimized for CPU (`DequantizeLinear` actually drags performance on GPU because it won't fuse into INT8 GEMM). To leverage GPU performance:

| Path | Changes | Expected RTF (RTX 4090) |
|------|------|--------:|
| ORT-CUDA + FP16 | `pip install onnxruntime-gpu` + `--variant fp16 --ort-provider CUDAExecutionProvider --device cuda:0` | **0.05 – 0.15** |
| TensorRT EP | Convert fp16 onnx to TRT engine + INT8 calibration | 0.03 – 0.08 |
| vLLM / TensorRT-LLM | Switch to LLM-specific stack (requires rewriting generate loop) | 0.02 – 0.05 |

> i.e., 4090 + ORT-CUDA + FP16 ≈ **20–50× faster** than current CPU INT8-HQ, and 7–20× faster than real-time.

---

## 10. Troubleshooting & Rollback

### 10.1 Single Variant Rollback

```bash
# Delete a specific LM variant (does not affect others)
rm -rf output/omnivoice_lm_int8_hq output/inference_demo/int8hq
rm -rf output/omnivoice_lm_int4    output/inference_demo/int4
rm -rf output/omnivoice_lm_int8    output/inference_demo/int8
rm -rf output/omnivoice_lm_fp16    output/inference_demo/fp16
```

### 10.2 Full Reset

```bash
rm -rf output/                          # Deletes all ONNX artifacts + test wav files
```

### 10.3 Common Errors

| Symptom | Cause | Solution |
|------|------|------|
| `IndexError: tuple index out of range` in `transformers/masking_utils.py` | Tracing stage wraps python int into 0-D tensor by transformers | Automatically monkey-patched via `_common.py::patch_transformers_sdpa_mask_for_tracing` |
| `Could not find an implementation for ConvInteger` | INT8 dynamic quantized Conv, but ORT-CPU lacks this op | `quantize_int8_dynamic` already restricts `op_types_to_quantize=["MatMul","Gemm","Attention"]` |
| `Unable to find data type for weight_name='...'` | quantize_dynamic failed to infer intermediate tensor dtype | `extra_options["DefaultTensorType"] = onnx.TensorProto.FLOAT` already added |
| `should be stored in .../model.onnx.data, but it doesn't exist` | External data sidecar not found | Loading order issue; fixed by `_normalize_sidecar` |
| Significant audio quality degradation during inference | Used `--variant int8` (standard dynamic / weight_only) | Switch to `--variant int8hq` |
| `--variant fp16` extremely slow on CPU | x86 lacks FP16 SIMD; FP16 is intended for GPU path | Switch back to `int8hq` or use `--ort-provider CUDAExecutionProvider` |
