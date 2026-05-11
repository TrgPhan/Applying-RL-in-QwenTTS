# Đặc tả triển khai: RL cho gwen-tts-0.6B (Vietnamese voice cloning)

**Version:** 2.0
**Target model:** `g-group-ai-lab/gwen-tts-0.6B` (finetune từ `Qwen/Qwen3-TTS-12Hz-0.6B-Base`)
**Target audience:** Code-generation model + researcher
**Framework:** PyTorch 2.x, transformers 4.40+, qwen-tts package
**Date:** 2026-05

**Update v2.0:** Bổ sung Section 15 (Đánh giá thực tế RL có cải thiện không), Section 16 (Tối ưu memory), Section 17 (Tối ưu thời gian train), Section 18 (Cải thiện performance không hi sinh chất lượng) — toàn bộ có căn cứ paper/tools 2024-2026.

---

## 0. Tóm tắt điều hành (TL;DR)

Spec này mô tả **roadmap 3 giai đoạn** áp dụng RL cho `gwen-tts-0.6B` để cải thiện WER và SIM trên tiếng Việt:

```
Gate 1: SFT continuation (LM CE loss + KL anchor)     → 1 tuần
         │
         ▼ [nếu chưa đủ target]
Gate 2: GRPO minimal (Whisper WER reward only)        → 1.5 tuần
         │
         ▼ [nếu concept work]
Gate 3: GRPO multi-reward (WER + SIM + length + entropy) → 2 tuần
```

**Vì sao chọn GRPO thay vì DiffRO/DPO?**
- **GRPO** có precedent rõ ràng trên LLaSA (single-codebook TTS LLM, kiến trúc gần với Qwen3-TTS): WER giảm 31%, SIM tăng từ 0.684→0.758 ([arXiv 2511.21270](https://arxiv.org/abs/2511.21270))
- **DiffRO** cần Gumbel-Softmax → numerical instability với bf16, chỉ cải thiện codebook 0 ([arXiv 2507.05911](https://arxiv.org/abs/2507.05911))
- **DPO** cần dataset preference → không có sẵn

**Kỳ vọng thực tế (không hứa hẹn):**
- WER giảm **10-25% relative** trên test Vietnamese
- SIM tăng **2-5%**
- Có rủi ro reward hacking → cần monitoring

**Compute requirement:** 1× GPU 40GB+ (A100/H100/A6000 48GB) với LoRA + bf16

---

## 1. Hiểu rõ kiến trúc gwen-tts-0.6B

### 1.1. Sơ đồ kiến trúc

```
Input: text (Vietnamese), ref_audio, ref_text
│
├─→ Text tokenizer (Qwen3 BPE)
│       text → text_tokens [T_text]
│
├─→ Speech tokenizer encoder (Qwen-TTS-Tokenizer-12Hz, FROZEN)
│       ref_audio → ref_code [T_ref, 16]   # 16 codebooks at 12.5Hz
│
├─→ Speaker encoder (FROZEN, optional)
│       ref_audio → ref_spk_embedding [d_spk]
│
├─→ Talker (28-layer Qwen3 transformer, ~600M params)
│       Inputs: text_tokens, ref_code (prepended), ref_spk_embedding
│       Outputs: Predicted codebook 0 tokens [T_gen]
│       At each step t:
│           hidden_t = Transformer(text + ref_code + generated_so_far)
│           logits_0_t = Linear(hidden_t)  # vocab=2048
│           token_0_t = sample(logits_0_t)
│
├─→ MTP / Code Predictor (5-layer transformer, ~80M params)
│       Inputs: hidden_t from talker
│       Outputs: Codebooks 1-15 in parallel [T_gen, 15]
│       For each codebook k=1..15:
│           logits_k_t = Linear_k(hidden_t)
│           token_k_t = sample(logits_k_t)
│
├─→ Concat: full_codes [T_gen, 16]
│
└─→ Mimi codec decoder (FROZEN)
        full_codes → wav_24k [T_wav]
```

### 1.2. Các module và param counts

| Module | Params | Train/Freeze (RL) |
|---|---|---|
| Text tokenizer | 0 (vocab table) | Frozen |
| Speech tokenizer encoder | ~50M | **Frozen luôn** |
| Speaker encoder | ~10M | **Frozen luôn** |
| **Talker (28 layers)** | **~550M** | **Trainable (LoRA)** |
| MTP / Code Predictor (5 layers) | ~80M | Frozen Gate 1-2, optional unfreeze Gate 3 |
| Mimi decoder | ~30M | **Frozen luôn** |

**Total trainable** với LoRA r=16 trên talker: ~5M params (~1% của 0.6B)

### 1.3. Generation config (chuẩn cho gwen-tts-0.6B)

```python
generation_config = dict(
    temperature=0.3,
    top_k=20,
    top_p=0.9,
    max_new_tokens=4096,
    repetition_penalty=2.0,
    subtalker_do_sample=True,
    subtalker_temperature=0.1,
    subtalker_top_k=20,
    subtalker_top_p=1.0,
)
```

**Quan trọng:** Giữ nguyên config này cho RL rollouts để match inference distribution.

### 1.4. Sampling rate và token rates

- **Audio sample rate:** 24000 Hz (output)
- **Codec frame rate:** 12.5 Hz (mỗi token = 80ms audio)
- **Codec codebooks:** 16 (1 semantic + 15 acoustic)
- **Codebook size:** 2048 (= 11 bits)
- **Total bitrate:** 12.5 × 16 × 11 = 2200 bps

Cho audio 5 giây:
- Codebook 0 sequence: 62 tokens
- Total tokens: 1000

→ AR sequence ngắn, GRPO/DiffRO khả thi.

---

## 2. Phân tích tính khả thi RL chi tiết

### 2.1. So sánh với precedents

| Paper | Model | Method | WER ↓ relative | SIM ↑ |
|---|---|---|---|---|
| [DiffRO (2507.05911)](https://arxiv.org/abs/2507.05911) | CosyVoice 2.0 (single codebook) | Gumbel-Softmax + ASR reward | **50%** | not reported |
| [Multi-Reward GRPO (2511.21270)](https://arxiv.org/abs/2511.21270) | LLaSA-8B (single codebook) | GRPO + 5 rewards | **31-46%** | **+11%** |
| [CosyVoice 2 DPO](https://funaudiollm.github.io/pdf/CosyVoice_2.pdf) | CosyVoice 2.0 SFT | DPO with preference pairs | **15-25%** | not reported |

### 2.2. Tại sao Qwen3-TTS có thể đạt cải thiện thấp hơn

**Yếu tố làm giảm kỳ vọng:**

1. **Multi-codebook architecture** — RL trên codebook 0 (semantic) chỉ ảnh hưởng 1/16 token stream. CosyVoice 2 single codebook → toàn bộ token stream được tối ưu.
2. **Đã finetune mạnh trên Vietnamese** — checkpoint đã ở local optimum gần tốt → marginal improvement nhỏ hơn.
3. **TikTok data có noise** — Whisper reward signal không clean như benchmark Seed-TTS.
4. **Vietnamese ASR khó hơn Chinese/English** — Whisper-large-v3 WER tiếng Việt ~7-10%, cao hơn Chinese (2-3%).

**Updated kỳ vọng:**
- WER giảm 10-25% relative
- SIM tăng 2-5%
- Naturalness có thể giữ nguyên hoặc giảm nhẹ (cần human eval)

### 2.3. Thách thức kỹ thuật và mitigation

| Thách thức | Mức độ | Mitigation |
|---|---|---|
| Reward hacking trên Whisper | **Cao** | Multi-reward (SIM + length + entropy) + KL trust region + early stopping |
| bf16 numerical instability | Trung bình | Mixed precision: model bf16, loss/reward fp32 |
| Distribution shift TikTok→Whisper | **Cao** | Filter data SNR ≥ 15dB; benchmark Whisper trên dataset trước |
| Voice cloning ref_code cut logic | Trung bình | Replicate exact cut logic từ inference vào RL loop |
| MTP gradient (nếu unfreeze) | Cao | Frozen ở Gate 1-2; chỉ unfreeze ở Gate 3 với LR ×0.1 |
| Repetition penalty mismatch | Thấp | Dùng same gen config cho rollouts và inference |
| Catastrophic forgetting đa ngôn ngữ | Trung bình | KL anchor mạnh, có thể train bilingual nếu cần |

### 2.4. Tại sao chọn GRPO làm phương án chính?

**5 lý do:**

1. **Có precedent gần** — LLaSA-8B + GRPO trong paper 2511.21270 là kiến trúc gần nhất với Qwen3-TTS (single-codebook AR LM). Architecture của Qwen3-TTS-12Hz tuy multi-codebook nhưng codebook 0 vẫn dominant cho semantic → có thể coi như single-codebook RL.

2. **Stable hơn DiffRO** — không cần Gumbel-Softmax (numerical issues với bf16). Group normalization handle reward scale automatically.

3. **Multi-reward natively** — có thể combine WER + SIM + length + entropy + prosody. DiffRO khó kết hợp vì khác scale.

4. **Không cần preference dataset** như DPO.

5. **Token-level reward** — natural credit assignment qua AR sequence (62 tokens cho 5s audio).

### 2.5. Khi nào KHÔNG nên dùng GRPO

- Nếu compute budget < 24GB VRAM → dùng DiffRO single-codebook
- Nếu có sẵn high-quality preference dataset (10K+ pairs) → DPO faster
- Nếu chỉ muốn improve naturalness (MOS) → Whisper reward không đủ, cần human feedback model

---

## 3. Notation và ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| \( x \) | Input text (Vietnamese, đã normalize) |
| \( c_{\text{ref}} \) | Reference codes [T_ref, 16] |
| \( s_{\text{ref}} \) | Speaker embedding |
| \( y_{\text{ref}} \) | Reference waveform (cho SIM reward) |
| \( T \) | Text transcript |
| \( o_t \) | Generated codebook 0 token tại step t |
| \( o_{1:T_{\text{gen}}} \) | Full generated codebook 0 sequence |
| \( c^{1:15}_t \) | Generated codebooks 1-15 tại step t (từ MTP) |
| \( y \) | Decoded waveform từ Mimi |
| \( \pi_\theta \) | Policy = talker với LoRA params \(\theta\) |
| \( \pi_{\text{ref}} \) | Reference policy (frozen copy của model gốc) |
| \( \pi_{\text{old}} \) | Policy của step trước (cho importance sampling trong GRPO) |
| \( G \) | Group size (số rollouts per prompt) |
| \( A_t \) | Advantage tại step t |
| \( R \) | Total reward (scalar per rollout) |
| \( r_{\text{wer}}, r_{\text{sim}}, r_{\text{len}}, r_{\text{ent}} \) | Component rewards |
| \( \alpha_i \) | Reward coefficients |
| \( \beta_{\text{KL}} \) | KL penalty coefficient |
| \( \varepsilon_{\text{clip}} \) | PPO/GRPO clipping ratio |

---

## 4. Dataset và preprocessing

### 4.1. Yêu cầu dataset

**Tối thiểu:**
- 5,000-10,000 utterances Vietnamese
- Mỗi utterance: `(text, ref_audio_path, ref_text, target_audio_path)` 
- Trong RL voice cloning: ref và target có thể là cùng 1 audio (self-reference) hoặc khác speaker

**Khuyến nghị:**
- Subset của dataset đã finetune (1000h TikTok của gwen-tts) — ~10K samples
- Filter SNR ≥ 15dB để giảm noise reward
- Filter duration 2-15 giây
- Vietnamese only cho Gate 1-2; multilingual cho Gate 3 nếu muốn maintain other languages

### 4.2. Preprocessing pipeline

```python
# File: preprocess_rl_data.py
import torchaudio
import torch
from qwen_tts import Qwen3TTSModel
from vietnamese_text_normalization import normalize_vi_text  # cần build hoặc dùng package

SR_TARGET = 24000  # Qwen3-TTS native rate

def preprocess_for_rl(items: list[dict], model: Qwen3TTSModel, output_path: str):
    """Preprocess data cho RL training."""
    processed = []
    for item in items:
        # 1. Load và normalize text
        text_normalized = normalize_vi_text(item["text"])
        ref_text_normalized = normalize_vi_text(item["ref_text"])
        
        # 2. Load audio và resample
        wav, sr = torchaudio.load(item["target_audio"])
        if sr != SR_TARGET:
            wav = torchaudio.transforms.Resample(sr, SR_TARGET)(wav)
        wav = wav.mean(dim=0) if wav.size(0) > 1 else wav.squeeze(0)
        
        # 3. Filter length
        duration = wav.size(0) / SR_TARGET
        if duration < 2.0 or duration > 15.0:
            continue
        
        # 4. Filter SNR (optional but recommended)
        snr_db = compute_snr(wav)  # implement với webrtc-vad
        if snr_db < 15:
            continue
        
        # 5. Pre-extract ref_code và speaker embedding (faster training)
        ref_audio_tensor, _ = load_and_resample(item["ref_audio"], SR_TARGET)
        with torch.no_grad():
            ref_code = model.model.speech_tokenizer.encode(ref_audio_tensor.unsqueeze(0))
            ref_spk_emb = model.model.extract_speaker_embedding(ref_audio_tensor)
        
        processed.append({
            "text": text_normalized,
            "ref_text": ref_text_normalized,
            "ref_code": ref_code.cpu(),               # [T_ref, 16]
            "ref_spk_embedding": ref_spk_emb.cpu(),   # [d_spk]
            "target_wav": wav.cpu(),                  # [T_wav], for SIM reward
            "duration": duration,
        })
    
    torch.save(processed, output_path)
    print(f"Saved {len(processed)} items to {output_path}")
```

### 4.3. Vietnamese text normalization

**Phải normalize trước khi train:**
- Số: "100" → "một trăm"
- Symbols: "5%" → "năm phần trăm"
- Abbreviations: "TP.HCM" → "thành phố Hồ Chí Minh"
- Punctuation: giữ dấu câu cho prosody

**Library options:**
- `vinorm` (https://github.com/v-nhandt21/Vinorm)
- Custom rules dựa trên dataset của bạn
- Hoặc dùng LLM (GPT-4o) cho subset nhỏ → cache

### 4.4. Data split

```
Train: 8,000-9,000 samples
Val:   500-1,000 samples (chỉ dùng cho metric tracking, KHÔNG cho gradient)
Test:  500 samples (chỉ chạy 1 lần ở cuối mỗi gate)
```

Random seed = 42 cho reproducibility.

---

## 5. GATE 1 — SFT Continuation (LM Cross-Entropy Loss)

### 5.1. Mục tiêu Gate 1

**Deliverables:**
1. Checkpoint sau SFT thêm với LM CE loss + KL anchor
2. Bảng metrics: WER, SIM, naturalness (subjective spot-check)
3. Quyết định gate exit

**Gate exit criteria:**
- WER trên test Vietnamese ≤ target (vd: 5%) → DỪNG, deploy SFT
- WER giảm ≥ 5% relative so với checkpoint gốc → đi tiếp Gate 2
- WER không cải thiện hoặc tăng → debug data/hyperparameters

### 5.2. Công thức toán học

**Loss function:**

\[
\mathcal{L}_{\text{SFT}}(\theta) = \mathcal{L}_{\text{CE}}(\theta) + \beta_{\text{KL}} \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
\]

Trong đó:

**Cross-entropy trên codebook 0:**

\[
\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{T_{\text{gen}}} \sum_{t=1}^{T_{\text{gen}}} \log \pi_\theta(o_t^* \mid x, c_{\text{ref}}, s_{\text{ref}}, o_{<t}^*)
\]

\(o_t^*\) = ground truth codebook 0 token tại step t (được encode từ target audio bằng frozen tokenizer).

**KL anchor:**

\[
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \sum_{t=1}^{T_{\text{gen}}} \sum_{v} \pi_\theta(v|...) \log \frac{\pi_\theta(v|...)}{\pi_{\text{ref}}(v|...)}
\]

Approximate qua sample average trong implementation.

**Tại sao thêm KL anchor?**
- Checkpoint gốc đã train tốt → drift quá xa = mất multilingual capability
- Anchor nhẹ (β_KL = 0.01-0.05) = balance giữa learn và preserve

### 5.3. Pseudo-code Gate 1

**File: `train_sft.py`**

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from qwen_tts import Qwen3TTSModel
import copy

# === Config ===
CONFIG_SFT = {
    "base_checkpoint": "g-group-ai-lab/gwen-tts-0.6B",
    "lr": 5e-5,                   # higher than VITS spec vì LoRA + LLM-style training
    "batch_size": 4,
    "gradient_accum_steps": 4,
    "max_steps": 10000,
    "val_every": 500,
    "save_every": 2000,
    "beta_kl": 0.02,
    "grad_clip": 1.0,
    "warmup_steps": 500,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "max_seq_len": 4096,
}


def build_sft_model(config):
    """Load gwen-tts-0.6B + apply LoRA on talker only."""
    # Load base model
    model = Qwen3TTSModel.from_pretrained(
        config["base_checkpoint"],
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    # Freeze everything by default
    for p in model.model.parameters():
        p.requires_grad = False
    
    # Apply LoRA chỉ trên talker (28-layer transformer)
    talker = model.model.talker  # adjust based on actual attribute name
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    talker = get_peft_model(talker, lora_config)
    model.model.talker = talker
    
    # Reference model (frozen copy, no LoRA)
    ref_model = Qwen3TTSModel.from_pretrained(
        config["base_checkpoint"],
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    for p in ref_model.model.parameters():
        p.requires_grad = False
    ref_model.eval()
    
    return model, ref_model


def sft_step(model, ref_model, batch, config):
    """Single SFT training step."""
    text_ids = batch["text_ids"].cuda()           # [B, T_text]
    ref_codes = batch["ref_codes"].cuda()         # [B, T_ref, 16]
    target_codes_0 = batch["target_codes_0"].cuda()  # [B, T_gen] — codebook 0 only
    target_lens = batch["target_lens"].cuda()
    spk_emb = batch["spk_emb"].cuda()
    
    # Build full input: text_ids + ref_codes (codebook 0) + target_codes_0[:-1] (teacher forcing)
    # Output: predict target_codes_0[1:]
    
    # Forward through talker
    outputs = model.model.talker.forward(
        text_ids=text_ids,
        ref_codes=ref_codes[..., 0],   # codebook 0 only for input
        spk_emb=spk_emb,
        target_codes=target_codes_0,
    )
    logits = outputs.logits   # [B, T_gen, 2048]
    
    # CE loss với padding mask
    mask = torch.arange(target_codes_0.size(1)).cuda()[None, :] < target_lens[:, None]
    
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_codes_0.view(-1),
        ignore_index=-100,
        reduction="none"
    ).view_as(target_codes_0)
    ce_loss = (ce_loss * mask.float()).sum() / mask.float().sum()
    
    # KL with reference
    with torch.no_grad():
        ref_outputs = ref_model.model.talker.forward(
            text_ids=text_ids,
            ref_codes=ref_codes[..., 0],
            spk_emb=spk_emb,
            target_codes=target_codes_0,
        )
        ref_logits = ref_outputs.logits
    
    kl = F.kl_div(
        F.log_softmax(logits.float(), dim=-1),
        F.log_softmax(ref_logits.float(), dim=-1),
        reduction="none",
        log_target=True,
    ).sum(dim=-1)  # [B, T_gen]
    kl = (kl * mask.float()).sum() / mask.float().sum()
    
    total_loss = ce_loss + config["beta_kl"] * kl
    
    return {
        "total": total_loss,
        "ce_loss": ce_loss.detach(),
        "kl": kl.detach(),
    }


def train_sft(config):
    model, ref_model = build_sft_model(config)
    
    train_loader = DataLoader(
        TTSRLDataset(config["train_path"]),
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=rl_collate_fn,
        num_workers=4,
    )
    val_loader = DataLoader(
        TTSRLDataset(config["val_path"]),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=rl_collate_fn,
    )
    
    # Optimizer chỉ trên LoRA params
    trainable_params = [p for p in model.model.talker.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config["lr"], betas=(0.9, 0.95), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, s / config["warmup_steps"])
    )
    
    # Training loop
    model.train()
    accum_loss = 0.0
    best_val_wer = float("inf")
    
    for step, batch in enumerate(cycle(train_loader)):
        losses = sft_step(model, ref_model, batch, config)
        loss = losses["total"] / config["gradient_accum_steps"]
        loss.backward()
        accum_loss += loss.item()
        
        if (step + 1) % config["gradient_accum_steps"] == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, config["grad_clip"]
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if step % 50 == 0:
                log({
                    "step": step,
                    "loss/total": accum_loss,
                    "loss/ce": losses["ce_loss"].item(),
                    "loss/kl": losses["kl"].item(),
                    "grad_norm": grad_norm.item(),
                    "lr": scheduler.get_last_lr()[0],
                })
            accum_loss = 0.0
        
        if step > 0 and step % config["val_every"] == 0:
            val_metrics = validate_sft(model, val_loader)
            log({"val/WER": val_metrics["WER"], "val/SIM": val_metrics["SIM"]})
            if val_metrics["WER"] < best_val_wer:
                best_val_wer = val_metrics["WER"]
                save_lora_checkpoint(model, f"{config['save_dir']}/gate1_best.pt")
        
        if step >= config["max_steps"]:
            break
```

### 5.4. Hyperparameters Gate 1

| Hyperparameter | Giá trị | Giải thích |
|---|---|---|
| `lr` | 5e-5 | Cao hơn VITS spec vì LoRA + standard LLM training |
| `batch_size` | 4 | Tối đa với 24GB VRAM |
| `gradient_accum_steps` | 4 | Effective batch = 16 |
| `max_steps` | 10,000 | ~12 epoch trên 8K samples |
| `warmup_steps` | 500 | Linear warmup |
| `beta_kl` | 0.02 | Anchor nhẹ |
| `grad_clip` | 1.0 | Standard cho LLM |
| `lora_r` | 16 | Sweet spot capacity vs efficiency |
| `lora_alpha` | 32 | 2× r theo recommendation |
| `lora_target` | q,k,v,o proj | Standard cho transformer LLM |

### 5.5. VRAM budget Gate 1

```
gwen-tts-0.6B (bf16):       ~1.2 GB
LoRA params (5M):           ~20 MB
Reference copy (bf16):      ~1.2 GB
Optimizer state (Adam):     ~80 MB (chỉ LoRA params)
Activations (batch=4):      ~4 GB
Gradient (LoRA only):       ~20 MB
Mimi decoder (bf16):        ~60 MB
Misc overhead:              ~1 GB
─────────────────────────
Total:                      ~7.6 GB
```

→ Chạy được trên RTX 3090/4090 24GB hoặc thấp hơn.

### 5.6. Debugging Gate 1

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| Loss NaN | bf16 overflow trong KL | Compute KL in fp32 |
| Loss giảm nhưng audio robotic | Overfit teacher forcing | Tăng beta_kl, giảm LR |
| Loss flat | LoRA r quá nhỏ | Tăng r=32, alpha=64 |
| OOM | Sequence quá dài | Truncate max_seq_len, gradient checkpointing |
| WER không đổi | KL anchor quá mạnh | Giảm beta_kl xuống 0.005 |
| Multilingual degrade | Catastrophic forgetting | Mix 10% non-Vietnamese trong batch |

### 5.7. Evaluation Gate 1

```python
@torch.no_grad()
def validate_sft(model, val_loader):
    """Generate audio + compute WER/SIM trên val set."""
    model.eval()
    whisper = load_whisper("medium")  # Vietnamese support
    wavlm = load_wavlm()
    
    wers, sims = [], []
    for batch in val_loader:
        # Generate audio với inference mode
        wavs, sr = model.generate_voice_clone(
            text=batch["text"],
            language="Vietnamese",
            ref_audio=batch["ref_audio_path"],
            ref_text=batch["ref_text"],
            **GENERATION_CONFIG,
        )
        
        # WER
        for i, wav in enumerate(wavs):
            transcript = whisper.transcribe(wav, language="vi")
            wer = compute_wer(transcript, batch["text"][i])
            wers.append(wer)
        
        # SIM
        for i, wav in enumerate(wavs):
            emb_gen = wavlm(wav).mean(dim=1)
            emb_ref = wavlm(batch["target_wav"][i]).mean(dim=1)
            sim = F.cosine_similarity(emb_gen, emb_ref, dim=-1)
            sims.append(sim.item())
    
    model.train()
    return {"WER": np.mean(wers), "SIM": np.mean(sims)}
```

---

## 6. GATE 2 — GRPO Minimal (Whisper WER reward only)

### 6.1. Mục tiêu Gate 2

**Deliverables:**
1. Checkpoint GRPO với chỉ WER reward
2. Verify GRPO machinery hoạt động trên Vietnamese
3. Quyết định: có scale lên Gate 3 không

**Gate exit criteria:**
- WER giảm ≥ 5% relative so với Gate 1 → đi tiếp Gate 3
- WER cải thiện < 5% hoặc tăng → debug, không scale lên

### 6.2. Công thức toán học GRPO

**GRPO objective** (theo DeepSeekMath 2402.03300):

\[
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{o^i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{T_i} \sum_{t=1}^{T_i} \min\left( \rho_t^i A_t^i, \text{clip}(\rho_t^i, 1-\varepsilon, 1+\varepsilon) A_t^i \right) - \beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]
\]

Trong đó:

**Importance sampling ratio:**

\[
\rho_t^i = \frac{\pi_\theta(o_t^i | x, o_{<t}^i)}{\pi_{\theta_{\text{old}}}(o_t^i | x, o_{<t}^i)}
\]

**Group-normalized advantage:**

\[
A_t^i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}
\]

với \(\mu_R, \sigma_R\) là mean/std của rewards trong group G samples.

**Reward (Gate 2 — chỉ WER):**

\[
R_i = r_{\text{wer}}(y_i, T_i) = 1 - \frac{D_{\text{lev}}(\hat{T}_i, T_i)}{|T_i|}
\]

với \(\hat{T}_i\) = transcript từ Whisper, \(D_{\text{lev}}\) = Levenshtein distance, \(|T_i|\) = độ dài text gốc.

Range: [0, 1] (clip âm về 0).

### 6.3. GRPO Algorithm chi tiết

```
Algorithm: GRPO Training Step
─────────────────────────────────────────────────────
Input: prompt batch B, group size G, current policy π_θ
       old policy π_θ_old (snapshot), ref policy π_ref

For each prompt x in B:
    # 1. Rollout G samples từ π_θ_old
    For i = 1..G:
        o^i ~ π_θ_old(·|x)         # generate codebook 0 sequence
        c^i_{1:15} = MTP(hidden^i)   # generate codebooks 1-15 (frozen MTP)
        y^i = MimiDecoder([o^i, c^i_{1:15}])  # decode to wav
    
    # 2. Compute rewards
    For i = 1..G:
        T̂^i = Whisper.transcribe(y^i)
        R^i = 1 - lev(T̂^i, T) / |T|
    
    # 3. Group normalization
    μ_R, σ_R = mean(R^{1:G}), std(R^{1:G})
    For i = 1..G, t = 1..T_i:
        A^i_t = (R^i - μ_R) / (σ_R + ε)   # token-level advantage = sequence-level advantage
    
    # 4. Compute current logprobs (with gradient)
    For i = 1..G, t = 1..T_i:
        log π_θ(o^i_t | ...) = forward(model, x, o^i_<=t)
        log π_θ_old(o^i_t | ...) = saved during rollout (no grad)
        log π_ref(o^i_t | ...) = forward(ref_model, x, o^i_<=t) (no grad)
    
    # 5. Compute loss
    For i = 1..G, t = 1..T_i:
        ρ^i_t = exp(log π_θ - log π_θ_old)
        L_clip^i_t = min(ρ^i_t * A^i_t, clip(ρ^i_t, 1-ε, 1+ε) * A^i_t)
    
    L_policy = -mean(L_clip)
    L_kl = β_KL * mean(log π_θ - log π_ref)   # approximation
    L_total = L_policy + L_kl
    
    # 6. Backward + update
    L_total.backward()
    optimizer.step()

# 7. Update old policy snapshot every K steps
if step % K == 0:
    π_θ_old = copy(π_θ)
─────────────────────────────────────────────────────
```

### 6.4. Pseudo-code Gate 2

**File: `train_grpo_minimal.py`**

```python
import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import copy
import jiwer

# === Config ===
CONFIG_GRPO_MIN = {
    "base_checkpoint": "./checkpoints/gate1_best",
    "whisper_model": "openai/whisper-medium",
    "lr": 5e-7,
    "batch_size": 2,
    "group_size": 8,                # G samples per prompt
    "max_steps": 5000,
    "val_every": 200,
    "save_every": 1000,
    "old_policy_update_interval": 100,
    "beta_kl": 0.05,
    "epsilon_clip": 0.2,            # PPO/GRPO clip
    "grad_clip": 1.0,
    "warmup_steps": 100,
    "lora_r": 16,
    "lora_alpha": 32,
    # Generation config (match inference)
    "gen_temperature": 0.3,
    "gen_top_k": 20,
    "gen_top_p": 0.9,
    "gen_max_new_tokens": 1024,    # giới hạn để tránh OOM
    "gen_repetition_penalty": 2.0,
}


class WhisperWERReward:
    """Compute WER reward via Whisper (non-differentiable, treat as black box)."""
    
    def __init__(self, model_name="openai/whisper-medium"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).cuda().eval()
        for p in self.model.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def compute_wer_reward(self, wavs: list, texts: list[str], language="vi") -> torch.Tensor:
        """
        Args:
            wavs: list of waveforms at 16kHz, length B
            texts: list of ground truth texts, length B
        Returns:
            rewards: [B] tensor in [0, 1]
        """
        rewards = []
        for wav, text in zip(wavs, texts):
            # Resample to 16kHz if needed
            if wav.size(-1) > 0:
                # Whisper inference
                inputs = self.processor(
                    wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
                ).input_features.cuda()
                
                forced_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
                generated = self.model.generate(
                    inputs, forced_decoder_ids=forced_ids, max_length=256
                )
                transcript = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
                
                # Compute WER
                try:
                    wer = jiwer.wer(text.lower().strip(), transcript.lower().strip())
                    reward = max(0.0, 1.0 - wer)  # clip negative
                except Exception:
                    reward = 0.0
            else:
                reward = 0.0
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)


def rollout_batch(model, batch, config):
    """
    Generate G samples per prompt, return tokens + log_probs.
    Returns:
        all_codes_0: [B*G, T_max] codebook 0 tokens
        all_log_probs: [B*G, T_max] log probs (no grad, for old policy)
        all_wavs: list of [B*G] decoded waveforms
        all_lens: [B*G] sequence lengths
    """
    model.eval()
    B = len(batch["text"])
    G = config["group_size"]
    
    all_codes_0 = []
    all_log_probs = []
    all_wavs = []
    all_lens = []
    
    for i in range(B):
        for g in range(G):
            with torch.no_grad():
                # Generate full output (codebook 0 + 1-15)
                wavs, sr, talker_codes, mtp_codes, log_probs = model.generate_voice_clone_with_logprobs(
                    text=batch["text"][i],
                    language="Vietnamese",
                    ref_audio_tensor=batch["ref_audio"][i],
                    ref_text=batch["ref_text"][i],
                    return_logprobs=True,                       # custom flag
                    temperature=config["gen_temperature"],
                    top_k=config["gen_top_k"],
                    top_p=config["gen_top_p"],
                    max_new_tokens=config["gen_max_new_tokens"],
                    repetition_penalty=config["gen_repetition_penalty"],
                )
            
            all_codes_0.append(talker_codes)              # [T]
            all_log_probs.append(log_probs)               # [T]
            all_wavs.append(wavs[0])                       # waveform
            all_lens.append(talker_codes.size(0))
    
    model.train()
    return {
        "codes_0": all_codes_0,
        "log_probs_old": all_log_probs,
        "wavs": all_wavs,
        "lens": all_lens,
        "B": B,
        "G": G,
    }


def grpo_step(model, ref_model, whisper_reward, batch, config, step):
    """Single GRPO training step."""
    
    # === 1. Rollout G samples per prompt ===
    rollouts = rollout_batch(model, batch, config)
    B, G = rollouts["B"], rollouts["G"]
    
    # === 2. Compute rewards ===
    texts_expanded = [t for t in batch["text"] for _ in range(G)]
    rewards = whisper_reward.compute_wer_reward(
        rollouts["wavs"], texts_expanded, language="vi"
    ).cuda()  # [B*G]
    
    # === 3. Group normalization ===
    rewards_grouped = rewards.view(B, G)
    mu = rewards_grouped.mean(dim=1, keepdim=True)
    sigma = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
    advantages = ((rewards_grouped - mu) / sigma).view(-1)  # [B*G]
    
    # === 4. Compute new log_probs (with gradient) ===
    # Re-run forward through talker với teacher forcing
    new_log_probs = []
    ref_log_probs = []
    
    for idx in range(B * G):
        codes = rollouts["codes_0"][idx].cuda()
        text_idx = idx // G
        
        # Forward through current model
        outputs = model.model.talker.forward_with_codes(
            text=batch["text"][text_idx],
            ref_codes=batch["ref_codes"][text_idx],
            spk_emb=batch["spk_emb"][text_idx],
            target_codes=codes,
        )
        logits = outputs.logits  # [T, V]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_log_probs = log_probs.gather(-1, codes.unsqueeze(-1)).squeeze(-1)  # [T]
        new_log_probs.append(token_log_probs)
        
        # Forward through reference (no grad)
        with torch.no_grad():
            ref_outputs = ref_model.model.talker.forward_with_codes(
                text=batch["text"][text_idx],
                ref_codes=batch["ref_codes"][text_idx],
                spk_emb=batch["spk_emb"][text_idx],
                target_codes=codes,
            )
            ref_log_probs_token = F.log_softmax(ref_outputs.logits.float(), dim=-1).gather(
                -1, codes.unsqueeze(-1)
            ).squeeze(-1)
        ref_log_probs.append(ref_log_probs_token)
    
    # === 5. Compute GRPO loss ===
    total_policy_loss = 0.0
    total_kl = 0.0
    total_tokens = 0
    
    for idx in range(B * G):
        new_lp = new_log_probs[idx]  # [T]
        old_lp = rollouts["log_probs_old"][idx].cuda()  # [T]
        ref_lp = ref_log_probs[idx]  # [T]
        adv = advantages[idx]  # scalar
        
        # Importance ratio
        ratio = torch.exp(new_lp - old_lp)
        clipped_ratio = torch.clamp(ratio, 1 - config["epsilon_clip"], 1 + config["epsilon_clip"])
        
        # Policy loss (negative because we maximize)
        policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()
        
        # KL with ref (token-level approximation)
        kl = (new_lp - ref_lp).mean()
        
        total_policy_loss += policy_loss
        total_kl += kl
        total_tokens += new_lp.size(0)
    
    total_policy_loss = total_policy_loss / (B * G)
    total_kl = total_kl / (B * G)
    
    total_loss = total_policy_loss + config["beta_kl"] * total_kl
    
    return {
        "total": total_loss,
        "policy_loss": total_policy_loss.detach(),
        "kl": total_kl.detach(),
        "reward_mean": rewards.mean().detach(),
        "reward_std": rewards.std().detach(),
        "advantage_abs_mean": advantages.abs().mean().detach(),
    }


def train_grpo_minimal(config):
    # Build model
    model, ref_model = build_grpo_model(config)
    
    # Old policy = snapshot of current model
    old_model = copy.deepcopy(model)
    for p in old_model.parameters():
        p.requires_grad = False
    old_model.eval()
    
    # Whisper reward
    whisper_reward = WhisperWERReward(config["whisper_model"])
    
    # Dataloaders
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    
    # Optimizer
    trainable_params = [p for p in model.model.talker.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config["lr"], betas=(0.9, 0.95), weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, s / config["warmup_steps"])
    )
    
    # Training loop
    model.train()
    best_val_wer = float("inf")
    
    for step, batch in enumerate(cycle(train_loader)):
        losses = grpo_step(model, ref_model, whisper_reward, batch, config, step)
        loss = losses["total"]
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config["grad_clip"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Update old policy
        if (step + 1) % config["old_policy_update_interval"] == 0:
            old_model.load_state_dict(model.state_dict())
        
        # Logging
        if step % 10 == 0:
            log({
                "step": step,
                "loss/total": loss.item(),
                "loss/policy": losses["policy_loss"].item(),
                "loss/kl": losses["kl"].item(),
                "reward/mean": losses["reward_mean"].item(),
                "reward/std": losses["reward_std"].item(),
                "advantage_abs_mean": losses["advantage_abs_mean"].item(),
                "grad_norm": grad_norm.item(),
                "lr": scheduler.get_last_lr()[0],
            })
        
        # Validation
        if step > 0 and step % config["val_every"] == 0:
            val_metrics = validate_full(model, val_loader)
            log({"val/WER": val_metrics["WER"], "val/SIM": val_metrics["SIM"]})
            if val_metrics["WER"] < best_val_wer:
                best_val_wer = val_metrics["WER"]
                save_lora_checkpoint(model, f"{config['save_dir']}/gate2_best.pt")
        
        if step >= config["max_steps"]:
            break
```

### 6.5. Hyperparameters Gate 2

| Hyperparameter | Giá trị | Giải thích |
|---|---|---|
| `lr` | 5e-7 | Tham khảo Multi-Reward GRPO paper (1e-6); giảm vì 0.6B nhỏ hơn 8B |
| `batch_size` | 2 | Mỗi prompt sinh G samples → effective batch lớn |
| `group_size G` | 8 | Theo precedent (4-12) |
| `max_steps` | 5,000 | Effective rollouts = 5000 × 2 × 8 = 80K |
| `epsilon_clip` | 0.2 | Standard PPO/GRPO |
| `beta_kl` | 0.05 | Cao hơn SFT vì RL có thể drift mạnh hơn |
| `old_policy_update_interval` | 100 | Update mỗi 100 step (giống PPO) |
| `gen_temperature` | 0.3 | Match inference config |
| `gen_repetition_penalty` | 2.0 | Match inference config |

### 6.6. VRAM budget Gate 2

```
Model + LoRA:               ~1.3 GB
Reference model:            ~1.2 GB
Old policy snapshot:        ~1.2 GB
Whisper-medium:             ~3.0 GB
Mimi decoder:               ~60 MB
Activations (B=2, G=8):     ~6 GB
Gradients (LoRA only):      ~80 MB
Optimizer state:            ~80 MB
Rollout buffer (G samples): ~500 MB
Misc:                       ~1 GB
─────────────────────────────────
Total:                      ~14.5 GB
```

→ Chạy được trên A100 40GB hoặc A6000 48GB. RTX 3090/4090 24GB cũng OK với gradient checkpointing.

### 6.7. Critical implementation notes

**1. Generation phải match inference distribution**

```python
# Đúng — match inference
wavs = model.generate_voice_clone(
    temperature=0.3, top_k=20, top_p=0.9,
    repetition_penalty=2.0,
    max_new_tokens=1024,
)

# Sai — sẽ làm train-test mismatch
wavs = model.generate_voice_clone(
    temperature=1.0,   # ❌ different from inference
)
```

**2. Voice cloning ref_code cut logic**

Generated audio có ref portion prepended → phải cut đúng trước khi feed Whisper:

```python
# Replicate logic từ generate_voice_clone()
def get_generated_only_wav(full_wav, ref_code_len, total_codes_len, total_wav_len):
    cut = int(ref_code_len / max(total_codes_len, 1) * total_wav_len)
    return full_wav[cut:]
```

**3. Whisper input format**

```python
# Resample 24kHz → 16kHz cho Whisper
wav_16k = torchaudio.transforms.Resample(24000, 16000)(wav_24k)

# Force language=vi
forced_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
```

**4. Token-level advantage assignment**

Sequence-level reward → broadcast ra token-level:

```python
# Tất cả tokens trong sequence i nhận cùng advantage
A_token_t = A_sequence  # for all t in sequence i
```

Điều này **không tối ưu** (không có temporal credit assignment) nhưng standard cho TTS. Alternative: dùng GAE với value model — phức tạp hơn, để Gate 4 nếu cần.

### 6.8. Debugging Gate 2

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| Reward mean không tăng | LR quá nhỏ hoặc KL penalty quá mạnh | LR ×2, β_KL /2 |
| Reward std rất nhỏ | Group samples quá similar | Tăng temperature lên 0.5 |
| Policy loss âm liên tục | Advantage normalization sai | Check group computation |
| KL divergence explode | Old policy update quá thưa | Giảm interval xuống 50 |
| Audio quality giảm | Reward hacking on Whisper | Đi sang Gate 3 với multi-reward |
| OOM during rollout | G quá lớn | Giảm G xuống 4 |
| Whisper transcribe rỗng | Audio < 0.5s hoặc silence | Filter trong rollout, gán reward=0 |
| Non-deterministic results | Seed không set | `torch.manual_seed`, set generator state |

### 6.9. Monitoring metrics

**Bắt buộc track:**
- `reward/mean` — should increase monotonically
- `reward/std` — should NOT collapse to 0 (cần diversity)
- `advantage_abs_mean` — should stay bounded (0.1-2.0)
- `loss/kl` — should stay positive but bounded
- `grad_norm` — stable (0.1-1.0)
- `val/WER` — primary metric

**Red flags:**
- `reward/std → 0` → mode collapse, restart with higher temperature
- `loss/kl > 5` → drift quá mạnh, tăng β_KL
- `val/WER` tăng but `reward/mean` tăng → reward hacking, đi Gate 3

---

## 7. GATE 3 — GRPO Multi-Reward (Full DRPO style)

### 7.1. Mục tiêu Gate 3

**Deliverables:**
1. Checkpoint với 4 rewards: WER + SIM + length + entropy
2. Ablation table contribution của từng reward
3. Final evaluation: WER, SIM, UTMOS, human MOS subset
4. Paper-ready results

**Gate exit criteria:**
- Cải thiện ≥ 5% so với Gate 2 → Victory
- Cải thiện 2-5% → acceptable, có ablation để publish
- Cải thiện < 2% → DRPO multi-reward không add value, dùng Gate 2

### 7.2. Công thức toán học Gate 3

**Total reward (theo Multi-Reward GRPO 2511.21270):**

\[
R_i = \alpha_{\text{wer}} r_{\text{wer}} + \alpha_{\text{sim}} r_{\text{sim}} + \alpha_{\text{len}} r_{\text{len}} + \alpha_{\text{ent}} r_{\text{ent}}
\]

**WER reward (như Gate 2):**

\[
r_{\text{wer}} = \max\left(0, 1 - \frac{D_{\text{lev}}(\hat{T}, T)}{|T|}\right)
\]

**SIM reward (cosine similarity qua WavLM):**

\[
r_{\text{sim}} = \frac{\langle E(y), E(y_{\text{ref}}) \rangle}{\|E(y)\| \|E(y_{\text{ref}})\|}
\]

với \(E\) = WavLM-large speaker verification model. Range: [-1, 1], thực tế [0.3, 0.95] cho speech.

**Length penalty reward:**

\[
r_{\text{len}} = \begin{cases}
1 & \text{if } \frac{T_{\text{audio}}/T_{\text{text}}}{r_{\text{ref}}} \in [0.85, 1.15] \\
0 & \text{otherwise}
\end{cases}
\]

trong đó \(r_{\text{ref}}\) = speaking rate của reference audio (giây/từ).

**Entropy regularization:**

\[
r_{\text{ent}} = -\lambda_{\text{ent}} \cdot \max(0, \bar{H} - H_{\text{target}})
\]

với \(\bar{H}\) = average token-level entropy, \(H_{\text{target}}\) ≈ 1.5-2.5 bits (estimate từ high-quality samples).

### 7.3. Reward coefficients

Theo paper Multi-Reward GRPO:
- \(\alpha_{\text{wer}} = 1.0\) (primary)
- \(\alpha_{\text{sim}} = 1.0\) (secondary)
- \(\alpha_{\text{len}} = 0.1\) (light)
- \(\alpha_{\text{ent}} = 1.0\) với \(\lambda_{\text{ent}} = 0.5\)

**Tuning notes:**
- Nếu WER không cải thiện → tăng α_wer = 2.0
- Nếu SIM giảm → tăng α_sim = 1.5
- Nếu audio robotic → tăng α_ent

### 7.4. Pseudo-code Gate 3 (chỉ những phần khác Gate 2)

```python
class MultiRewardGRPO:
    def __init__(self, config):
        self.whisper = WhisperWERReward(config["whisper_model"])
        self.wavlm = self._build_wavlm()
        self.alpha = {
            "wer": config["alpha_wer"],
            "sim": config["alpha_sim"],
            "len": config["alpha_len"],
            "ent": config["alpha_ent"],
        }
        self.lambda_ent = config["lambda_ent"]
        self.h_target = config["h_target"]
    
    def _build_wavlm(self):
        from transformers import AutoModel
        model = AutoModel.from_pretrained("microsoft/wavlm-large").cuda().eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
    
    def compute_all_rewards(self, wavs, texts, ref_wavs, ref_speaking_rates,
                            log_probs_per_token):
        """
        Args:
            wavs: list[Tensor] of generated waveforms
            texts: list[str] of ground truth
            ref_wavs: list[Tensor] of reference waveforms
            ref_speaking_rates: list[float] of seconds/word
            log_probs_per_token: list[Tensor] of log probs (for entropy)
        Returns:
            total_rewards: [B] tensor
            component_rewards: dict
        """
        B = len(wavs)
        
        # 1. WER
        r_wer = self.whisper.compute_wer_reward(wavs, texts, language="vi").cuda()
        
        # 2. SIM
        r_sim = []
        for w, w_ref in zip(wavs, ref_wavs):
            with torch.no_grad():
                w_16k = resample_to_16k(w)
                w_ref_16k = resample_to_16k(w_ref)
                e = self.wavlm(w_16k.unsqueeze(0)).last_hidden_state.mean(1)
                e_ref = self.wavlm(w_ref_16k.unsqueeze(0)).last_hidden_state.mean(1)
                sim = F.cosine_similarity(e, e_ref, dim=-1).item()
            r_sim.append(sim)
        r_sim = torch.tensor(r_sim).cuda()
        
        # 3. Length penalty
        r_len = []
        for i, (w, t) in enumerate(zip(wavs, texts)):
            audio_dur = w.size(-1) / 24000
            text_words = len(t.split())
            actual_rate = audio_dur / max(text_words, 1)
            ratio = actual_rate / ref_speaking_rates[i]
            r_len.append(1.0 if 0.85 <= ratio <= 1.15 else 0.0)
        r_len = torch.tensor(r_len).cuda()
        
        # 4. Entropy
        r_ent = []
        for lp in log_probs_per_token:
            # average entropy from log probs
            # entropy approximation: -E[log p] ≈ -log_p_chosen
            # But we want full entropy: need full distribution
            # Use approximation: avg entropy ≈ -log_p_avg
            avg_h = -lp.mean().item()
            penalty = max(0, avg_h - self.h_target)
            r_ent.append(-self.lambda_ent * penalty)
        r_ent = torch.tensor(r_ent).cuda()
        
        # Total
        total = (
            self.alpha["wer"] * r_wer
            + self.alpha["sim"] * r_sim
            + self.alpha["len"] * r_len
            + self.alpha["ent"] * r_ent
        )
        
        return total, {
            "wer": r_wer.mean().item(),
            "sim": r_sim.mean().item(),
            "len": r_len.mean().item(),
            "ent": r_ent.mean().item(),
        }
```

### 7.5. Hyperparameters Gate 3

| Hyperparameter | Giá trị | Giải thích |
|---|---|---|
| `lr` | 3e-7 | Thấp hơn Gate 2 vì task phức tạp hơn |
| `batch_size` | 2 | Same as Gate 2 |
| `group_size` | 8 | Same |
| `max_steps` | 8,000 | Dài hơn Gate 2 |
| `alpha_wer` | 1.0 | Primary |
| `alpha_sim` | 1.0 | Secondary |
| `alpha_len` | 0.1 | Light |
| `alpha_ent` | 1.0 | Regularizer |
| `lambda_ent` | 0.5 | Inside r_ent |
| `h_target` | 2.0 | Bits, calibrate từ checkpoint |
| `beta_kl` | 0.03 | Giảm vì có nhiều rewards stabilize |

### 7.6. VRAM budget Gate 3

```
Gate 2 baseline:            14.5 GB
+ WavLM-large:              + 1.5 GB
+ Reference wavs in batch:  + 200 MB
─────────────────────────────────
Total:                      ~16 GB
```

→ Vẫn fit A100 40GB / A6000 48GB.

### 7.7. Ablation protocol

Chạy 5 configs để hiểu contribution:

| Config | WER | SIM | Len | Ent | Mục đích |
|---|---|---|---|---|---|
| A: Gate 2 baseline | ✓ | ✗ | ✗ | ✗ | Baseline |
| B: + SIM | ✓ | ✓ | ✗ | ✗ | Test SIM contribution |
| C: + Len | ✓ | ✓ | ✓ | ✗ | Test length penalty |
| D: + Ent | ✓ | ✓ | ✓ | ✓ | Full Gate 3 |
| E: w/o WER | ✗ | ✓ | ✓ | ✓ | Sanity: WER reward critical? |

### 7.8. Optional: Prosody reward (advanced)

Nếu muốn improve naturalness MOS:

\[
r_{\text{pro}} = \begin{cases}
1 & \text{if } \hat{P}(y) \in \{P(T)\} \\
0 & \text{otherwise}
\end{cases}
\]

Trong đó:
- \(\{P(T)\}\) = set of valid pause structures, annotate offline bằng LLM (vd: GPT-4o, DeepSeek-R1) trên text Vietnamese
- \(\hat{P}(y)\) = pause structure detected từ generated audio (Whisper word timestamps → silence durations → discrete pause symbols)

**Implementation challenge:** Vietnamese không có chuẩn pause marker như Chinese (#1-#4). Cần custom rules:
- Pause < 50ms = no pause
- Pause 50-200ms = #1 (intra-phrase)
- Pause 200-500ms = #2 (phrase boundary)
- Pause > 500ms = #3 (sentence boundary)

→ Khuyến nghị: skip prosody reward ở v1.0, thêm sau nếu cần.

---

## 8. Infrastructure dùng chung

### 8.1. Environment setup

```bash
# File: environment.yml
name: gwen_tts_rl
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.12
  - pytorch=2.3.0
  - pytorch-cuda=12.1
  - torchaudio=2.3.0
  - pip
  - pip:
    - qwen-tts                         # Qwen3-TTS package
    - transformers==4.40.0
    - peft==0.10.0                     # for LoRA
    - accelerate==0.30.0
    - flash-attn==2.5.8                # for attention efficiency
    - librosa==0.10.2
    - soundfile==0.12.1
    - jiwer==3.0.4                     # WER computation
    - vinorm                           # Vietnamese text normalization
    - wandb==0.17.0
    - numpy==1.26.4
    - scipy==1.13.0
```

### 8.2. Directory structure

```
gwen_tts_rl/
├── configs/
│   ├── gate1_sft.yaml
│   ├── gate2_grpo_min.yaml
│   └── gate3_grpo_full.yaml
├── data/
│   ├── raw/                          # original audio + transcripts
│   ├── processed/                    # after preprocessing
│   │   ├── train.pt
│   │   ├── val.pt
│   │   └── test.pt
│   └── normalized_texts.json
├── rewards/
│   ├── whisper_wer.py
│   ├── wavlm_sim.py
│   └── length_entropy.py
├── scripts/
│   ├── preprocess.py
│   ├── train_sft.py                  # Gate 1
│   ├── train_grpo_minimal.py         # Gate 2
│   ├── train_grpo_full.py            # Gate 3
│   └── evaluate.py
├── utils/
│   ├── lora.py
│   ├── checkpoint.py
│   ├── log.py
│   └── audio.py
├── checkpoints/
│   ├── gate1/
│   ├── gate2/
│   └── gate3/
└── logs/
```

### 8.3. Checkpoint loading utility

```python
# File: utils/checkpoint.py
import torch
from peft import PeftModel
from qwen_tts import Qwen3TTSModel

def load_with_lora(base_model_id, lora_path=None):
    """Load gwen-tts-0.6B với LoRA optional."""
    model = Qwen3TTSModel.from_pretrained(
        base_model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    if lora_path:
        model.model.talker = PeftModel.from_pretrained(
            model.model.talker, lora_path
        )
    
    return model


def save_lora_checkpoint(model, path):
    """Save chỉ LoRA params."""
    model.model.talker.save_pretrained(path)
    print(f"Saved LoRA to {path}")
```

### 8.4. Sanity checks trước train

```python
def sanity_checks(model, ref_model):
    # 1. LoRA params đúng được unfrozen
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.0f}M ({100*trainable/total:.2f}%)")
    assert 1e6 < trainable < 50e6, "LoRA trainable count suspicious"
    
    # 2. Reference model fully frozen
    for p in ref_model.parameters():
        assert not p.requires_grad, "Ref model not frozen"
    
    # 3. Generation works (smoke test)
    wavs, sr = model.generate_voice_clone(
        text="Xin chào, đây là bài test.",
        language="Vietnamese",
        ref_audio="test_ref.wav",
        ref_text="Reference text.",
    )
    assert wavs[0].size > 0 and sr == 24000, "Generation broken"
    
    # 4. Whisper VN works
    whisper = WhisperWERReward("openai/whisper-medium")
    rewards = whisper.compute_wer_reward([wavs[0]], ["Xin chào"], language="vi")
    print(f"Smoke test WER reward: {rewards.item():.3f}")
    
    print("✓ All sanity checks passed")
```

---

## 9. Evaluation protocol

### 9.1. Metrics bắt buộc

```python
def evaluate_full(model, test_loader, output_dir):
    """Run full evaluation."""
    metrics = {
        "WER": [],
        "CER": [],
        "SIM_WavLM": [],
        "RTF": [],          # real-time factor
        "duration_ratio": [],
    }
    
    whisper = load_whisper("large-v3")
    wavlm = load_wavlm()
    
    for batch in test_loader:
        import time
        start = time.time()
        wavs, sr = model.generate_voice_clone(
            text=batch["text"],
            language="Vietnamese",
            ref_audio=batch["ref_audio_path"],
            ref_text=batch["ref_text"],
            **GENERATION_CONFIG,
        )
        elapsed = time.time() - start
        
        for i, wav in enumerate(wavs):
            audio_dur = wav.shape[-1] / sr
            metrics["RTF"].append(elapsed / audio_dur)
            
            # WER + CER
            transcript = whisper.transcribe(wav, language="vi")
            metrics["WER"].append(jiwer.wer(batch["text"][i], transcript))
            metrics["CER"].append(jiwer.cer(batch["text"][i], transcript))
            
            # SIM
            emb_gen = wavlm(wav).mean(1)
            emb_ref = wavlm(batch["target_wav"][i]).mean(1)
            metrics["SIM_WavLM"].append(F.cosine_similarity(emb_gen, emb_ref).item())
            
            # Duration ratio
            target_dur = batch["target_wav"][i].shape[-1] / sr
            metrics["duration_ratio"].append(audio_dur / target_dur)
    
    # Save 50 random samples cho human eval
    save_for_human_eval(model, test_loader, output_dir / "human_eval", n=50)
    
    return {k: float(np.mean(v)) for k, v in metrics.items()}
```

### 9.2. Bảng kết quả mẫu

| Model | WER ↓ | CER ↓ | SIM ↑ | RTF ↓ | MOS-Naturalness | MOS-Similarity |
|---|---|---|---|---|---|---|
| gwen-tts-0.6B (baseline) | ? | ? | ? | ? | ? | ? |
| Gate 1 (SFT) | ? | ? | ? | ? | ? | ? |
| Gate 2 (GRPO-WER) | ? | ? | ? | ? | ? | ? |
| Gate 3 (GRPO-Full) | ? | ? | ? | ? | ? | ? |
| Gate 3 w/o SIM | ? | ? | ? | ? | ? | ? |
| Gate 3 w/o Len | ? | ? | ? | ? | ? | ? |

### 9.3. Human evaluation

- 50 samples random từ test set
- 3+ Vietnamese native evaluators
- MOS scale 1-5: naturalness, similarity, intelligibility
- Blind comparison với baseline

---

## 10. Decision tree và timeline

### 10.1. Decision flow

```
Start
  │
  ▼
Gate 1 (SFT 1 tuần) ──► WER target?
                          │
                          ├─── Yes → Deploy SFT
                          │
                          └─── No  → Gate 2
                                       │
                                       ▼
                          Cải thiện ≥ 5%? ──── No → Stop, dùng Gate 1
                                       │
                                       └─── Yes → Gate 3
                                                    │
                                                    ▼
                                       Cải thiện ≥ 5% so Gate 2?
                                            │
                                            ├─── Yes → Victory
                                            │
                                            └─── No → Use Gate 2 + ablation paper
```

### 10.2. Timeline tổng

| Gate | Setup | Training | Eval | Buffer | Total |
|---|---|---|---|---|---|
| Gate 1 (SFT) | 1 ngày | 2 ngày | 1 ngày | 2 ngày | **~1 tuần** |
| Gate 2 (GRPO-min) | 1 ngày | 4 ngày | 1 ngày | 3 ngày | **~1.5 tuần** |
| Gate 3 (GRPO-full) | 2 ngày | 5 ngày | 2 ngày | 4 ngày | **~2 tuần** |
| **Tổng** | | | | | **~4.5 tuần** |

### 10.3. Compute budget

| Gate | GPU-hours (A100) | Storage |
|---|---|---|
| Gate 1 | ~50h | 5 GB |
| Gate 2 | ~100h | 15 GB |
| Gate 3 | ~150h | 30 GB |
| **Tổng** | **~300 GPU-hours** | **50 GB** |

A100 cloud rent ~$1.5/h: **~$450 compute cost**

---

## 11. Risks và mitigation tổng hợp

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Reward hacking trên Whisper | Cao | Cao | Multi-reward Gate 3 + KL trust + early stop + human eval |
| Vietnamese ASR signal noisy | Trung | Trung | Filter SNR ≥ 15dB; benchmark Whisper trước |
| bf16 numerical instability | Trung | Cao | Mixed precision, clip, NaN check mỗi 100 step |
| MTP frozen → acoustic không cải thiện | Cao | Trung | Optional unfreeze MTP ở Gate 3; benchmark có/không |
| Catastrophic forgetting đa ngôn ngữ | Trung | Trung | KL anchor + mix 10% non-VN data |
| Voice cloning prefix cut sai | Thấp | Cao | Unit test cut logic |
| OOM với G=8 | Trung | Trung | Gradient checkpointing; giảm G xuống 4 |
| Whisper bias với accent | Cao | Trung | Cân nhắc Vietnamese-finetuned ASR |
| Mode collapse (low std reward) | Trung | Cao | Tăng temperature; entropy reward Gate 3 |

---

## 12. References

### 12.1. Papers core (must read)

- **Multi-Reward GRPO for TTS (foundation cho Gate 2-3):** [arXiv 2511.21270](https://arxiv.org/html/2511.21270v1)
- **DiffRO (alternative method):** [arXiv 2507.05911](https://arxiv.org/abs/2507.05911)
- **CosyVoice 2 DPO (alternative method):** [CosyVoice2 Tech Report](https://funaudiollm.github.io/pdf/CosyVoice_2.pdf)
- **DeepSeekMath GRPO (foundation):** [arXiv 2402.03300](https://arxiv.org/abs/2402.03300)

### 12.2. Models

- **gwen-tts-0.6B:** [HuggingFace](https://huggingface.co/g-group-ai-lab/gwen-tts-0.6B)
- **Qwen3-TTS-12Hz-0.6B-Base:** [HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- **Qwen3-TTS Tech Report:** [arXiv 2601.15621](https://arxiv.org/html/2601.15621v1)
- **Whisper (reward model):** [arXiv 2212.04356](https://arxiv.org/abs/2212.04356)
- **WavLM (SIM reward):** [arXiv 2110.13900](https://arxiv.org/abs/2110.13900)
- **Mimi codec:** [arXiv 2410.00037 (Moshi paper)](https://arxiv.org/abs/2410.00037)

### 12.3. Code references

- **Qwen3-TTS GitHub:** https://github.com/QwenLM/Qwen3-TTS
- **CosyVoice DPO Notes:** https://github.com/ScottishFold007/Cosyvoice_DPO_NOTES
- **vLLM-Omni Qwen3-TTS:** https://docs.vllm.ai/projects/vllm-omni/en/stable/user_guide/examples/offline_inference/qwen3_tts/

### 12.4. Vietnamese resources

- **Vinorm (text normalization):** https://github.com/v-nhandt21/Vinorm
- **Whisper VN benchmark:** Test trên VIVOS dataset trước khi train

---

## 13. Notes for code-gen model

**Khi implement spec này:**

1. **Bắt buộc** đọc Section 1 (kiến trúc) trước khi viết code — Qwen3-TTS có đặc điểm rất cụ thể (talker + MTP + Mimi) cần xử lý đúng.

2. **Bắt đầu từ Gate 1**, KHÔNG skip. Gate 1 verify infrastructure (data loading, LoRA, checkpoint save/load) trước khi RL phức tạp.

3. **Custom forward function** cần thiết cho rollout với log_probs return — cần đọc code Qwen3TTSForConditionalGeneration để biết hook đúng chỗ. Có thể cần monkey-patch hoặc subclass.

4. **Voice cloning prefix cut logic** từ `generate_voice_clone()` phải được replicate chính xác trong evaluation. Nếu cut sai, WER sẽ cao bất thường.

5. **Unit tests bắt buộc:**
   - Sanity check (Section 8.4)
   - Smoke test rollout với G=2 trước khi full training
   - Whisper VN benchmark trên 50 samples trước khi train

6. **Logging với wandb** từ step 0 — không train mù.

7. **Save checkpoint mỗi 1000 step** + always save best val WER.

---

## 14. Changelog

- **v1.0 (2026-05-11):** Initial spec với 3 gates roadmap, dựa trên paper Multi-Reward GRPO 2511.21270 và DiffRO 2507.05911.
- **v2.0 (2026-05-11):** Bổ sung Section 15-18: phân tích nghiêm túc RL có thực sự cải thiện không, tối ưu memory (Liger/vLLM colocate/8-bit AdamW/PEFT-no-ref), tối ưu time (vLLM rollout speedup 5-10×, async generation, FlashAttn-2), nâng cấp performance (rsLoRA/DoRA, GSPO sequence-level, MO-GRPO multi-objective normalization, DAPO clip-higher, FP16 thay BF16, Dr.GRPO length-norm fix).

---

## 15. Đánh giá thực tế: RL CÓ thực sự cải thiện gwen-tts-0.6B không?

**Câu hỏi quan trọng nhất** trước khi đầu tư 300 GPU-hours và 4.5 tuần. Trả lời thẳng thắn dựa trên căn cứ.

### 15.1. Bằng chứng RL cải thiện được TTS LM (positive)

| Bằng chứng | Mô hình | Kết quả | Nguồn |
|---|---|---|---|
| Multi-Reward GRPO trên LLaSA-8B (single codebook AR LM, kiến trúc gần Qwen3-TTS) | LLaSA-8B | WER 1.59→1.10 (giảm 31%), SIM 0.684→0.758 (+11%) | [arXiv 2511.21270](https://arxiv.org/abs/2511.21270) |
| DiffRO trên CosyVoice 2.0 | CosyVoice 2.0 | WER giảm 50% relative trên SeedTTS-Eval | [arXiv 2507.05911](https://arxiv.org/abs/2507.05911) |
| Flow-GRPO trên SD3.5-M (image gen) — chứng tỏ GRPO không chỉ tốt cho text | SD3.5-M | GenEval 63%→95%, visual text 59%→92%, **"very little reward hacking"** | [Flow-GRPO NeurIPS 2025](https://neurips.cc/virtual/2025/poster/116065) |
| FlowSE-GRPO trên speech enhancement | flow-matching SE | Multi-metric reward giảm reward hacking | [arXiv 2601.16483](https://arxiv.org/html/2601.16483v1) |
| LoRA + RL ngang full FT (2/3 resources) | Llama RL | Match FFT performance | [Thinking Machines blog](https://www.reddit.com/r/LocalLLaMA/comments/1nturn1/) |

→ **Kết luận:** Có precedent rõ ràng RL cải thiện được TTS AR LM, đặc biệt cho intelligibility (WER) và speaker similarity (SIM).

### 15.2. Lý do RL có thể KHÔNG cải thiện gwen-tts-0.6B nhiều (negative)

Đây là phần phải nhìn thẳng — nếu mong đợi 50% relative WER reduction như DiffRO trên CosyVoice 2 thì sẽ thất vọng:

#### 15.2.1. "GRPO is a sharpener, not a teacher" (ICLR 2025 finding)

["Can GRPO Help LLMs Transcend Their Pretraining Origin?"](https://openreview.net/forum?id=9fwvcl0Jur) chứng minh: **GRPO chỉ sharpening pretraining biases, không tạo capabilities mới**. OOD improvement chỉ xuất hiện khi target task align với pretraining bias; in-distribution gain giảm dần khi performance saturate.

**Hệ quả cho gwen-tts-0.6B:**
- gwen-tts đã finetune trên ~1000h tiếng Việt → đã near-saturation cho Vietnamese voice cloning
- WER baseline có thể đã ở ~5-8% → headroom giảm WER có giới hạn vật lý (Whisper-large-v3 trên speech sạch tiếng Việt cũng ~5-7%)
- Reward signal có **floor** = Whisper VN error rate. Không thể giảm WER thấp hơn Whisper baseline.

#### 15.2.2. Multi-codebook diluion

Gwen3-TTS có 16 codebooks; RL chỉ train codebook 0 (semantic, talker). 15 codebooks acoustic do MTP frozen sinh ra → **RL chỉ ảnh hưởng 1/16 token stream**. So với CosyVoice 2 single-codebook (1/1) hay LLaSA single-codebook (1/1) — gwen-tts-0.6B có "reward leverage" thấp hơn 16×.

#### 15.2.3. RL ≠ better acoustic quality

WER reward đo intelligibility, không đo naturalness. Có 2 rủi ro thực:
- **Reward hacking:** model học trick Whisper (over-articulation, monotone) → WER↓ nhưng MOS-naturalness↓
- **MOS-N drop có thể đi kèm WER↓**, đặc biệt với β_KL thấp

["Why Does RL Generalize Better Than SFT? A Data-Centric Perspective"](https://arxiv.org/html/2602.10815v1) cho thấy **DC-SFT (Data-Curation SFT) có thể outperform GRPO** với stability và computational efficiency tốt hơn — đặc biệt với model nhỏ.

#### 15.2.4. Whisper VN noise floor

Whisper-medium có WER tiếng Việt ~10-15% trên speech sạch (cao hơn Chinese 2-3%, English 4-5%). Reward noise lớn → policy gradient noisy → có thể không học được signal khác noise.

### 15.3. Câu trả lời thẳng thắn

**RL có thể tốt hơn base model nếu:**
1. ✅ Whisper-medium WER trên test set của bạn ≥ 8% (đủ headroom)
2. ✅ Bạn dùng multi-reward (không chỉ WER) để chống reward hacking
3. ✅ Bạn dùng KL anchor đủ mạnh (β_KL ≥ 0.03) để giữ acoustic quality
4. ✅ Bạn run human MOS evaluation bên cạnh WER
5. ✅ Bạn so sánh với **SFT continuation** (Gate 1) như baseline mạnh — không chỉ với raw checkpoint

**RL có thể KHÔNG tốt hơn base model nếu:**
1. ❌ Test set baseline WER đã < 5% — RL gain sẽ < 1% absolute
2. ❌ Chỉ dùng single reward (WER) → reward hacking → MOS drop
3. ❌ KL anchor quá yếu → catastrophic forgetting tiếng Việt vùng phương ngữ
4. ❌ Không filter SNR — Whisper noise dominate signal

### 15.4. Khuyến nghị mới (trước Gate 1)

**Phải làm trước khi bắt đầu Gate 1:**

```python
# Step 0: Kiểm tra điều kiện đầu tư RL
# Run trên 200 samples test Vietnamese
results = {
    "baseline_WER": evaluate_wer(model, test_set),         # cần ≥ 5%
    "baseline_SIM": evaluate_sim(model, test_set),         # cần ≤ 0.85 (có headroom)
    "whisper_self_WER": evaluate_whisper_on_clean(test_set),  # noise floor
    "whisper_consistency": whisper_test_retest_variance(),    # < 2% std
}

# Decision rule:
if results["baseline_WER"] < 5.0:
    print("⚠ Headroom thấp. Cân nhắc DC-SFT thay vì RL")
if results["whisper_self_WER"] > results["baseline_WER"] * 0.7:
    print("⚠ Whisper là bottleneck, không phải model. RL sẽ không giúp")
if results["baseline_SIM"] > 0.88:
    print("⚠ SIM đã cao. SIM reward sẽ không gain nhiều")
```

**Ngưỡng go/no-go:**
- Baseline WER ≥ 5% AND Whisper noise floor < 50% baseline WER → **Go RL** (kỳ vọng giảm 10-25%)
- Baseline WER 3-5% → **Go DC-SFT trước**, RL chỉ là bonus
- Baseline WER < 3% → **Skip RL**, đầu tư vào prosody/MOS với reward khác (UTMOS, NISQA)

### 15.5. Alternative: DC-SFT (Data-Curation SFT) trước RL

Thêm Gate 0.5 trước Gate 1 (chi phí thấp, có thể bỏ qua RL):

```python
# Gate 0.5: Curate medium-difficulty samples từ training set
# Theo arXiv 2602.10815: train chỉ medium-difficulty samples
# beats both standard SFT và GRPO

for sample in train_set:
    # Difficulty = -log P(target | input) under current model
    difficulty = compute_negative_log_likelihood(model, sample)
    sample["difficulty"] = difficulty

# Drop hardest 20% (often noise) và easiest 30% (no info)
medium_set = [s for s in train_set 
              if percentile(s["difficulty"], 30) <= s["difficulty"] <= percentile(s["difficulty"], 80)]

# Train SFT chỉ trên medium set với LoRA + KL anchor
train_sft(model, medium_set, ...)
```

Nếu DC-SFT đã đạt target → có thể skip Gate 2-3. Compute < 50 GPU-h, rủi ro thấp.

---

## 16. Tối ưu bộ nhớ (Memory Optimization)

Mục tiêu: giảm peak VRAM để chạy được trên GPU 24GB hoặc thậm chí 16GB, không hi sinh chất lượng nhiều (< 0.5 WER absolute).

### 16.1. Bảng so sánh kỹ thuật memory

| Kỹ thuật | VRAM saved | Speed cost | Quality cost | Áp dụng Gate |
|---|---|---|---|---|
| **PEFT no-ref (disable LoRA cho ref)** | ~1.2 GB | 0% | 0% | 1, 2, 3 |
| **8-bit AdamW (bitsandbytes)** | ~75% optimizer state | 0-5% | < 0.1 quality drop | 1, 2, 3 |
| **Gradient checkpointing** | ~30-50% activations | +20-30% time | 0% | 2, 3 |
| **Liger Kernel chunked GRPO loss** | **40% peak** | -5-10% time (faster) | 0% | 2, 3 |
| **FlashAttention-2** | ~20% activations | -10-30% time | 0% | All |
| **vLLM colocate + sleep** | 30-50% generation memory | -20% time | 0% | 2, 3 |
| **CPU offload optimizer (DeepSpeed ZeRO-2 offload)** | Toàn bộ optim state | +10-30% time | 0% | Nếu OOM |
| **bf16 → fp16 mixed (sensitive parts)** | 0% | 0% | tăng stability | All |

### 16.2. PEFT no-ref trick (BIGGEST WIN, 0 cost)

**Vấn đề:** Spec hiện tại load 2 model copies (policy + reference) = 2.4 GB.

**Giải pháp:** Khi dùng LoRA, **disable adapters** sẽ cho output base model gốc — không cần load reference riêng. Tiết kiệm 1.2 GB **không hi sinh chất lượng**.

```python
# OLD (spec v1.0): 2.4 GB cho 2 model copies
model = load_with_lora(...)
ref_model = load_base_only(...)  # ❌ duplicate memory

with torch.no_grad():
    ref_logits = ref_model(input_ids)

# NEW (v2.0): 1.2 GB, single model
model = load_with_lora(...)

with torch.no_grad():
    with model.disable_adapter():    # ✓ tạm tắt LoRA = base model
        ref_logits = model(input_ids)
# Adapter tự re-enable sau context
```

**Căn cứ:** [TRL PEFT integration docs](https://huggingface.co/docs/trl/en/peft_integration), [Liger GRPO blog](https://huggingface.co/blog/liger-grpo) ("using PEFT in GRPO allows one to forgo loading a separate reference model").

**Caveat:** Có conflict đã biết với `sync_ref_model` trong TRL ([issue #3108](https://github.com/huggingface/trl/issues/3108)) khi LoRA + ZeRO3 — nếu dùng TRL phải tắt `sync_ref_model`. Implementation tự viết không ảnh hưởng.

### 16.3. 8-bit AdamW (bitsandbytes) — 75% optimizer memory saved

**Lý do:** AdamW giữ 2 state (m, v) per param, mỗi state fp32 = 8 byte/param. Với LoRA 5M params → 40MB. Nhưng nếu unfreeze MTP (80M params) ở Gate 3 → 640MB → đáng giá.

```python
import bitsandbytes as bnb

# OLD: standard AdamW
optimizer = torch.optim.AdamW(
    trainable_params, lr=config["lr"], betas=(0.9, 0.95), weight_decay=0.01
)

# NEW: 8-bit AdamW (drop-in replacement)
optimizer = bnb.optim.AdamW8bit(
    trainable_params, lr=config["lr"], betas=(0.9, 0.95), weight_decay=0.01,
    is_paged=True,    # paged memory: tự động offload to CPU khi không dùng
)
```

**Căn cứ:** [bitsandbytes docs](https://huggingface.co/docs/bitsandbytes/v0.43.0/en/optimizers): "75% less memory, same performance, 4x faster than regular Adam". Đã được verify trên hàng nghìn LLM finetuning runs.

**Quality impact:** < 0.1% (block-wise quantization với dynamic range estimation, giữ exact statistics cho elements lớn).

### 16.4. Liger Kernel — 40% GRPO loss memory + faster training

**Vấn đề chính:** GRPO loss tính `log_softmax(logits)` trên full vocab × full sequence. Với gwen-tts vocab 2048 × 1024 tokens × batch×G = 16384 → logits matrix 32 MB per sample × G=8 × bf16 = ~5 GB peak chỉ cho logits.

**Giải pháp Liger Chunked GRPO Loss** ([HF blog](https://huggingface.co/blog/liger-grpo)): chunk batch dimension, compute gradient từng chunk trong forward pass, accumulate — không bao giờ store full logits.

```python
# Drop-in với TRL:
from trl import GRPOConfig

training_args = GRPOConfig(
    use_liger_loss=True,    # ✓ chỉ thế thôi
    # ... rest of config
)
```

Hoặc nếu tự viết loop:

```python
# pip install liger-kernel
from liger_kernel.transformers import apply_liger_kernel_to_qwen3

apply_liger_kernel_to_qwen3(
    rope=True,
    swiglu=True,
    cross_entropy=False,           # dùng FusedLinearCrossEntropy thay
    fused_linear_cross_entropy=True,
    rms_norm=True,
)
```

**Lợi ích đo thực tế:** memory saved 40%, throughput +20%, batch size có thể tăng 1.5-1.8× ([HF Liger GRPO blog](https://huggingface.co/blog/liger-grpo)).

**Caveat cho gwen-tts:** Liger có Qwen3 support sẵn, nhưng custom layers (talker forward với ref_code prepend) có thể cần adapt. Test trên smoke run trước.

### 16.5. Gradient checkpointing có chiến lược

Không phải bật blanket — hi sinh 20-30% time. Chỉ bật cho activation-heavy layers:

```python
model.model.talker.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
# Không bật cho MTP (chỉ 5 layer, activation nhỏ)
# Không bật cho Mimi decoder (frozen)
```

**Selective checkpointing:** chỉ checkpoint mỗi 2 layer thay vì mọi layer (tradeoff memory vs speed):

```python
from torch.utils.checkpoint import checkpoint_sequential
# Checkpoint mỗi 2 layer của 28-layer talker
x = checkpoint_sequential(model.talker.layers, segments=14, x)
```

→ Memory saved ~40% thay vì 50%, nhưng speed cost chỉ +10-15% thay vì +25-30%.

### 16.6. vLLM colocate cho rollout (huge win cho Gate 2-3)

**Vấn đề:** Generation chiếm ~70% thời gian Gate 2-3. PyTorch native generate không tối ưu cho batch sampling với G=8 rollouts.

**Giải pháp vLLM colocate mode** ([HF blog](https://huggingface.co/blog/vllm-colocate)):

```python
# TRL config
training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,    # vLLM dùng 30% VRAM, train dùng 70%
    vllm_enable_sleep_mode=True,         # tự động free memory khi không generate
    vllm_lora_sync=True,                 # sync LoRA adapter file thay vì NCCL broadcast (40MB vs 1.4GB)
)
```

**Lợi ích:**
- Generation speed: 5-10× với PagedAttention + continuous batching
- LoRA sync: 200ms (filesystem) thay vì 350ms-5s (NCCL broadcast)
- Sleep mode level 2: free toàn bộ vLLM memory giữa các step → policy + vLLM share VRAM

**Caveat cho gwen-tts:**
- vLLM-Omni hỗ trợ Qwen3-TTS ([docs](https://docs.vllm.ai/projects/vllm-omni/en/stable/user_guide/examples/offline_inference/qwen3_tts/)) — nhưng custom voice cloning với ref_code prepend phải được test
- Nếu vLLM-Omni chưa support voice cloning đầy đủ, fallback: tự viết batched rollout với KV cache reuse

### 16.7. Tổng hợp memory budget mới (Gate 2 với tất cả tối ưu)

```
Kỹ thuật v1.0 (cũ):                 v2.0 (tối ưu):
─────────────────────────────────────────────────────
Model + LoRA:        1.3 GB         1.3 GB
Reference model:     1.2 GB    →    0.0 GB (PEFT no-ref)
Old policy snapshot: 1.2 GB    →    0.0 GB (importance sampling correction)
Whisper-medium:      3.0 GB    →    1.5 GB (whisper-large-v3-turbo, faster + smaller)
Mimi decoder:        0.06 GB        0.06 GB
Activations:         6.0 GB    →    3.6 GB (Liger 40% saved)
Gradients (LoRA):    0.08 GB        0.08 GB
Optimizer state:     0.08 GB    →   0.02 GB (8-bit AdamW)
Rollout buffer:      0.5 GB     →    0.5 GB (colocate vLLM, sleep mode)
Misc:                1.0 GB         0.8 GB
─────────────────────────────────────────────────────
TOTAL:               14.5 GB    →    7.8 GB  (giảm 46%)
```

→ Gate 2 fit thoải mái trên RTX 4090 24GB / RTX 3090 24GB, hoặc thậm chí RTX 4080 16GB nếu giảm G xuống 4.

→ Gate 3: 7.8 + 1.5 (WavLM) = ~9.3 GB → vẫn fit 16GB GPU.

### 16.8. Old policy snapshot — bỏ được không?

**Spec v1.0 yêu cầu π_θ_old = full model copy (1.2 GB).**

**v2.0 alternative:** Importance Sampling Correction (Axolotl pattern, [docs](https://docs.axolotl.ai/docs/vllm_serving.html)):

```python
# Thay vì giữ full old model:
# 1. Save log_probs từ vLLM rollout (ít memory, ~T tokens × G samples)
# 2. Importance sampling correction qua threshold

old_log_probs = saved_during_rollout    # [B*G, T] from vLLM
new_log_probs = forward_with_grad(model, codes)  # current

ratio = torch.exp(new_log_probs - old_log_probs)

# Mask sequences quá off-policy
is_off_policy = (ratio - 1).abs() > 0.5
advantages[is_off_policy] = 0  # ignore stale samples
```

**Tiết kiệm:** 1.2 GB và 1 lần forward pass. **Cost:** một số sample bị drop (acceptable, throughput vẫn cao).

---

## 17. Tối ưu thời gian train (Speed Optimization)

### 17.1. Bottleneck phân tích

Đo profiler trên RL TTS thấy phân bố thời gian:

| Phase | % time (v1.0) | Tối ưu hóa |
|---|---|---|
| Generation (rollout) | **65-75%** | vLLM, async generation, batched G samples |
| Reward computation (Whisper) | 10-15% | batched Whisper, faster-whisper, whisper-turbo |
| Forward (compute new log_probs) | 8-12% | FlashAttention-2, gradient checkpointing tuned |
| Backward + optimizer | 3-5% | 8-bit AdamW, fused optimizer |
| Mimi decoding | 2-3% | batch decode |

**→ Generation là bottleneck KHỔNG LỒ. Tối ưu nó = giảm 50%+ wall time.**

### 17.2. vLLM rollout speedup (5-10×)

Xem Section 16.6. Với gwen-tts-0.6B, vLLM colocate giảm rollout time từ ~30s/batch xuống ~3-5s/batch.

**Lưu ý quan trọng:** vLLM dùng FP16 cho computation by default; gwen-tts-0.6B train với BF16. Hai engine có **training-inference mismatch** — đây là vấn đề đã được paper ["Defeating the Training-Inference Mismatch via FP16"](https://arxiv.org/html/2510.26788v1) giải quyết bằng cách switch toàn bộ sang FP16:

```python
model = Qwen3TTSModel.from_pretrained(
    config["base_checkpoint"],
    dtype=torch.float16,    # ✓ FP16 thay BF16
    attn_implementation="flash_attention_2",
)
```

**Tradeoff BF16 vs FP16 cho RL:**
- BF16: range cao (e^±127), precision thấp (7 mantissa bits) → mismatch giữa training và inference engine
- FP16: range thấp (e^±15), precision cao (10 mantissa bits) → 8× nhiều giá trị hơn → ít rounding error hơn

Paper cho thấy FP16 **virtually eliminates training-inference mismatch** trong RL setting. Đối với gwen-tts-0.6B (model nhỏ), FP16 không có overflow issue (overflow phổ biến chỉ với model > 7B layer-norm gradient).

**Khuyến nghị:** Test FP16 trong sanity check; nếu loss stable, dùng FP16 cho Gate 2-3.

### 17.3. Async generation pattern

Thay vì sequential `rollout → reward → forward → backward`, làm parallel:

```python
# v1.0 (sequential): 30s rollout → 5s reward → 5s forward → 1s backward = 41s/step
# v2.0 (async): max(30s rollout, 10s reward+forward+backward) = 30s/step

import asyncio

async def grpo_step_async(model, batch, config):
    # Spawn rollout task
    rollout_task = asyncio.create_task(
        async_rollout(model, batch, config)
    )
    
    # Khi rollout đang chạy, prepare batch tiếp theo
    next_batch = await fetch_next_batch_from_loader()
    
    # Wait rollout xong
    rollouts = await rollout_task
    
    # Compute reward + train song song
    reward_task = asyncio.create_task(compute_rewards(rollouts))
    forward_task = asyncio.create_task(compute_new_logprobs(rollouts))
    
    rewards = await reward_task
    new_logprobs = await forward_task
    
    # Loss + backward
    loss = grpo_loss(rewards, new_logprobs, ...)
    loss.backward()
    optimizer.step()
```

**Lưu ý:** Async chỉ hữu ích khi reward model trên GPU khác hoặc CPU pinned. Nếu cùng GPU thì tuần tự nhanh hơn (tránh contention).

[ms-swift recommendation](https://github.com/modelscope/ms-swift/issues/3848): `--async_generate true --sleep_level 1` cho colocate.

### 17.4. Batched Whisper transcription

**v1.0 (slow):** transcribe từng wav 1 lần → 10-15s cho B*G=16 samples.

**v2.0 (fast):** dùng `faster-whisper` (CTranslate2 backend, 4× faster) + batched API:

```python
from faster_whisper import BatchedInferencePipeline, WhisperModel

# faster-whisper-medium giảm 2.5GB → 1.2GB, 4× faster
whisper_model = WhisperModel(
    "medium", device="cuda", compute_type="float16"
)
batched_pipeline = BatchedInferencePipeline(model=whisper_model)

# Batched transcription
transcripts = []
for wav_batch in chunks(wavs, batch_size=8):
    segments_list = batched_pipeline.transcribe_batched(
        wav_batch, language="vi", batch_size=8
    )
    transcripts.extend([" ".join(s.text for s in segs) for segs in segments_list])
```

**Hoặc whisper-large-v3-turbo** (Distill, 809M params, gần ngang large-v3 cho VN):
- Memory: 1.6GB (vs 3.0GB medium)
- Speed: 4× faster than large-v3
- Quality: WER tăng 0.5-1% so với large-v3 nhưng vẫn tốt hơn medium

### 17.5. FlashAttention-2 (mặc định bật)

Đã được include trong spec v1.0 với `attn_implementation="flash_attention_2"`. Confirm version ≥ 2.5.8 để có Qwen3 support.

### 17.6. Compile model với torch.compile

Thêm 5-15% throughput, miễn phí:

```python
model.model.talker = torch.compile(
    model.model.talker,
    mode="reduce-overhead",    # cho inference-heavy GRPO
    fullgraph=False,
)
```

**Caveat:** không compile lúc rollout nếu dùng vLLM (vLLM tự compile nội bộ).

### 17.7. Tổng hợp speedup ước lượng

| Optimization | Speedup |
|---|---|
| vLLM colocate (rollout) | **5×** trên rollout (chiếm 70% time) → 3.5× tổng |
| FP16 thay BF16 (eliminate mismatch) | 1.1× (ít re-compute logprobs) |
| faster-whisper batched | 1.05× (reward chỉ 10% time) |
| 8-bit paged AdamW | 1.05× |
| Liger fused kernels | 1.1-1.2× |
| torch.compile | 1.05-1.15× |
| Async generate (overlap) | 1.1-1.15× |
| **Tổng compounded** | **~5-6×** |

Gate 2 v1.0 dự kiến 1.5 tuần → v2.0 ~3 ngày. Gate 3: 2 tuần → ~4 ngày. **Tổng dự án: 4.5 tuần → ~10 ngày**.

---

## 18. Cải thiện performance (không hi sinh chất lượng)

Mục tiêu: nâng cao chất lượng cuối cùng vượt v1.0, có căn cứ paper.

### 18.1. Tổng quan các kỹ thuật và quality impact

| Kỹ thuật | Quality gain | Memory cost | Speed cost | Risk |
|---|---|---|---|---|
| **rsLoRA scaling** | +1-3% (rank cao hơn không degrade) | 0% | 0% | Rất thấp |
| **DoRA thay LoRA** | +2-4% | +0.01% params | +20% train (0% inference) | Thấp |
| **GSPO thay GRPO** | Stability tăng, ~1-2% gain | 0% | 0% | Trung (mới hơn) |
| **MO-GRPO multi-reward normalization** | Giảm reward hacking | 0% | 0% | Rất thấp |
| **DAPO Clip-Higher (ε_high=0.28)** | +2-5% (chống entropy collapse) | 0% | 0% | Thấp |
| **DAPO Token-level loss** | +1-3% (chống length bias) | 0% | 0% | Rất thấp |
| **Dr.GRPO length-norm fix** | Giảm verbosity bias | 0% | 0% | Rất thấp |
| **FP16 thay BF16** | Stability +, quality không đổi | 0% | -10% | Thấp |
| **Per-codebook reward (advanced)** | +3-5% acoustic | +50% reward time | -20% | Trung-cao |
| **DC-SFT data filtering** | OOD generalization +2-3% | 0% | 0% | Rất thấp |

### 18.2. rsLoRA — sửa scaling factor LoRA

**Vấn đề LoRA gốc:** scaling factor `α/r`. Khi tăng r (rank), gradient bị shrink → high-rank LoRA học chậm. Spec v1.0 dùng r=16, α=32 → scale = 2.0 (ổn).

**Nếu muốn tăng r=32, 64 cho more capacity:** dùng `α/√r` thay vì `α/r` ([rsLoRA paper](https://arxiv.org/abs/2312.03732), [HF blog](https://huggingface.co/blog/damjan-k/rslora)):

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=64,                         # tăng rank cho capacity
    lora_alpha=16,                # alpha cũ
    use_rslora=True,              # ✓ scale = 16/√64 = 2.0 (stable)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)
```

**Quality gain:** [rsLoRA paper](https://arxiv.org/abs/2312.03732) cho thấy higher rank với rsLoRA outperform LoRA r=16 by 1-3% với cùng compute (chỉ memory tăng).

**Khuyến nghị:** chỉ áp dụng rsLoRA nếu LoRA r=16 không đủ capacity (loss flat) ở Gate 1.

### 18.3. DoRA — Weight-Decomposed LoRA

[DoRA paper (arXiv 2402.09353)](https://arxiv.org/abs/2402.09353) decomposes weight = magnitude (1D vector) × direction (matrix), train LoRA chỉ cho direction. Closer to full FT learning capacity.

**Kết quả:** DoRA outperform LoRA 2-4% trên LLaMA, LLaVA across tasks.

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    use_dora=True,                # ✓ DoRA
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

**Caveat:** Train chậm ~20% hơn LoRA (decompose magnitude tốn extra forward). Inference cost = 0 (merge giống LoRA).

**Khuyến nghị:** Dùng DoRA cho Gate 1 (SFT), bonus quality không hi sinh inference. Cho Gate 2-3 (GRPO), test trên smoke run trước (DoRA + RL ít precedent).

### 18.4. GSPO — Sequence-level thay token-level (Qwen3 native)

**Quan trọng:** [GSPO](https://arxiv.org/abs/2507.18071) là RL algorithm Qwen team dùng cho **Qwen3 production**. Thay vì importance ratio token-level, GSPO dùng sequence-level:

\[
s_i(\theta) = \left( \frac{\pi_\theta(o_i | x)}{\pi_{\theta_{\text{old}}}(o_i | x)} \right)^{1/|o_i|}
\]

**Lợi ích:**
- **Stable hơn GRPO** trên sequence dài (gwen-tts có 60-1000+ tokens) — token-level ratio variance tăng theo length, GSPO length-normalized
- **Tolerant với precision mismatch** (BF16/FP16 mismatch → GSPO ít sensitive hơn)
- **Eliminate Routing Replay** cho MoE (không relevant gwen-tts)
- Qwen team report "superior training efficiency and performance compared to GRPO"

**Implementation diff với GRPO:**

```python
# GRPO (v1.0): per-token ratio
ratio_token = torch.exp(new_log_probs - old_log_probs)  # [B*G, T]
loss_token = torch.min(
    ratio_token * advantages.unsqueeze(-1),
    torch.clamp(ratio_token, 1-ε, 1+ε) * advantages.unsqueeze(-1)
)
loss = -loss_token.mean()

# GSPO (v2.0): sequence-level ratio length-normalized
seq_log_ratio = (new_log_probs - old_log_probs).sum(dim=-1) / seq_lens.float()  # [B*G]
ratio_seq = torch.exp(seq_log_ratio)  # [B*G]
loss_seq = torch.min(
    ratio_seq * advantages,
    torch.clamp(ratio_seq, 1-ε, 1+ε) * advantages
)
loss = -loss_seq.mean()
```

**Khuyến nghị:** Dùng GSPO cho Gate 2-3 thay GRPO. Giữ multi-reward (MO-GRPO style normalization) để chống reward hacking.

### 18.5. MO-GRPO — Multi-objective normalization (anti reward-hacking)

[MO-GRPO (arXiv 2509.22047)](https://arxiv.org/html/2509.22047v1) chỉ ra: vanilla GRPO trên multi-reward bị **bias về reward có variance cao** (low-variance rewards bị ignore).

**Hệ quả cho gwen-tts:** WER reward variance ~0.1 (range [0,1] với mean ~0.85, std ~0.05). SIM reward variance ~0.02. Length penalty binary (variance 0.25). → GRPO sẽ over-optimize WER + length, ignore SIM.

**Fix MO-GRPO:** Normalize từng reward bằng group std TRƯỚC khi combine:

```python
# v1.0 (bị bias):
R_combined = α_wer * R_wer + α_sim * R_sim + α_len * R_len + α_ent * R_ent
A = (R_combined - μ) / (σ + ε)  # ❌ R_wer dominate

# v2.0 MO-GRPO (each reward normalized separately):
A_wer = (R_wer - μ_wer) / (σ_wer + ε)        # group-normalize từng cái
A_sim = (R_sim - μ_sim) / (σ_sim + ε)
A_len = (R_len - μ_len) / (σ_len + ε)
A_ent = (R_ent - μ_ent) / (σ_ent + ε)
A = α_wer * A_wer + α_sim * A_sim + α_len * A_len + α_ent * A_ent  # ✓ balanced
```

Kết quả MO-GRPO: chứng minh giải quyết reward hacking trong machine translation, instruction following — pattern áp dụng được cho TTS multi-reward.

### 18.6. DAPO Clip-Higher — chống entropy collapse

[DAPO (arXiv 2503.14476)](https://arxiv.org/pdf/2503.14476) phát hiện: **upper clip ε=0.2 hạn chế exploration** vì nó cap probability tăng cho low-prob tokens. Asymmetric clip giúp đáng kể:

\[
\text{loss} = -\min\left( \rho A, \text{clip}(\rho, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}) A \right)
\]

Với `ε_low=0.2` (giữ nguyên), `ε_high=0.28` (nới thoáng cho exploration).

**Quality gain:** DAPO trên Qwen2.5-32B đạt 50% AIME accuracy — vanilla GRPO chỉ 30%. Token-level loss bonus giảm length bias.

```python
# Trong grpo_step():
ratio = torch.exp(new_lp - old_lp)
clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.28)   # ✓ asymmetric
loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()
```

### 18.7. Dr.GRPO — Sửa length normalization bias

Vấn đề GRPO standard: normalize loss bằng `1/|o_i|` per response → token-level learning rate **phụ thuộc length**. Sequence dài → mỗi token contribute ít → bias model về sequence ngắn hoặc dài tùy advantage sign.

[Dr.GRPO](https://www.emergentmind.com/topics/dr-grpo) thay bằng group mean length:

```python
# v1.0 GRPO bị length bias:
loss = (ratio * advantage).sum() / |o_i|    # ❌ per-response

# v2.0 Dr.GRPO:
group_mean_len = sum(seq_lens) / len(seq_lens)
loss = (ratio * advantage).sum() / group_mean_len  # ✓ uniform scaling
```

**Cho TTS:** Length bias = bias về audio dài/ngắn. Quan trọng nếu reward correlate với length (vd: dài hơn → Whisper transcribe đầy đủ hơn → WER thấp giả tạo). Gwen-tts có length penalty reward đã giảm risk này; Dr.GRPO là extra layer of correction.

### 18.8. Per-codebook reward (advanced — chỉ làm sau Gate 3)

Spec v1.0 chỉ train codebook 0 (1/16 token stream). MTP frozen → acoustic không cải thiện.

**Advanced approach:** Unfreeze MTP ở Gate 3 + add **per-codebook auxiliary reward**:

```python
# Reward riêng cho từng codebook k=1..15 (acoustic detail)
# Sử dụng MOS predictor (UTMOS, NISQA) làm signal

r_acoustic = utmos_predictor(generated_wav)   # [B] in [1, 5]
r_acoustic_normalized = (r_acoustic - 1) / 4   # [B] in [0, 1]

R_total += α_acoustic * r_acoustic_normalized
```

**Combined với DiffRO Gumbel-Softmax** cho codebooks 1-15: cho phép gradient flow qua MTP. Risk: bf16 instability. Mitigation: dùng FP16 (Section 17.2).

→ Để v3.0 sau khi confirm Gate 1-3 với codebook 0 hoạt động.

### 18.9. DC-SFT data curation (Gate 0.5, áp dụng cho mọi Gate)

Theo [arXiv 2602.10815](https://arxiv.org/html/2602.10815v1) (xem Section 15.5), filter data trước RL:

1. Score difficulty mỗi sample bằng base model
2. Drop top 20% hardest (often noise/outliers) và bottom 30% easiest (no info)
3. Train trên medium-difficulty 50% còn lại

**Tác dụng:** matches/exceeds RL OOD generalization với compute thấp hơn nhiều, **stable hơn**.

### 18.10. NEFTune (free quality boost cho SFT)

[NEFTune (arXiv 2310.05914)](https://arxiv.org/abs/2310.05914): inject Gaussian noise vào input embedding TTS → trick prevent overfit, free 5-10% improvement trên downstream tasks.

```python
# Trong sft_step Gate 1:
if training:
    embed_noise = torch.randn_like(text_embeds) * (alpha_neftune / sqrt(seq_len * dim))
    text_embeds = text_embeds + embed_noise
# alpha_neftune = 5 (default)
```

Áp dụng cho Gate 1 SFT only (không cho RL).

### 18.11. Configuration tổng hợp v2.0 (recommended)

```python
CONFIG_V2_OPTIMAL = {
    # === Memory ===
    "use_peft_no_ref": True,                # disable_adapter context cho ref
    "optimizer": "adamw_8bit_paged",        # bitsandbytes
    "liger_loss": True,
    "gradient_checkpointing": True,
    "gc_segments": 14,                       # selective
    
    # === Speed ===
    "use_vllm": True,
    "vllm_mode": "colocate",
    "vllm_sleep_level": 2,
    "vllm_lora_sync": True,
    "vllm_gpu_memory_utilization": 0.3,
    "async_generate": True,
    "whisper_model": "whisper-large-v3-turbo",   # faster + smaller
    "whisper_batch_size": 8,
    "compile_model": True,
    
    # === Numerical ===
    "dtype": "float16",                      # FP16 thay BF16 cho RL
    "attn_implementation": "flash_attention_2",
    
    # === LoRA ===
    "lora_r": 16,
    "lora_alpha": 32,
    "use_dora": True,                        # DoRA cho extra quality
    "use_rslora": False,                     # bật nếu r > 32
    
    # === RL algorithm (v2.0 GSPO + MO-GRPO + DAPO) ===
    "rl_algorithm": "gspo",                  # sequence-level thay token-level
    "clip_low": 0.2,                         # DAPO asymmetric
    "clip_high": 0.28,
    "length_normalization": "group_mean",    # Dr.GRPO
    "multi_reward_normalization": "per_reward",  # MO-GRPO
    "loss_aggregation": "token_mean",        # DAPO recommendation
    
    # === Anti-hacking ===
    "beta_kl": 0.05,
    "early_stop_metric": "val_mos",          # human MOS, không chỉ WER
    "reward_hacking_check_every": 200,
    
    # === Data ===
    "dc_sft_filter": True,                   # medium-difficulty filtering
    "dc_sft_drop_hardest": 0.20,
    "dc_sft_drop_easiest": 0.30,
    "snr_threshold_db": 15,
    
    # === Gate 1 SFT bonus ===
    "neftune_alpha": 5.0,                    # chỉ Gate 1
}
```

### 18.12. Expected improvement v2.0 vs v1.0

| Metric | v1.0 baseline | v2.0 optimized | Gain |
|---|---|---|---|
| Peak VRAM (Gate 2) | 14.5 GB | 7.8 GB | **−46%** |
| Wall time (full project) | ~4.5 tuần | ~10 ngày | **−68%** |
| WER reduction | 10-25% rel | **15-30% rel** (DoRA + DAPO + GSPO) | +20% relative |
| SIM gain | 2-5% | **3-7%** (MO-GRPO balanced) | +40% relative |
| Stability (training crash rate) | medium | **high** | GSPO + FP16 |
| Reward hacking incidents | medium-high | **low** | MO-GRPO normalization |
| MOS-Naturalness drop | -0.1 to -0.3 | **-0.0 to -0.1** | KL anchor + DC-SFT |

→ Tất cả gain có căn cứ paper cụ thể, không phải hứa hẹn vô căn cứ.

### 18.13. Thứ tự áp dụng (priority order)

Không apply tất cả cùng lúc. Theo thứ tự rủi ro thấp → cao:

**Tier 1 (apply ngay, rủi ro 0):**
1. PEFT no-ref (Section 16.2) — saves 1.2 GB
2. 8-bit AdamW paged (Section 16.3) — saves 75% optim
3. faster-whisper-turbo (Section 17.4) — 4× reward speed
4. FlashAttn-2 confirmed bật
5. MO-GRPO normalization (Section 18.5) — chống reward hacking
6. DAPO clip-higher (Section 18.6) — anti entropy collapse
7. DC-SFT data filter (Section 18.9) — better data

**Tier 2 (apply sau smoke test, rủi ro thấp):**
8. Liger Kernel GRPO loss (Section 16.4)
9. vLLM colocate (Section 16.6) — biggest speed win
10. DoRA (Section 18.3)
11. Dr.GRPO length-norm fix (Section 18.7)
12. NEFTune cho Gate 1 (Section 18.10)

**Tier 3 (apply sau Gate 1 success, rủi ro trung bình):**
13. GSPO thay GRPO (Section 18.4)
14. FP16 thay BF16 (Section 17.2)
15. Gradient checkpointing selective (Section 16.5)

**Tier 4 (chỉ nếu cần thêm gain, advanced):**
16. Per-codebook reward + unfreeze MTP (Section 18.8)
17. rsLoRA với r=64 (Section 18.2)

---

## 19. References bổ sung v2.0

### 19.1. Memory & speed optimization

- **Liger Kernel GRPO:** [HF blog](https://huggingface.co/blog/liger-grpo), [GitHub linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — 40% memory saved
- **vLLM colocate TRL:** [HF blog](https://huggingface.co/blog/vllm-colocate), [vLLM weight transfer docs](https://docs.vllm.ai/en/latest/training/weight_transfer/)
- **vLLM-Omni Qwen3-TTS:** [docs.vllm.ai](https://docs.vllm.ai/projects/vllm-omni/en/stable/user_guide/examples/offline_inference/qwen3_tts/)
- **8-bit AdamW:** [bitsandbytes docs](https://huggingface.co/docs/bitsandbytes/v0.43.0/en/optimizers)
- **PEFT disable_adapter:** [PEFT docs](https://huggingface.co/docs/peft/v0.7.1/package_reference/lora)
- **faster-whisper:** https://github.com/SYSTRAN/faster-whisper
- **FP16 vs BF16 cho RL:** [arXiv 2510.26788](https://arxiv.org/html/2510.26788v1)
- **Axolotl GRPO LoRA sync:** [docs](https://docs.axolotl.ai/docs/vllm_serving.html)

### 19.2. RL algorithm improvements

- **GSPO (Qwen3 production):** [arXiv 2507.18071](https://arxiv.org/abs/2507.18071), [Qwen blog](https://qwenlm.github.io/blog/gspo/)
- **DAPO:** [arXiv 2503.14476](https://arxiv.org/pdf/2503.14476), [verl docs](https://verl.readthedocs.io/en/latest/algo/dapo.html)
- **Dr.GRPO:** [emergentmind](https://www.emergentmind.com/topics/dr-grpo)
- **MO-GRPO multi-objective:** [arXiv 2509.22047](https://arxiv.org/html/2509.22047v1)
- **GTPO no reference model:** [arXiv 2508.03772](https://arxiv.org/html/2508.03772v3)
- **ProGRPO entropy collapse fix:** [arXiv 2602.05281](https://arxiv.org/html/2602.05281v1)

### 19.3. PEFT methods

- **rsLoRA:** [arXiv 2312.03732](https://arxiv.org/abs/2312.03732), [HF blog](https://huggingface.co/blog/damjan-k/rslora)
- **DoRA:** [arXiv 2402.09353](https://arxiv.org/abs/2402.09353), [NVIDIA paper](https://research.nvidia.com/labs/lpr/publication/liu2024dora/)
- **NEFTune:** [arXiv 2310.05914](https://arxiv.org/abs/2310.05914)

### 19.4. RL fundamentals & limitations

- **Can GRPO transcend pretraining?** [OpenReview ICLR 2026](https://openreview.net/forum?id=9fwvcl0Jur) — sharpener không phải teacher
- **DC-SFT beats RL:** [arXiv 2602.10815](https://arxiv.org/html/2602.10815v1)
- **Flow-GRPO no reward hacking:** [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/116065)
- **FlowSE-GRPO speech enhancement:** [arXiv 2601.16483](https://arxiv.org/html/2601.16483v1)

---

**END OF SPEC v2.0**

**Thứ tự thực hiện khuyến nghị:**

1. **Đọc Section 15** (đánh giá thực tế RL có cải thiện không) — quyết định go/no-go
2. **Run sanity checks** (Section 8.4) trên gwen-tts-0.6B + benchmark Whisper
3. **Apply Tier 1 optimizations** (Section 18.13): PEFT no-ref, 8-bit AdamW, faster-whisper, MO-GRPO, DAPO clip-higher, DC-SFT filter — TẤT CẢ rủi ro 0
4. **Gate 0.5: DC-SFT** (Section 15.5) — nếu đủ cải thiện, có thể skip RL
5. **Gate 1: SFT continuation** với DoRA + NEFTune
6. **Gate 2: GSPO minimal** với Liger + vLLM colocate (Tier 2)
7. **Gate 3: GSPO multi-reward** với MO-GRPO normalization, FP16
8. **Final: human MOS evaluation** — không chỉ WER

**Compute budget mới (v2.0):** ~80-100 GPU-hours A100 (vs 300h v1.0), ~$120-150 cloud cost.

**Timeline mới (v2.0):** ~10-12 ngày (vs 4.5 tuần v1.0).
