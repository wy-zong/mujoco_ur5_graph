# SmolVLA 與 XVLA 模型架構詳細介紹

> 本文件基於 LeRobot 專案原始碼分析，詳細說明 **SmolVLA** 和 **XVLA** 兩種 Vision-Language-Action (VLA) 模型的架構設計與系統建立方式。

---

## 目錄

1. [概述](#概述)
2. [SmolVLA 架構](#smolvla-架構)
3. [XVLA 架構](#xvla-架構)
4. [兩者比較](#兩者比較)
5. [訓練與微調指南](#訓練與微調指南)

---

## 概述

SmolVLA 和 XVLA 都屬於 **Vision-Language-Action (VLA)** 模型，其核心理念是將視覺語言模型 (VLM) 的理解能力與機器人動作生成結合，實現「看圖讀指令→輸出動作序列」的端到端控制。

兩者都使用 **Flow Matching** 技術進行動作去噪生成，但在 VLM 骨幹、動作頭設計和適應性機制上有顯著差異。

| 特性 | SmolVLA | XVLA |
|------|---------|------|
| **開發者** | Hugging Face | 2toINF + Hugging Face |
| **VLM 骨幹** | SmolVLM2 (Gemma-based) | Florence-2 (Encoder-only) |
| **視覺編碼器** | SigLIP | DaViT |
| **動作生成頭** | Action Expert (小型 LM) | SoftPromptedTransformer |
| **核心機制** | Cross-Attention KV Cache | Soft Prompts + Domain ID |
| **參數量** | ~450M | ~900M |
| **動作空間** | 固定維度 + padding | Action Registry (可註冊多種模式) |

---

## SmolVLA 架構

### 整體結構圖

```
┌──────────────────────────────┐
│                 actions      │
│                    ▲         │
│ ┌─────────┐      ┌─|────┐   │
│ |         │────► │      │   │
│ |         │ kv   │      │   │
│ |         │────► │Action │   │
│ |   VLM   │cache │Expert│   |
│ │         │────► |      │   │
│ │         │      │      │   │
│ └▲──▲───▲─┘      └───▲──┘   |
│  │  |   |            │      |
│  |  |   |          noise    │
│  │  │ state                 │
│  │ language tokens          │
│  image(s)                   │
└──────────────────────────────┘
```

### 核心模組詳解

#### 1. VLM 骨幹：SmolVLMWithExpertModel

**原始碼位置：** `lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py`

SmolVLA 的 VLM 是基於 `SmolVLM2-500M-Video-Instruct`，屬於 Gemma 系列的輕量視覺語言模型。

**建立流程：**

```python
# 1. 載入或建立 VLM
if load_vlm_weights:
    self.vlm = AutoModelForImageTextToText.from_pretrained(
        model_id, device_map=device, torch_dtype="bfloat16"
    )
else:
    config = AutoConfig.from_pretrained(model_id)
    self.vlm = SmolVLMForConditionalGeneration(config=config)

# 2. 裁剪 VLM 層數 (可選)
if num_vlm_layers > 0:
    self.get_vlm_model().text_model.layers = \
        self.get_vlm_model().text_model.layers[:num_vlm_layers]

# 3. 建立 Action Expert (縮小版 LM)
lm_expert_config = copy.deepcopy(config.text_config)
lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # 0.75x
lm_expert_config.intermediate_size = get_intermediate_size(...)
self.lm_expert = AutoModel.from_config(lm_expert_config)

# 4. Cross-Attention 機制：將 Expert 的 K/V 投影改為接收 VLM 維度
for layer_idx in range(len(self.lm_expert.layers)):
    if layer_idx % self_attn_every_n_layers == 0:
        continue  # 保留自注意力層
    # 修改 K/V 投影層：input_dim = VLM hidden_size
    self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
        vlm_kv_dim, expert_kv_dim, bias=attention_bias
    )
    self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
        vlm_kv_dim, expert_kv_dim, bias=attention_bias
    )
```

**關鍵設計：**
- VLM 和 Expert 每層交替使用 **Self-Attention** 和 **Cross-Attention**
- Cross-Attention 時：Q 來自 Expert，K/V 來自 VLM 的 KV Cache
- Self-Attention 時：QKV 全部來自 Expert + VLM 的拼接 Tokens
- 使用 RoPE (Rotary Position Embedding) 進行位置編碼

#### 2. Flow Matching 動作生成：VLAFlowMatching

**原始碼位置：** `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`

Flow Matching 是一種生成模型技術，透過學習一個向量場 `v_t` 來將噪聲 `x_T ~ N(0,1)` 逐步推向目標動作 `x_0`。

**訓練 Forward Pass：**

```python
def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise, time):
    # 1. 建構噪聲軌跡
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions  # 插值
    u_t = noise - actions  # 目標向量場

    # 2. 嵌入前綴 (images + language + state)
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )

    # 3. 嵌入後綴 (noisy_action + timestep)
    suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

    # 4. 建構 2D 注意力遮罩 (prefix 不看 suffix, suffix 可看 prefix)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

    # 5. VLM + Expert 聯合前向傳播
    (_, suffix_out), _ = self.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        inputs_embeds=[prefix_embs, suffix_embs],
    )

    # 6. 投影輸出並計算 MSE 損失
    v_t = self.action_out_proj(suffix_out)
    losses = F.mse_loss(u_t, v_t, reduction="none")
```

**推理 (去噪迭代)：**

```python
def sample_actions(self, ...):
    # 1. 先計算 prefix 的 KV cache (只算一次)
    _, past_key_values = self.vlm_with_expert.forward(
        inputs_embeds=[prefix_embs, None],
        fill_kv_cache=True,
    )

    # 2. 迭代去噪 (預設 10 步)
    x_t = noise                  # shape: (B, chunk_size, max_action_dim)
    for step in range(num_steps):
        time = 1.0 + step * dt   # 從 1.0 → 0.0
        v_t = self.denoise_step(x_t, prefix_pad_masks, past_key_values, time)
        x_t = x_t + dt * v_t     # Euler 步進
    return x_t
```

#### 3. Prefix / Suffix 嵌入

**Prefix (給 VLM 處理)：**
- **影像嵌入**：SigLIP 編碼 → Connector 投影 → 乘以 `√dim` 縮放
- **語言嵌入**：Tokenizer → Embedding 層 → 乘以 `√dim` 縮放
- **狀態嵌入**：Linear 投影到 VLM hidden_size

**Suffix (給 Expert 處理)：**
- **動作嵌入**：Linear 投影到 expert_hidden_size
- **時間嵌入**：正弦餘弦位置編碼 (sinusoidal)
- 動作 + 時間 拼接後通過 2 層 MLP (SiLU 啟動)

#### 4. 注意力遮罩設計

```
Prefix:  [img_1] [img_2] ... [lang_1] [lang_2] ... [state]  ← 互相可見
Suffix:  [action_1] [action_2] ... [action_N]                ← 可看 prefix + 自身

att_masks: [0 0 0 ... 0   0   ...  1       1 1 ... 1]
           ← prefix tokens →     state   ← suffix tokens →
```

- `att_mask = 0`：與前面所有 token 共享注意力（雙向）
- `att_mask = 1`：因果遮罩（只看自身及之前的 token）

#### 5. 配置參數

**原始碼位置：** `lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py`

```python
class SmolVLAConfig:
    n_obs_steps: int = 1              # 觀測步數
    chunk_size: int = 50              # 動作 chunk 長度
    n_action_steps: int = 50          # 每次推理執行的動作步數
    max_state_dim: int = 32           # 狀態維度 (不足則 padding)
    max_action_dim: int = 32          # 動作維度 (不足則 padding)
    resize_imgs_with_padding: (512, 512)  # 影像大小
    num_steps: int = 10               # Flow Matching 去噪步數
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    freeze_vision_encoder: bool = True   # 凍結視覺編碼器
    train_expert_only: bool = True       # 只訓練 Expert
    num_expert_layers: int = -1          # Expert 層數 (-1 = 與 VLM 相同)
    num_vlm_layers: int = 16            # VLM 使用前 N 層
    self_attn_every_n_layers: int = 2   # 每 N 層插入自注意力
    expert_width_multiplier: float = 0.75  # Expert 隱藏維度比例
    attention_mode: str = "cross_attn"   # 注意力模式
```

#### 6. 前後處理管線

**原始碼位置：** `lerobot/src/lerobot/policies/smolvla/processor_smolvla.py`

```
輸入管線：
  RenameObservations → AddBatchDimension → SmolVLANewLine
  → Tokenizer(SmolVLM2) → Device → Normalizer(MEAN_STD)

輸出管線：
  Unnormalizer → Device(CPU)
```

---

## XVLA 架構

### 整體結構圖

```
                    ┌─────────────────────────────────────┐
                    │            XVLA Pipeline            │
                    └─────────────────────────────────────┘

     ┌────────┐     ┌──────────────────┐     ┌─────────────────────────┐
     │ Images │────►│   Florence-2     │────►│   SoftPromptedTransformer│
     │ (多視角)│     │  Vision+Language │     │                         │
     └────────┘     │    Encoder       │     │  ┌─────────────────┐    │
                    │                  │     │  │ Action Encoder  │    │
     ┌────────┐     │  ┌────────────┐  │     │  │ (domain-aware)  │    │
     │ Text   │────►│  │ DaViT      │  │     │  └────────┬────────┘    │
     │(指令)  │     │  │ + BART enc │  │     │           │             │
     └────────┘     │  └────────────┘  │     │  ┌────────▼────────┐    │
                    └──────┬───────────┘     │  │ Transformer     │    │
                           │                 │  │ Blocks (×24)    │    │
                    vlm_features             │  │ + Soft Prompts  │    │
                    aux_visual               │  └────────┬────────┘    │
                           │                 │           │             │
                           └────────────────►│  ┌────────▼────────┐    │
     ┌────────┐                              │  │ Action Decoder  │    │
     │ Proprio│─────────────────────────────►│  │ (domain-aware)  │    │──►  Actions
     │ (狀態) │                              │  └─────────────────┘    │
     └────────┘                              └─────────────────────────┘
     ┌────────┐                                        ▲
     │Domain  │────────────────────────────────────────┘
     │  ID    │
     └────────┘
```

### 核心模組詳解

#### 1. VLM 骨幹：Florence-2

**原始碼位置：** `lerobot/src/lerobot/policies/xvla/modeling_florence2.py`

XVLA 使用 **Florence-2** 作為視覺語言骨幹。Florence-2 具有：
- **視覺塔 (Vision Tower)**：DaViT (Dual Attention Vision Transformer)
- **語言模型**：基於 BART 的 Encoder（僅使用 Encoder，Decoder 被刪除）
- **多模態融合**：影像特徵與文字嵌入合併後送入 BART Encoder

```python
class XVLAModel(nn.Module):
    def __init__(self, config, florence_config, proprio_dim):
        # 1. 建立 Florence-2 VLM
        self.vlm = Florence2ForConditionalGeneration(florence_config)

        # 2. 刪除不需要的 Decoder 部分
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        # 3. 建立 SoftPromptedTransformer
        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,           # 1024
            multi_modal_input_size=projection_dim,
            depth=config.depth,                        # 24
            num_heads=config.num_heads,                # 16
            num_domains=config.num_domains,            # 30
            dim_action=self.dim_action,
            dim_propio=self.dim_proprio,
            len_soft_prompts=config.len_soft_prompts,  # 32
        )
```

**VLM Forward：**

```python
def forward_vlm(self, input_ids, pixel_values, image_mask):
    # 1. 攤平多視角影像，只編碼有效視角
    flat_images = pixel_values.flatten(0, 1)
    valid_images = flat_images[flat_mask]
    valid_feats = self.vlm._encode_image(valid_images)

    # 2. 文字嵌入 + 影像特徵合併
    inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
    merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
        image_features[:, 0], inputs_embeds
    )

    # 3. 通過 BART Encoder
    enc_out = self.vlm.language_model.model.encoder(
        attention_mask=attention_mask, inputs_embeds=merged_embeds
    )[0]

    # 4. 輔助影像特徵 (額外視角)
    aux_visual_inputs = image_features[:, 1:].reshape(B, -1, hidden_dim)

    return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}
```

#### 2. SoftPromptedTransformer

**原始碼位置：** `lerobot/src/lerobot/policies/xvla/soft_transformer.py`

這是 XVLA 的核心動作頭，融合多模態特徵並生成動作。

**關鍵組件：**

```python
class SoftPromptedTransformer(nn.Module):
    def __init__(self, ...):
        # Transformer Blocks (×24, pre-LN)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # 多模態投影 (可選 domain-aware)
        self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
        self.aux_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)

        # 可學習位置嵌入
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size))

        # Domain-Aware 動作編碼器/解碼器
        self.action_encoder = DomainAwareLinear(
            dim_action + dim_time + dim_propio, hidden_size, num_domains
        )
        self.action_decoder = DomainAwareLinear(hidden_size, dim_action, num_domains)

        # Soft Prompt Hub (每個 domain 有獨立的 prompt)
        self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
```

**Forward 流程：**

```python
def forward(self, domain_id, vlm_features, aux_visual_inputs, action_with_noise, proprio, t):
    # 1. 編碼動作 token = [action + proprio + time_embedding]
    time_emb = timestep_embedding(t, dim_time)          # 正弦餘弦
    action_tokens = cat([action_with_noise, proprio, time_emb], dim=-1)
    x = self.action_encoder(action_tokens, domain_id)   # Domain-Aware

    # 2. 投影 VLM 特徵並拼接
    x = cat([x, vlm_proj(vlm_features), aux_visual_proj(aux_visual_inputs)], dim=1)

    # 3. 加入位置嵌入
    x = x + self.pos_emb[:, :seq_len, :]

    # 4. 附加 Soft Prompts
    soft_prompts = self.soft_prompt_hub(domain_id)      # [B, 32, hidden_size]
    x = cat([x, soft_prompts], dim=1)

    # 5. 通過 Transformer Blocks
    for block in self.blocks:
        x = block(x)    # Pre-LN → MHSA → Residual → Pre-LN → MLP → Residual

    # 6. 只解碼動作段
    return self.action_decoder(self.norm(x[:, :num_actions]), domain_id)
```

#### 3. DomainAwareLinear

XVLA 的獨特設計——**Domain-Aware 線性層**：每個 domain 擁有獨立的權重和偏置。

```python
class DomainAwareLinear(nn.Module):
    def __init__(self, input_size, output_size, num_domains=20):
        self.fc = nn.Embedding(num_domains, output_size * input_size)  # 權重
        self.bias = nn.Embedding(num_domains, output_size)             # 偏置

    def forward(self, x, domain_id):
        weight = self.fc(domain_id).view(B, input_size, output_size)
        bias = self.bias(domain_id).view(B, output_size)
        return matmul(x, weight) + bias
```

#### 4. Action Hub (動作空間註冊系統)

**原始碼位置：** `lerobot/src/lerobot/policies/xvla/action_hub.py`

XVLA 使用註冊機制支援多種動作空間：

| 模式 | 動作維度 | 說明 |
|------|---------|------|
| `ee6d` | 20 | 末端執行器 (xyz + 6D 旋轉 + gripper) |
| `joint` | 14 | 關節空間 + gripper |
| `agibot_ee6d` | 20 | AGI-bot 變體 (純 MSE 損失) |
| `so101_bimanual` | 20 (模型) / 12 (實際) | 雙臂 SO101 機器人 |
| `auto` | 自動偵測 | **推薦** - 從資料集推斷維度 |

每種模式定義了：
- `compute_loss()` — 不同分量使用不同損失 (MSE for joints, BCE for grippers)
- `preprocess()` — 前處理 (如歸零 gripper channels)
- `postprocess()` — 後處理 (如 sigmoid 轉換 gripper)

#### 5. Flow Matching 訓練與推理

**訓練：**
```python
def forward(self, input_ids, image_input, image_mask, domain_id, proprio, action):
    # 1. 編碼視覺語言
    enc = self.forward_vlm(input_ids, image_input, image_mask)

    # 2. Stratified timestep 採樣 (比均勻採樣更穩定)
    t = (rand(1) + arange(B) / B) % (1 - 1e-5)

    # 3. 加噪
    action_noisy = randn_like(action) * t + action * (1 - t)

    # 4. 前處理 (action_space 特定)
    proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

    # 5. Transformer 預測
    pred_action = self.transformer(domain_id, action_noisy_m, t, proprio_m, **enc)

    # 6. 計算損失 (action_space 特定)
    return self.action_space.compute_loss(pred_action, action)
```

**推理 (Euler 積分去噪)：**
```python
def generate_actions(self, ..., steps=10):
    enc = self.forward_vlm(input_ids, image_input, image_mask)
    x1 = randn(B, chunk_size, dim_action)  # 初始噪聲
    action = zeros_like(x1)

    for i in range(steps, 0, -1):
        t = i / steps
        x_t = x1 * t + action * (1 - t)
        proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
        action = self.transformer(domain_id, x_t_m, proprio_m, t, **enc)

    return self.action_space.postprocess(action)
```

#### 6. 配置參數

**原始碼位置：** `lerobot/src/lerobot/policies/xvla/configuration_xvla.py`

```python
class XVLAConfig:
    n_obs_steps: int = 1
    chunk_size: int = 32               # 動作 chunk 長度
    n_action_steps: int = 32

    # Florence-2 配置
    florence_config: dict = {}          # 包含 vision_config + text_config
    tokenizer_name: str = "facebook/bart-large"
    tokenizer_max_length: int = 64

    # Transformer Head
    hidden_size: int = 1024
    depth: int = 24                    # Transformer 層數
    num_heads: int = 16
    num_domains: int = 30              # 支援的 domain 數量
    len_soft_prompts: int = 32         # Soft Prompt 長度

    # 動作與本體感知
    action_mode: str = "ee6d"          # 動作空間模式
    num_denoising_steps: int = 10
    use_proprio: bool = True
    max_state_dim: int = 32
    max_action_dim: int = 20

    # 凍結選項
    freeze_vision_encoder: bool = False
    freeze_language_encoder: bool = False
    train_policy_transformer: bool = True
    train_soft_prompts: bool = True

    # 差異化學習率
    optimizer_lr: float = 1e-4
    # VLM 參數使用 1/10 學習率
```

#### 7. 前後處理管線

**原始碼位置：** `lerobot/src/lerobot/policies/xvla/processor_xvla.py`

```
輸入管線：
  AddBatchDimension → ImageScale(×255) → Tokenizer(BART)
  → ImageNetNormalize → AddDomainId → Device → Normalizer

輸出管線：
  ActionSpacePostprocess → Unnormalizer → Device(CPU)
```

特殊處理步驟：
- `XVLAImageScaleProcessorStep` — 將 [0,1] 影像轉為 [0,255]
- `XVLAImageToFloatProcessorStep` — 將 [0,255] 影像轉為 [0,1]
- `XVLAImageNetNormalizeProcessorStep` — ImageNet 正規化 (VLM 需要)
- `LiberoProcessorStep` — LIBERO 環境專用處理 (旋轉矩陣→6D表示)

---

## 兩者比較

### 架構設計哲學

| 設計面向 | SmolVLA | XVLA |
|---------|---------|------|
| **導向** | 輕量、快速微調 | 跨具身泛化 |
| **VLM-動作頭連接** | Cross-Attention + KV Cache | 特徵拼接 + 共同 Self-Attention |
| **推理效率** | KV Cache 只算一次 prefix | 每步都重新編碼 VLM (但 VLM 較小) |
| **多機器人支援** | 需重新微調 | Soft Prompts + Domain ID |
| **參數效率** | 只訓練 Expert (~25% 參數) | 可只訓練 Soft Prompts (~1% 參數) |

### 動作生成比較

| 項目 | SmolVLA | XVLA |
|------|---------|------|
| **chunk_size** | 50 | 32 |
| **去噪步數** | 10 | 10 |
| **時間採樣** | Beta(1.5, 1.0) | Stratified Uniform |
| **損失函數** | 統一 MSE | 分量式 (MSE + BCE) |
| **動作維度處理** | Zero-padding | Action Registry |

### Attention 機制比較

**SmolVLA：**
```
VLM layers:    L1 ──── L2 ──── L3 ──── L4 ──── ...
                │        │        │        │
Expert layers: E1(SA) ─ E2(CA) ─ E3(SA) ─ E4(CA) ─ ...

SA = Self-Attention (Expert 自身 tokens)
CA = Cross-Attention (Q=Expert, K/V=VLM cache)
```

**XVLA：**
```
[action_tokens | vlm_features | aux_visual | soft_prompts]
  └───────────全部拼接後進入統一 Self-Attention────────────┘
```

---

## 訓練與微調指南

### SmolVLA 微調

```bash
# 安裝依賴
pip install -e ".[smolvla]"

# 從預訓練模型微調
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=YOUR_DATASET \
  --batch_size=64 \
  --steps=20000 \
  --policy.device=cuda

# 從頭訓練 (載入 VLM 權重，隨機初始化 Expert)
lerobot-train \
  --policy.type=smolvla \
  --dataset.repo_id=YOUR_DATASET \
  --batch_size=64 \
  --steps=200000
```

### XVLA 微調

```bash
# 安裝依賴
pip install -e ".[xvla]"

# 推薦配置：不凍結 VLM，使用 auto 動作模式
lerobot-train \
  --policy.path="lerobot/xvla-base" \
  --dataset.repo_id=YOUR_DATASET \
  --policy.dtype=bfloat16 \
  --policy.action_mode=auto \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --steps=20000 \
  --policy.device=cuda
```

### Domain ID 配置 (XVLA)

| Dataset | Domain ID |
|---------|-----------|
| Bridge | 0 |
| RT1 | 1 |
| Calvin | 2 |
| LIBERO | 3 |
| WidowX-Air | 4 |
| AIR-AGILEX-HQ | 5 |
| RobotWin2 | 6 |
| RoboCasa | 7 |
| VLABench | 8 |
| AGIBOT | 9 |
| AIR-AGILEX | 10 |
| AIRBOT | 18 |

---

## 原始碼路徑總覽

### SmolVLA

| 檔案 | 功能 |
|------|------|
| `lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py` | 配置類別 |
| `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` | SmolVLAPolicy + VLAFlowMatching |
| `lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py` | SmolVLMWithExpertModel (VLM + Expert) |
| `lerobot/src/lerobot/policies/smolvla/processor_smolvla.py` | 前後處理管線 |

### XVLA

| 檔案 | 功能 |
|------|------|
| `lerobot/src/lerobot/policies/xvla/configuration_xvla.py` | 配置類別 |
| `lerobot/src/lerobot/policies/xvla/modeling_xvla.py` | XVLAPolicy + XVLAModel |
| `lerobot/src/lerobot/policies/xvla/soft_transformer.py` | SoftPromptedTransformer |
| `lerobot/src/lerobot/policies/xvla/action_hub.py` | Action Registry 系統 |
| `lerobot/src/lerobot/policies/xvla/modeling_florence2.py` | Florence-2 VLM 實作 |
| `lerobot/src/lerobot/policies/xvla/configuration_florence2.py` | Florence-2 配置 |
| `lerobot/src/lerobot/policies/xvla/processor_xvla.py` | 前後處理管線 |
| `lerobot/src/lerobot/policies/xvla/utils.py` | 工具函式 (旋轉表示轉換等) |

---

## 參考文獻

- **SmolVLA Paper**: [https://huggingface.co/papers/2506.01844](https://huggingface.co/papers/2506.01844)
- **XVLA Paper**: [https://arxiv.org/pdf/2510.10274](https://arxiv.org/pdf/2510.10274)
- **SmolVLA 預訓練模型**: [lerobot/smolvla_base](https://hf.co/lerobot/smolvla_base)
- **XVLA 預訓練模型**: [lerobot/xvla-base](https://hf.co/lerobot/xvla-base)
