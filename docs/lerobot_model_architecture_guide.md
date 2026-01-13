# LeRobot 模型架構與推論指南

> 本文件整理自對 LeRobot 專案的深入探索，涵蓋 ACT、SmolVLA、PI0 三種核心模型的架構設計、訓練流程、推論機制，以及相關的 Transformer 基礎概念。

---

## 目錄

1. [模型總覽](#1-模型總覽)
2. [ACT (Action Chunking Transformer)](#2-act-action-chunking-transformer)
3. [SmolVLA (Small Vision-Language-Action)](#3-smolvla-small-vision-language-action)
4. [PI0 (Physical Intelligence 0)](#4-pi0-physical-intelligence-0)
5. [SmolVLA vs PI0 詳細比較](#5-smolvla-vs-pi0-詳細比較)
6. [Expert 模型解析](#6-expert-模型解析)
7. [Attention Mode 詳解](#7-attention-mode-詳解)
8. [Transformer 核心概念](#8-transformer-核心概念)
9. [推論流程解析](#9-推論流程解析)
10. [LeRobot 推理指令比較](#10-lerobot-推理指令比較)
11. [SmolVLA 完整訓練參數](#11-smolvla-完整訓練參數)
12. [VLM 在 VLA 中的輸出](#12-vlm-在-vla-中的輸出)
13. [VLM 微調機制詳解](#13-vlm-微調機制詳解)

---

## 1. 模型總覽

LeRobot 支援多種機器人控制策略，以下是三種主要模型的對比：

| 模型 | 核心技術 | 基礎架構 | 訓練目標 | 模型大小 |
|------|---------|---------|---------|---------|
| **ACT** | Transformer | ResNet + Transformer Encoder/Decoder | L1 Loss + KL Divergence (VAE) | ~10M 參數 |
| **SmolVLA** | Flow Matching | SmolVLM (VLM) + Action Expert | MSE Loss (去噪) | ~500M 參數 |
| **PI0** | Flow Matching | PaliGemma (VLM) + Gemma Expert | MSE Loss (去噪) | ~3B+ 參數 |

### 訓練的基本概念

所有模型的訓練都遵循相同的基本流程：

```
輸入 (觀察)               輸出 (預測動作)           損失計算
┌───────────────┐       ┌──────────────┐       ┌──────────────┐
│ • 相機圖片     │       │              │       │              │
│ • 機器人狀態   │  ───► │    模型      │  ───► │ 預測 vs 真實  │
│ • 語言指令     │       │              │       │    ↓         │
└───────────────┘       └──────────────┘       │  計算誤差    │
                                                │    ↓         │
                                                │  反向傳播    │
                                                └──────────────┘
```

---

## 2. ACT (Action Chunking Transformer)

### 2.1 架構圖

```
                                 Transformer
                                 用於推理 (訓練時作為VAE decoder)
                                ┌───────────────────────┐
                                │             輸出      │
                                │              ▲        │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │   VAE   │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                 輸入     └─────┼──┘  │ 圖像嵌入        │
                                │   狀態嵌入            │
                                └───────────────────────┘
```

### 2.2 核心組件

**關鍵檔案位置**：`lerobot/policies/act/modeling_act.py`

```python
class ACT(nn.Module):
    def __init__(self, config: ACTConfig):
        # 1. 視覺骨幹 (ResNet18)
        self.backbone = IntermediateLayerGetter(
            resnet18(weights="IMAGENET1K_V1"),
            return_layers={"layer4": "feature_map"}
        )
        
        # 2. VAE Encoder (可選，用於變分訓練)
        self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
        
        # 3. Transformer Encoder-Decoder
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)
        
        # 4. 動作輸出頭
        self.action_head = nn.Linear(dim_model, action_dim)
```

### 2.3 訓練流程

```python
def forward(self, batch):
    # 1. 圖片 → ResNet → 特徵圖
    cam_features = self.backbone(image)["feature_map"]
    
    # 2. 如果使用 VAE：編碼動作序列 → 潛在變數
    latent_sample = sample_from_vae_encoder(actions)
    
    # 3. Transformer Encoder 處理：[latent, state, image_features]
    encoder_out = self.encoder([latent, state_emb, cam_features])
    
    # 4. Transformer Decoder 預測動作序列
    decoder_out = self.decoder(encoder_out)
    actions_hat = self.action_head(decoder_out)
    
    # 5. 計算損失
    l1_loss = |actions_hat - actions|
    kl_loss = KL(latent_pdf || N(0,1))
    total_loss = l1_loss + kl_weight * kl_loss
```

### 2.4 配置參數

**關鍵檔案位置**：`lerobot/policies/act/configuration_act.py`

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `chunk_size` | 100 | 預測的動作序列長度 |
| `n_action_steps` | 100 | 執行的動作步數 |
| `vision_backbone` | "resnet18" | 視覺編碼器 |
| `dim_model` | 256 | Transformer 隱藏維度 |
| `n_heads` | 8 | 注意力頭數 |
| `n_encoder_layers` | 4 | Encoder 層數 |
| `n_decoder_layers` | 1 | Decoder 層數 |
| `use_vae` | True | 是否使用 VAE |
| `latent_dim` | 32 | VAE 潛在空間維度 |

---

## 3. SmolVLA (Small Vision-Language-Action)

### 3.1 架構圖

```
┌──────────────────────────────┐
│                 actions      │
│                    ▲         │
│ ┌─────────┐      ┌─|────┐    │
│ |         │────► │      │    │
│ |         │ kv   │      │    │
│ |         │────► │Action│    │
│ |   VLM   │cache │Expert│    |
│ │         │────► |      │    │
│ │         │      │      │    │
│ └▲──▲───▲─┘      └───▲──┘    |
│  │  |   |            │       |
│  |  |   |          noise     │
│  │  │ state                  │
│  │ language tokens           │
│  image(s)                    │
└──────────────────────────────┘
```

### 3.2 核心組件

**關鍵檔案位置**：`lerobot/policies/smolvla/modeling_smolvla.py`

```python
class VLAFlowMatching(nn.Module):
    def __init__(self, config):
        # 1. VLM + Action Expert 雙塔架構
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            freeze_vision_encoder=True,
            train_expert_only=config.train_expert_only,
        )
        
        # 2. 狀態投影層
        self.state_proj = nn.Linear(max_state_dim, hidden_size)
        
        # 3. 動作投影層
        self.action_in_proj = nn.Linear(max_action_dim, expert_size)
        self.action_out_proj = nn.Linear(expert_size, max_action_dim)
        
        # 4. 時間步 + 動作融合 MLP
        self.action_time_mlp_in = nn.Linear(expert_size * 2, expert_size)
        self.action_time_mlp_out = nn.Linear(expert_size, expert_size)
```

### 3.3 Flow Matching 訓練流程

```python
def forward(self, images, lang_tokens, state, actions, noise=None, time=None):
    # 1. 採樣噪聲和時間
    noise = sample_gaussian_noise(actions.shape)
    time = sample_from_beta_distribution(bsize)  # 時間 t ∈ [0, 1]
    
    # 2. 線性插值：從乾淨動作到噪聲的路徑
    x_t = time * noise + (1 - time) * actions
    
    # 3. 目標：從噪聲到動作的「速度向量」
    u_t = noise - actions
    
    # 4. 嵌入 prefix (圖片 + 語言 + 狀態)
    prefix_embs = embed_prefix(images, lang_tokens, state)
    
    # 5. 嵌入 suffix (帶噪動作 + 時間步)
    suffix_embs = embed_suffix(x_t, time)
    
    # 6. VLM + Expert 前向傳播
    suffix_out = vlm_with_expert(prefix_embs, suffix_embs)
    
    # 7. 預測速度向量
    v_t = action_out_proj(suffix_out)
    
    # 8. MSE 損失
    losses = MSE(u_t, v_t)
    return losses
```

### 3.4 推理時的去噪過程

```python
def sample_actions(self, images, lang_tokens, state, noise=None):
    num_steps = 10  # 去噪步數
    dt = -1.0 / num_steps
    
    x_t = noise  # 從純噪聲開始
    
    for step in range(num_steps):
        time = 1.0 + step * dt  # 時間從 1 → 0
        
        # 網路預測當前時間的速度向量
        v_t = model.denoise_step(x_t, time)
        
        # 歐拉法積分
        x_t = x_t + dt * v_t
    
    return x_t  # 最終的乾淨動作
```

### 3.5 配置參數

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `chunk_size` | 50 | 預測的動作序列長度 |
| `num_steps` | 10 | 去噪步數 |
| `resize_imgs_with_padding` | (512, 512) | 圖像解析度 |
| `attention_mode` | "cross_attn" | 注意力模式 |
| `self_attn_every_n_layers` | 2 | 自注意力層間隔 |
| `freeze_vision_encoder` | True | 是否凍結視覺編碼器 |
| `train_expert_only` | True | 是否只訓練 Expert |
| `optimizer_lr` | 1e-4 | 學習率 |

---

## 4. PI0 (Physical Intelligence 0)

### 4.1 架構圖

```
┌───────────────────────────────────────────┐
│                                           │
│  ┌─────────────┐      ┌─────────────┐     │
│  │ PaliGemma   │      │   Gemma     │     │
│  │   (VLM)     │─────►│  Expert     │     │
│  │             │ KV   │  (Action)   │     │
│  └─────▲───────┘cache └──────▲──────┘     │
│        │                     │            │
│  ┌─────┴─────┐        ┌──────┴──────┐     │
│  │ 圖片+語言  │        │ 狀態+動作+時間│    │
│  └───────────┘        └─────────────┘     │
│                                           │
└───────────────────────────────────────────┘
```

### 4.2 核心組件

**關鍵檔案位置**：`lerobot/policies/pi0/modeling_pi0.py`

```python
class PI0Pytorch(nn.Module):
    def __init__(self, config):
        # 1. PaliGemma VLM (處理圖片+語言)
        paligemma_config = get_gemma_config("gemma_2b")
        
        # 2. Gemma Action Expert (專門處理動作)
        action_expert_config = get_gemma_config("gemma_300m")
        
        # 3. 雙模型架構
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            freeze_vision_encoder=True,
            train_expert_only=config.train_expert_only,
        )
        
        # 4. 投影層
        self.action_in_proj = nn.Linear(max_action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, max_action_dim)
        self.state_proj = nn.Linear(max_state_dim, expert_width)
```

### 4.3 配置參數

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `paligemma_variant` | "gemma_2b" | VLM 變體 |
| `action_expert_variant` | "gemma_300m" | Expert 變體 |
| `image_resolution` | (224, 224) | 圖像解析度 |
| `num_inference_steps` | 10 | 推理時的去噪步數 |
| `gradient_checkpointing` | False | 梯度檢查點 |
| `compile_model` | False | 是否使用 torch.compile |
| `freeze_vision_encoder` | False | 是否凍結視覺編碼器 |
| `train_expert_only` | False | 是否只訓練 Expert |
| `optimizer_lr` | 2.5e-5 | 學習率 |

---

## 5. SmolVLA vs PI0 詳細比較

### 5.1 配置參數差異

| 參數 | SmolVLA | PI0 | 說明 |
|------|---------|-----|------|
| **圖像解析度** | 512×512 | 224×224 | SmolVLA 使用更高解析度 |
| **梯度裁剪** | 10.0 | **1.0** | PI0 使用更保守的梯度裁剪 |
| **Weight Decay** | 1e-10 | **0.01** | PI0 用更強的正則化 |
| **學習率** | 1e-4 | 2.5e-5 | SmolVLA 學習率更高 |
| **凍結視覺編碼器** | ✅ 預設凍結 | ❌ 預設不凍結 | 不同的微調策略 |

### 5.2 架構設計差異

#### SmolVLA 的 Expert 架構

```
VLM (SmolVLM) 和 Action Expert 共享注意力機制
┌─────────┐     Cross-Attention      ┌─────────┐
│   VLM   │ ─────────────────────►  │  Expert  │
└─────────┘     每 N 層交錯          └─────────┘

self_attn_every_n_layers = 2  (交錯自注意力層)
expert_width_multiplier = 0.75 (Expert 比 VLM 窄)
```

#### PI0 的 Expert 架構

```
VLM 和 Expert 是獨立的 Gemma 模型
┌──────────────┐         ┌──────────────┐
│  PaliGemma   │  ──►    │    Gemma     │
│   (VLM)      │ KV      │   Expert     │
│  gemma_2b    │ cache   │  gemma_300m  │
└──────────────┘         └──────────────┘

兩個模型層數、寬度可以完全不同
```

### 5.3 狀態嵌入位置不同

```python
# SmolVLA: 狀態放在 prefix（VLM 處理）
def embed_prefix(images, lang_tokens, state):
    embs = [image_emb, lang_emb, state_emb]  # ← 狀態在這裡
    return torch.cat(embs)

# PI0: 狀態放在 suffix（Expert 處理）
def embed_prefix(images, lang_tokens):
    embs = [image_emb, lang_emb]  # ← 沒有狀態
    return torch.cat(embs)

def embed_suffix(state, noisy_actions, timestep):
    embs = [state_emb, action_time_emb]  # ← 狀態在這裡
    return torch.cat(embs)
```

### 5.4 為什麼 SmolVLA 凍結 VLM 而 PI0 可以微調？

#### SmolVLA 的理念

```
「保護預訓練知識，只訓練專門的 Expert」

優點：
• 避免災難性遺忘
• 訓練更穩定
• 計算資源需求低
• 快速適應新任務

適合場景：
• 資源有限
• 多任務泛化
```

#### PI0 的理念

```
「機器人控制是特殊任務，需要深度適應」

防止災難性遺忘的措施：
• freeze_vision_encoder = False (但可選擇凍結)
• train_expert_only = False (但可選擇只訓練 Expert)
• 極低的學習率 (2.5e-5 vs SmolVLA 的 1e-4)
• 更強的 weight decay (0.01 vs 1e-10)

優點：
• VLM 可以學習更好的空間理解
• 適應機器人視角的圖像
• 單任務可達更高性能

適合場景：
• 大量計算資源
• 大規模數據集
• 追求極致性能
```

---

## 6. Expert 模型解析

### 6.1 什麼是 Expert？

**Expert (專家模型)** 是一個專門處理動作生成的小型神經網路，它與主要的 VLM (視覺語言模型) 並行工作。

```
┌────────────────────────────────────────────────────────────────┐
│                                                                 │
│   VLM (Vision-Language Model)        Action Expert              │
│   ┌───────────────────────┐         ┌──────────────────┐       │
│   │ • 理解圖片             │         │ • 處理動作預測    │       │
│   │ • 理解語言指令         │  ───►   │ • 專注於機器人    │       │
│   │ • 提取視覺語義特徵     │  資訊   │   控制任務        │       │
│   │                       │  傳遞   │                  │       │
│   │ 模型大小：~500M-3B     │         │ 模型大小：~75M    │       │
│   └───────────────────────┘         └──────────────────┘       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 為什麼需要 Expert？

| 問題 | 解決方案 |
|------|----------|
| VLM 太大，微調成本高 | Expert 小，只需微調 Expert |
| VLM 是通用模型，不擅長控制 | Expert 專門為機器人動作設計 |
| 避免災難性遺忘 | VLM 凍結，Expert 學習新任務 |

### 6.3 VLM 在機器人策略中的角色

**重要澄清**：VLM 在這兩個模型中都**不是用來輸出語言的**！

```
VLM 原始設計（聊天機器人）：
輸入: "這張圖片裡有什麼？" + [圖片]
輸出: "圖片中有一隻貓坐在桌子上..."

在 SmolVLA/PI0 中（機器人控制）：
輸入: "抓起紅色方塊" + [圖片] + [機器人狀態]
      ↓
    VLM (只當作編碼器！)
      ↓
輸出: [隱藏狀態向量]  ← 不是文字！
      ↓
    Expert
      ↓
輸出: [動作序列] ← 關節角度
```

VLM 的角色是「理解器」，把圖片和語言轉換成向量表示，然後 Expert 用這些向量來生成動作。

---

## 7. Attention Mode 詳解

### 7.1 什麼是 Attention Mode？

**Attention Mode** 決定了 VLM 和 Expert 之間**資訊如何流動**。

### 7.2 三種模式

#### 自注意力 (self_attn)

```
VLM tokens + Expert tokens → 合併 → 一起做自注意力

[圖片] [語言] [狀態] [動作] [動作] [動作]
   ↑      ↑      ↑      ↑      ↑      ↑
   └──────┴──────┴──────┴──────┴──────┘
               全部一起計算注意力

優點：資訊流動最自由
缺點：計算量大 O(n²)
```

#### 交叉注意力 (cross_attn) - SmolVLA 預設

```
VLM 單獨處理                Expert 從 VLM 取資訊
┌──────────────┐           ┌──────────────┐
│ [圖片][語言] │    K,V    │  [動作預測]   │
│    ↓  ↓      │ ───────►  │      ↓       │
│  自注意力    │           │  交叉注意力   │
└──────────────┘           └──────────────┘

Expert 的 Query 去查詢 VLM 的 Key, Value

優點：VLM 可以完全凍結，計算效率高
缺點：Expert 無法影響 VLM 的表示
```

### 7.3 SmolVLA 的 self_attn_every_n_layers

```python
# 配置選項
self_attn_every_n_layers: int = 2  # 每 2 層插入一次自注意力

# 效果
Layer 0: 🔵 self_attn   (VLM + Expert 一起)
Layer 1: 🟡 cross_attn  (Expert 查詢 VLM)
Layer 2: 🔵 self_attn   (VLM + Expert 一起)
Layer 3: 🟡 cross_attn  (Expert 查詢 VLM)
...
```

### 7.4 調整注意力的影響

| 場景 | 建議設定 | 預期效果 |
|------|---------|---------|
| GPU 記憶體有限 | `cross_attn`, `n_layers=4` | 省記憶體，速度快 |
| 追求最高精度 | `self_attn` | 資訊流動最自由，但慢 |
| 微調預訓練模型 | `cross_attn`, `n_layers=2` | 平衡效能和精度 |
| 從頭訓練 | `self_attn` | Expert 和 VLM 一起學習 |
| 快速推理 | `cross_attn`, `use_cache=True` | KV Cache 加速 |

---

## 8. Transformer 核心概念

### 8.1 Query, Key, Value

用圖書館比喻：

```
📖 Query (查詢)：你想找的書的描述
   "我想找一本關於機器人控制的書"

🏷️ Key (鍵)：每本書的標籤/索引
   "書名: 機器人學", "書名: 深度學習", "書名: 控制理論"...

📚 Value (值)：書的實際內容
   書裡面真正的知識和資訊

💡 注意力機制的運作：
   1. 用 Query 和所有 Key 比對相似度
   2. 相似度高的 Key → 對應的 Value 權重高
   3. 加權平均所有 Value → 得到你需要的資訊
```

### 8.2 數學公式

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V

其中：
• Q × K^T：計算 Query 和每個 Key 的相似度
• softmax：把相似度轉成權重 (加總 = 1)
• × V：用權重加權 Value
```

### 8.3 KV Cache

在自回歸生成時，每次生成新詞都要重新計算之前所有詞的 K 和 V，這很浪費。

**解決方案：KV Cache**

```
生成 "我"：
    計算 K₁, V₁
    存入 Cache: [K₁, V₁]

生成 "愛"：
    從 Cache 取出: K₁, V₁  ✅ 不用重算
    計算 K₂, V₂
    存入 Cache: [K₁, V₁, K₂, V₂]

生成 "機器人"：
    從 Cache 取出: K₁, V₁, K₂, V₂  ✅ 不用重算
    計算 K₃, V₃

速度提升 3-10 倍！
```

### 8.4 在 SmolVLA/PI0 中的 KV Cache 應用

```python
# 推理時的流程

# 第一步：處理圖片+語言 (prefix)，建立 KV Cache
_, past_key_values = self.vlm_with_expert.forward(
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
    fill_kv_cache=True,
)

# 第二步：去噪循環，使用快取的 KV
for step in range(num_steps):
    suffix_out = self.vlm_with_expert.forward(
        inputs_embeds=[None, suffix_embs],
        past_key_values=past_key_values,  # ← 使用快取
        use_cache=True,
        fill_kv_cache=False,
    )
```

---

## 9. 推論流程解析

### 9.1 完整流程圖

```
┌─────────────────────────────────────────────────────────────────┐
│  1️⃣ 初始化階段                                                  │
├─────────────────────────────────────────────────────────────────┤
│   a. 載入 Policy 模型 (config.json + model.safetensors)         │
│   b. 建立 Environment (模擬器或真實機器人)                       │
│   c. 建立 Pre/Post Processors (資料正規化、裝置轉換)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2️⃣ Rollout 迴圈 (每個 episode)                                 │
├─────────────────────────────────────────────────────────────────┤
│   policy.reset()           # 重置內部狀態                       │
│   observation = env.reset() # 重置環境                          │
│                                                                  │
│   while not done:                                                │
│       ① 預處理觀察 → preprocessor(observation)                  │
│       ② 模型推論   → policy.select_action(observation)          │
│       ③ 後處理動作 → postprocessor(action)                      │
│       ④ 執行動作   → env.step(action)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Action Chunking

```
模型一次預測 50 步動作，但逐步執行：

第 1 次呼叫: 推論 [a₀, a₁, a₂, ..., a₄₉] → 回傳 a₀
第 2 次呼叫: 不推論，直接回傳 a₁
第 3 次呼叫: 不推論，直接回傳 a₂
...
第 50 次呼叫: 不推論，直接回傳 a₄₉
第 51 次呼叫: 重新推論新的 chunk！

效果：減少推論次數，提高效率！
```

### 9.3 Flow Matching 去噪過程

```
時間 t=1.0: x_t = [噪聲噪聲噪聲噪聲]
     ↓ v_t
時間 t=0.9: x_t = [有點像動作的東西]
     ↓ v_t
時間 t=0.8: x_t = [更像動作了]
     ↓ v_t
...
     ↓ v_t
時間 t=0.1: x_t = [動作₀, 動作₁, ..., 動作₄₉]

最終輸出：50 步乾淨的動作序列！
```

---

## 10. LeRobot 推理指令比較

### 10.1 三種推理方式

| 指令 | 用途 | 運行環境 |
|------|------|---------|
| `lerobot-eval` | 模擬環境評估 | 虛擬環境 |
| `lerobot-record` | 真實機器人 + 記錄數據 | 真實機器人 |
| `lerobot-replay` | 重播已錄製的動作 | 真實機器人 |

### 10.2 lerobot-eval

```bash
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10
```

**特點**：
- 在模擬器中評估策略性能
- 自動 reset 環境
- 自動判斷成功/失敗
- 可並行執行多個環境
- 快速評估、可重複

**為什麼只支援虛擬環境**：
- 需要 `env.reset()` 自動重置
- 需要自動判斷 `is_success`
- 需要並行執行
- 真實機器人無法滿足這些需求

### 10.3 lerobot-record

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=COM5 \
    --policy.path=outputs/model/pretrained_model \
    --teleop.type=so101_leader \
    --teleop.port=COM6 \
    --dataset.repo_id=local/my_dataset \
    --dataset.single_task="Grab the red cube"
```

**特點**：
- 控制真實機器人
- 同時記錄數據
- 可用 Policy 或 Teleop 控制
- 可混合使用（隨時切換）

**這是做真實機器人推理的標準方式！**

### 10.4 lerobot-replay

```bash
lerobot-replay \
    --robot.type=so101_follower \
    --replay.dataset=local/my_dataset \
    --replay.episode=0
```

**特點**：
- 重播已錄製的動作
- 用於驗證錄製的數據是否正確
- 沒有推理，只是播放

---

## 11. SmolVLA 完整訓練參數

LeRobot 提供完整的 SmolVLA 訓練配置，所有參數都可透過命令行覆蓋。

### 11.1 輸入/輸出結構

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `n_obs_steps` | 1 | 觀察歷史步數 |
| `chunk_size` | 50 | 預測的動作序列長度 |
| `n_action_steps` | 50 | 執行的動作步數 |
| `max_state_dim` | 32 | 最大狀態維度 (會 padding) |
| `max_action_dim` | 32 | 最大動作維度 (會 padding) |

### 11.2 圖像預處理

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `resize_imgs_with_padding` | (512, 512) | 圖像解析度 |
| `empty_cameras` | 0 | 添加空白相機數量 |

### 11.3 VLM + Expert 架構

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `vlm_model_name` | "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" | VLM 骨幹 |
| `load_vlm_weights` | False | 是否載入 VLM 權重 (從 smolvla_base 時為 True) |
| `num_vlm_layers` | 16 | VLM 使用的層數 |
| `num_expert_layers` | -1 | Expert 層數 (-1 = 與 VLM 相同) |
| `expert_width_multiplier` | 0.75 | Expert 寬度比例 |

### 11.4 注意力機制

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `attention_mode` | "cross_attn" | 注意力模式 (cross_attn / self_attn) |
| `self_attn_every_n_layers` | 2 | 每 N 層插入自注意力 |
| `use_cache` | True | 使用 KV Cache |

### 11.5 微調設定

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `freeze_vision_encoder` | True | 凍結視覺編碼器 |
| `train_expert_only` | True | 只訓練 Expert |
| `train_state_proj` | True | 訓練狀態投影層 |

### 11.6 優化器設定

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `optimizer_lr` | 1e-4 | 學習率 |
| `optimizer_betas` | (0.9, 0.95) | Adam betas |
| `optimizer_eps` | 1e-8 | Adam epsilon |
| `optimizer_weight_decay` | 1e-10 | 權重衰減 |
| `optimizer_grad_clip_norm` | 10 | 梯度裁剪 |

### 11.7 學習率調度器

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `scheduler_warmup_steps` | 1,000 | 預熱步數 |
| `scheduler_decay_steps` | 30,000 | 衰減步數 |
| `scheduler_decay_lr` | 2.5e-6 | 最終學習率 |

### 11.8 官方推薦訓練指令

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/mydataset \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --policy.device=cuda \
  --wandb.enable=true
```

### 11.9 自訂參數範例

```bash
# 調整學習率和訓練步數
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/my_dataset \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=500 \
  --steps=15000

# 調整 Expert 架構
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/my_dataset \
  --policy.num_expert_layers=8 \
  --policy.expert_width_multiplier=0.5

# 完全微調 (不只訓練 Expert)
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/my_dataset \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false
```

---

## 12. VLM 在 VLA 中的輸出

### 12.1 VLM 輸出什麼？

**答案：Hidden States (隱藏狀態向量) + KV Cache**

```
輸入                           VLM                           輸出
┌────────────────┐       ┌──────────────┐       ┌────────────────────┐
│ 📷 圖片         │       │              │       │ Hidden States      │
│ 💬 語言指令     │  ───► │   SmolVLM    │  ───► │ (隱藏狀態向量)      │
│ 🤖 機器人狀態   │       │   PaliGemma  │       │                    │
└────────────────┘       └──────────────┘       │ 形狀:               │
                                                 │ (batch, seq_len,   │
                                                 │  hidden_size)      │
                                                 │                    │
                                                 │ 例如:              │
                                                 │ (1, 512, 1152)     │
                                                 └────────────────────┘
                                                         │
                                                         ▼
                                                 ┌────────────────────┐
                                                 │ + KV Cache         │
                                                 │ (用於推理加速)      │
                                                 └────────────────────┘
```

### 12.2 Hidden States 代表什麼？

```
Hidden States 是一個「理解向量」，它編碼了：

├── 圖片內容：場景中有什麼物體、在哪裡
├── 語言意圖：使用者想讓機器人做什麼
├── 狀態資訊：機器人目前的姿態
└── 多模態對齊：文字和圖像的關聯

這不是文字！是一個高維度的數值向量
```

### 12.3 原始 VLM vs VLA 中的 VLM

```
原始 VLM (聊天機器人)：
輸入: "這張圖片裡有什麼？" + [圖片]
      ↓
    VLM (全部)
      ↓
    LM Head (語言輸出層)
      ↓
輸出: "圖片中有一隻貓..." (文字 tokens)


VLA 中的 VLM (機器人控制)：
輸入: "抓起紅色方塊" + [圖片] + [狀態]
      ↓
    VLM (只用 Hidden States)
      ↓
    ❌ LM Head (不使用！)
      ↓
    Action Expert (使用 Hidden States)
      ↓
輸出: [0.1, 0.3, -0.2, ...] (動作向量)
```

### 12.4 程式碼中的實際輸出

```python
# SmolVLA 的 VLM 前向傳播
def forward(...):
    outputs, past_key_values = self.vlm_with_expert.forward(
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
        fill_kv_cache=True,
    )
    
    # outputs 形狀: (batch, seq_len, hidden_size)
    # 例如: (1, 512, 1152) for SmolVLM-500M
    
    # past_key_values: 每一層的 (K, V) 快取
    # 用於 Expert 做交叉注意力
```

---

## 13. VLM 微調機制詳解

### 13.1 梯度反向傳播流程

```
前向傳播 (Forward Pass)

輸入                      VLM                    Expert              輸出
[圖片]                     │                       │
[語言]  ──────►  ┌─────────▼─────────┐     ┌──────▼──────┐     預測動作
[狀態]           │  Hidden States    │────►│   Action    │────►  â
                 │  (理解向量)        │     │   Expert    │
                 └───────────────────┘     └─────────────┘


反向傳播 (Backward Pass)

                                                              Loss
                                                         L = MSE(â, a)
                                                              │
                                                              ▼
                 ┌───────────────────┐     ┌─────────────┐   ∂L/∂â
                 │                   │◄────│   Action    │◄────
                 │       VLM         │     │   Expert    │
                 │                   │     │             │
                 └───────────────────┘     └─────────────┘
                         │                       │
                         ▼                       ▼
                    ∂L/∂W_vlm               ∂L/∂W_expert
                    (VLM 權重梯度)          (Expert 權重梯度)
```

### 13.2 凍結 vs 解凍 VLM

#### 凍結 VLM (`freeze_vision_encoder=True`, `train_expert_only=True`)

```
    VLM                    Expert
  ┌─────────┐            ┌─────────┐
  │ 🔒 凍結  │  ────────► │ 🔓 訓練  │
  │ 不更新   │  hidden    │ 更新權重 │
  └─────────┘  states    └─────────┘
       ↑                      ↑
   grad = 0              grad ≠ 0

效果：
• VLM 保持原本的「理解能力」
• Expert 學習如何把這個理解轉換成動作
• 訓練快、記憶體少
```

#### 解凍 VLM (`freeze_vision_encoder=False`, `train_expert_only=False`)

```
    VLM                    Expert
  ┌─────────┐            ┌─────────┐
  │ 🔓 訓練  │  ────────► │ 🔓 訓練  │
  │ 更新權重 │  hidden    │ 更新權重 │
  └─────────┘  states    └─────────┘
       ↑                      ↑
   grad ≠ 0              grad ≠ 0

梯度流：
Loss = MSE(預測動作, 真實動作)
  │
  ▼
∂L/∂(Action Expert weights)  ← 直接更新
  │
  ▼
∂L/∂(Hidden States)  ← 繼續往後傳
  │
  ▼
∂L/∂(VLM weights)  ← VLM 也被更新！

效果：
• VLM 學習產生「對動作預測更有用」的表示
• 可能學到更好的空間理解
• 但可能忘記原本的語言/視覺能力 (災難性遺忘)
```

### 13.3 解凍 VLM 時，VLM 學到什麼？

```
梯度信號來自：
「這個 hidden state 讓 Expert 預測出錯誤的動作」

所以 VLM 會調整以：
「產生讓 Expert 更容易預測正確動作的 hidden state」

具體變化：
┌────────────────────────────────────────────────────────────┐
│                                                             │
│  Before (預訓練的 VLM)                                      │
│  ────────────────────                                       │
│  • 擅長描述圖片內容                                         │
│  • 擅長回答問題                                             │
│  • 不知道什麼資訊對機器人控制重要                           │
│                                                             │
│  After (微調後的 VLM)                                       │
│  ─────────────────────                                      │
│  • 學會強調物體位置資訊                                     │
│  • 學會理解相對空間關係                                     │
│  • 學會從機器人視角編碼場景                                 │
│  • 可能變得不擅長聊天了                                     │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### 13.4 程式碼中的實現

```python
class SmolVLMWithExpertModel(nn.Module):
    def __init__(self, ..., train_expert_only=True, freeze_vision_encoder=True):
        
        # 載入 VLM
        self.vlm = AutoModelForImageTextToText.from_pretrained(...)
        
        # 凍結視覺編碼器
        if freeze_vision_encoder:
            for param in self.vlm.model.vision_model.parameters():
                param.requires_grad = False  # 🔒 不計算梯度
        
        # 只訓練 Expert
        if train_expert_only:
            for param in self.vlm.parameters():
                param.requires_grad = False  # 🔒 凍結 VLM
            
            # 但解凍 Expert
            for param in self.lm_expert.parameters():
                param.requires_grad = True  # 🔓 訓練 Expert
```

### 13.5 設定比較

| 設定 | VLM 梯度 | Expert 梯度 | 效果 |
|------|---------|------------|------|
| `train_expert_only=True` | ❌ 不計算 | ✅ 計算 | 快速微調，保留 VLM 能力 |
| `train_expert_only=False` | ✅ 計算 | ✅ 計算 | 深度適應，可能更高性能 |
| `freeze_vision_encoder=True` | 視覺 ❌ | ✅ 計算 | 保護視覺編碼器 |
| `freeze_vision_encoder=False` | 視覺 ✅ | ✅ 計算 | 完全微調 |

**關鍵點**：Loss 永遠都是 `MSE(預測動作, 真實動作)`。差別只在於**哪些參數會根據這個 Loss 被更新**！

---

## 附錄：快速參考

### 模型選擇指南

| 場景 | 推薦模型 | 原因 |
|------|---------|------|
| 簡單操控任務 | ACT | 模型小、訓練快 |
| 需要語言指令 | SmolVLA | 原生支援語言 |
| 追求最高性能 | PI0 | 模型最大、表達能力最強 |
| 資源有限 | SmolVLA | 平衡效能和資源 |

### 常用配置調整

```python
# 記憶體優化
SmolVLAConfig(
    attention_mode="cross_attn",
    self_attn_every_n_layers=4,
    use_cache=True,
)

# 高精度
SmolVLAConfig(
    attention_mode="self_attn",
    self_attn_every_n_layers=1,
)

# PI0 類似 SmolVLA 模式
PI0Config(
    freeze_vision_encoder=True,
    train_expert_only=True,
)
```

---

*本文件由對 LeRobot 專案的深入探索整理而成。*
