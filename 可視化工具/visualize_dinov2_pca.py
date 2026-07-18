import torch
import numpy as np
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
import cv2
from PIL import Image

def main():
    # 1. 設定參數
    dataset_id = "wuc1/bi_so101_flatten-and-fold-the-rag-then-place-0416-0417-merge"
    camera_key = "observation.images.left_camera3" 
    model_id = "facebook/dinov2-base" # 常用的 DINOv2 基礎模型 (Patch size = 14)
    
    # 抽取第 0, 50, 100 步的影像 (可根據您的資料集長度調整，以觀看不同動作階段)
    frame_indices = [0, 50, 100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000] 

    print(f"正在載入資料集: {dataset_id}...")
    # 使用 LeRobotDataset 自動處理影片格式解碼
    dataset = LeRobotDataset(dataset_id, video_backend="pyav")
    
    # 收集影像
    images = []
    for idx in frame_indices:
        # dataset[idx] 會返回一個包含所有感測器資料的字典
        item = dataset[idx]
        
        # 取得特定相機的張量，LeRobotDataset 返回的通常是 (C, H, W) 且值在 [0.0, 1.0] 的 float32 tensor
        img_tensor = item[camera_key]
        
        # 將其轉換為 PIL Image，以符合 DINOv2 processor 的輸入格式
        img_np = (img_tensor.numpy() * 255).astype(np.uint8)
        img_np = np.transpose(img_np, (1, 2, 0)) # (C, H, W) -> (H, W, C)
        images.append(Image.fromarray(img_np))
            
    if not images:
        print("找不到影像，請確認 camera_key 是否正確。")
        return

    print(f"正在載入 DINOv2 模型: {model_id}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval() # 設定為評估模式

    all_patch_tokens = []
    processed_images_info = []

    print("正在提取特徵...")
    with torch.no_grad():
        for img in images:
            # DINOv2 預設會將影像 Resize 到 224x224 (或其他 14 的倍數)
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            
            # 獲取最後一層的隱藏狀態
            last_hidden_states = outputs.last_hidden_state
            
            # 移除第一個 CLS token，剩下的就是影像 Patch 的特徵
            patch_tokens = last_hidden_states[0, 1:, :] 
            
            # 計算特徵圖的寬高
            grid_size = int(np.sqrt(patch_tokens.shape[0])) 
            
            all_patch_tokens.append(patch_tokens.numpy())
            processed_images_info.append({
                "original": img,
                "grid_size": grid_size
            })

    print("正在進行 PCA 降維計算...")
    all_tokens_concat = np.concatenate(all_patch_tokens, axis=0)
    
    # 將高維度 (如 768 維) 降至 3 維 (對應 R, G, B)
    pca = PCA(n_components=3)
    pca.fit(all_tokens_concat)

    # 建立畫布準備繪圖
    fig, axes = plt.subplots(len(images), 2, figsize=(10, 4 * len(images)))
    fig.suptitle("DINOv2 Feature PCA Visualization", fontsize=16)

    for idx, (tokens, info) in enumerate(zip(all_patch_tokens, processed_images_info)):
        # 轉換該張圖片的特徵
        pca_features = pca.transform(tokens)
        
        # 最小最大正規化 (Min-Max Scaling) 到 [0, 1] 之間，才能作為 RGB 顏色顯示
        pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0))
        
        # 重塑回 2D 網格狀
        pca_image = pca_features.reshape(info["grid_size"], info["grid_size"], 3)
        
        # 為了視覺效果，將小尺寸的 PCA 圖放大回原始影像大小
        orig_w, orig_h = info["original"].size
        pca_image_resized = cv2.resize(pca_image, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 繪圖
        ax_orig = axes[idx, 0] if len(images) > 1 else axes[0]
        ax_pca = axes[idx, 1] if len(images) > 1 else axes[1]

        ax_orig.imshow(info["original"])
        ax_orig.set_title(f"Original Image (Frame {frame_indices[idx]})")
        ax_orig.axis("off")

        ax_pca.imshow(pca_image_resized)
        ax_pca.set_title("DINOv2 PCA Features")
        ax_pca.axis("off")

    plt.tight_layout()
    plt.savefig("dinov2_pca_visualization.png")
    print("完成！請查看產生的 dinov2_pca_visualization.png 圖片。")

if __name__ == "__main__":
    main()