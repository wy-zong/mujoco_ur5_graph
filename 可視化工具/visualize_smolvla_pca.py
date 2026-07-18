import torch
import numpy as np
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoImageProcessor, AutoModelForImageTextToText, AutoProcessor
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from sklearn.decomposition import PCA
import cv2
from PIL import Image
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize SmolVLA or Base SmolVLM Vision PCA")
    parser.add_argument(
        "--dataset_id", 
        type=str, 
        default="wuc1/bi_so101_flatten-and-fold-the-rag-then-place-0416-0417-merge",
        help="Dataset ID"
    )
    parser.add_argument(
        "--camera_key", 
        type=str, 
        default="observation.images.left_camera1",
        help="Camera key in the dataset"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="wuc1/bi_so101_flatten-and-fold-the-rag-then-place-0416-0417-merge0420-model",
        help="HuggingFace repo ID or local path for the trained SmolVLA model (or base VLM)."
    )
    parser.add_argument(
        "--use_base_vlm", 
        action="store_true",
        help="If set, load the base SmolVLM directly from AutoModelForImageTextToText instead of a trained SmolVLA policy."
    )
    args = parser.parse_args()

    # 1. 設定參數
    dataset_id = args.dataset_id
    camera_key = args.camera_key
    model_id = args.model_id
    
    # 抽取特定步數的影像，這裡縮小區間以便觀察動作細節
    frame_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400]

    print(f"正在載入資料集: {dataset_id}...")
    # 使用 LeRobotDataset 自動處理影片格式解碼
    dataset = LeRobotDataset(dataset_id, video_backend="pyav")
    
    # 收集影像
    images = []
    for idx in frame_indices:
        try:
            item = dataset[idx]
        except IndexError:
            print(f"警告：資料集總長度小於指定的步數 {idx}，將提早結束擷取。")
            break
        img_tensor = item[camera_key]
        img_np = (img_tensor.numpy() * 255).astype(np.uint8)
        img_np = np.transpose(img_np, (1, 2, 0)) # (C, H, W) -> (H, W, C)
        images.append(Image.fromarray(img_np))
            
    if not images:
        print("找不到影像，請確認 camera_key 是否正確。")
        return

    print(f"正在載入模型: {model_id}...")
    
    # 載入 Processor
    # 對於 SmolVLA 或 SmolVLM，我們都可以用基礎模型的 Processor (因為影像處理邏輯相同)
    base_vlm_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    try:
        processor = AutoImageProcessor.from_pretrained(base_vlm_id)
    except Exception:
        processor = AutoProcessor.from_pretrained(base_vlm_id).image_processor

    # 載入模型權重
    if args.use_base_vlm:
        print("載入基礎 SmolVLM 模型...")
        model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float32)
        vision_model = model.model.vision_model
        connector = model.model.connector
        config_source = model.config
    else:
        print("載入已訓練的 SmolVLA 策略模型...")
        # 由於 LeRobot 架構，我們透過 SmolVLAPolicy 載入
        policy = SmolVLAPolicy.from_pretrained(model_id)
        # 提取內部的 vision_model 與 connector
        vlm_model = policy.model.vlm_with_expert.get_vlm_model()
        vision_model = vlm_model.vision_model
        connector = vlm_model.connector
        config_source = vlm_model.config
        
        # 將權重轉為 float32 以便後續推論
        vision_model.to(torch.float32)
        connector.to(torch.float32)

    vision_model.eval()
    connector.eval()

    all_connector_tokens = []
    all_vision_tokens = []
    processed_images_info = []

    print("正在提取並處理視覺特徵 (Vision Encoder -> Connector)...")
    with torch.no_grad():
        for img in images:
            # 處理影像格式以符合 SmolVLM 的輸入要求
            inputs = processor(images=img, return_tensors="pt")
            
            # SmolVLM / Idefics3 的 pixel_values 形狀為 (batch_size, num_crops, channels, height, width)
            # vision_model 期望接收 4D 張量 (batch_size * num_crops, channels, height, width)
            pixel_values_5d = inputs.pixel_values
            
            # 為了方便可視化空間對應，我們只取「全局縮放圖」(Global crop，通常是最後一個)
            global_pixel_values = pixel_values_5d[:, -1, :, :, :] # 形狀: (1, 3, H, W)
            
            # 將輸入放到與模型相同的設備上
            device = vision_model.device
            global_pixel_values = global_pixel_values.to(device)
            
            # 1. 透過 Vision Model 提取特徵
            outputs = vision_model(pixel_values=global_pixel_values)
            last_hidden_states = outputs.last_hidden_state # (1, num_patches, hidden_size)
            
            # 2. 透過 Connector 進行模態轉換與空間降採樣
            # 這是真正送入語言模型與 Expert 模組的視覺特徵
            projected_states = connector(last_hidden_states)
            
            # 取第一筆，並轉回 CPU 以便進行 PCA 與 numpy 操作
            patch_tokens = projected_states[0].cpu() # (num_projected_patches, connector_hidden_size)
            vision_tokens = last_hidden_states[0].cpu() # (num_patches, vision_hidden_size)
            
            # 推算降採樣後的空間維度
            if hasattr(vision_model.config, 'patch_size'):
                patch_size = vision_model.config.patch_size
                _, _, h, w = global_pixel_values.shape
                
                # 原始 Vision Encoder 的 Patch 網格
                raw_grid_h = h // patch_size
                raw_grid_w = w // patch_size
                
                # Connector 通常會再做空間 Pooling (例如 Idefics3/SmolVLM 會使用 scale_factor)
                scale_factor = getattr(config_source, 'scale_factor', 2) # 通常是 2 或 4
                
                grid_h = raw_grid_h // scale_factor
                grid_w = raw_grid_w // scale_factor
            else:
                # 備案：假設是方陣
                grid_size = int(np.sqrt(patch_tokens.shape[0]))
                grid_h = grid_size
                grid_w = grid_size
                
                raw_grid_size = int(np.sqrt(vision_tokens.shape[0]))
                raw_grid_h = raw_grid_size
                raw_grid_w = raw_grid_size
            
            # 若計算與實際大小有出入，強制退回開根號推算
            if grid_h * grid_w != patch_tokens.shape[0]:
                grid_size = int(np.sqrt(patch_tokens.shape[0]))
                grid_h = grid_size
                grid_w = grid_size
                
            if raw_grid_h * raw_grid_w != vision_tokens.shape[0]:
                raw_grid_size = int(np.sqrt(vision_tokens.shape[0]))
                raw_grid_h = raw_grid_size
                raw_grid_w = raw_grid_size

            all_connector_tokens.append(patch_tokens.numpy())
            all_vision_tokens.append(vision_tokens.numpy())
            processed_images_info.append({
                "original": img,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "raw_grid_h": raw_grid_h,
                "raw_grid_w": raw_grid_w
            })

    def perform_pca_and_plot(tokens_list, info_list, title, output_filename, is_vision_encoder=False):
        print(f"正在進行 PCA 降維計算 ({title})...")
        all_tokens_concat = np.concatenate(tokens_list, axis=0)
        
        # 將高維度特徵降至 3 維 (對應 R, G, B)，捕捉最重要的空間與語意特徵變異
        pca = PCA(n_components=3)
        pca.fit(all_tokens_concat)

        # 建立畫布準備繪圖
        fig, axes = plt.subplots(len(images), 2, figsize=(10, 4 * len(images)))
        fig.suptitle(f"{title}\nModel: {model_id}", fontsize=14)

        for idx, (tokens, info) in enumerate(zip(tokens_list, info_list)):
            # 轉換該張圖片的特徵
            pca_features = pca.transform(tokens)
            
            # 最小最大正規化 (Min-Max Scaling) 到 [0, 1] 之間，才能作為 RGB 顏色顯示
            pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0))
            
            # 重塑回 2D 網格狀，重現影像的空間結構
            grid_h_key = "raw_grid_h" if is_vision_encoder else "grid_h"
            grid_w_key = "raw_grid_w" if is_vision_encoder else "grid_w"
            pca_image = pca_features.reshape(info[grid_h_key], info[grid_w_key], 3)
            
            # 為了視覺效果，將小尺寸的 PCA 圖放大回原始影像大小
            orig_w, orig_h = info["original"].size
            pca_image_resized = cv2.resize(pca_image, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # 繪圖設定
            ax_orig = axes[idx, 0] if len(images) > 1 else axes[0]
            ax_pca = axes[idx, 1] if len(images) > 1 else axes[1]

            ax_orig.imshow(info["original"])
            ax_orig.set_title(f"Original (Frame {frame_indices[idx]})")
            ax_orig.axis("off")

            ax_pca.imshow(pca_image_resized)
            ax_pca.set_title(f"PCA Features")
            ax_pca.axis("off")

        plt.tight_layout()
        plt.savefig(output_filename, dpi=150)
        print(f"完成！已產生 {output_filename}")
        
    title_prefix = "Base SmolVLM" if args.use_base_vlm else "Trained SmolVLA"

    # 1. 繪製 Connector 輸出 (原本的功能)
    conn_title = f"{title_prefix} (Connector Output)"
    conn_output = f"smolvla_pca_visualization_connector_{'base' if args.use_base_vlm else 'trained'}.png"
    perform_pca_and_plot(all_connector_tokens, processed_images_info, conn_title, conn_output, is_vision_encoder=False)
    
    # 2. 繪製 Vision Encoder 輸出 (新增的功能)
    vis_title = f"{title_prefix} (Vision Encoder Output)"
    vis_output = f"smolvla_pca_visualization_vision_encoder_{'base' if args.use_base_vlm else 'trained'}.png"
    perform_pca_and_plot(all_vision_tokens, processed_images_info, vis_title, vis_output, is_vision_encoder=True)
    
    print("這兩張圖分別展示了純視覺編碼器的特徵，以及經過 Connector 融合、降採樣後送入 LLM 的最終空間特徵。")

if __name__ == "__main__":
    main()

