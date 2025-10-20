#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSAD 单模板推理（ℳ_comp 记忆库）
输入：单张正常图 + 逻辑/结构异常图
输出：只跑异常图，全部生成到 once_wnpsad/
"""
import os, numpy as np
from PIL import Image
import torch, torch.nn.functional as F
import cv2
from torchvision import transforms as T
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_psad_seg import PSADSegNet, get_coord_map
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt

# ---------- 参数 ----------
WEIGHTS      = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/psad_infer/best_seg.pth"
NUM_CLASSES  = 7
RESIZE       = 512
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ONCE_DIR     = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/once_wnpsad"
PHOTO_DIR    = os.path.join(ONCE_DIR, "photo")
PICTURE_DIR  = os.path.join(ONCE_DIR, "picture")

SEG_MAP      = os.path.join(PICTURE_DIR, "segmentation_map")
SEG_ABN_MAP  = os.path.join(PICTURE_DIR, "segmentation_abnormalmap")
CLOUD_ABN    = os.path.join(PICTURE_DIR, "abnormal_cloud")
SEG_ANOM     = os.path.join(PICTURE_DIR, "segmented_map")
HEAT_DIR     = os.path.join(PICTURE_DIR, "heatmap")
NPY_DIR      = os.path.join(PICTURE_DIR, "npy")

for d in [SEG_MAP, SEG_ABN_MAP, CLOUD_ABN, SEG_ANOM, HEAT_DIR, NPY_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- 1. 加载分割模型 ----------
model = PSADSegNet(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE)["model"])
model.eval()
tv_resize = T.Resize((RESIZE, RESIZE), interpolation=Image.BILINEAR)

# ---------- 2. 单张正常模板 ----------
NORMAL_TEMPLATE = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/normal_breakfastbox/usual.png"

def build_comp_memory():
    """只用单张正常图当模板，不再跑训练集"""
    mem_file = os.path.join(PHOTO_DIR, "once_comp_memory.npy")
    if os.path.exists(mem_file):
        return np.load(mem_file)

    print("【模板】用单张正常图生成 comp 模板 ...")
    img = Image.open(NORMAL_TEMPLATE).convert("RGB")
    img_t = T.Compose([tv_resize, T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        coord = get_coord_map(1, RESIZE, RESIZE, DEVICE)
        logits = model(img_t, coord)
        prob = F.softmax(logits, dim=1)
        pred = prob.argmax(1).squeeze(0).cpu().numpy()

    # 高维特征
    img_3c = img_t.squeeze(0).cpu()
    img_pil = T.ToPILImage()(img_3c)
    img_tensor = T.Compose([T.Resize((RESIZE, RESIZE)), T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])(img_pil).unsqueeze(0).to(DEVICE)

    feat_ext = PSADSegNet(NUM_CLASSES).to(DEVICE)
    feat_ext.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE)["model"])
    feat_ext.eval()
    with torch.no_grad():
        coord = get_coord_map(1, RESIZE, RESIZE, DEVICE)
        ft_map = feat_ext(img_tensor, coord)                      # (1, C, H, W)

    # 获取特征维度
    feat_dim = ft_map.shape[1]
    print(f"特征维度: {feat_dim}")

    comp_vec = []
    for c in range(NUM_CLASSES):
        mask = torch.from_numpy((pred == c).astype(np.float32))
        if mask.sum() == 0:
            vec = np.zeros(ft_map.shape[1])
        else:
            mask = mask.to(DEVICE)
            vec = (ft_map * mask.unsqueeze(0)).sum(dim=(2, 3)) / mask.sum()
            vec = vec.cpu().numpy()
        comp_vec.append(vec)
    comp_vec = np.stack(comp_vec, axis=0)   # (C, feat_dim)

    mem = comp_vec.reshape(1, -1)           # (1, C*feat_dim)
    np.save(mem_file, mem)
    return mem

COMP_MEMORY = build_comp_memory()   # (1, C*feat_dim)

# ---------- 3. 单图推理（只跑异常图） ----------
def infer_one(image_path, tag):
    base = os.path.splitext(os.path.basename(image_path))[0]

    # ① 读图 & 分割
    img = Image.open(image_path).convert("RGB")
    img_t = T.Compose([tv_resize, T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        coord = get_coord_map(1, RESIZE, RESIZE, DEVICE)
        logits = model(img_t, coord)
        prob = F.softmax(logits, dim=1)
        pred = prob.argmax(1).squeeze(0).cpu().numpy()

    # ① 保存分割图（异常图目录）
    seg_dir = SEG_ABN_MAP
    colored = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]], dtype=np.uint8)[pred]
    Image.fromarray(colored).save(os.path.join(seg_dir, f"{base}_color.png"))
    jet = cv2.applyColorMap(np.clip(pred * 36, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(os.path.join(seg_dir, f"{base}.png"))

    # 高维特征
    img_3c = img_t.squeeze(0).cpu()
    img_pil = T.ToPILImage()(img_3c)
    img_tensor = T.Compose([T.Resize((RESIZE, RESIZE)), T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])(img_pil).unsqueeze(0).to(DEVICE)

    feat_ext = PSADSegNet(NUM_CLASSES).to(DEVICE)
    feat_ext.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE)["model"])
    feat_ext.eval()
    with torch.no_grad():
        coord = get_coord_map(1, RESIZE, RESIZE, DEVICE)
        ft_map = feat_ext(img_tensor, coord)                      # (1, C, H, W)
    
    # 获取特征维度
    feat_dim = ft_map.shape[1]

    comp_vec = []
    for c in range(NUM_CLASSES):
        mask = torch.from_numpy((pred == c).astype(np.float32))
        if mask.sum() == 0:
            vec = np.zeros(ft_map.shape[1])
        else:
            mask = mask.to(DEVICE)
            vec = (ft_map * mask.unsqueeze(0)).sum(dim=(2, 3)) / mask.sum()
            vec = vec.cpu().numpy()
        comp_vec.append(vec)
    comp_vec = np.stack(comp_vec, axis=0)   # (C, feat_dim)

    # 像素距离（单模板）
    train_vecs = COMP_MEMORY.reshape(-1, feat_dim)  # (C, feat_dim)
    distances = np.linalg.norm(train_vecs - comp_vec, axis=1)  # (C,)
    comp_scores = distances  # (C,)
    score = comp_scores.mean()
    np.save(os.path.join(NPY_DIR, f"{base}_score.npy"), score)
    print(f"[{tag}] 异常分数 = {score:.4f}")

    # 像素级距离图（单模板）
    # 修复索引越界问题
    pixel_scores = np.zeros_like(pred, dtype=np.float32)
    for i in range(NUM_CLASSES):
        mask = (pred == i)
        if np.any(mask):
            # 为每个像素分配其对应类别的距离值
            distance_value = distances[i]
            if hasattr(distance_value, '__len__') and len(distance_value) > 0:
                pixel_scores[mask] = distance_value[0] if hasattr(distance_value, '__getitem__') else distance_value.item()
            else:
                pixel_scores[mask] = distance_value

    pixel_scores = (pixel_scores - pixel_scores.min()) / (pixel_scores.max() - pixel_scores.min() + 1e-8)

    # 阈值：单模板 95% 分位
    thresh = np.percentile(pixel_scores, 95)
    mask = (pixel_scores > thresh).astype(np.uint8)  # 修正数据类型
    Image.fromarray(mask).save(os.path.join(SEG_ANOM, f"{base}_mask.png"))

    # 热力图
    pixel_scores_norm = (pixel_scores - pixel_scores.min()) / (pixel_scores.max() - pixel_scores.min() + 1e-8)
    jet = cv2.applyColorMap(np.uint8(pixel_scores_norm * 255), cv2.COLORMAP_JET)
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (RESIZE, RESIZE))
    heatmap_on_original = cv2.addWeighted(original_img, 0.5, jet, 0.5, 0)
    cv2.imwrite(os.path.join(HEAT_DIR, f"{base}_heatmap_on_original.png"), heatmap_on_original)

    return score

# ---------- 4. 主入口（只跑异常图） ----------
def main():
    print("=== PSAD 单模板推理（ℳ_comp 记忆库）===")
    print(f"正常模板：{NORMAL_TEMPLATE}")

    scores = []
    for tag, path in [("logical", "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/logical_anomaly_breakfastbox/logical.png"),
                      ("structural", "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/structural_anomaly_breakfastbox/structural.png")]:
        score = infer_one(path, tag)
        scores.append(score)

    avg_score = np.mean(scores)
    print(f"【汇总】两张异常图平均异常分数 = {avg_score:.4f}")
    print("✅ 全部结果已保存至", PICTURE_DIR)

if __name__ == "__main__":
    main()