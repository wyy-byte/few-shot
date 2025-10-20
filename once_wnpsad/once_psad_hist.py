#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSAD 单模板推理（仅 hist 记忆库）
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

# ---------- 参数 ----------
WEIGHTS      = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/psad_infer/best_seg.pth"
NUM_CLASSES  = 7
RESIZE       = 512
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ONCE_DIR     = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/once_wnpsad"
PHOTO_DIR    = os.path.join(ONCE_DIR, "photo")
PICTURE_DIR  = os.path.join(ONCE_DIR, "picture")

SEG_MAP      = os.path.join(PHOTO_DIR, "segmentation_map")
SEG_ABN_MAP  = os.path.join(PHOTO_DIR, "segmentation_abnormalmap")
HIST_NOR     = os.path.join(PHOTO_DIR, "normal_histogram")
HIST_ABN     = os.path.join(PHOTO_DIR, "abnormal_histogram")
SEG_ANOM     = os.path.join(PHOTO_DIR, "segmented_map")
HEAT_DIR     = os.path.join(PHOTO_DIR, "heatmap")

for d in [SEG_MAP, SEG_ABN_MAP, HIST_NOR, HIST_ABN, SEG_ANOM, HEAT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- 1. 加载分割模型 ----------
model = PSADSegNet(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE)["model"])
model.eval()
tv_resize = T.Resize((RESIZE, RESIZE), interpolation=Image.BILINEAR)

# ---------- 2. 单张正常模板 ----------
NORMAL_TEMPLATE = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/normal_breakfastbox/usual.png"

def build_hist_memory():
    """只用单张正常图当模板，不再跑训练集"""
    mem_file = os.path.join(PHOTO_DIR, "once_hist_memory.npy")
    if os.path.exists(mem_file):
        return np.load(mem_file)

    print("【模板】用单张正常图生成 hist 模板 ...")
    img = Image.open(NORMAL_TEMPLATE).convert("RGB")
    img_t = T.Compose([tv_resize, T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        coord = get_coord_map(1, RESIZE, RESIZE, DEVICE)
        logits = model(img_t, coord)
        prob = F.softmax(logits, dim=1)
        hist = prob.mean(dim=(2, 3)).squeeze(0).cpu().numpy()  # (7,)

    mem = hist.reshape(1, -1)  # (1, 7)
    np.save(mem_file, mem)
    return mem

HIST_MEMORY = build_hist_memory()  # (1, 7)

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

    # ① 保存分割图
    seg_dir = SEG_ABN_MAP
    colored = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]], dtype=np.uint8)[pred]
    Image.fromarray(colored).save(os.path.join(seg_dir, f"{base}_color.png"))
    jet = cv2.applyColorMap(np.clip(pred * 42, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(os.path.join(seg_dir, f"{base}.png"))

    hist = prob.mean(dim=(2, 3)).squeeze(0).cpu().numpy()  # (7,)
    hist_dir = HIST_ABN
    np.save(os.path.join(hist_dir, f"{base}.npy"), hist)
    import matplotlib.pyplot as plt
    plt.bar(range(7), hist, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink'])
    plt.title(f"{tag} class histogram")
    plt.xlabel("class"); plt.ylabel("proportion")
    plt.savefig(os.path.join(hist_dir, f"{base}.png"), dpi=150)
    plt.close()

    # ③ 异常分数（单模板距离）
    dists = np.linalg.norm(HIST_MEMORY - hist, axis=1)
    score = dists.min()
    np.save(os.path.join(hist_dir, f"{base}_score.npy"), score)
    print(f"[{tag}] 异常分数 = {score:.4f}")

    # ④ 像素级距离图（单模板）
    # 为每个类别计算与正常模板的距离
    class_scores = np.linalg.norm(HIST_MEMORY - hist, axis=1)  # (1,)
    # 对于每个像素，根据其预测类别获取对应的距离
    pixel_scores = np.zeros_like(pred, dtype=np.float32)
    
    # 计算每个像素与其预测类别的距离
    for i in range(NUM_CLASSES):
        mask = (pred == i)
        if np.any(mask):
            # 使用整个直方图与当前类别的距离作为该类别的异常分数
            pixel_scores[mask] = np.linalg.norm(HIST_MEMORY[0, i] - hist[i])

    pixel_scores = (pixel_scores - pixel_scores.min()) / (pixel_scores.max() - pixel_scores.min() + 1e-8)

    # 阈值：单模板 95% 分位
    thresh = np.percentile(pixel_scores, 95)
    mask = (pixel_scores > thresh).astype(np.uint8) * 255
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
    print("=== PSAD 单模板推理（仅 hist 记忆库）===")
    print(f"正常模板：{NORMAL_TEMPLATE}")

    scores = []
    for tag, path in [("logical", "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/logical_anomaly_breakfastbox/logical.png"),
                      ("structural", "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/structural_anomaly_breakfastbox/structural.png")]:
        score = infer_one(path, tag)
        scores.append(score)

    avg_score = np.mean(scores)
    print(f"【汇总】两张异常图平均异常分数 = {avg_score:.4f}")
    print("✅ 全部结果已保存至", PHOTO_DIR)

if __name__ == "__main__":
    main()