#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSAD 单模板自适应最终异常分
输入：单张正常图 + 逻辑/结构异常图
输出：只跑异常图，分母 = 单模板自身，0-3 区间
"""
import os, sys, numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T

# 添加包含 train_psad_seg.py 的目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_psad_seg import PSADSegNet, get_coord_map

# ---------- 参数 ----------
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESIZE       = 512
NUM_CLASSES  = 7
WEIGHTS      = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/psad_infer/best_seg.pth"

ONCE_DIR     = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/once_wnpsad"
NPY_DIR      = os.path.join(ONCE_DIR, "npy")
os.makedirs(NPY_DIR, exist_ok=True)

# ---------- 1. 加载分割模型 ----------
model = PSADSegNet(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE)["model"])
model.eval()
tv_resize = T.Resize((RESIZE, RESIZE), interpolation=Image.BILINEAR)

# 创建特征提取模型
feat_model = PSADSegNet(NUM_CLASSES).to(DEVICE)
feat_model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE)["model"])
feat_model.eval()

# ---------- 2. 单张正常模板 ----------
NORMAL_TEMPLATE = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/normal_breakfastbox/usual.png"

def build_once_memory():
    """只用单张正常图当模板，不再跑训练集"""
    mem_file = os.path.join(NPY_DIR, "once_memory.npy")
    if os.path.exists(mem_file):
        return np.load(mem_file)

    print("【模板】用单张正常图生成模板 ...")
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

    # 提取特征
    with torch.no_grad():
        coord = get_coord_map(1, RESIZE, RESIZE, DEVICE)
        ft_map = feat_model(img_tensor, coord)                      # (1, C, H, W)

    feat_dim = ft_map.shape[1]
    comp_vec = []
    for c in range(NUM_CLASSES):
        mask = torch.from_numpy((pred == c).astype(np.float32))
        if mask.sum() == 0:
            vec = np.zeros(feat_dim)
        else:
            mask = mask.to(DEVICE)
            vec = (ft_map * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(0, 2, 3)) / mask.sum()
            vec = vec.cpu().numpy()
        comp_vec.append(vec)
    comp_vec = np.stack(comp_vec, axis=0)   # (NUM_CLASSES, feat_dim)

    mem = comp_vec.reshape(1, -1)           # (1, NUM_CLASSES*feat_dim)
    np.save(os.path.join(NPY_DIR, "once_memory.npy"), mem)
    return mem

COMP_MEMORY = build_once_memory()   # (1, NUM_CLASSES*feat_dim)

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

    # 高维特征
    img_3c = img_t.squeeze(0).cpu()
    img_pil = T.ToPILImage()(img_3c)
    img_tensor = T.Compose([T.Resize((RESIZE, RESIZE)), T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])(img_pil).unsqueeze(0).to(DEVICE)

    # 提取特征
    with torch.no_grad():
        coord = get_coord_map(1, RESIZE, RESIZE, DEVICE)
        ft_map = feat_model(img_tensor, coord)                      # (1, C, H, W)

    feat_dim = ft_map.shape[1]
    comp_vec = []
    for c in range(NUM_CLASSES):
        mask = torch.from_numpy((pred == c).astype(np.float32))
        if mask.sum() == 0:
            vec = np.zeros(feat_dim)
        else:
            mask = mask.to(DEVICE)
            vec = (ft_map * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(0, 2, 3)) / mask.sum()
            vec = vec.cpu().numpy()
        comp_vec.append(vec)
    comp_vec = np.stack(comp_vec, axis=0)   # (NUM_CLASSES, feat_dim)

    # 单模板记忆库：一条 7×feat_dim 向量
    train_vecs = COMP_MEMORY[0].reshape(NUM_CLASSES, -1)  # (NUM_CLASSES, feat_dim)
    distances = np.linalg.norm(train_vecs - comp_vec, axis=1)  # (NUM_CLASSES,)
    comp_scores = distances
    score = comp_scores.mean()
    np.save(os.path.join(NPY_DIR, f"{base}_score.npy"), score)
    print(f"[{tag}] 异常分数 = {score:.4f}")

    # 像素级距离图（单模板）
    pixel_scores = distances[pred]          # (H, W)
    pixel_scores = (pixel_scores - pixel_scores.min()) / (pixel_scores.max() - pixel_scores.min() + 1e-8)

    # 阈值：单模板 95% 分位
    thresh = np.percentile(pixel_scores, 95)
    # 不生成图，只保存分数
    np.save(os.path.join(NPY_DIR, f"{base}_pixel_scores.npy"), pixel_scores)

    return score

# ---------- 4. 主入口（只跑异常图） ----------
def main():
    print("=== PSAD 单模板自适应最终异常分 ===")
    print(f"正常模板：{NORMAL_TEMPLATE}")

    scores = []
    for tag, path in [("logical", "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/logical_anomaly_breakfastbox/logical.png"),
                      ("structural", "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/structural_anomaly_breakfastbox/structural.png")]:
        score = infer_one(path, tag)
        scores.append(score)

    avg_score = np.mean(scores)
    print(f"【汇总】两张异常图平均异常分数 = {avg_score:.4f}")
    print("✅ 全部结果已保存至", NPY_DIR)

if __name__ == "__main__":
    main()