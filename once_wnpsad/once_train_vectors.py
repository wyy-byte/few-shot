#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSAD 单模板生成三库「原始分数向量」
输入：单张正常图
输出：hist / comp / patch 三库原始分数向量（只用单张模板）
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
NPY_OUT      = os.path.join(ONCE_DIR, "train_scores")
os.makedirs(NPY_OUT, exist_ok=True)

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
    mem_file = os.path.join(NPY_OUT, "once_memory.npy")
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
    np.save(os.path.join(NPY_OUT, "once_memory.npy"), mem)
    return mem

COMP_MEMORY = build_once_memory()   # (1, NUM_CLASSES*feat_dim)

# ---------- 3. 生成三库向量 ----------
def generate_vectors():
    print("【推理】生成三库向量 ...")
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

    # hist 向量：类别分布 (NUM_CLASSES,)
    hist_vec = np.bincount(pred.flatten(), minlength=NUM_CLASSES) / pred.size
    hist_vec = hist_vec.reshape(1, -1)  # (1, NUM_CLASSES)

    # comp 向量：类别平均特征 (NUM_CLASSES, feat_dim) -> (1, NUM_CLASSES*feat_dim)
    comp_vec_flat = comp_vec.reshape(1, -1)

    # patch 向量：与模板比较得到的差异 (NUM_CLASSES,)
    train_vecs = COMP_MEMORY[0].reshape(NUM_CLASSES, -1)  # (NUM_CLASSES, feat_dim)
    distances = np.linalg.norm(train_vecs - comp_vec, axis=1)  # (NUM_CLASSES,)
    patch_vec = distances.reshape(1, -1)  # (1, NUM_CLASSES)

    # 保存三库向量
    np.save(os.path.join(NPY_OUT, "hist_vec.npy"), hist_vec)
    np.save(os.path.join(NPY_OUT, "comp_vec.npy"), comp_vec_flat)
    np.save(os.path.join(NPY_OUT, "patch_vec.npy"), patch_vec)

    print(f"hist_vec:  {hist_vec.shape} = {hist_vec}")
    print(f"comp_vec:  {comp_vec_flat.shape}")
    print(f"patch_vec: {patch_vec.shape} = {patch_vec}")

if __name__ == "__main__":
    generate_vectors()
    print("✅ 三库向量已保存至", NPY_OUT)