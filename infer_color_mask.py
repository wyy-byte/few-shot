#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单张→批量 彩色分割掩码生成（永不覆盖已有文件）
"""
import os, sys, pathlib, torch, numpy as np
from PIL import Image
import torchvision.transforms as T

# 把当前目录加进 PYTHONPATH 后再导入
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_psad_sup_only import PSADBackbone, RePRIPixelClassifier

# ---------- 配置 ----------
WEIGHTS_PATH = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/best_finetune.pth"
IMG_DIR      = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/psad_data/Segmentation_map/juice_bottle/test/good/003.png"
SAVE_DIR     = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/yalian"
NUM_CLASSES  = 5

RESIZE       = 512
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 模型 ----------
backbone = PSADBackbone().to(device)
head     = RePRIPixelClassifier(1792, NUM_CLASSES).to(device)
ckpt     = torch.load(WEIGHTS_PATH, map_location=device)
backbone.load_state_dict(ckpt["backbone"])
head.load_state_dict(ckpt["head"])
backbone.eval(); head.eval()

# ---------- 预处理 ----------
transform = T.Compose([
    T.Resize((RESIZE, RESIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- 调色板 ----------
# 5 类高对比调色板（RGB）
palette = [0,0,0,        # 0 背景 黑
           255,0,0,      # 1 红
           0,255,0,      # 4 绿
           0,0,255,      # 5 蓝
           255,255,0] + [0]*((256-5)*3)   # 6 黄
palette += [0] * (256 * 3 - len(palette))
# ---------- 防覆盖工具 ----------
def unique_name(stem, suffix="_color.png"):
    """给定 stem，返回 SAVE_DIR 下不重复的文件路径"""
    base = os.path.join(SAVE_DIR, stem + suffix)
    if not os.path.exists(base):
        return base
    counter = 1
    while True:
        path = os.path.join(SAVE_DIR, f"{stem}_{counter}{suffix}")
        if not os.path.exists(path):
            return path
        counter += 1

# ---------- 推理 ----------
@torch.no_grad()
def infer_one(img_path):
    
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    pred = head(backbone(tensor)).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    print("真实类别索引：", np.unique(pred))
    color_mask = Image.fromarray(pred, mode='P')
    color_mask.putpalette(palette)
    color_mask = color_mask.convert("RGB")

    save_path = unique_name(pathlib.Path(img_path).stem)  # 永不覆盖
    color_mask.save(save_path)
    print("saved →", save_path)

# ---------- 单张 or 批量 ----------
if os.path.isfile(IMG_DIR):
    infer_one(IMG_DIR)
else:
    for f in pathlib.Path(IMG_DIR).glob("*.png"):
        infer_one(str(f))