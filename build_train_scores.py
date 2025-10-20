# 三库训练集原始分数向量就是「自适应缩放」的「尺子」——用它们的最大值把三张测试图的原始距离拉齐到同一把 0-1 刻度，
# 再相加，避免某一库量级大就垄断最终异常分。
# 训练集向量 = 自适应缩放的「分母」
# 没有它，PSAD 就无法「自己定义正常范围」，也就谈不上自适应。
"""
一次性生成 PSAD 三库「训练集原始分数向量」
仅跑 good 目录，输出三个 .npy
"""
import os, glob, numpy as np, torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

# ========== 复用你已有脚本的核心函数 ==========
from train_psad_seg import PSADSegNet, get_coord_map, PSADSegDataset, collate
from psad_hist_inference import infer_one as hist_infer  # 返回 hist_score
from psad_comp_inference import infer_one as comp_infer  # 返回 comp_score
from patchcore_inference import load_patchcore_scores, get_anomaly_score  # 返回 patch_score

# ---------- 参数 ----------
GOOD_DIR   = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/psad_data/MVTec_LOCO_AD_512size/orig_512/breakfast_box/train/good"
RESIZE     = 512
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NPY_OUT    = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/train_scores"  # 输出目录
os.makedirs(NPY_OUT, exist_ok=True)

# ---------- 1. 加载分割模型（复用你权重） ----------
WEIGHTS = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/psad_infer/best_seg.pth"
NUM_CLASSES = 7
model = PSADSegNet(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE)["model"])
model.eval()

# ---------- 2. 加载 PatchCore 预计算分数 ----------
patchcore_dict = load_patchcore_scores()  # 来自你 ADscore.txt

# ---------- 3. 遍历 good 目录 ----------
good_list = sorted(glob.glob(os.path.join(GOOD_DIR, "*.png")))
hist_scores, comp_scores, patch_scores = [], [], []

for img_path in tqdm(good_list, desc="Extract train scores"):
    # 为了复用你已有函数，我们直接调 infer_one（只取返回的 score）
    tag = "normal"  # 仅标记，不影响计算
    hist_s  = hist_infer(img_path, tag)   # 返回 hist 距离
    comp_s  = comp_infer(img_path, tag)   # 返回 comp 距离
    # PatchCore：用文件名（无后缀）当 key 取分数
    name    = os.path.splitext(os.path.basename(img_path))[0]
    patch_s = get_anomaly_score(patchcore_dict, "normal", name)  # 返回 patch 距离
    if patch_s is None:  # 缺 key 就用中位数兜底
        patch_s = np.median(list(patchcore_dict.values()))
    hist_scores.append(hist_s)
    comp_scores.append(comp_s)
    patch_scores.append(patch_s)

# ---------- 4. 保存三个向量 ----------
np.save(os.path.join(NPY_OUT, "train_hist_score.npy"), np.array(hist_scores))
np.save(os.path.join(NPY_OUT, "train_comp_score.npy"), np.array(comp_scores))
np.save(os.path.join(NPY_OUT, "train_patch_score.npy"), np.array(patch_scores))

print("✅ 三库训练集原始分数向量已生成：")
print(f"  hist: {len(hist_scores)} 张, 保存至 {NPY_OUT}/train_hist_score.npy")
print(f"  comp: {len(comp_scores)} 张, 保存至 {NPY_OUT}/train_comp_score.npy")
print(f"  patch: {len(patch_scores)} 张, 保存至 {NPY_OUT}/train_patch_score.npy")
