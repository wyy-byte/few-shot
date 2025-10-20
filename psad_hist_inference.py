# hist 库分数 < 0.1 是常态，
# 只要 normal < logical 即可，无需担心绝对值小。
"""
PSAD 完整推理流水线（仅 hist 记忆库）
输入：normal / logical / structural 各一张
输出：①-⑥ 全部自动保存
"""
import os, glob, argparse, numpy as np
from PIL import Image
import torch, torch.nn.functional as F
import cv2
from torchvision import transforms as T
from tqdm import tqdm
from train_psad_seg import PSADSegNet, get_coord_map   # 复用训练脚本


WEIGHTS      = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/psad_infer/best_seg.pth"
MEM_BANK     = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/psad_infer/hist_memory.npy"
NUM_CLASSES  = 7
RESIZE       = 512
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PHOTO_DIR    = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/photo"
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

# ---------- 2. 构建 / 加载 hist 记忆库 ----------
def build_hist_memory():
    
    if os.path.exists(MEM_BANK):
        return np.load(MEM_BANK)
    from train_psad_seg import LABELED_NAMES, UNLABELED_NAMES, PSADSegDataset, collate
    from torch.utils.data import DataLoader
    ds = PSADSegDataset(LABELED_NAMES + UNLABELED_NAMES, with_label=False, aug=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)
    bank = []
    with torch.no_grad():
        for img, _, _ in tqdm(loader, desc="Build hist memory"):
            img = img.to(DEVICE)
            coord = get_coord_map(1, RESIZE, RESIZE, DEVICE)
            logits = model(img, coord)
            prob = F.softmax(logits, dim=1)
            hist = prob.mean(dim=(2, 3)).squeeze(0).cpu().numpy()  # (6,)
            bank.append(hist)
    bank = np.stack(bank)  # (N, 6)
    np.save(MEM_BANK, bank)
    return bank

HIST_MEMORY = build_hist_memory()  # (N, 6)

# ---------- 3. 单图推理 ----------
def infer_one(image_path, tag):
    base = os.path.splitext(os.path.basename(image_path))[0]
    is_normal = tag == "normal"

    # ① 读图 & 分割
    img = Image.open(image_path).convert("RGB")
    img_t = T.Compose([tv_resize, T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        coord = get_coord_map(1, RESIZE, RESIZE, DEVICE)
        logits = model(img_t, coord)
        prob   = F.softmax(logits, dim=1)
        pred   = prob.argmax(1).squeeze(0).cpu().numpy()  # 512×512

    # ① 保存分割图
    seg_dir = SEG_MAP if is_normal else SEG_ABN_MAP
    # 原 6 行 → 改为 8 行（背景 + 7 组件）
    colored = np.array(
        [[0, 0, 0],          # 0 背景
         [255, 0, 0],        # 1
         [0, 255, 0],        # 2
         [0, 0, 255],        # 3
         [255, 255, 0],      # 4
         [255, 0, 255],      # 5
         [0, 255, 255],      # 6  ← 新增    
        ], dtype=np.uint8
    )[pred]
    Image.fromarray(colored).save(os.path.join(seg_dir, f"{base}_color.png"))
    jet = cv2.applyColorMap(np.clip(pred * 42, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(os.path.join(seg_dir, f"{base}.png"))

    hist = prob.mean(dim=(2, 3)).squeeze(0).cpu().numpy()  # (6,)
    hist_dir = HIST_NOR if is_normal else HIST_ABN
    np.save(os.path.join(hist_dir, f"{base}.npy"), hist)
    import matplotlib.pyplot as plt
    plt.bar(range(NUM_CLASSES), hist, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'])
    plt.title(f"{tag} class histogram")
    plt.xlabel("class"); plt.ylabel("proportion")
    plt.savefig(os.path.join(hist_dir, f"{base}.png"), dpi=150)
    plt.close()

    # ③ 异常分数（仅 hist 距离）
    dists = np.linalg.norm(HIST_MEMORY - hist, axis=1)
    score = dists.min()  # 统一异常分
    np.save(os.path.join(hist_dir, f"{base}_score.npy"), score)
    print(f"[{tag}] 异常分数 = {score:.4f}")

    # 计算像素级距离图
   # 每个类别的异常分数（离训练集越远离异常）
    class_scores = np.linalg.norm(HIST_MEMORY - hist, axis=1)  # (C,)
# 将类别分数映射到每个像素
    pixel_scores = class_scores[pred]          # (H, W)
# 归一化到 0-1
    pixel_scores = (pixel_scores - pixel_scores.min()) / (pixel_scores.max() - pixel_scores.min())
  
    # ④ 异常掩膜（阈值 = 训练集 95% 分位 和 测试图像 95% 分位 的较小值）
    thresh = np.percentile(dists, 95)  # 训练集的 95% 分位数
    test_thresh = np.percentile(pixel_scores, 95)  # 测试图像的 95% 分位数  # 测试图像的 95% 分位数
    final_thresh = min(thresh, test_thresh)  # 选择较小的阈值

    # 使用像素级距离图生成异常掩膜
    mask = (pixel_scores > final_thresh).astype(np.uint8) * 255
    Image.fromarray(mask).save(os.path.join(SEG_ANOM, f"{base}_mask.png"))

    # 归一化像素级距离图
    pixel_heat_map_normalized = (pixel_scores - pixel_scores.min()) / (pixel_scores.max() - pixel_scores.min())
    pixel_scores_norm = (pixel_scores - pixel_scores.min()) / (pixel_scores.max() - pixel_scores.min())
    jet = cv2.applyColorMap(np.uint8(pixel_scores_norm * 255), cv2.COLORMAP_JET)
    mask = (pixel_scores > np.percentile(pixel_scores, 95)).astype(np.uint8) * 255

    # 将热力图叠加到原始异常图像上
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (RESIZE, RESIZE))
    heatmap_on_original = cv2.addWeighted(original_img, 0.5, jet, 0.5, 0)

    # 保存叠加后的热力图
    cv2.imwrite(os.path.join(HEAT_DIR, f"{base}_heatmap_on_original.png"), heatmap_on_original)
    return score
# ---------- 4. 主入口 ----------
def main():
    class Args:
        normal = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/normal_breakfastbox/usual.png"
        logical = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/logical_anomaly_breakfastbox/logical.png"
        structural = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/structural_anomaly_breakfastbox/structural.png"
    args = Args()

    print("=== PSAD 完整推理（仅 hist 记忆库）===")
   
    scores = []
    
    for tag, path in [("normal", args.normal), ("logical", args.logical), ("structural", args.structural)]:
        score = infer_one(path, tag)
        scores.append(score)
    avg_score = np.mean(scores[1:])  # 只平均两张异常图
    print(f"【汇总】两张异常图平均异常分数 = {avg_score:.4f}")
    print("✅ 全部结果已保存至", PHOTO_DIR)
if __name__ == "__main__":
    main()