#normal ≈ struct < logical，与论文里 逻辑异常更难、更全局 的结论一致，顺序正确即可。
"""
PSAD 组件记忆库（ℳ_comp）推理流水线
输入：normal / logical / structural 各一张
输出：①-⑥ 全部自动保存
"""
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os, glob, argparse, numpy as np
from PIL import Image
import torch, torch.nn.functional as F
import cv2
from torchvision import transforms as T
from tqdm import tqdm
from train_psad_seg import PSADSegNet, get_coord_map
# ===== 以下原本在 feature_extract.py 的全部内容 =====
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self, avgpool_size=5):
        super().__init__()
        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)
        self.model = models.resnet101(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.layer1[-1].register_forward_hook(hook_t)
        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)
        self.model.layer4[-1].register_forward_hook(hook_t)
        self.ks = avgpool_size
        self.ps = self.ks // 2

    def init_features(self):
        self.features = []

    def extract_ft(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def forward(self, x):
        features = self.extract_ft(x)
        features[0] = F.avg_pool2d(features[0], kernel_size=self.ks, padding=self.ps, stride=1)
        features[1] = F.avg_pool2d(features[1], kernel_size=self.ks, padding=self.ps, stride=1)
        features[2] = F.avg_pool2d(features[2], kernel_size=self.ks, padding=self.ps, stride=1)
        f0 = F.interpolate(features[0], align_corners=True, mode="bilinear", size=x.shape[-2:])
        f1 = F.interpolate(features[1], align_corners=True, mode="bilinear", size=x.shape[-2:])
        f2 = F.interpolate(features[2], align_corners=True, mode="bilinear", size=x.shape[-2:])
        f3 = F.interpolate(features[3], align_corners=True, mode="bilinear", size=x.shape[-2:])
        ft = torch.cat([f0, f1, f2], dim=1)   # 默认 1792-D
        return ft, [f0, f1, f2, f3]
# ===== 原 feature_extract.py 结束 =====

# ---------- 参数 ----------
WEIGHTS      = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/psad_infer/best_seg.pth"
MEM_BANK     = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/comp_memory.npy"   # ℳ_comp 路径
NUM_CLASSES  = 7                       # 含背景共 7 类
RESIZE       = 512
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PHOTO_DIR    = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/picture"
SEG_MAP      = os.path.join(PHOTO_DIR, "segmentation_map")
SEG_ABN_MAP  = os.path.join(PHOTO_DIR, "segmention_abnormalmap")
CLOUD_NOR    = os.path.join(PHOTO_DIR, "cloud")
CLOUD_ABN    = os.path.join(PHOTO_DIR, "abnormal_cloud")
SEG_ANOM     = os.path.join(PHOTO_DIR, "segmented_map")
HEAT_DIR     = os.path.join(PHOTO_DIR, "heatmap")
NPY_DIR      = os.path.join(PHOTO_DIR, "npy")

for d in [SEG_MAP, SEG_ABN_MAP, CLOUD_NOR, CLOUD_ABN, SEG_ANOM, HEAT_DIR, NPY_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- 1. 加载分割模型 ----------
model = PSADSegNet(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE)["model"])
model.eval()

tv_resize = T.Resize((RESIZE, RESIZE), interpolation=Image.BILINEAR)

# ---------- 2. 构建 / 加载 ℳ_comp 记忆库（高维特征向量云） ----------
def build_comp_memory():
    if os.path.exists(MEM_BANK):
        return np.load(MEM_BANK)          # (N, C, 1792)
    from train_psad_seg import LABELED_NAMES, UNLABELED_NAMES, PSADSegDataset, collate
    from torch.utils.data import DataLoader
    ds = PSADSegDataset(LABELED_NAMES + UNLABELED_NAMES, with_label=False, aug=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)
    feat_ext = FeatureExtractor().to(DEVICE)
    feat_ext.eval()
    bank = []
    with torch.no_grad():
        for img, _, _ in tqdm(loader, desc="Build comp memory"):
            img = img.to(DEVICE)
            coord = get_coord_map(1, RESIZE, RESIZE, DEVICE)
            logits = model(img, coord)
            prob   = F.softmax(logits, dim=1)
            pred   = prob.argmax(1).squeeze(0)           # 512×512
            # 高维特征
            img_3c = img.squeeze(0).cpu()
            img_pil = T.ToPILImage()(img_3c)
            img_tensor = T.Compose([T.Resize((RESIZE, RESIZE)), T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])(img_pil).unsqueeze(0).to(DEVICE)
            ft_map, _ = feat_ext(img_tensor)             # (1, 1792, H, W)
            # 按类别取平均 → (C, 1792)
            comp_vec = []
            for c in range(NUM_CLASSES):
                mask = (pred == c).float()
                if mask.sum() == 0:
                    vec = torch.zeros(ft_map.shape[1]).to(DEVICE)
                else:
                    vec = (ft_map * mask.unsqueeze(0)).sum(dim=(2, 3)) / mask.sum()
                comp_vec.append(vec.cpu().numpy())
            bank.append(np.stack(comp_vec, axis=0))   # (C, 1792)
    bank = np.stack(bank)                           # (N, C, 1792)
    np.save(MEM_BANK, bank)
    return bank

COMP_MEMORY = build_comp_memory()   # (N, C, 1792)

# ---------- 3. 单图推理 ----------
def infer_one(image_path, tag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        pred   = prob.argmax(1).squeeze(0).cpu().numpy()           # 512×512

    # ① 保存分割图
    seg_dir = SEG_MAP if is_normal else SEG_ABN_MAP
    colored = np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255]], dtype=np.uint8)[pred]
    Image.fromarray(colored).save(os.path.join(seg_dir, f"{base}_color.png"))
    jet = cv2.applyColorMap(np.clip(pred * 36, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(os.path.join(seg_dir, f"{base}.png"))

    # ② 构建当前图的部件高维向量（与记忆库同格式）
    img_3c = img_t.squeeze(0).cpu()
    img_pil = T.ToPILImage()(img_3c)
    img_tensor = T.Compose([T.Resize((RESIZE, RESIZE)), T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])(img_pil).unsqueeze(0).to(DEVICE)
    feat_ext = FeatureExtractor().to(DEVICE)
    feat_ext.eval()
    with torch.no_grad():
        ft_map, _ = feat_ext(img_tensor)                      # (1, 1792, H, W)
    comp_vec = []
    for c in range(NUM_CLASSES):
        mask = torch.from_numpy((pred == c).astype(np.float32))
        if mask.sum() == 0:
            vec = np.zeros(ft_map.shape[1])
        else:
          mask = torch.from_numpy((pred == c).astype(np.float32)).to(device)
          vec   = (ft_map * mask.unsqueeze(0)).sum(dim=(2, 3)) / mask.sum()
          vec   = vec.cpu().numpy()
        comp_vec.append(vec)
    comp_vec = np.stack(comp_vec, axis=0)   # (C, 1792)

    # ② 随机投影散点图（2D 可视化高维云）
    cloud_dir = CLOUD_NOR if is_normal else CLOUD_ABN
    
    # 创建固定的随机投影模型（只在第一次运行时创建）
    if not hasattr(infer_one, 'rp'):
        from sklearn.random_projection import GaussianRandomProjection
        infer_one.rp = GaussianRandomProjection(n_components=2, random_state=42)
        # 只对训练数据进行拟合一次
        train_flat = COMP_MEMORY.reshape(-1, 1792)  # (N*C, 1792)
        infer_one.cloud_2d = infer_one.rp.fit_transform(train_flat)
    
    # 使用固定的投影模型对测试数据进行变换
   # 对每个类别分别投影，再取平均
    test_2d_list = []
    for c in range(NUM_CLASSES):
        c_vec = comp_vec[c].reshape(1, -1)          # (1, 1792)
        if np.linalg.norm(c_vec) == 0:              # 全零向量跳过
            continue
        test_2d_list.append(infer_one.rp.transform(c_vec))
    test_2d = np.array(test_2d_list).mean(axis=0)   # (1, 2)
    
    plt.figure(figsize=(4, 4))
    plt.scatter(infer_one.cloud_2d[:, 0], infer_one.cloud_2d[:, 1], c='silver', s=8, label='Train cloud')
    plt.scatter(test_2d[:, 0], test_2d[:, 1], c='red', s=60, label='Current')
    plt.legend()
    plt.title(f"{tag} component cloud")
    plt.savefig(os.path.join(cloud_dir, f"{base}_cloud.png"), dpi=150)
    plt.close()

    comp_scores = []
    
    for c in range(NUM_CLASSES):
        # 获取当前图像第c类的特征向量并展平
        test_vec_c = comp_vec[c].flatten()  # 从 (1, 1792) 变为 (1792,)
        
        # 获取记忆库中第c类的所有特征向量并重塑
        train_vec_c = COMP_MEMORY[:, c, :, :].reshape(-1, 1792)  # 从 (351, 1, 1792) 变为 (351, 1792)
        
        # 检查测试向量是否为零向量
        if np.linalg.norm(test_vec_c) == 0:
            comp_scores.append(0.0)  # 零向量与零向量距离为0
            continue
        
        # 严格按照PSAD方法：使用欧氏距离计算最近邻
        distances = np.linalg.norm(train_vec_c - test_vec_c, axis=1)
        min_distance = np.min(distances)
        
        comp_scores.append(min_distance)
    
    comp_scores = np.array(comp_scores)  # (C,)
    
    # 计算整体异常分数（所有组件异常分数的平均值）
    score = comp_scores.mean()
    np.save(os.path.join(NPY_DIR, f"{base}_score.npy"), score)
    print(f"[{tag}] 异常分数 = {score:.4f}")
    pixel_scores = comp_scores[pred]                   # (H, W)
    pixel_scores = (pixel_scores - pixel_scores.min()) / (pixel_scores.max() - pixel_scores.min() + 1e-8)
    thresh = np.percentile(pixel_scores, 95)
    mask = (pixel_scores > thresh).astype(np.uint8) * 255
    Image.fromarray(mask).save(os.path.join(SEG_ANOM, f"{base}_mask.png"))

    # ⑤ 热力图（Jet 叠加原图）
    jet = cv2.applyColorMap(np.uint8(pixel_scores * 255), cv2.COLORMAP_JET)
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (RESIZE, RESIZE))
    heatmap_on_original = cv2.addWeighted(original_img, 0.5, jet, 0.5, 0)
    cv2.imwrite(os.path.join(HEAT_DIR, f"{base}_heatmap_on_original.png"), heatmap_on_original)

    # ⑥ npy 存档
    np.save(os.path.join(NPY_DIR, f"{base}_pixel_scores.npy"), pixel_scores)
    np.save(os.path.join(NPY_DIR, f"{base}_comp_vec.npy"), comp_vec)

    return score

# ---------- 4. 主入口 ----------
def main():
    # 固定参数，不再从命令行读取
    args = type('Args', (), {
        'weights': "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/psad_infer/best_seg.pth",
        'normal':  "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/normal_breakfastbox/usual.png",
        'logical': "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/logical_anomaly_breakfastbox/logical.png",
        'struct':  "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/structural_anomaly_breakfastbox/structural.png"
    })()
    print("=== PSAD Component 记忆库推理 ===")
    scores = []
    for tag, path in [("normal", args.normal), ("logical", args.logical), ("struct", args.struct)]:
        score = infer_one(path, tag)
        scores.append(score)
    avg_score = np.mean(scores[1:])
    print(f"【汇总】两张异常图平均异常分数 = {avg_score:.4f}")
    print("✅ 全部结果已保存至", PHOTO_DIR)

if __name__ == "__main__":
    main()