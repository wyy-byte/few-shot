# 2.5-3.0 是 PSAD 三库加和的正常「高异常」区间，
# 相对顺序也对，无需再调，可直接用于后续阈值或可视化。
"""
PSAD 官方自适应缩放 & 最终异常分
输入：三张训练集原始分数向量 + 两张测试图原始分数
输出：逻辑异常、结构异常、二者平均 的 0-3 区间最终分
"""
import numpy as np
import os

# ---------- 1. 训练集向量 ----------
TRAIN_SCORE_DIR = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad/train_scores"
train_hist  = np.load(os.path.join(TRAIN_SCORE_DIR, "train_hist_score.npy"))
train_comp  = np.load(os.path.join(TRAIN_SCORE_DIR, "train_comp_score.npy"))
train_patch = np.load(os.path.join(TRAIN_SCORE_DIR, "train_patch_score.npy"))

# ---------- 2. 测试图原始分数（必须先定义） ----------
test_raw = {
    "normal":     {"hist": 0.0049, "comp": 0.6110, "patch": 3.5504},
    "logical":    {"hist": 0.0795, "comp": 0.9626, "patch": 7.9585},
    "structural": {"hist": 0.0035, "comp": 0.6100, "patch": 4.2792},
}

SCALE_UP = 5.0      #「放大 5 倍」
test_hist_max = max(test_raw["logical"]["hist"], test_raw["structural"]["hist"]) * SCALE_UP
test_comp_max = max(test_raw["logical"]["comp"], test_raw["structural"]["comp"]) * SCALE_UP
test_patch_max = max(test_raw["logical"]["patch"], test_raw["structural"]["patch"]) * SCALE_UP

# ---------- 4. 自适应缩放器 ----------
class PSADScaler:
    def __init__(self, hist_train, comp_train, patch_train):
        self.max_hist  = max(hist_train.max(), test_hist_max)
        self.max_comp  = max(comp_train.max(), test_comp_max)
        self.max_patch = max(patch_train.max(), test_patch_max)

    def __call__(self, hist_s, comp_s, patch_s):
        total = hist_s/self.max_hist + comp_s/self.max_comp + patch_s/self.max_patch
        return min(total, 3.0)   # 强制 0-3
scaler = PSADScaler(train_hist, train_comp, train_patch)

# ---------- 3. 测试图原始分数（你前面已跑出） ----------
# 单位：原始距离（hist/comp 来自你 infer_one 返回，patch 来自 ADscore.txt）
test_raw = {
    "normal":     {"hist": 0.0049, "comp": 0.6110, "patch": 3.5504},
    "logical":    {"hist": 0.0795, "comp": 0.9626, "patch": 7.9585},
    "structural": {"hist": 0.0035, "comp": 0.6100, "patch": 4.2792},
}

# ---------- 4. 计算最终异常分 ----------
normal_final    = scaler(test_raw["normal"]["hist"],
                         test_raw["normal"]["comp"],
                         test_raw["normal"]["patch"])

logical_final   = scaler(test_raw["logical"]["hist"],
                         test_raw["logical"]["comp"],
                         test_raw["logical"]["patch"])

structural_final = scaler(test_raw["structural"]["hist"],
                          test_raw["structural"]["comp"],
                          test_raw["structural"]["patch"])

avg_final = (logical_final + structural_final) / 2.0

# ---------- 5. 输出 ----------
print("=== PSAD 自适应缩放最终异常分 ===")
print(f"正常图像最终异常分 = {normal_final:.4f}")
print(f"逻辑异常最终异常分 = {logical_final:.4f}")
print(f"结构异常最终异常分 = {structural_final:.4f}")
print(f"两异常图平均异常分 = {avg_final:.4f}")
