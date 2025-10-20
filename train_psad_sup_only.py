import os, glob, argparse, numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import wide_resnet101_2
from torchvision import transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- 路径/超参 ----------
IMG_DIR = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/psad_data/MVTec_LOCO_AD_512size/orig_512/juice_bottle/train/good"
LBL_DIR = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/psad_data/MVTec_LOCO_AD_512size/Annotations_fewshot_512/juice_bottle"
SAVE_DIR = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad"
REPRI_DIR = os.path.join(SAVE_DIR, "repri_loss")
FINE_DIR  = os.path.join(SAVE_DIR, "finetune_loss")
NUM_CLASSES = 5
RESIZE = 512
BATCH = 2
REPRI_EPOCHS = 50
FINE_EPOCHS  = 50
LR_REPRI = 1e-3
LR_FINE  = 2.5e-5
WD = 1e-4
LAMBDA_CE  = 1.0
LAMBDA_DICE = 0.2

for d in [SAVE_DIR, REPRI_DIR, FINE_DIR]:
    os.makedirs(d, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# ---------- 数据 ----------
def get_five_names():
    names = sorted([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{IMG_DIR}/*.png")])[:3]
    assert len(names) == 3, f"需要 3 张图，实际 {len(names)}"
    return names
FIVE_NAMES = get_five_names()

tv_resize = T.Resize((RESIZE, RESIZE), interpolation=Image.BILINEAR)
tv_resize_lbl = T.Resize((RESIZE, RESIZE), interpolation=Image.NEAREST)

class PSADSegDataset(Dataset):
    def __init__(self, names, aug=True):
        self.names = names; self.aug = aug
    def __len__(self): return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(f"{IMG_DIR}/{name}.png").convert("RGB")
        lbl = Image.open(f"{LBL_DIR}/{name}.png")
        img = tv_resize(img); lbl = tv_resize_lbl(lbl)
        if self.aug:
            img = T.ColorJitter(0.2, 0.2, 0.2, 0.05)(img)
            if torch.rand(1) < 0.5:
                img = T.functional.hflip(img); lbl = T.functional.hflip(lbl)
        img = T.ToTensor()(img)
        img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        lbl = torch.from_numpy(np.array(lbl)).long().clamp(0, NUM_CLASSES - 1)
        return img, lbl

# ---------- 坐标图 ----------
def get_coord_map(b, h, w, device):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                            torch.linspace(-1, 1, w, device=device), indexing='ij')
    return torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)   # B,2,H,W

# ---------- 模型 ----------
class RePRIPixelClassifier(nn.Module):
    def __init__(self, in_ch, num_cls):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_cls, 1)
    def forward(self, x): return self.conv(x)

class PSADBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        wr = wide_resnet101_2(weights="IMAGENET1K_V1")
        # 第一层改 5 通道
        self.layer0 = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False),
            wr.bn1, wr.relu, wr.maxpool
        )
        self.layer1 = wr.layer1
        self.layer2 = wr.layer2
        self.layer3 = wr.layer3

    def forward(self, x):
        b, _, h, w = x.shape
        coord = get_coord_map(b, h, w, x.device)
        x = torch.cat([x, coord], dim=1)   # 3+2=5
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        f1 = F.interpolate(x1, size=(h, w), mode="bilinear", align_corners=False)
        f2 = F.interpolate(x2, size=(h, w), mode="bilinear", align_corners=False)
        f3 = F.interpolate(x3, size=(h, w), mode="bilinear", align_corners=False)
        return torch.cat([f1, f2, f3], dim=1)   # B,1792,H,W

def dice_loss(pred, target, eps=1e-6):
    pred = F.softmax(pred, dim=1)
    target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()

# ---------- 通用工具 ----------
def plot_curve(loss, miou, save_dir, stage):
    txt = os.path.join(save_dir, f'{stage}_log.txt')
    np.savetxt(txt, np.column_stack([np.arange(len(loss)), loss, miou]),
               header='epoch,loss,miou', delimiter=',', comments='')
    plt.figure(figsize=(7, 4))
    ax1, ax2 = plt.gca(), plt.gca().twinx()
    ax1.plot(loss, color='tab:red', marker='o', label='Loss')
    ax2.plot(miou, color='tab:blue', marker='s', label='mIoU')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss', color='tab:red')
    ax2.set_ylabel('mIoU', color='tab:blue')
    plt.title(f'{stage} training curve')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{stage}_curve.png'), dpi=200)
    plt.close()

@torch.no_grad()
def validate(backbone, head):
    backbone.eval(); head.eval()
    ds = PSADSegDataset(FIVE_NAMES, aug=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    iou_list = []
    for img, lbl in dl:
        img, lbl = img.to(device), lbl.to(device)
        feats = backbone(img)
        logits = head(feats)
        pred = logits.argmax(1)
        inter = ((pred == lbl) & (lbl != 0)).sum(dim=(1, 2))
        union = ((pred != 0) | (lbl != 0)).sum(dim=(1, 2))
        iou_list.append((inter / (union + 1e-8)).mean().item())
    return np.mean(iou_list) if iou_list else 0.

# ---------- RePRI 阶段 ----------
def run_repri():
    print("========== RePRI 预训练阶段（含坐标图） ==========")
    backbone = PSADBackbone().to(device).eval()
    for p in backbone.parameters(): p.requires_grad = False
    head = RePRIPixelClassifier(1792, NUM_CLASSES).to(device)
    optimizer = AdamW(head.parameters(), lr=LR_REPRI, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=REPRI_EPOCHS)
    ds = PSADSegDataset(FIVE_NAMES, aug=True)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
    loss_rec, miou_rec = [], []
    best = 0.
    for epoch in range(REPRI_EPOCHS):
        head.train()
        pbar = tqdm(dl, desc=f"RePRI E{epoch}")
        loss_sum = 0
        for img, lbl in pbar:
            img, lbl = img.to(device), lbl.to(device)
            with torch.no_grad():
                feats = backbone(img)
            logits = head(feats)
            loss = F.cross_entropy(logits, lbl) + LAMBDA_DICE * dice_loss(logits, lbl)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()
            pbar.set_postfix(loss=loss.item())
        scheduler.step()
        miou = validate(backbone, head)
        loss_rec.append(loss_sum / len(dl))
        miou_rec.append(miou)
        print(f"RePRI Epoch {epoch:03d}  loss {loss_rec[-1]:.4f}  mIoU {miou:.3f}")
        if miou > best:
            best = miou
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "epoch": epoch}, os.path.join(SAVE_DIR, "best_repri.pth"))
    plot_curve(loss_rec, miou_rec, REPRI_DIR, "repri")
    print("RePRI 完成，最佳 mIoU:", best)
    return backbone, head

# ---------- 微调阶段 ----------
def finetune(backbone, head):
    print("========== CE+Dice 微调阶段（含坐标图） ==========")
    for p in backbone.parameters(): p.requires_grad = True
    optimizer = AdamW(list(backbone.parameters()) + list(head.parameters()),
                      lr=LR_FINE, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=FINE_EPOCHS)
    ds = PSADSegDataset(FIVE_NAMES, aug=True)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
    loss_rec, miou_rec = [], []
    best = 0.
    for epoch in range(FINE_EPOCHS):
        backbone.train(); head.train()
        pbar = tqdm(dl, desc=f"Fine E{epoch}")
        loss_sum = 0
        for img, lbl in pbar:
            img, lbl = img.to(device), lbl.to(device)
            feats = backbone(img)
            logits = head(feats)
            loss = F.cross_entropy(logits, lbl) + LAMBDA_DICE * dice_loss(logits, lbl)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()
            pbar.set_postfix(loss=loss.item())
        scheduler.step()
        miou = validate(backbone, head)
        loss_rec.append(loss_sum / len(dl))
        miou_rec.append(miou)
        print(f"Fine Epoch {epoch:03d}  loss {loss_rec[-1]:.4f}  mIoU {miou:.3f}")
        if miou > best:
            best = miou
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "epoch": epoch}, os.path.join(SAVE_DIR, "best_finetune.pth"))
        if (epoch + 1) % 10 == 0:
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "epoch": epoch}, os.path.join(SAVE_DIR, "checkpoint_fine.pth"))
    plot_curve(loss_rec, miou_rec, FINE_DIR, "finetune")
    print("微调完成，最佳 mIoU:", best)

# ---------- 推理 ----------
def inference(weights, img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(weights, map_location=device)
    backbone = PSADBackbone().to(device).eval()
    head = RePRIPixelClassifier(1792, NUM_CLASSES).to(device).eval()
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    transform = T.Compose([T.Resize((RESIZE, RESIZE)), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for name in tqdm(os.listdir(img_dir)):
        if not name.endswith(".png"): continue
        img = Image.open(os.path.join(img_dir, name)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = head(backbone(tensor)).argmax(1).squeeze(0).cpu().numpy()
        Image.fromarray(pred.astype(np.uint8)).save(os.path.join(out_dir, name))
    print("推理完成 →", out_dir)

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["repri", "fine", "both"], default="both")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--weights", default="")
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()

    if args.eval:
        assert args.weights and args.image_dir and args.out_dir
        inference(args.weights, args.image_dir, args.out_dir)
        return

    backbone, head = None, None
    if args.stage in ["repri", "both"]:
        backbone, head = run_repri()
    if args.stage in ["fine", "both"]:
        if backbone is None:
            ckpt = torch.load(os.path.join(SAVE_DIR, "best_repri.pth"), map_location=device)
            backbone = PSADBackbone().to(device)
            head = RePRIPixelClassifier(1792, NUM_CLASSES).to(device)
            backbone.load_state_dict(ckpt["backbone"])
            head.load_state_dict(ckpt["head"])
        finetune(backbone, head)

if __name__ == "__main__":
    main()