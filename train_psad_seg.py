#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_psad_seg.py  |  PSAD 少样本语义分割
强制 512×512 + 完整主函数 + 继续训练/推理开关
"""
import os, glob, argparse, numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import wide_resnet101_2
from torchvision import transforms as T

# ---------------- 配置 ----------------
IMG_DIR = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/psad_data/orig_512/breakfast_box/train/good"
LBL_DIR = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/psad_data/MVTec_LOCO_AD_512size/Annotations_fewshot_512/breakfast_box"
SAVE_DIR = "/home/ps/few-shot-research/PSAD_logical_anomaly_detection/wnpsad"
NUM_CLASSES = 7
RESIZE = 512
BATCH = 5
EPOCHS = 100
LR = 1e-3
WD = 1e-4
LAMBDA_CE = 10
LAMBDA_DICE = 1
LAMBDA_H = 10
LAMBDA_HIST = 10
WARM_EPOCH = 50

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/logs", exist_ok=True)
writer = SummaryWriter(f"{SAVE_DIR}/logs")
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# ---------- 数据 ----------
def get_split():
    names = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{IMG_DIR}/*.png")]
    labeled = [n for n in names if os.path.exists(f"{LBL_DIR}/{n}.png")]
    unlabeled = [n for n in names if n not in labeled]
    assert len(labeled) == 5, f"需要 5 张标注，实际 {len(labeled)}"
    return labeled, unlabeled

LABELED_NAMES, UNLABELED_NAMES = get_split()

tv_resize = T.Resize((RESIZE, RESIZE), interpolation=Image.BILINEAR)
tv_resize_lbl = T.Resize((RESIZE, RESIZE), interpolation=Image.NEAREST)

class PSADSegDataset(Dataset):
    def __init__(self, names, with_label=False, aug=True):
        self.names = names
        self.with_label = with_label
        self.aug = aug

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(f"{IMG_DIR}/{name}.png").convert("RGB")
        lbl = Image.open(f"{LBL_DIR}/{name}.png") if self.with_label else Image.new("L", img.size)

        img = tv_resize(img)
        lbl = tv_resize_lbl(lbl)

        if self.aug:
            img = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)(img)
            if torch.rand(1) < 0.5:
                img = T.functional.hflip(img)
                lbl = T.functional.hflip(lbl)

        img = T.ToTensor()(img)
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        lbl = torch.from_numpy(np.array(lbl)).long().clamp(0, NUM_CLASSES - 1)
        return img, lbl, self.with_label

def collate(batch):
    imgs, lbls, with_label = zip(*batch)
    return torch.stack(imgs), torch.stack(lbls), with_label[0]

# ---------- 模型 ----------
class SegHead(nn.Module):
    def __init__(self, in_ch, num_cls):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(in_ch + 2, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, num_cls, 1)
        )

    def forward(self, x, coord):
        x = torch.cat([x, coord], dim=1)
        return self.dec(x)

class PSADSegNet(nn.Module):
    def __init__(self, num_cls):
        super().__init__()
        wr = wide_resnet101_2(weights="IMAGENET1K_V1")
        self.layer1 = nn.Sequential(wr.conv1, wr.bn1, wr.relu, wr.maxpool, wr.layer1)
        self.layer2 = wr.layer2
        self.layer3 = wr.layer3
        self.head = SegHead(256 + 512 + 1024, num_cls)

    def forward(self, x, coord):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        h, w = x.shape[2:]
        f1 = F.interpolate(x1, size=(h, w), mode="bilinear", align_corners=False)
        f2 = F.interpolate(x2, size=(h, w), mode="bilinear", align_corners=False)
        f3 = F.interpolate(x3, size=(h, w), mode="bilinear", align_corners=False)
        v = torch.cat([f1, f2, f3], dim=1)
        return self.head(v, coord)

def get_coord_map(b, h, w, device):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                            torch.linspace(-1, 1, w, device=device), indexing='ij')
    return torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)

def dice_loss(pred, target, eps=1e-6):
    pred = F.softmax(pred, dim=1)
    target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()

def hist_match_loss(prob_u, ref_onehot):
    # prob_u:     (B, 4, H, W)
    # ref_onehot: (B, 4, H, W)
    hist_l = ref_onehot.float().mean(dim=(2, 3))          # (B, 4)
    hist_u = prob_u.mean(dim=(2, 3))                      # (B, 4)
    return F.mse_loss(hist_u, hist_l)
def train_one_epoch(model, loader_l, loader_u, optimizer, epoch):
    model.train()
    pbar = tqdm(total=len(loader_l) + len(loader_u), desc=f"Epoch {epoch}")
    loss_sum = 0
    for img_l, lbl_l, _ in loader_l:
        img_l, lbl_l = img_l.to(device), lbl_l.to(device)
        coord = get_coord_map(img_l.shape[0], img_l.shape[2], img_l.shape[3], device)
        logits = model(img_l, coord)
        loss = LAMBDA_CE * F.cross_entropy(logits, lbl_l) + LAMBDA_DICE * dice_loss(logits, lbl_l)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_sum += loss.item(); pbar.update(1)

    if epoch >= WARM_EPOCH:
        for img_u, _, _ in loader_u:
            img_u = img_u.to(device)
            coord = get_coord_map(img_u.shape[0], img_u.shape[2], img_u.shape[3], device)
            logits_u = model(img_u, coord)
            prob_u = F.softmax(logits_u, dim=1)
            loss_h = -torch.mean(torch.sum(prob_u * torch.log(prob_u + 1e-8), dim=1))
            ref_img, ref_lbl, _ = next(iter(loader_l))
            ref_onehot = F.one_hot(ref_lbl.to(device), NUM_CLASSES).permute(0, 3, 1, 2).float()
            loss_hist = hist_match_loss(prob_u, ref_onehot) # 确保 (1, 4)
            loss_ul = LAMBDA_H * loss_h + LAMBDA_HIST * loss_hist
            optimizer.zero_grad(); loss_ul.backward(); optimizer.step()
            loss_sum += loss_ul.item(); pbar.update(1)
    pbar.close()
    return loss_sum / (len(loader_l) + len(loader_u))

@torch.no_grad()
def validate(model, loader):
    model.eval()
    miou_list = []
    for img, lbl, _ in loader:
        img, lbl = img.to(device), lbl.to(device)
        coord = get_coord_map(img.shape[0], img.shape[2], img.shape[3], device)
        pred = model(img, coord).argmax(1)
        inter = ((pred == lbl) & (lbl != 0)).sum(dim=(1, 2))
        union = ((pred != 0) | (lbl != 0)).sum(dim=(1, 2))
        miou_list.append((inter / (union + 1e-8)).mean().item())
    model.train()
    return np.mean(miou_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5, help="继续训练学习率")
    parser.add_argument("--epochs", type=int, default=150, help="继续训练 epoch 数")
    parser.add_argument("--dice_thresh", type=float, default=0.2, help="Dice 阈值") 
    parser.add_argument("--resume", default="", help="继续训练权重路径")
    parser.add_argument("--eval", action="store_true", help="推理模式")
    parser.add_argument("--weights", default="", help="推理权重")
    parser.add_argument("--image_dir", default="", help="推理图片目录")
    parser.add_argument("--out_dir", default="", help="推理保存目录")
    args = parser.parse_args()

    
    model = PSADSegNet(NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start = ckpt["epoch"] + 1
        print(f"继续训练 epoch {start}")

    if args.eval:
        assert args.weights and args.image_dir and args.out_dir
        os.makedirs(args.out_dir, exist_ok=True)
        model.load_state_dict(torch.load(args.weights, map_location=device)["model"])
        model.eval()
        transform = T.Compose([T.Resize((RESIZE, RESIZE)), T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        for name in tqdm(os.listdir(args.image_dir)):
            if not name.endswith(".png"): continue
            img = np.array(Image.open(os.path.join(args.image_dir, name)).convert("RGB"))
            tensor = transform(img).unsqueeze(0).to(device)
            coord = get_coord_map(1, RESIZE, RESIZE, device)
            pred = model(tensor, coord).argmax(1).squeeze(0).cpu().numpy()
            Image.fromarray(pred.astype(np.uint8)).save(os.path.join(args.out_dir, name))
        print("推理完成 →", args.out_dir)
        return

    ds_l = PSADSegDataset(LABELED_NAMES, with_label=True, aug=True)
    ds_u = PSADSegDataset(UNLABELED_NAMES, with_label=False, aug=True)
    loader_l = DataLoader(ds_l, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate, drop_last=True)
    loader_u = DataLoader(ds_u, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate, drop_last=True)

    best = 0
    for epoch in range(start, EPOCHS):
        train_loss = train_one_epoch(model, loader_l, loader_u, optimizer, epoch)
        scheduler.step()
        writer.add_scalar("loss/train", train_loss, epoch)
        miou = validate(model, loader_l)
        writer.add_scalar("mIoU/val", miou, epoch)
        print(f"Epoch {epoch}  平均损失 {train_loss:.4f}  mIoU {miou:.3f}")
        if miou > best:
            best = miou
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch}, f"{SAVE_DIR}/best_seg.pth")
        if (epoch + 1) % 10 == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch}, f"{SAVE_DIR}/checkpoint.pth")
    writer.close()

if __name__ == "__main__":
    main()