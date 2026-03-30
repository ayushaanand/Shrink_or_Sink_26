"""
train.py
--------
Implements the full training pipeline to faithfully reproduce the submitted model.
Leverages Knowledge Distillation from a robust ResNet-50 Teacher.

Usage:
    python train.py --dataset-path ./data --teacher-path teacher_best.pth --model-path student_final.pth
"""

import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image


from model import DynamicNet

def set_seed(seed=42):
    """Ensures deterministic, reproducible training as mandated by the rulebook."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    """Ensures each DataLoader worker process has an independent, reproducible random seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def kd_loss(logits, teacher_logits, labels, T=4.0, alpha=0.9):
    """Knowledge Distillation Loss combining hard labels and soft teacher probabilities.
    Unlabeled samples (y=-1) use the teacher's argmax as a pseudo-label for CE,
    matching the reference train_recipe.py implementation exactly.
    """
    loss_kd = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(logits/T, dim=1),
        F.softmax(teacher_logits/T, dim=1)
    ) * (T * T)
    
    # Substitute teacher pseudo-labels for unlabeled images (y == -1)
    hard_labels = torch.where(labels == -1, teacher_logits.argmax(dim=1), labels)
    loss_ce = F.cross_entropy(logits, hard_labels, label_smoothing=0.1)
    
    return (1. - alpha) * loss_ce + alpha * loss_kd

class RAMCachedSTL10(Dataset):
    """Stores all 105k uint8 images into RAM for hyperspeed distillation. Disabled natively on 'mps' fallback."""
    def __init__(self, stl10_datasets, transform=None, teacher_logits=None):
        self.transform = transform
        raw_data = np.concatenate([ds.data for ds in stl10_datasets], axis=0)
        self.data = np.transpose(raw_data, (0, 2, 3, 1))
        self.labels = np.concatenate([ds.labels for ds in stl10_datasets], axis=0)
        self.teacher_logits = teacher_logits
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        target = int(self.labels[index])
        if self.teacher_logits is not None:
            return img, target, self.teacher_logits[index]
        return img, target

def get_teacher(path, device, retries=3):
    """Loads the pre-trained Ultimate Teacher (ResNet-50) robustly against network I/O failures."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"\n🚨 CRITICAL ERROR: The teacher weights file was not found at '{path}'. "
                                f"Please make sure you have attached the Dataset correctly in the Kaggle UI!")
        
    for attempt in range(retries):
        try:
            print(f"Loading Teacher Model from '{path}' (Attempt {attempt+1}/{retries})...")
            teacher = models.resnet50(weights=None)
            teacher.fc = nn.Linear(teacher.fc.in_features, 10)
            teacher.load_state_dict(torch.load(path, map_location=device))
            teacher = teacher.to(device)
            teacher.eval()
            print(f"✅ Teacher securely loaded from '{path}'\n")
            return teacher
        except Exception as e:
            print(f"⚠️ Warning: Failed to load teacher: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            
    raise RuntimeError("🚨 CRITICAL ERROR: Failed to load teacher after multiple attempts. The file might be corrupted.")

def train(args):
    set_seed(42)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Device: {device}")

    # STL-10 augmentations (ColorJitter + RandCrop/Flip)
    train_tf = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
    ])
    
    print(f"Loading native STL-10 labeled (5k) + unlabeled (100k) datasets to '{args.dataset_path}'...")
    train_ds = datasets.STL10(root=args.dataset_path, split="train", download=not args.no_download, transform=train_tf)
    unlab_ds = datasets.STL10(root=args.dataset_path, split="unlabeled", download=not args.no_download, transform=train_tf)
    
    teacher = get_teacher(args.teacher_path, device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for Teacher!")
        teacher = nn.DataParallel(teacher)

    if device.type != 'mps':
        print("\n⚡ [Kaggle Hyperspeed Mode]: Caching 105k images into RAM and precomputing Teacher Logits...")
        raw_chw = np.concatenate([train_ds.data, unlab_ds.data], axis=0)
        _norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713])])
        
        class _InferDs(Dataset):
            def __init__(self, chw): self.d = np.transpose(chw, (0, 2, 3, 1))
            def __len__(self): return len(self.d)
            def __getitem__(self, i): return _norm(Image.fromarray(self.d[i]))
            
        infer_ld = DataLoader(_InferDs(raw_chw), batch_size=512, shuffle=False, num_workers=2)
        all_logits = []
        with torch.no_grad():
            for x in infer_ld:
                all_logits.append(teacher(x.to(device)).cpu().half())
        teacher_logits = torch.cat(all_logits, dim=0)
        
        print("✅ Logits successfully cached natively into RAM! System will train at Maximum IPC.")
        # Override the transform inside the class internally so we don't double transform
        combined_ds = RAMCachedSTL10([train_ds, unlab_ds], transform=train_tf, teacher_logits=teacher_logits)
    else:
        from torch.utils.data import ConcatDataset
        print("🍎 [Apple MPS Fallback]: Executing via native PyTorch dataset pipelines for maximum stability.")
        combined_ds = ConcatDataset([train_ds, unlab_ds])
    
    g = torch.Generator()
    g.manual_seed(42)
    
    train_ld = DataLoader(
        combined_ds, 
        batch_size=128, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    print(f"Initializing Student Model (Widths: {args.widths}, Depths: {args.depths})...")
    student = DynamicNet(widths=args.widths, depths=args.depths).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for Student!")
        student = nn.DataParallel(student)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    if os.path.exists(args.checkpoint):
        print(f"\n[RESUME] Found checkpoint at '{args.checkpoint}'. Resuming training...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(student, nn.DataParallel):
            student.module.load_state_dict(ckpt['model_state'])
        else:
            student.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f"[RESUME] ✅ Resumed smoothly from Epoch {ckpt['epoch']}\n")

    print(f"Starting Knowledge Distillation from Epoch {start_epoch} to {args.epochs}...")
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_ld):
            if len(batch) == 3:
                x, y, t_logits = batch
                x, y = x.to(device), y.to(device)
                t_logits = t_logits.to(device).float()
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    t_logits = teacher(x)
                
            optimizer.zero_grad()
            s_logits = student(x)
            
            loss = kd_loss(s_logits, t_logits, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_ld):.4f}")
        
        # ── Save Unconditional Checkpoint ──
        save_state = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
        temp_ckpt = args.checkpoint + ".tmp"
        torch.save({
            'epoch': epoch,
            'model_state': save_state,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }, temp_ckpt)
        os.replace(temp_ckpt, args.checkpoint)

    # ── Final Student Evaluation ──
    print(f"\n✅ Distillation Loop Complete. Running Final Verification Pass...")
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
    ])
    test_ds = datasets.STL10(root=args.dataset_path, split="test", download=not args.no_download, transform=test_transform)
    test_ld = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    student.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_ld:
            x, y = x.to(device), y.to(device)
            preds = student(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            
    final_acc = (correct / len(test_ds)) * 100.0
    print(f"The Final Distilled Student Accuracy is: {final_acc:.2f}%\n")

    print(f"Saving final weights (FP16 Compressed) to '{args.model_path}'...")
    student.half() # Cast weights to 16-bit to cut file size by 50%
    save_state = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
    torch.save(save_state, args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Path for STL-10 dataset")
    parser.add_argument("--teacher-path", type=str, required=True, help="Path to teacher weights")
    parser.add_argument("--model-path", type=str, default="student_final.pth", help="Output filename")
    parser.add_argument("--checkpoint", type=str, default="student_checkpoint.pth", help="Checkpoint file string")
    parser.add_argument("--epochs", type=int, default=100, help="Training length")
    parser.add_argument("--widths", type=int, nargs='+', default=[32, 64, 128, 256], help="Widths for DynamicNet stages")
    parser.add_argument("--depths", type=int, nargs='+', default=[2, 2, 2, 2], help="Depths for DynamicNet stages")
    parser.add_argument("--no-download", action="store_true", help="Skip dataset download (for Kaggle/Local already downloaded)")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    train(parser.parse_args())
