"""
test.py — Mandatory Evaluation Script for APEIREON: Shrink or Sink
=================================================================
This script performs model inference on the STL-10 test set and reports 
the final classification accuracy, as required by the competition rulebook.

Usage:
    python test.py --dataset-path ./data --model-path student_final.pth
"""

import argparse
import re
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model import DynamicNet

def infer_architecture(state_dict: dict):
    """Detects and reconstructs architecture parameters from state_dict."""
    # Normalize keys (strip 'module.' prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    state_dict = new_state_dict

    # ── ResNet-50 Teacher detection ──────────────────────────────────
    if "layer1.0.conv1.weight" in state_dict:
        return None, None, state_dict, "teacher"
    
    # ── DynamicNet Student detection ──────────────────────────────────
    if "conv1.0.weight" in state_dict:
        w0 = state_dict["conv1.0.weight"].shape[0]
        stage_out = {}
        stage_max_layer = {}
        stage_pw_pat  = re.compile(r"^features\.(\d+)\.stage\.(\d+)\.pw_bn\.weight$")
        for key, tensor in state_dict.items():
            m = stage_pw_pat.match(key)
            if m:
                s, d = int(m.group(1)), int(m.group(2))
                stage_out[s] = tensor.shape[0]
                stage_max_layer[s] = max(stage_max_layer.get(s, 0), d)
        
        n_stages = max(stage_out.keys()) + 1
        widths_param = [stage_out[s] for s in range(n_stages)]
        depths_param = [stage_max_layer[s] + 1 for s in range(n_stages)]
        return widths_param, depths_param, state_dict, "student"

    raise KeyError("Unknown Model Architecture in provided weights.")

def main():
    parser = argparse.ArgumentParser(description="APEIREON Model Evaluation")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to STL-10 dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .pth model weights")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    # ── Device Selection (CUDA -> MPS -> CPU) ─────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Load Weights & Infer Model ────────────────────────────────────
    ckpt = torch.load(args.model_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Invalid weight file format.")

    # Handle common wrappers
    raw_sd = ckpt.get("model", ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt)))
    widths, depths, cleaned_sd, model_type = infer_architecture(raw_sd)

    if model_type == "teacher":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        model = DynamicNet(widths=widths, depths=depths)

    model.load_state_dict(cleaned_sd, strict=True)
    model.to(device).eval()

    # ── Dataset Loading ───────────────────────────────────────────────
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
    ])
    test_ds = torchvision.datasets.STL10(root=args.dataset_path, split="test", download=True, transform=transform)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ── Inference Pass ────────────────────────────────────────────────
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"\nEvaluation Results:")
    print(f"-------------------")
    print(f"Total Samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.2f}%")
    print(f"-------------------")

if __name__ == "__main__":
    main()
