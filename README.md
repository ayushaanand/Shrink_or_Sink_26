# 📉 Shrink or Sink: Dynamo Submission

**Team Name**: Dynamo  
**Final Accuracy**: 70.84% (vs. 67.0% Baseline)  
**Weights Size**: 26.4 KB (`student_final.pth`)  
**Total Inference Size**: ~31.2 KB (Weights + `test.py` + `model.py`)  

---

Our submission aggressively targets the APEIREON scoring metrics by deploying **DynamicNet**, a bespoke, handcrafted Depthwise-Separable Convolutional architecture explicitly engineered to map against the STL-10 feature dimensions using the absolute minimum parameter count possible. 

By shifting from traditional heavy convolution blocks to highly-optimized depthwise-separable matrices aligned on power-of-2 dimensions (**`w = [16, 32, 64, 64]`** and **`d = [1, 1, 1, 1]`**), and implementing state-of-the-art **Knowledge Distillation**, we are able to mathematically compress our final weights into an incredibly tiny footprint on disk (**26.4 KB**) while firmly exceeding the `67.0%` accuracy baseline.

## 🏆 Final Results

| Model | Accuracy | Epochs | Params | Disk Size |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-50 Teacher** | **87.96%** | 300* | ~23.5M | ~90 MB |
| **DynamicNet Student** | **70.84%** | 100 | ~10K | **26.4 KB** |

*\*Replicating the exact 87.96% Teacher from scratch is highly stochastic and computationally demanding. The specific weights used for this submission were achieved over a 4-day training period that included 2 to 3 manual warm restarts (SGDR) and dynamic re-selection of the most confident unlabeled images for pseudo-labeling. The provided training script contains the core, un-restarted baseline engine that natively reaches ~83% within 1.2 hours.*

## 📂 Repository Structure (Mandatory Components)

- **`student_final.pth`**: FP16-quantized weights saved in an ordered-list format (tensor names are not stored).
- **`model.py`**: Defines the `DynamicNet` architecture utilizing Depthwise-Separable blocks.
- **`test.py`**: Mandatory evaluation script featuring a Hybrid Loader for our optimized model format.
- **`train.py`**: The complete Knowledge Distillation training pipeline. Includes fixed deterministic RNG seeds (`42`).
- **`teacher_train.py`**: Optional helper script to train a ResNet-50 teacher model *from scratch* (extra; not required for evaluation).

---

## 🛠 Model Compression Techniques

1. **Depthwise Separable Bottlenecks**: De-couples spatial and channel convolutions, drastically reducing structural parameters by ~8x compared to a native CNN.
2. **Feature Manifold Alignment**: The topology (**`[16, 32, 64, 64]`** with depths **`[1, 1, 1, 1]`**) was empirically engineered to downsample STL-10 images cleanly without wasting space.
3. **ResNet-50 Knowledge Distillation**: The student model mimics the "dark knowledge" of our 87.96% Teacher through soft-targets (`T=4.0`, `Alpha=0.9`).   
4. **Ordered-List FP16 Serialization**: The student weights are saved as an ordered list of FP16 tensors, so tensor parameter names are not stored in `student_final.pth`.

---

## 🚀 Execution & Verification

### 1. Verification (Accuracy Check)
To verify the accuracy of the generated `.pth` file on the official `test` split:
```bash
python test.py --dataset-path ./data --model-path student_final.pth
```
*(The script will automatically detect the optimized format and reconstruct the model.)*

### 2. Training reproduction
To reproduce the student weights using Knowledge Distillation from our teacher:
```bash
python train.py --dataset-path ./data --teacher-path ./teacher_final.pth --widths 16 32 64 64 --depths 1 1 1 1 --epochs 100
```

---

## ⚠️ Hardware & Performance Disclaimer

While the training scripts are technically universal and cross-platform (CUDA/MPS/CPU), we optimized the pipeline for GPU-friendly training on Kaggle-like T4 x2 hardware. Runtime can vary by environment.

*   **Observed Speed**: In our experiments on Kaggle T4 x2, training took approximately **81 seconds per epoch** after the initial in-memory caching of the dataset.
*   **MPS / CPU**: On hardware without dedicated NVIDIA GPUs, the evaluation script will successfully fallback to MPS / CPU inference, though most of the scripts are tailored to exploit the Kaggle T4 2x hardware.
*   **RAM Management**: `train.py` uses an in-memory cached dataset; on machines with limited system RAM (<8GB), you may need to adjust batch size or stop/avoid in-memory caching to prevent OOM errors.
*   **Determinism**: Fully reproducible training via fixed seeds (42) and CuDNN determinism enabled.

---

## 💾 Resuming & Checkpoints

The `--weights` flag can be used to instigate a warm restart from a raw weights file for `train_teacher.py`. 

The training script is designed to be restartable. If a run is interrupted, simply run the same command again. `train.py` will look for its checkpoint file (`student_checkpoint.pth`) and restore the model/optimizer/scheduler state from the last saved epoch. Same goes for `train_teacher.py`.
