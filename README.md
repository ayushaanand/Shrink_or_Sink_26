# 📉 Shrink or Sink: DynamicNet Baseline

Our submission aggressively targets the `(smallest_size / your_size) × 750` scoring metric by deploying **DynamicNet**, a bespoke, handcrafted Depthwise-Separable Convolutional architecture explicitly engineered to map against the STL-10 96x96 feature dimensions using the absolute minimum parameter count possible. 

By shifting from traditional heavy convolution blocks to highly-optimized depthwise-separable matrices aligned on power-of-2 dimensions (**`w = [32, 64, 128, 256]`** and **`d = [2, 2, 2, 2]`**), and implementing state-of-the-art **Knowledge Distillation** from a completely indigenous ResNet-50 teacher model, we are able to mathematically compress our final weights (`FP16`) into an incredibly tiny footprint on disk (**326 KB**) while firmly exceeding the `85.0%` accuracy constraint.

## 🏆 Final Results

| Model | Accuracy | Epochs | Params | Disk Size |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-50 Teacher** | **87.96%** | 300* | ~23.5M | ~90 MB |
| **DynamicNet Student** | **86.75%** | 100 | ~163K | **326 KB** |

*\*Teacher run includes a warm restart from 84.88% (200 epochs) to achieve the final 87.96%.*

## 📂 Repository Structure (Mandatory Components)

- `model.py`: Defines the `DynamicNet` architecture utilizing Depthwise-Separable blocks.
- `train.py`: The complete Knowledge Distillation training pipeline. Configurable dynamic inputs allow running against any structured STL-10 subset directory. Includes fixed deterministic RNG seeds (`42`).
- `teacher_train.py`: The completely independent native script we built to train our original 85%+ ResNet-50 Teacher model *from scratch* (utilizing 100k test-clean Pseudo-Labeling on the STL-10 dataset). We provide this for absolute transparency regarding our teacher weights.
- `test.py`: Performs model inference on test subset evaluation, reporting the final classification accuracy as mandated by the APEIREON rulebook.
- `requirements.txt`: Exceedingly minimal list of dependencies required for verification.

---

## 🛠 Model Compression Techniques

1. **Depthwise Separable Bottlenecks**: De-couples spatial and channel convolutions, drastically reducing structural parameters by ~8x compared to a native CNN.
2. **Feature Manifold Alignment**: The width/depth topology (**`[32, 64, 128, 256]`** with depths **`[2, 2, 2, 2]`**) was empirically engineered to downsample STL-10 images cleanly without excess memory buffers wasting space parameters.
3. **ResNet-50 Knowledge Distillation**: The student model fundamentally lacks the capacity to map the complex STL-10 boundaries independently. We trained our ResNet-50 Teacher model completely *from scratch* (utilizing the STL-10 100k Unlabeled dataset for Pseudo-labeling; zero ImageNet weights). 
   - **Phase 1**: We completed an initial 200-epoch `teacher_train.py` run (50-epoch labeled Burn-In + 150-epoch Mastery at 95% pseudo-label strictness), achieving **84.88%** validation accuracy.
   - **Phase 2 (Restart)**: To squeeze out the absolute maximum accuracy required to train the tiny student, we initiated a 100-epoch warm restart using the Phase 1 weights, aggressively lowering strictness to capture harder features. We completed the final checkpoints with 98% strictness, maximizing our Teacher at an incredible **87.96%** Final Accuracy.
   - We ultimately distilled this "dark knowledge" through soft-targets (`T=4.0`, `Alpha=0.9`) to force the tiny student model to mimic its massive internal topological understandings.
4. **FP16 Serialization**: At the conclusion of training, the raw `state_dict()` weights are natively type-casted down into 16-bit half-precision floating points. This universally halves the raw output `.pth` size on disk automatically (Final Size: **326 KB**), with absolutely 0% impact on inference accuracy.

---

## 🚀 Execution & Verification

### 1. Training the Teacher Model
To reproduce the teacher weights from scratch (requires ~105k images):
```bash
python teacher_train.py --data ./data --out ./teacher_best.pth --epochs 200
```

### 2. Training the Student Model (Distillation)
To reproduce the student weights using Knowledge Distillation from our teacher:
```bash
python train.py --dataset-path ./data --teacher-path ./teacher_final.pth --model-path ./student_final.pth --epochs 100
```
*(Note: `train.py` handles the STL-10 dataset downloading automatically.)*

### 4. Research & Development (Architecture Search)
You can now dynamically override the model architecture via the command line to test different model sizes:
```bash
# Example: Re-training with a 4-stage "Ultra-Tiny" configuration
python train.py --widths 16 32 64 128 --depths 1 1 1 1 --dataset-path ./data
```
*(Defaults to the winning `[32, 64, 128, 256]` and `[2, 2, 2, 2]` if omitted.)*

### 3. Verification (Accuracy Check)
To verify the accuracy of the generated `.pth` file on the unseen `test` split:
```bash
python test.py --dataset-path ./data --model-path ./student_final.pth
```

---

## ⚠️ Hardware & Performance Disclaimer

While the training scripts are technically universal and cross-platform (CUDA/MPS/CPU), the pipeline has been **specifically optimized to exploit Kaggle T4 x2 GPU environments**. 

*   **Benchmarked Speed**: On Kaggle, training the student model requires approximately **81 seconds per epoch** after the initial RAM-caching of the 105k images.
*   **Other Hardware**: Systems without high-speed NVMe/RAM bandwidth or multi-GPU support may encounter significantly longer epoch times. If running on a local machine with limited VRAM, consider disabling the RAM-caching feature in `train.py`.

---

## 💾 Resuming & Checkpoints

Both training scripts are designed to be "unkillable" and will save their state every epoch.

### 1. Automatic Resumption
If a training run is interrupted, simply run the same command again. Both `train.py` and `teacher_train.py` will look for their respective checkpoint files (`student_checkpoint.pth` and `teacher_checkpoint.pth`) and resume exactly from the last completed epoch.

### 2. Warm Starting (Teacher Only)
If you have a trained model but want to start a fresh 300-epoch run using those weights as a starting point (e.g., for refined pseudo-labeling):
```bash
python teacher_train.py --weights teacher_best.pth --epochs 300
```
This will start from **Epoch 1** but with the "intelligence" of your previous model already loaded.
