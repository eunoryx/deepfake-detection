# Deepfake detection in facial videos using TALL++

A PyTorch-based model for video frame classification using a **ResNet18 backbone** enhanced with a **Graph Reasoning Block (GRB)** for patch-level feature interactions and semantic consistency.

---

## Features

- ResNet18 backbone for feature extraction

- Graph Reasoning Block for patch-level attention

- Semantic consistency loss to align features

- Weighted loss to handle class imbalance

- Gradient accumulation for memory-efficient training

---

## Requirements

- Python 3.8+

- PyTorch 2.0+

- torchvision

- scikit-learn

- numpy

- matplotlib

Install dependencies:

```bash
pip install torch torchvision scikit-learn numpy matplotlib
```

## Usage

Train the model:

```bash
python  train_resnet.py  --root_dir  path/to/dataset  --img_size  224
```

- **--root_dir:** Path to dataset root directory
- **--img_size:** Image size (default 224)

After training:  
-Best model saved as best_resnet_model.pth  
-Training plots saved as resnet_training_history.png  
-Evaluation metrics saved as resnet_results.txt

## Dataset

-Input: Video frames  
-Labels: Binary (0 = real, 1 = fake)  
-Preprocessing: Resize to 224x224, normalize with ImageNet statistics
