# Zendo test

## Overview
This repository provides the dataset and source code used in our study on anomaly detection on EM images.

---

## Dataset Description

The dataset consists of grayscale images in JPG and PNG format.

- Total samples: 200
- Normal samples: 180
- Abnormal samples: 20 (including false positives)
- Image format: JPG, PNG
- Resolution: [e.g., 512×512]

### Directory Structure
```shell
datasets/
├── train/
```

---

## Code Description

This repository includes Python code for training and inference.

### Main Files
- `train.py` – training pipeline
- `validate.py` – validate script
- `model_pool.py` – model_pool definition

---

## Installation

```bash
conda create -n cv python=3.12 -y
conda activate cv
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install opencv-python
```
---

### Usage

#### Training
```shell
python src/vision/train.py
```

#### Inference
```shell
python src/vision/infer.py
```

---
## Reproducibility

To ensure reproducibility:

- Validation threshold: 0.5
- Inference threshold: 0.70

Hardware: [optional, e.g., NVIDIA 4090, NVIDIA 4060ti]

---
## License
This project is licensed under the MIT License.

---

## Contact

For questions, please contact: [abcn@qq.com]