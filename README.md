# Zendo test

## Overview
This repository provides the dataset and source code used in our study on [your task, e.g., anomaly detection on EM images].

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
data/
├── train/
```

## Data Access

All data is included in this repository under the `data/` directory.

If you are using a subset or need preprocessing, refer to:

***scripts/preprocess.py***

---

## Code Description

This repository includes Python code for training and inference.

### Main Files
- `train.py` – training pipeline
- `infer.py` – inference script
- `model.py` – model definition

---

## Installation

```bash
pip install -r requirements.txt
```
---

### Usage

#### Training
```shell
python src/vision/train.py --data_path data/train/
```

#### Inference
```shell
python src/vision/infer.py --input examples/demo.jpg
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

### Contact

For questions, please contact: [xxx@xxx.com]