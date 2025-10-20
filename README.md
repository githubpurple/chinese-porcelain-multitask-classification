 # Multi-Task Deep Learning for Chinese Porcelain Classification

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]
[![PyTorch](https://img.shields.io/badge/pytorch-1.12+-red.svg)]
[![License](https://img.shields.io/badge/license-MIT-green.svg)]

Official implementation of "Multi-Task Deep Learning for Chinese Porcelain Classification" (paper, 2025)

## Features
- Multi-task classification (Dynasty, Kiln, Glaze, Type)
- Transfer learning with MobileNetV2/ResNet/VGG16/InceptionV3
- Gradient-based interpretability analysis

## Quick Start
```python
from models import MobileNetMultiTask
model = MobileNetMultiTask.from_pretrained('path/to/weights')
predictions = model.predict(image_path)
```

## Citation
If you use this code, please cite:
```bibtex
@article{ling2025multi,
  title={Multi-task Learning for Identification of Porcelain in Song and Yuan Dynasties},
  author={Ling, Ziyao and Delnevo, Giovanni and Salomoni, Paola and Mirri, Silvia},
  journal={arXiv preprint arXiv:2503.14231},
  year={2025}
}
```
