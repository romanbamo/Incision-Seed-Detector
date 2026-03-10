# Incision Seed Detector

This repository contains a Deep Learning model based on **EfficientNet-B1** designed to automatically detect the starting "seed" (x, y coordinates) for a **Region Growing** algorithm in surgical incision images.



## Project Overview
The goal is to automate the initialization of segmentation algorithms. Instead of manually picking a starting point, this CNN predicts the most likely center of a surgical incision to begin the region-growing process.

- **Architecture:** EfficientNet-B1 (Backbone) + Linear Regression Head.
- **Input:** RGB Image (resized to 240x240).
- **Output:** Normalized (x, y) coordinates of the incision seed.
- **Framework:** PyTorch.

## Repository Structure
* `src/model.py`: Model architecture definition.
* `src/dataset.py`: Custom PyTorch Dataset for medical images and coordinate labels.
* `src/train.py`: Two-phase training script (Transfer Learning + Fine-Tuning).
* `requirements.txt`: Necessary Python libraries.

## Getting Started

### Prerequisites
* Python 3.8+
* CUDA-enabled GPU (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/romanbamo/Incision-Seed-Detector.git](https://github.com/romanbamo/Incision-Seed-Detector.git)
   cd Incision-Seed-Detector
