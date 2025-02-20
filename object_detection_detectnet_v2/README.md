# NVIDIA TAO DetectNet_v2 for Infineon PSOC EDGE Devices

![NVIDIA TAO](https://img.shields.io/badge/NVIDIA-TAO_Toolkit-76B900?style=flat-square&logo=nvidia)
![Infineon](https://img.shields.io/badge/Infineon-Tooling-0058CC?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square)

This repository contains a detailed Jupyter notebook that demonstrates how to train, optimize, and deploy object detection models from NVIDIA TAO (Train Adapt Optimize) to Infineon PSOC EDGE devices. The workflow showcases a complete end-to-end pipeline from model training to edge deployment, specifically optimized for resource-constrained IoT and edge devices.

**Note: An NVIDIA GPU is required to run this notebook.**

## <details><summary>Table of Contents</summary>

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Workflow Steps](#workflow-steps)
  - [Environment Setup](#environment-setup)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Model Optimization](#model-optimization)
  - [Model Deployment](#model-deployment)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Acknowledgements](#acknowledgements)
</details>

## Overview

This notebook demonstrates how to leverage NVIDIA TAO (Train Adapt Optimize) Toolkit to build an efficient object detection model using DetectNet_v2 architecture, and then optimize it for deployment on Infineon PSOC EDGE devices. The workflow covers the entire process from data preparation to deployment, including critical optimization techniques such as pruning and quantization-aware training.

The resulting models are specifically tailored for edge deployment, balancing accuracy with the resource constraints of edge devices. The integration with Infineon's toolchain enables seamless deployment to PSOC EDGE devices, allowing developers to bring AI capabilities to IoT and embedded systems.

[Back to Top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)

## Prerequisites

- **Hardware**:
  - NVIDIA GPU with at least 8GB memory (recommended)
  - 16GB+ system RAM
  - 50GB+ free disk space

- **Software**:
  - NVIDIA driver 455.0 or later
  - Docker CE 19.03.5 or later
  - Docker API 1.40+
  - nvidia-container-toolkit 1.3.0-1 or later
  - nvidia-container-runtime 3.4.0-1 or later
  - nvidia-docker2 2.5.0-1 or later
  - Python 3.7-3.10
  - Jupyter Notebook or JupyterLab

- **Accounts**:
  - NGC account for accessing NVIDIA's model registry

[Back to Top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Infineon/tao-psoc-edge.git
   cd tao-psoc-edge
   ```

2. **Install the TAO launcher**:
   ```bash
   pip install nvidia-tao
   ```

3. **Set up Docker access to NGC registry**:
   ```bash
   docker login nvcr.io
   ```
   When prompted, use `$oauthtoken` as username and your NGC API key as password.

4. **Launch the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

5. **Open the notebook** `DetectNet_v2_for_Infineon_PSOC_EDGE.ipynb`

[Back to Top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)

## Workflow Steps

The notebook guides you through the following major steps:

### Environment Setup

This section helps you configure the environment variables, directories, and TAO toolkit for the workflow. It includes setting up:
- Environment variables for data and experiment directories
- Docker mappings for persistent storage
- TAO launcher installation and verification

**Time estimate**: 10-15 minutes

[Back to Top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)

### Data Preparation

Here you'll download and prepare the COCO dataset for training:
- Downloading the dataset
- Verifying the dataset structure
- Converting the dataset to TFRecords format
- Downloading a pre-trained ResNet-18 model from NVIDIA's NGC registry

**Time estimate**: 30-45 minutes (dependent on internet speed)

[Back to Top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)

### Model Training

This section walks through:
- Configuring training parameters via specification files
- Training the DetectNet_v2 model on the COCO dataset
- Evaluating the trained model's performance
- Visualizing inference results

**Time estimate**: 4-8 hours (dependent on GPU performance)

[Back to Top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)

### Model Optimization

Learn how to optimize the model for edge deployment:
- Pruning the model to reduce size and computational requirements
- Retraining the pruned model to recover accuracy
- Applying Quantization-Aware Training (QAT) for INT8 inference
- Evaluating the optimized model's performance

**Time estimate**: 2-4 hours

[Back to Top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)

### Model Deployment

The final steps to prepare and deploy your model:
- Exporting the optimized model to ONNX format
- Installing the Infineon toolchain dependencies
- Converting the model for Infineon PSOC EDGE using IFX Tooling
- Generating deployment-ready artifacts

**Time estimate**: 30-45 minutes

[Back to Top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)

## Performance Considerations

- **Training time** can be reduced by using more powerful GPUs or multiple GPUs
- **Pruning threshold** affects the trade-off between model size and accuracy
- **Quantization** further reduces model size but may slightly impact accuracy
- **Edge performance** depends on the specific Infineon PSOC EDGE device configuration

Typical performance metrics for optimized models on PSOC EDGE devices:
- Model size reduction: 60-80% (after pruning and quantization)
- Inference speed: 5-15 FPS (depending on input resolution)
- Power consumption: 50-200 mW during inference

[Back to Top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)

## Troubleshooting

### Common Issues

1. **Docker permissions**:
   - Solution: Ensure your user is in the docker group (`sudo usermod -aG docker $USER`)

2. **NGC authentication**:
   - Solution: Verify your NGC API key is correct and not expired

3. **Out of memory during training**:
   - Solution: Reduce batch size in the training specification file

4. **Conversion failures**:
   - Solution: Check input shapes and ensure the model architecture is compatible with the Ethos-U55 NPU

For additional support, please open an issue on the GitHub repository.

[Back to Top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)

## References

- [NVIDIA TAO Toolkit Documentation](https://docs.nvidia.com/tao/tao-toolkit/index.html)
- [Infineon PSOC EDGE Documentation](https://www.infineon.com/cms/en/product/promopages/next-generation-mcu/)
- [DetectNet_v2 Architecture](https://docs.nvidia.com/tao/tao-toolkit/text/cv_finetuning/tensorflow_1/object_detection/detectnet_v2.html)
- [Model Optimization Techniques](https://docs.nvidia.com/tao/tao-toolkit/text/model_optimization.html)

[Back to Top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)

## Acknowledgements

This project leverages NVIDIA's Train Adapt Optimize (TAO) Toolkit, which provides a streamlined workflow for training, adapting, and optimizing deep learning models. We acknowledge NVIDIA's contributions to the field of AI and edge computing through tools like TAO that make the deployment of AI on edge devices more accessible.

The workflow demonstrated in this notebook is based on the COCO dataset, and we acknowledge the contribution of the COCO consortium in providing this valuable resource for the computer vision community.

[Back to Top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)