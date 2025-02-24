# NVIDIA TAO DetectNet_v2 for Infineon PSOC EDGE Devices

![NVIDIA TAO](https://img.shields.io/badge/NVIDIA-TAO_Toolkit-76B900?style=flat-square&logo=nvidia)
![Infineon](https://img.shields.io/badge/Infineon-Tooling-0058CC?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square)

This repository contains a detailed Jupyter notebook that demonstrates how to train, optimize, and deploy object detection models from NVIDIA Train Adapt Optimize (TAO) to Infineon PSOC&trade; Edge MCU devices. The workflow showcases a complete end-to-end pipeline from model training to edge deployment, specifically optimized for resource-constrained IoT and edge devices.

> **Note:** An NVIDIA GPU is required to run this notebook.


## <details><summary>Table of contents</summary>

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Getting started](#getting-started)
- [Workflow steps](#workflow-steps)
  - [Environment setup](#environment-setup)
  - [Data preparation](#data-preparation)
  - [Model training](#model-training)
  - [Model optimization](#model-optimization)
  - [Model deployment](#model-deployment)
- [Performance considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Acknowledgements](#acknowledgements)
</details>


## Overview

This notebook demonstrates how to leverage the NVIDIA Train Adapt Optimize (TAO) Toolkit to build an efficient object detection model using DetectNet_v2 architecture and optimize it for deployment on Infineon PSOC&trade; Edge MCU devices. The workflow covers the entire process from data preparation to deployment, including critical optimization techniques, such as pruning and quantization-aware training.

The resulting models are specifically tailored for edge deployment, balancing accuracy with the resource constraints of edge devices. The integration with Infineon's software enables seamless deployment to PSOC&trade; Edge MCU devices, allowing developers to bring AI capabilities to IoT and embedded systems.

[Back to top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)


## Prerequisites

- **Hardware:**
  - NVIDIA GPU with at least 8 GB memory (recommended)
  - 16 GB+ system RAM
  - 50 GB+ free disk space

- **Software:**
  - NVIDIA driver 455.0 or later
  - Docker CE 19.03.5 or later
  - Docker API 1.40+
  - nvidia-container-toolkit 1.3.0-1 or later
  - nvidia-container-runtime 3.4.0-1 or later
  - nvidia-docker2 2.5.0-1 or later
  - Python 3.7-3.10
  - Jupyter Notebook or JupyterLab

- **Accounts:**
  - NGC account for accessing NVIDIA's model registry

[Back to top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)


## Getting started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Infineon/psoc-edge-nvidia-tao.git
   cd psoc-edge-nvidia-tao
   ```

2. **Install the TAO launcher:**
   ```bash
   pip install nvidia-tao
   ```

3. **Set up Docker access to NGC registry:**
   ```bash
   docker login nvcr.io
   ```
   When prompted, use `$oauthtoken` as username and your NGC API key as password.

4. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

5. **Open the notebook** `DetectNet_v2_for_Infineon_PSOC_EDGE.ipynb`

[Back to top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)


## Workflow steps

The notebook guides you through the following steps:


### Environment setup

This section helps you to configure the environment variables, directories, and TAO toolkit for the workflow. It includes setting up:
- Environment variables for data and experiment directories
- Docker mappings for persistent storage
- TAO launcher installation and verification

**Time estimate:** 10 to 15 minutes

[Back to top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)


### Data preparation

You will download and prepare the COCO dataset for training:
- Downloading the dataset
- Verifying the dataset structure
- Converting the dataset to TFRecords format
- Downloading a pre-trained ResNet-18 model from NVIDIA's NGC registry

**Time estimate:** 30 to 45 minutes (dependent on internet speed)

[Back to top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)


### Model training

This section provides:
- Configuring training parameters via specification files
- Training the DetectNet_v2 model on the COCO dataset
- Evaluating the trained model's performance
- Visualizing inference results

**Time estimate:** 4 to 8 hours (dependent on GPU performance)

[Back to top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)


### Model optimization

Learn how to optimize the model for edge deployment:
- Pruning the model to reduce size and computational requirements
- Retraining the pruned model to recover accuracy
- Applying Quantization-Aware Training (QAT) for INT8 inference
- Evaluating the optimized model's performance

**Time estimate:** 2 to 4 hours

[Back to top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)


### Model deployment

The final steps to prepare and deploy your model:
- Exporting the optimized model to ONNX format
- Installing the Infineon software dependencies
- Converting the model for Infineon PSOC&trade; Edge MCU using Infineon tooling (TFlite format)
- Generating deployment-ready artifacts

**Time estimate:** 30 to 45 minutes

[Back to top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)


## Performance considerations

- **Training time:** Reduced by using more powerful GPUs or multiple GPUs
- **Pruning threshold:** Affects the trade-off between model size and accuracy
- **Quantization:** Reduces model size but may slightly impact accuracy
- **Edge performance:** Depends on the specific Infineon PSOC&trade; Edge MCU device configuration

Typical performance metrics for optimized models on PSOC&trade; Edge MCU devices:
- **Model size reduction:** 60% to 80% (after pruning and quantization)
- **Inference speed:**  5 FPS to 15 FPS (depending on input resolution)
- **Power consumption:** 50 mW to 200 mW during inference

[Back to top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)


## Troubleshooting

### Common issues

- **Docker permissions:**
   - Ensure that you are in the docker group (`sudo usermod -aG docker $USER`)

- **NGC authentication:**
   - Verify that your NGC API key is correct and not expired

- **Out of memory during training:**
   - Reduce batch size in the training specification file

- **Conversion failures:**
   - Check input shapes and ensure the model architecture is compatible with the Ethos-U55 NPU

For additional support, open an issue on the GitHub repository.

[Back to top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)


## References

- [NVIDIA TAO Toolkit documentation](https://docs.nvidia.com/tao/tao-toolkit/index.html)
- [Infineon PSOC&trade; Edge MCU documentation](https://www.infineon.com/cms/en/product/promopages/next-generation-mcu/)
- [DetectNet_v2 architecture](https://docs.nvidia.com/tao/tao-toolkit/text/cv_finetuning/tensorflow_1/object_detection/detectnet_v2.html)
- [Model optimization techniques](https://docs.nvidia.com/tao/tao-toolkit/text/model_optimization.html)

[Back to top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)


## Acknowledgements

This project leverages NVIDIA's Train Adapt Optimize (TAO) Toolkit that provides a streamlined workflow for training, adapting, and optimizing deep learning models. Acknowledging NVIDIA's contributions to the field of AI and edge computing through tools like TAO that make the deployment of AI on edge devices more accessible.

The workflow demonstrated in this notebook is based on the COCO dataset and acknowledge the contribution of the COCO consortium in providing this valuable resource for the computer vision community.

[Back to top](#nvidia-tao-detectnet_v2-for-infineon-psoc-edge-devices)