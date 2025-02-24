# Infineon PSOC&trade; Edge AI with NVIDIA TAO Toolkit

![NVIDIA TAO](https://img.shields.io/badge/NVIDIA-TAO_Toolkit-76B900?style=flat-square&logo=nvidia)
![Infineon](https://img.shields.io/badge/Infineon-PSOC_Edge-0058CC?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-FF6F00?style=flat-square&logo=tensorflow)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue?style=flat-square&logo=python)

A comprehensive toolkit for training, optimizing, and deploying AI models on Infineon PSOC&trade; Edge MCU devices using NVIDIA's Train Adapt Optimize (TAO) Toolkit. This repository provides end-to-end workflows for building efficient computer vision applications that run on resource-constrained edge devices.

> **Note:** An NVIDIA GPU is required to run the notebooks in this repository.

## Repository structure

This repository contains two main workflows:

- **[object_detection_detectnet_v2](./object_detection_detectnet_v2/):** Complete pipeline for training custom object detection models using NVIDIA's DetectNet_v2 architecture and deploying to Infineon PSOC&trade; Edge devices.

- **[peoplenet_tao](./peoplenet_tao/):** Streamlined workflow for acquiring and converting NVIDIA's pre-trained PeopleNet models for deployment on Infineon PSOC&trade; Edge devices.

  - **[peoplenet_tflite_demo](./peoplenet_tao/peoplenet_tflite_demo/):** A Python implementation for running the converted models on edge devices with TensorFlow Lite.


## What is Infineon PSOC&trade; Edge MCU?

[Infineon PSOC&trade; Edge MCU](https://www.infineon.com/cms/en/product/promopages/next-generation-mcu/) is a family of high-performance, low-power microcontrollers designed specifically for edge computing and IoT applications. Key features include:

- **Arm&reg; Cortex&reg;-M55 processors** with optional Arm&reg; Ethos-U55 Neural Processing Unit (NPU)

- **Ultra-low power consumption** for battery-powered applications

- **Enhanced security features** for IoT device protection

- **Integrated peripherals** for sensor interfaces and connectivity

- **Memory configurations** optimized for AI/ML workloads

PSOC&trade; Edge MCU devices are ideal for applications requiring on-device AI inference like smart sensors, predictive maintenance, computer vision, audio processing, and industrial automation.


## NVIDIA TAO Toolkit: AI for the edge

[NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit) (Train, Adapt, Optimize) is a low-code AI toolkit that simplifies and accelerates the development of production-quality AI models for edge deployment. TAO enables developers to:

- **Train** models using transfer learning from NVIDIA's pre-trained models

- **Adapt** models to specific use cases with custom datasets

- **Optimize** models for deployment on resource-constrained edge devices


### Why TAO for edge deployment?

Traditional deep learning models are often too large and computationally intensive for edge devices. TAO addresses this challenge through:

- **Pruning:** Systematically removes redundant parameters from models

- **Quantization:** Reduces precision of weights from FP32 to INT8/FP16

- **Optimization:** Architecture-specific optimizations for efficient inference

- **Pre-trained models:** High-quality starting points that reduce training data requirements

These optimizations can reduce the model size by 60-90% and increase inference speed by 2-5x, making advanced AI capabilities possible on edge devices like Infineon PSOC&trade; Edge MCU.


## Getting started

### Prerequisites

- **Hardware:**
  - NVIDIA GPU with at least 8 GB memory (for training/optimization)
  - 16 GB+ system RAM
  - 50 GB+ free disk space

- **Software:**
  - NVIDIA drivers (455.0+)
  - Docker CE (19.03.5+)
  - Python 3.7-3.9
  - NGC account (for accessing NVIDIA's pre-trained models)


### Quick start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Infineon/psoc-edge-nvidia-tao.git
   cd psoc-edge-nvidia-tao
   ```

2. **Choose your workflow:**
   - For training custom object detection models:
     ```bash
     cd object_detection_detectnet_v2
     jupyter notebook DetectNet_v2_for_Infineon_PSOC_EDGE.ipynb
     ```
   
   - For using pre-trained PeopleNet models:
     ```bash
     cd peoplenet_tao
     jupyter notebook PeopleNet.ipynb
     ```

3. **Follow the notebook instructions** to train/optimize your models

4. **Deploy and run** the optimized models using the provided Python implementation in the *peoplenet_tao/peoplenet_tflite_demo* directory


## Workflow overview

### Training custom models with DetectNet_v2

The `object_detection_detectnet_v2` workflow provides a complete pipeline for:

- Setting up the TAO environment
- Preparing your custom dataset
- Training DetectNet_v2 models
- Pruning and quantizing for edge deployment
- Converting to ONNX format
- Deploying to Infineon PSOC&trade; Edge MCU devices

See the [DetectNet_v2 README](./object_detection_detectnet_v2/README.md) for detailed instructions.


### Using pre-trained PeopleNet models

The `peoplenet_tao` workflow offers a streamlined approach for:

- Downloading NVIDIA's pre-trained PeopleNet models

- Converting models using Infineon's tooling

- Deploying to Infineon hardware with Ethos-U acceleration

See the [PeopleNet README](./peoplenet_tao/README.md) for step-by-step guidance.


### Running models on edge devices

The `peoplenet_tflite_demo` provides a Python implementation for:

- Loading converted models with TensorFlow Lite

- Processing inputs from cameras or video files

- Running inference with hardware acceleration

- Visualizing the detection results

Check the [TFLite demo README](./peoplenet_tao/peoplenet_tflite_demo/README.md) for usage details.


## Edge deployment performance

The optimized models achieve impressive performance metrics on Infineon PSOC&trade; Edge MCU devices:

Model | Size reduction | Inference speed
-------|----------------|-----------------
DetectNet_v2 (Pruned + INT8) | 60-80% | 5-15 FPS
PeopleNet (Pruned + INT8) | 70-85% | 8-20 FPS

<br> 

> **Note:** Performance varies based on input resolution, model configuration, memory placement, and the specific PSOC&trade; Edge MCU device used.

## Development workflow

```
┌───────────────┐      ┌───────────────┐      ┌────────────────┐
│ Train/download│      │   Optimize    │      │    Deploy      │
│   TAO model   │ ───▶│  for Ethos-U   │────▶│  to PSOC Edge  │
└───────────────┘      └───────────────┘      └────────────────┘
       │                      │                     │
       ▼                      ▼                     ▼
┌───────────────┐     ┌───────────────┐      ┌───────────────┐
│  Pre-trained  │     │    Pruning    │      │   TFLite or   │
│    models     │     │ quantization  │      │  ONNX runtime │
└───────────────┘     └───────────────┘      └───────────────┘
```

## Use cases

- **Smart retail:** People counting, queue management

- **Industrial safety:** Personal protective equipment detection, restricted area monitoring

- **Smart buildings:** Occupancy detection, energy management

- **Security applications:** Intrusion detection, perimeter monitoring

- **Automotive:** Passenger detection, driver monitoring


## License

This project is licensed under Infineon License - see the [LICENSE](./LICENSE.txt) file for details.

By pulling and using the NVIDIA TAO Toolkit, you accept the terms and conditions of these [licenses](https://www.nvidia.com/en-us/data-center/products/nvidia-ai-enterprise/eula/).

## Acknowledgements

- NVIDIA for the TAO Toolkit and pre-trained models
- Arm&reg; for the Ethos-U NPU architecture
- TensorFlow team for TensorFlow Lite
- ONNX community for the runtime and tooling


## References

- [NVIDIA TAO Toolkit documentation](https://docs.nvidia.com/tao/tao-toolkit/index.html)

- [Infineon PSOC&trade; Edge MCU documentation](https://www.infineon.com/cms/en/product/promopages/next-generation-mcu/)

- [Arm&reg; Ethos-U NPU](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u55)

- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)