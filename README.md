# Infineon PSOC Edge AI with NVIDIA TAO Toolkit

![NVIDIA TAO](https://img.shields.io/badge/NVIDIA-TAO_Toolkit-76B900?style=flat-square&logo=nvidia)
![Infineon](https://img.shields.io/badge/Infineon-PSOC_Edge-0058CC?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-FF6F00?style=flat-square&logo=tensorflow)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue?style=flat-square&logo=python)

A comprehensive toolkit for training, optimizing, and deploying AI models on Infineon PSOC Edge devices using NVIDIA's Train Adapt Optimize (TAO) Toolkit. This repository provides end-to-end workflows for building efficient computer vision applications that run on resource-constrained edge devices.

## Repository Structure

This repository contains two main workflows:

- **[object_detection_detectnet_v2](./object_detection_detectnet_v2/)**: Complete pipeline for training custom object detection models using NVIDIA's DetectNet_v2 architecture and deploying to Infineon PSOC Edge devices.

- **[peoplenet_tao](./peoplenet_tao/)**: Streamlined workflow for acquiring and converting NVIDIA's pre-trained PeopleNet models for deployment on Infineon hardware.
  - **[peoplenet_tflite_demo](./peoplenet_tao/peoplenet_tflite_demo/)**: A Python implementation for running the converted models on edge devices with TensorFlow Lite.

## What is Infineon PSOC Edge?

[Infineon PSOC Edge](https://www.infineon.com/cms/en/product/promopages/next-generation-mcu/) is a family of high-performance, low-power microcontrollers designed specifically for edge computing and IoT applications. Key features include:

- **Arm® Cortex®-M55 processors** with optional Arm® Ethos™-U55 NPU (Neural Processing Unit)
- **Ultra-low power consumption** for battery-powered applications
- **Enhanced security features** for IoT device protection
- **Integrated peripherals** for sensor interfaces and connectivity
- **Memory configurations** optimized for AI/ML workloads

PSOC Edge devices are ideal for applications requiring on-device AI inference like smart sensors, predictive maintenance, computer vision, audio processing, and industrial automation.

## NVIDIA TAO Toolkit: AI for the Edge

[NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit) (Train, Adapt, Optimize) is a low-code AI toolkit that simplifies and accelerates the development of production-quality AI models for edge deployment. TAO enables developers to:

- **Train** models using transfer learning from NVIDIA's pre-trained models
- **Adapt** models to specific use cases with custom datasets
- **Optimize** models for deployment on resource-constrained edge devices

### Why TAO for Edge Deployment?

Traditional deep learning models are often too large and computationally intensive for edge devices. TAO addresses this challenge through:

1. **Pruning**: Systematically removes redundant parameters from models
2. **Quantization**: Reduces precision of weights from FP32 to INT8/FP16
3. **Optimization**: Architecture-specific optimizations for efficient inference
4. **Pre-trained Models**: High-quality starting points that reduce training data requirements

These optimizations can reduce model size by 60-90% and increase inference speed by 2-5x, making advanced AI capabilities possible on edge devices like Infineon PSOC Edge.

## Getting Started

### Prerequisites

- **Hardware**:
  - NVIDIA GPU with at least 8GB memory (for training/optimization)
  - 16GB+ system RAM
  - 50GB+ free disk space

- **Software**:
  - NVIDIA drivers (455.0+)
  - Docker CE (19.03.5+)
  - Python 3.7-3.9
  - NGC account (for accessing NVIDIA's pre-trained models)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Infineon/tao-psoc-edge.git
   cd tao-psoc-edge
   ```

2. **Choose your workflow**:
   - For training custom object detection models:
     ```bash
     cd object_detection_detectnet_v2
     jupyter notebook DetectNet_v2_for_Infineon_PSOC_EDGE.ipynb
     ```
   
   - For using pre-trained PeopleNet models:
     ```bash
     cd peoplenet_tao
     jupyter notebook PeopleNet_Conversion_Notebook.ipynb
     ```

3. **Follow the notebook instructions** to train/optimize your models

4. **Deploy and run** the optimized models using the provided Python implementation in the `peoplenet_tflite_demo` directory

## Workflow Overview

### Training Custom Models with DetectNet_v2

The `object_detection_detectnet_v2` workflow provides a complete pipeline for:

1. Setting up the TAO environment
2. Preparing your custom dataset
3. Training DetectNet_v2 models
4. Pruning and quantizing for edge deployment
5. Converting to ONNX format
6. Deploying to Infineon PSOC Edge devices

Refer to the [DetectNet_v2 README](./object_detection_detectnet_v2/README.md) for detailed instructions.

### Using Pre-trained PeopleNet Models

The `peoplenet_tao` workflow offers a streamlined approach for:

1. Downloading NVIDIA's pre-trained PeopleNet models
2. Converting models using Infineon's tooling
3. Deploying to Infineon hardware with Ethos-U acceleration

See the [PeopleNet README](./peoplenet_tao/README.md) for step-by-step guidance.

### Running Models on Edge Devices

The `peoplenet_tflite_demo` provides a Python implementation for:

1. Loading converted models with TensorFlow Lite
2. Processing input from cameras or video files
3. Running inference with hardware acceleration
4. Visualizing detection results

Check the [TFLite Demo README](./peoplenet_tao/peoplenet_tflite_demo/README.md) for usage details.

## Edge Deployment Performance

The optimized models achieve impressive performance metrics on Infineon PSOC Edge devices:

| Model | Size Reduction | Inference Speed | 
|-------|----------------|-----------------|
| DetectNet_v2 (Pruned + INT8) | 60-80% | 5-15 FPS | 
| PeopleNet (Pruned + INT8) | 70-85% | 8-20 FPS | 

*Note: Performance varies based on input resolution, model configuration, and specific PSOC Edge device used.*

## Development Workflow

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Train/Download│     │   Optimize    │     │    Deploy     │
│   TAO Model   │────▶│  for Ethos-U  │────▶│  to PSOC Edge │
└───────────────┘     └───────────────┘     └───────────────┘
       │                      │                     │
       ▼                      ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Pre-trained  │     │    Pruning    │     │   TFLite or   │
│    Models     │     │ Quantization  │     │  ONNX Runtime │
└───────────────┘     └───────────────┘     └───────────────┘
```

## Use Cases

- **Smart retail**: People counting, queue management
- **Industrial safety**: PPE detection, restricted area monitoring
- **Smart buildings**: Occupancy detection, energy management
- **Security applications**: Intrusion detection, perimeter monitoring
- **Automotive**: Passenger detection, driver monitoring

## Contributing

Contributions to improve the tooling and workflows are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgements

- NVIDIA for the TAO Toolkit and pre-trained models
- Arm for the Ethos-U NPU architecture
- TensorFlow team for TensorFlow Lite
- ONNX community for the runtime and tooling

## References

- [NVIDIA TAO Toolkit Documentation](https://docs.nvidia.com/tao/tao-toolkit/index.html)
- [Infineon PSOC Edge Documentation](https://www.infineon.com/cms/en/product/promopages/next-generation-mcu/)
- [Arm Ethos-U NPU](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u55)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)