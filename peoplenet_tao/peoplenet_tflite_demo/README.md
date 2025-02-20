# People Detection System

![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8.0-orange)

A robust, high-performance people detection system that leverages NVIDIA TAO-trained models with TensorFlow Lite and Infineon tooling. The implementation supports hardware acceleration via Ethos-U NPU when available.

## Features

- **High-Performance Detection**: Optimized for real-time performance using ResNet-34 backbone
- **Multi-Class Support**: Detects people, bags, and faces with configurable thresholds
- **Hardware Acceleration**: Optional Ethos-U NPU support for edge devices
- **Post-Processing Pipeline**: Complete detection pipeline from raw outputs to visualized results
- **Non-Maximum Suppression**: Built-in NMS to filter overlapping detections
- **Rich Visualization**: Real-time display with configurable output options
- **Video Recording**: Save detection results to video files

## Architecture

The system consists of several key components:

1. **DetectNetV2 Model**: Main object detection network (ResNet34-based)
2. **NMS Model**: Non-Maximum Suppression to filter overlapping detections
3. **Post-Processing Model**: Converts raw model outputs to properly formatted bounding boxes
4. **People Detector**: Main application that orchestrates the detection pipeline

## Installation

### Prerequisites

- Python 3.7 or higher
- OpenCV
- TensorFlow 2.8 or TensorFlow Lite Runtime 2.8
- NVIDIA TAO Toolkit (for model training/conversion)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Infineon/tao-psoc-edge.git
   cd tao-psoc-edge/peoplenet_tao
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Use pre-trained models generated in `PeopleNet` notebook:

## Usage

### Basic Usage

Run the people detector with default settings:

```bash
python people_detector.py
```

### Advanced Options

```bash
python people_detector.py --camera 0 --model models/resnet34_peoplenet_int8_ex.tflite --nms-model models/nms_model.tflite
```

### Configuration Parameters

The application supports numerous configuration options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Path to DetectNetV2 model file | resnet34_peoplenet_int8_ex.tflite |
| `--nms-model` | Path to NMS model file | nms_model.tflite |
| `--camera` | Camera device index | 1 |
| `--width` | Camera capture width | 1920 |
| `--height` | Camera capture height | 1080 |
| `--display-scale` | Scale factor for display window | 1.0 |
| `--fullscreen` | Run in fullscreen mode | False |
| `--no-fps` | Hide FPS counter | False |
| `--min-height` | Minimum height for valid detections | 20 |
| `--person-threshold` | Score threshold for person detections | 0.4 |
| `--bag-threshold` | Score threshold for bag detections | 0.2 |
| `--face-threshold` | Score threshold for face detections | 0.2 |
| `--output` | Path to save output video | None |
| `--verbose` | Enable verbose logging | False |

### Building the NMS Model

For optimal performance, you can build a custom NMS model with parameters tailored to your use case:

```bash
python nms_builder.py --height 34 --width 60 --quantize --output models/custom_nms_model.tflite
```

NMS builder parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--height` | Height of the detection grid | 34 |
| `--width` | Width of the detection grid | 60 |
| `--output` | Output path for the TFLite model | nms_model.tflite |
| `--quantize` | Apply post-training quantization | False |
| `--verbose` | Enable verbose logging | False |

## Ethos-U NPU Acceleration

The system automatically detects and uses Ethos-U NPU acceleration when available. The required delegate library should be installed at `/usr/lib/libethosu_delegate.so` or specified via the environment.

## Model Information

### DetectNetV2

The primary model is based on NVIDIA's DetectNetV2 architecture with a ResNet-34 backbone, optimized using NVIDIA TAO toolkit and Infineon tools. The default model has been quantized to INT8 for optimal performance on edge devices.

Input size: 960×540 pixels
Grid size: 34×60 (H×W)
Classes: Person, Bag, Face

### Non-Maximum Suppression (NMS)

The NMS model filters overlapping detections based on IoU (Intersection over Union) and confidence thresholds. It's implemented as a separate TensorFlow Lite model for optimal performance.

## Development

### Project Structure

```
people-detection/
├── people_detector.py     # Main application
├── detect_net_v2_model.py # DetectNetV2 model wrapper
├── nms_model.py           # NMS model wrapper
├── post_processing_model.py # Post-processing utilities
├── nms_builder.py         # NMS model builder
├── requirements.txt       # Project dependencies
├── models/                # Pre-trained model files
│   ├── resnet34_peoplenet_int8_ex.tflite
│   └── nms_model.tflite
└── README.md              # This file
```

### Extending the System

#### Adding New Detection Classes

1. Modify the `DetectionConfig` class in `people_detector.py`
2. Add new class labels and corresponding thresholds
3. Update visualization colors

#### Custom Model Integration

To use a custom trained model:

1. Train your model using NVIDIA TAO toolkit
2. Export to TensorFlow Lite format
3. Update the grid size and other parameters in the configuration

## Performance Optimization

For best performance:

1. Use quantized INT8 models
2. Enable Ethos-U acceleration when available
3. Adjust the detection thresholds for your specific use case
4. Consider reducing the camera resolution for higher FPS

## Troubleshooting

### Common Issues

1. **"Failed to import TensorFlow Lite"**:
   - Install either the full TensorFlow package or tflite-runtime

2. **"Could not open camera"**:
   - Verify the camera index is correct
   - Check camera permissions

3. **"Ethos-U delegate library not found"**:
   - Install the Ethos-U delegate library or disable NPU acceleration

4. **Low frame rate**:
   - Reduce camera resolution
   - Use quantized models
   - Enable hardware acceleration


## Acknowledgments

- NVIDIA for the TAO toolkit and DetectNetV2 architecture
- TensorFlow team for TensorFlow Lite